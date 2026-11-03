"""透明的单动作控制循环。

数据集留在环境侧；Provider 只接收 build_view 或专用核对请求。
每次动作提交连同消息、状态和预测写进同一个 SQLite 事务。
"""
from __future__ import annotations

from copy import deepcopy
import json
import random
import re
import time

from pydantic import ValidationError

from .data import fingerprint
from .providers import prompt_text, response_schema
from .retrieval import search
from .schemas import Relation, SupportCheck, action_schema, parse_action


class StopEpisode(Exception):
    """预算、上下文或无法恢复的服务错误，需要保存部分结果。"""


class UncertainCall(Exception):
    """远端可能已执行，默认暂停，不能自动重复计费。"""


class ToolError(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message


def new_role():
    return {"visible": {}, "candidates": {}, "committed": {}, "relations": {},
            "gaps": {}, "delivered_messages": [], "last_tool_result": None}


def new_state(sample):
    task = sample["task"]
    roles = ("archivist", "interpreter") if task == "T4" else ("solo",)
    return {"sample_id": sample["sample_id"], "doc_id": sample["doc_id"], "task": task,
            "status": "RUNNING", "termination_reason": None,
            "role_states": {role: new_role() for role in roles},
            "current_turn": 1, "actor": roles[0], "scheduler_round": 1,
            "activation_steps": 0, "turn_steps": 0, "activation_started": False,
            "step": 0, "event_seq": 0, "invalid_streak": 0, "messages": [],
            "turns": [], "turn_previous_ids": [], "access_violations": [],
            "call_keys": {}, "counted_calls": [], "initialized": False,
            "usage": {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "usage_complete": True}, "final": None}


class Engine:
    def __init__(self, sample, config, provider, store, log=print, retry_uncertain=False):
        self.sample, self.config, self.provider, self.store = sample, config, provider, store
        self.log, self.retry_uncertain = log, retry_uncertain
        self.state = store.get_state(sample["sample_id"]) or new_state(sample)
        self.evidence = {e["evidence_id"]: {key: e[key] for key in ("evidence_id", "text", "start", "end")}
                         for e in sample["document"]["evidence"]}
        self.random = random.Random(config["runtime"]["seed"])

    @property
    def role(self):
        return self.state["role_states"][self.state["actor"]]

    def reveal(self, ids, role, origin, message_id=None):
        for eid in ids:
            # 一个来源字段不覆盖此前来源；重复投递不重复增加权限。
            self.state["role_states"][role]["visible"].setdefault(eid, {
                **self.evidence[eid], "access_origin": origin,
                "received_message_id": message_id})

    def initialize(self):
        if self.state["initialized"]:
            return
        task = self.sample["task"]
        if task in ("T1", "T2"):
            self.reveal(self.evidence, "solo", "initial")
        elif task == "T3":
            self.reveal(self.sample["events"][0]["visible_evidence_ids"], "solo", "revealed")
        else:
            for role, ids in self.sample["roles"].items():
                self.reveal(ids, role, "initial")
        self.state["initialized"] = True
        self.checkpoint("INITIALIZED", {})

    def build_view(self):
        """白名单构造请求：T3/T4 根本不创建 document_text 字段。"""
        s, role = self.state, self.role
        question = (self.sample["events"][s["current_turn"] - 1]["question"]
                    if s["task"] == "T3" else self.sample["question"])
        view = {"episode_id": s["sample_id"], "task": s["task"], "role": s["actor"],
                "question": question, "work_title": self.sample["document"]["work_title"],
                "current_turn": s["current_turn"], "visible_evidence": list(role["visible"].values()),
                "current_candidates": list(role["candidates"].values()),
                "committed_claims": list(role["committed"].values()),
                "relations": list(role["relations"].values()), "gaps": list(role["gaps"].values()),
                "delivered_messages": deepcopy(role["delivered_messages"]),
                "remaining_budget": {
                    "actions": self.config["limits"]["max_actions_per_episode"] - s["step"],
                    "model_calls": self.config["limits"]["max_model_calls_per_episode"] - s["usage"]["model_calls"]},
                "last_tool_result": role["last_tool_result"]}
        if s["task"] in ("T1", "T2"):
            view["document_text"] = self.sample["document"]["text"]
        return view

    def checkpoint(self, kind, payload, prediction=None):
        s = self.state
        s["event_seq"] += 1
        event = {"sample_id": s["sample_id"], "event_seq": s["event_seq"],
                 "kind": kind, "role": s["actor"], "turn": s["current_turn"], **payload}
        self.store.save_state(s["sample_id"], s, event=event,
                              messages=s["messages"], prediction=prediction)

    def _usage(self, call_id, response):
        if call_id in self.state["counted_calls"]:
            return
        self.state["counted_calls"].append(call_id)
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if type(value) is int and value >= 0:
                previous = self.state["usage"][key]
                if previous is not None:
                    self.state["usage"][key] += value
            else:
                self.state["usage"][key] = None
                self.state["usage"]["usage_complete"] = False

    def call(self, stage, request, suffix=""):
        """先记调用意图再访问远端；崩溃恢复复用已保存的响应。"""
        limits, s = self.config["limits"], self.state
        system = prompt_text(stage) + "\n只输出 JSON。输出结构：\n" + json.dumps(response_schema(stage), ensure_ascii=False)
        request_chars = len(system) + len(json.dumps(request, ensure_ascii=False))
        if request_chars > limits["max_input_chars_per_request"]:
            raise StopEpisode("CONTEXT_LIMIT")
        for attempt in range(limits["max_api_retries"] + 1):
            logical = f'{s["step"]}:{stage}:{suffix}:{attempt}'
            call_id = s["call_keys"].get(logical)
            if call_id is None:
                if s["usage"]["model_calls"] >= limits["max_model_calls_per_episode"]:
                    raise StopEpisode("BUDGET_EXHAUSTED")
                tokens = s["usage"]["total_tokens"]
                if tokens is not None and tokens >= limits["max_reported_total_tokens_per_episode"]:
                    raise StopEpisode("TOKEN_BUDGET_EXHAUSTED")
                s["usage"]["model_calls"] += 1
                call_id = f'{s["sample_id"]}:{s["usage"]["model_calls"]:04d}'
                s["call_keys"][logical] = call_id
                self.store.save_state(s["sample_id"], s)
            existing = self.store.get_call(call_id)
            if existing and existing["status"] == "COMPLETED":
                response = existing["response"]
                self._usage(call_id, response)
                return response["content"]
            if existing and existing["status"] == "STARTED":
                s["usage"]["usage_complete"] = False
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    s["usage"][key] = None
                if not self.retry_uncertain:
                    raise UncertainCall(call_id)
                self.log(f'  [恢复] {call_id} 重试不确定调用，可能重复计费')
                self.store.fail_call(call_id, {"code": "UNCERTAIN_RETRY_AUTHORIZED"})
                # 新调用 ID；旧不确定调用也保留在实际尝试计数中。
                del s["call_keys"][logical]
                return self.call(stage, request, suffix)
            if existing and existing["status"] == "FAILED":
                if existing["error"].get("retryable") and attempt < limits["max_api_retries"]:
                    continue
                raise StopEpisode(existing["error"].get("code", "API_ERROR"))
            self.store.start_call(call_id, s["sample_id"], stage, request)
            self.log(f'  [模型] {s["actor"]} / {stage} / 调用 {s["usage"]["model_calls"]}')
            try:
                response = self.provider.complete(stage, request, call_id)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
                retryable |= type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectionError")
                # 不持久化异常原文，避免 SDK 的请求头或密钥进入日志。
                error = {"code": f'API_{status or type(exc).__name__}', "retryable": retryable}
                self.store.fail_call(call_id, error)
                s["usage"]["usage_complete"] = False
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    s["usage"][key] = None
                self.store.save_state(s["sample_id"], s)
                if retryable and attempt < limits["max_api_retries"]:
                    time.sleep(min(2 ** attempt + self.random.random(), 10))
                    continue
                raise StopEpisode(error["code"]) from None
            self.store.complete_call(call_id, response)
            self._usage(call_id, response)
            self.store.save_state(s["sample_id"], s)
            return response["content"]
        raise StopEpisode("API_RETRIES_EXHAUSTED")

    def structured(self, stage, request, parser, schema):
        for repair in range(self.config["limits"]["max_format_repairs_per_response"] + 1):
            text = self.call(stage, request, str(repair))
            try:
                return parser(text)
            except (ValueError, ValidationError, TypeError):
                # 不把错误响应再发送给其他角色；只给相同请求一次结构修复提示。
                request = {**request, "format_repair": {"instruction": "上次响应不符合结构；请重新输出一个合法 JSON 对象。", "schema": schema}}
        raise ToolError("INVALID_ACTION" if stage == "ACTION" else "INVALID_ARGUMENTS", "模型响应格式不合法")

    def require_visible(self, ids):
        if not set(ids) <= self.role["visible"].keys():
            # 不区分不存在、尚未公开或属于同伴，以免泄漏可用性。
            raise ToolError("ACCESS_DENIED", "请求包含不可访问的证据")

    def claims(self, ids):
        all_claims = {**self.role["candidates"], **self.role["committed"]}
        if not set(ids) <= all_claims.keys():
            raise ToolError("UNKNOWN_CANDIDATE", "主张 ID 不存在")
        return [all_claims[cid] for cid in ids]

    def prediction(self, args=None):
        role = self.state["role_states"].get("interpreter", self.state["role_states"].get("solo"))
        ids = args["claim_ids"] if args is not None else list(role["committed"])
        claims = [role["committed"][cid] for cid in ids]
        evidence_ids = sorted({eid for c in claims for eid in c["evidence_ids"]})
        # 未产生主张的暂停不能仅靠程序标记变成一份非空答卷。
        # 真正失败仍由状态计零；已有提交的部分答案继续明确标记来源。
        answer = (args["answer"] if args is not None else
                  ("【部分结果：程序汇总已有提交，非正常模型完成】\n" +
                   "\n".join(c["text"] + ''.join(f'[{eid}]' for eid in c["evidence_ids"]) for c in claims)
                   if claims else ""))
        relation_ids = args["relation_ids"] if args is not None else list(role["relations"])
        return {"claims": deepcopy(claims), "answer": answer, "evidence_ids": evidence_ids,
                "relations": [deepcopy(role["relations"][rid]) for rid in relation_ids],
                "unresolved_gaps": [deepcopy(g) for g in role["gaps"].values() if g["status"] != "RESOLVED"]}

    def envelope(self):
        s = self.state
        return {"schema_version": "1.0", "sample_id": s["sample_id"], "doc_id": s["doc_id"],
                "task": s["task"], "status": s["status"], "termination_reason": s["termination_reason"],
                "prediction": s["final"] or self.prediction(), "turns": deepcopy(s["turns"]),
                "messages": deepcopy(s["messages"]), "access_violations": deepcopy(s["access_violations"]),
                "usage": deepcopy(s["usage"])}

    def execute(self, action):
        name, args = action["action"], action["arguments"]
        role, s = self.role, self.state
        if name == "READ":
            self.require_visible(args["evidence_ids"])
            return {"evidence": [role["visible"][eid] for eid in args["evidence_ids"]]}
        if name == "SEARCH":
            k = min(args.get("top_k", self.config["retrieval"]["default_top_k"]), self.config["retrieval"]["max_top_k"])
            return {"evidence": search(args["query"], list(role["visible"].values()), k)}
        if name == "PLAN_GAPS":
            for gap in args["gaps"]:
                self.claims(gap["related_claim_ids"] + gap["resolution_claim_ids"])
                gid = gap["gap_id"]
                if gid and gid not in role["gaps"]:
                    raise ToolError("INVALID_ARGUMENTS", "缺口 ID 不存在")
                if gap["status"] == "RESOLVED" and not gap["resolution_claim_ids"]:
                    raise ToolError("INVALID_ARGUMENTS", "解决缺口须给出主张依据")
                gid = gid or f'G{len(role["gaps"]) + 1:03d}'
                role["gaps"][gid] = {**gap, "gap_id": gid}
            if sum(g["status"] == "OPEN" for g in role["gaps"].values()) > self.config["memory"]["max_active_gaps"]:
                raise ToolError("INVALID_ARGUMENTS", "活动缺口超过限制")
            return {"gaps": list(role["gaps"].values())}
        if name == "PROPOSE":
            if len(args["claims"]) > self.config["memory"]["max_claims_per_propose"]:
                raise ToolError("INVALID_ARGUMENTS", "单次候选过多")
            ids = []
            for claim in args["claims"]:
                self.require_visible(claim["evidence_ids"])
                claim["evidence_ids"] = sorted(set(claim["evidence_ids"]))
                existing = next((c for c in role["candidates"].values() if c["fingerprint"] == fingerprint(claim)), None)
                if existing:
                    ids.append(existing["predicted_claim_id"])
                    continue
                if len(role["candidates"]) >= self.config["memory"]["max_candidates"]:
                    raise ToolError("BUDGET_EXHAUSTED", "候选数量达到上限")
                prefix = {"solo": "", "archivist": "A:", "interpreter": "I:"}[s["actor"]]
                cid = f'{prefix}P{len(role["candidates"]) + 1:03d}'
                role["candidates"][cid] = {**claim, "predicted_claim_id": cid, "status": "CANDIDATE",
                                           "fingerprint": fingerprint(claim), "support_check": None,
                                           "created_event_seq": s["event_seq"] + 1}
                ids.append(cid)
            return {"candidate_ids": ids}
        if name == "COMMIT":
            cid = args["candidate_id"]
            candidate = self.claims([cid])[0]
            if cid in role["committed"]:
                return {"committed": cid, "replayed": True}
            self.require_visible(candidate["evidence_ids"])
            if len(role["committed"]) >= self.config["memory"]["max_committed_claims"]:
                raise ToolError("BUDGET_EXHAUSTED", "提交主张达到上限")
            request = {"candidate": deepcopy(candidate),
                       "evidence": [role["visible"][eid] for eid in candidate["evidence_ids"]]}
            check = self.structured("SUPPORT_CHECK", request,
                                    lambda t: SupportCheck.model_validate_json(t).model_dump(), SupportCheck.model_json_schema())
            if check["candidate_id"] != cid or set(check["evidence_ids"]) != set(candidate["evidence_ids"]):
                raise ToolError("INVALID_ARGUMENTS", "支持核对不得改换主张或证据")
            accepted = check["support"] == "SUPPORTED" and all(check[k] for k in ("scope_consistent", "voice_preserved", "modality_preserved"))
            candidate["support_check"] = check
            if not accepted:
                candidate["status"] = "REJECTED" if check["support"] == "UNSUPPORTED" else "DEFERRED"
                return {"ok": False, "error_code": "SUPPORT_NOT_CONFIRMED", "message": "支持未确认，未提交", "retryable": False}
            candidate["status"] = "COMMITTED"
            role["committed"][cid] = deepcopy(candidate)
            return {"committed": cid, "support_check": check}
        if name == "COMPARE":
            claims = self.claims(args["claim_ids"])
            self.require_visible(args["evidence_ids"])
            request = {"claims": deepcopy(claims), "evidence": [role["visible"][eid] for eid in args["evidence_ids"]], "question": args["question"]}
            result = self.structured("SCOPE_COMPARE", request,
                                     lambda t: Relation.model_validate_json(t).model_dump(), Relation.model_json_schema())
            if set(result["claim_ids"]) != set(args["claim_ids"]) or not set(result["evidence_ids"]) <= set(args["evidence_ids"]):
                raise ToolError("INVALID_ARGUMENTS", "范围比较只能使用请求中的主张与证据")
            rid = f'R{len(role["relations"]) + 1:03d}'
            role["relations"][rid] = {**result, "relation_id": rid}
            return role["relations"][rid]
        if name == "DEFER":
            if args["gap_id"] not in role["gaps"]:
                raise ToolError("INVALID_ARGUMENTS", "缺口 ID 不存在")
            role["gaps"][args["gap_id"]]["status"] = "DEFERRED"
            return {"deferred": args["gap_id"]}
        if name in ("SEND_EVIDENCE", "REQUEST_PEER", "YIELD"):
            if s["task"] != "T4":
                raise ToolError("INVALID_ACTION", "协作动作只用于 T4")
            if name == "YIELD":
                return {"yield": True}
            if len(s["messages"]) >= self.config["limits"]["max_t4_messages"]:
                raise ToolError("BUDGET_EXHAUSTED", "消息预算耗尽")
            peer = "interpreter" if s["actor"] == "archivist" else "archivist"
            message = {"message_id": f'{s["sample_id"]}:MSG{len(s["messages"]) + 1:04d}',
                       "kind": "EVIDENCE" if name == "SEND_EVIDENCE" else "REQUEST",
                       "from_agent": s["actor"], "to_agent": peer, "evidence": [],
                       "request_id": args.get("request_id"), "sent_event_seq": s["event_seq"] + 1,
                       "delivered_event_seq": None}
            if name == "SEND_EVIDENCE":
                self.require_visible(args["evidence_ids"])
                if args["request_id"] and not any(m["message_id"] == args["request_id"] and m["kind"] == "REQUEST" for m in role["delivered_messages"]):
                    raise ToolError("INVALID_ARGUMENTS", "请求 ID 不在已收到的请求中")
                message["evidence"] = [deepcopy(self.evidence[eid]) for eid in dict.fromkeys(args["evidence_ids"])]
                message["purpose"] = args["purpose"]
            else:
                message["question"] = args["question"]
            s["messages"].append(message)
            return {"queued_message_id": message["message_id"], "delivered": False}
        if name == "FINALIZE":
            return self.finalize(args)
        raise ToolError("INVALID_ACTION", "未知动作")

    def finalize(self, args):
        s, role = self.state, self.role
        if s["task"] == "T4" and s["actor"] != "interpreter":
            raise ToolError("INVALID_ACTION", "联合最终回答者是 interpreter")
        if len(args["claim_ids"]) != len(set(args["claim_ids"])) or not set(args["claim_ids"]) <= role["committed"].keys():
            raise ToolError("CLAIM_NOT_COMMITTED", "只能选择已提交且不重复的主张")
        if not set(args["relation_ids"]) <= role["relations"].keys():
            raise ToolError("INVALID_ARGUMENTS", "关系 ID 不存在")
        if s["task"] == "T3" and set(args["claim_ids"]) != role["committed"].keys():
            raise ToolError("CLAIM_NOT_COMMITTED", "T3 必须输出全部累计主张")
        claims = [role["committed"][cid] for cid in args["claim_ids"]]
        required = {eid for c in claims for eid in c["evidence_ids"]}
        citations = set(re.findall(r'\[(E\d+)\]', args["answer"]))
        if not citations <= role["visible"].keys():
            kind = "future_citation" if citations <= self.evidence.keys() else "unknown_citation"
            raise ToolError(kind.upper(), "回答含不可引用的证据")
        if citations != required:
            raise ToolError("INVALID_CITATION", "回答引用须等于所选主张证据并集")
        if s["task"] == "T2" and (len(required) < 2 or not args["relation_ids"]):
            raise ToolError("INVALID_ARGUMENTS", "T2 需要多证据及至少一项范围比较解释")
        pred = self.prediction(args)
        if s["task"] == "T3":
            old = set(s["turn_previous_ids"])
            new = [cid for cid in role["committed"] if cid not in old]
            operation = ("ADD" if new else None) if s["current_turn"] == 1 else ("EXPAND" if new else "DEFER")
            s["turns"].append({"turn": s["current_turn"], "operation": operation,
                               "memory_claim_ids": list(role["committed"]), "new_claim_ids": new,
                               "claims": deepcopy(pred["claims"]), "answer": pred["answer"],
                               "complete": operation is not None, "visible_evidence_ids": list(role["visible"])})
            if operation is None:
                raise StopEpisode("NO_INITIAL_MEMORY")
            if s["current_turn"] < len(self.sample["events"]):
                s["turn_previous_ids"] = list(role["committed"])
                s["current_turn"] += 1
                s["turn_steps"] = -1  # 外层动作结束后加一。
                self.reveal(self.sample["events"][s["current_turn"] - 1]["visible_evidence_ids"], "solo", "revealed")
                return {"turn_finalized": True, "next_turn": s["current_turn"]}
        s["final"] = pred
        s["status"], s["termination_reason"] = "COMPLETED", "FINALIZED"
        return {"finalized": True}

    def start_activation(self):
        s = self.state
        if s["activation_started"]:
            return
        for msg in s["messages"]:
            if msg["to_agent"] == s["actor"] and msg["delivered_event_seq"] is None:
                msg["delivered_event_seq"] = s["event_seq"] + 1
                self.reveal([e["evidence_id"] for e in msg["evidence"]], s["actor"], "received", msg["message_id"])
                self.role["delivered_messages"].append(deepcopy(msg))
        s["activation_started"] = True
        self.checkpoint("ACTIVATION_STARTED", {})

    def advance_actor(self):
        s = self.state
        if s["actor"] == "archivist":
            s["actor"] = "interpreter"
        else:
            s["actor"] = "archivist"
            s["scheduler_round"] += 1
        s["activation_steps"], s["activation_started"] = 0, False

    def stop(self, reason, status="PARTIAL"):
        s = self.state
        s["status"], s["termination_reason"] = status, reason
        s["final"] = self.prediction()
        if status != "PAUSED_UNCERTAIN" and s["task"] == "T3" and not any(t["turn"] == s["current_turn"] for t in s["turns"]):
            pred = s["final"]
            s["turns"].append({"turn": s["current_turn"], "operation": None,
                               "memory_claim_ids": list(self.role["committed"]),
                               "new_claim_ids": [], "claims": deepcopy(pred["claims"]),
                               "answer": pred["answer"], "complete": False,
                               "visible_evidence_ids": list(self.role["visible"])})
        self.checkpoint("STOPPED", {"reason": reason}, self.envelope())

    def run(self):
        s, limits = self.state, self.config["limits"]
        if s["status"] == "PENDING":
            s["status"] = "RUNNING"
        if s["status"] == "PAUSED_UNCERTAIN":
            s["status"] = "RUNNING"
            s["final"] = None
            s["termination_reason"] = None
        if s["status"] != "RUNNING":
            return self.envelope()
        self.initialize()
        while s["status"] == "RUNNING":
            action = {"action": "INVALID", "arguments": {}, "reason": ""}
            try:
                if s["step"] >= limits["max_actions_per_episode"]:
                    raise StopEpisode("ACTION_BUDGET_EXHAUSTED")
                if s["task"] == "T3" and s["turn_steps"] >= limits["max_actions_per_t3_turn"]:
                    raise StopEpisode("TURN_BUDGET_EXHAUSTED")
                if s["task"] == "T4":
                    if s["scheduler_round"] > limits["max_t4_rounds"]:
                        raise StopEpisode("SCHEDULER_BUDGET_EXHAUSTED")
                    self.start_activation()
                view = self.build_view()
                self.log(f'[{s["sample_id"]}] 第{s["current_turn"]}轮 / {s["actor"]} / 决策 {s["step"] + 1}')
                action = self.structured("ACTION", view, parse_action, action_schema())
                # 工具失败只回滚业务状态；模型调用、用量和日志不能回滚。
                before = deepcopy(s)
                try:
                    result = {"ok": True, **self.execute(action)}
                except ToolError:
                    for key in ("role_states", "messages", "turns", "current_turn", "turn_previous_ids", "turn_steps", "final", "status", "termination_reason"):
                        s[key] = before[key]
                    raise
                self.log(f'  [动作] {action["action"]} → {"成功" if result.get("ok") else result.get("error_code")}')
            except ToolError as exc:
                result = {"ok": False, "error_code": exc.code, "message": exc.message, "retryable": False}
                if exc.code in ("ACCESS_DENIED", "FUTURE_CITATION", "UNKNOWN_CITATION"):
                    kind = {"ACCESS_DENIED": "access_denied", "FUTURE_CITATION": "future_citation", "UNKNOWN_CITATION": "unknown_citation"}[exc.code]
                    s["access_violations"].append({"kind": kind, "step": s["step"]})
                self.log(f'  [拒绝] {exc.code}: {exc.message}')
            except UncertainCall:
                self.stop("CALL_UNCERTAIN", "PAUSED_UNCERTAIN")
                break
            except StopEpisode as exc:
                self.stop(str(exc))
                break
            self.role["last_tool_result"] = result
            s["step"] += 1
            s["turn_steps"] += 1
            s["invalid_streak"] = 0 if result.get("ok") else s["invalid_streak"] + 1
            event_role = s["actor"]
            if s["task"] == "T4" and s["status"] == "RUNNING":
                s["activation_steps"] += 1
                if action["action"] == "YIELD" or s["activation_steps"] >= limits["max_actions_per_t4_activation"]:
                    self.advance_actor()
            self.checkpoint("ACTION", {"action": action, "result": result, "acting_role": event_role},
                            self.envelope() if s["status"] == "COMPLETED" else None)
            if s["status"] == "RUNNING" and s["invalid_streak"] >= limits["max_consecutive_invalid_actions"]:
                self.stop("CONSECUTIVE_INVALID_ACTIONS")
        return self.envelope()

