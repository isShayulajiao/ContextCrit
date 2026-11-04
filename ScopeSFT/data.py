"""CultCrit supervision -> legal ScopeCrit ACTION trajectories, without remote calls.

Gold annotations are teacher labels, not observations. These are deterministic
silver trajectories, NOT model rollouts or proof of annotation correctness.
ScopeCrit stays the sole authority for visibility, IDs, delivery and action rules.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import warnings

_SCOPECRIT = Path(__file__).resolve().parents[1] / "ScopeCrit"
if str(_SCOPECRIT) not in sys.path:
    sys.path.insert(0, str(_SCOPECRIT))

from scopecrit.config import Config
from scopecrit.data import _validate_row, fingerprint
from scopecrit.engine import Engine
from scopecrit.providers import prompt_text, response_schema
from scopecrit.schemas import Claim, parse_action

CLAIM_FIELDS = ("text", "aspect", "target_scope", "stance", "evidence_ids")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=1)
def action_system_prompt():
    """Keep one exact runtime schema string instead of regenerating per action."""
    return prompt_text("ACTION") + "\n只输出 JSON。输出结构：\n" + json.dumps(response_schema("ACTION"), ensure_ascii=False)


def _strict_constant(value):
    raise ValueError(f"非法 JSON 数值: {value}")


def _claim(value):
    claim = Claim.model_validate({k: value[k] for k in CLAIM_FIELDS}).model_dump()
    # Engine.PROPOSE canonicalizes citations before computing its fingerprint.
    # Use the identical set ordering for teacher lookup; source JSON is unchanged.
    claim["evidence_ids"] = sorted(set(claim["evidence_ids"]))
    return claim


def _action(name, arguments, reason="依据当前可见材料执行此操作"):
    return parse_action(canonical({"action": name, "arguments": arguments, "reason": reason}))


def _span(action, field):
    """Absolute Unicode character span of the sole PROPOSE claim's field value."""
    text = canonical(action)
    marker = canonical(field) + ":"
    start = text.index(marker) + len(marker)
    value = action["arguments"]["claims"][0][field]
    assert isinstance(value, str)
    # Include escaped value contents, exclude JSON quotation marks.
    return [start + 1, start + len(canonical(value)) - 1]


def _negative(sample, claim, gold_id):
    """Conservative, auditable source rules; all other cases retain NLL only.

    Auto pairs assert only that a stated limited observation/prediction does not
    justify an unconditional whole-book/certain conclusion. They do not assert
    that the actual unseen whole book has the opposite quality.
    """
    cited = [e for e in sample["document"]["evidence"] if e["evidence_id"] in claim["evidence_ids"]]
    support = "\n".join(e["text"] for e in cited)
    # Optional reviewed pairs are labels supplied by the dataset author. They are
    # never marked independently human-verified by this code.
    for pair in sample.get("scope_ct_negatives", []):
        if pair.get("claim_id") != gold_id:
            continue
        if pair.get("reviewed") is not True or not pair.get("justification", "").strip():
            raise ValueError("scope_ct_negatives 必须含 reviewed:true 和非空 justification")
        field = pair.get("semantic_field")
        if field not in ("text", "target_scope", "stance"):
            raise ValueError("semantic_field 必须为 text、target_scope 或 stance")
        negative = _claim(pair["rejected_claim"])
        if negative == claim or negative[field] == claim[field]:
            raise ValueError("人工审核负例必须在指定语义字段产生差异")
        if any(negative[k] != claim[k] for k in ("aspect", "evidence_ids")):
            raise ValueError("负例不得更改 aspect 或证据编号")
        evidence_ids = pair.get("evidence_ids", [])
        if not evidence_ids or not set(evidence_ids) <= {e["evidence_id"] for e in cited}:
            raise ValueError("人工审核负例须给出属于该主张可见引用的 evidence_ids")
        return negative, field, {"kind": "dataset_reviewed", "rule": "author_supplied", "human_verified_by_code": False,
                                 "justification": pair["justification"], "evidence_ids": evidence_ids}
    # Quoted voice -> narrator voice is accepted only with BOTH an explicit
    # gold attribution/disclaimer and a matching source attribution followed by
    # an actual opening quotation marker in the next cited source span.
    attributed = re.match(r"^文中转引(?P<name>[\u4e00-\u9fff·]{1,12})的文章称", claim["text"])
    disclaimer = re.search(r"[；;]这不是本文叙述者直接给出的小说总评[。.!！]?$", claim["text"])
    if attributed and disclaimer and claim["target_scope"].startswith("被引评论"):
        name = attributed.group("name")
        ordered = sorted(cited, key=lambda evidence: evidence["start"])
        proof = None
        for introduction, quotation in zip(ordered, ordered[1:]):
            marker = name + "先生的文章为证"
            between = sample["document"]["text"][introduction["end"]:quotation["start"]]
            if (marker in introduction["text"] and introduction["text"].rstrip().endswith(("——", "：", ":"))
                    and quotation["text"].lstrip().startswith(("“", "「", "『"))
                    and quotation["start"] >= introduction["end"] and not between.strip()):
                proof = (introduction, quotation)
                break
        if proof:
            negative = deepcopy(claim)
            body = claim["text"][attributed.end():disclaimer.start()].rstrip("。.!！")
            negative["text"] = "本文叙述者直接断言" + body + "。"
            negative["target_scope"] = claim["target_scope"].replace("被引评论", "本文叙述者", 1)
            return negative, "text", {"kind": "rule_generated", "rule": "explicit_quoted_voice_to_narrator",
                "human_verified_by_code": False, "attributed_speaker": name,
                "evidence_ids": [e["evidence_id"] for e in proof], "source_quotes": [e["text"] for e in proof],
                "justification": "原文明确引入具名他人的文章并紧接引号，正例也明确排除本文叙述者直接总评；负例只将同一判断的声音归属改为本文叙述者。"}
    # This narrowly matches the uploaded dataset's explicitly observed two parts.
    # Change both the prose and scope, preserving a single semantic error factor.
    if (claim["target_scope"] in ("前两段", "第一、二段", "第一第二段")
            and re.search(r"只登(?:了)?两段|第一第二段", support)):
        positive_text = claim["text"]
        replacements = (("第一、二段", "全书"), ("第一第二段", "全书"), ("前两段", "全书"),
                        ("只登两段便", "全书都"), ("只登了两段便", "全书都"))
        for before, after in replacements:
            if before in positive_text:
                negative = deepcopy(claim)
                negative["text"] = positive_text.replace(before, after, 1)
                negative["target_scope"] = "全书"
                return negative, "text", {"kind": "rule_generated", "rule": "observed_two_parts_to_whole_book",
                    "human_verified_by_code": False, "evidence_ids": [e["evidence_id"] for e in cited],
                    "source_quotes": [e["text"] for e in cited],
                    "justification": "引用仅提供已刊两段的评价，不足以确定未刊全书；文本与范围同步作同一范围扩张。"}
    if "预期" in claim["text"] and "预约券" in support and "该是" in support:
        negative = deepcopy(claim)
        negative["text"] = negative["text"].replace("预期", "确认", 1).replace("该是", "已经是", 1)
        negative["target_scope"] = negative["target_scope"].replace("预期", "确定评价")
        return negative, "text", {"kind": "rule_generated", "rule": "explicit_prediction_to_certainty",
            "human_verified_by_code": False, "evidence_ids": [e["evidence_id"] for e in cited],
            "source_quotes": [e["text"] for e in cited],
            "justification": "原文以预约券和该是明确表示预期，不能改成已经确认的事实。"}
    return None


class _MemoryStore:
    def get_state(self, sample_id):
        return None

    def save_state(self, *args, **kwargs):
        pass


class _TeacherEngine(Engine):
    """Reuse the real engine's state transitions; replace only external labels."""
    def __init__(self, sample):
        n = len(sample["gold"]["claims"])
        config = Config().model_dump()
        config["limits"].update(max_actions_per_episode=max(64, n * 4 + 40),
                                max_actions_per_t3_turn=max(16, n * 2 + 8),
                                max_model_calls_per_episode=max(80, n * 8 + 40),
                                max_t4_rounds=max(6, n + 4))
        super().__init__(sample, config, None, _MemoryStore(), log=lambda *_: None)
        self.records = []
        self.planned = set()
        self.read = set()
        self.gold_by_id = {c["claim_id"]: c for c in sample["gold"]["claims"]}

    def _desired(self):
        if self.sample["task"] == "T3":
            ids = self.sample["events"][self.state["current_turn"] - 1]["gold_memory_state"]
            return [self.gold_by_id[cid] for cid in ids]
        return self.sample["gold"]["claims"]

    def _teacher_action(self):
        s, role, sample = self.state, self.role, self.sample
        previous = role.get("last_tool_result")
        if previous and previous.get("ok") is False:
            raise ValueError(f"{sample['sample_id']}: 教师工具动作失败，停止生成而非重复动作：{previous}")
        if sample["task"] == "T4":
            # Fulfil annotated exchanges, plus deliver all answer evidence to
            # interpreter. Never reveal them to the recipient before activation.
            needed = {x["evidence_id"] for x in sample["required_exchanges"] if x["from_agent"] == s["actor"]}
            if s["actor"] == "archivist":
                needed |= set(sample["gold"]["required_evidence_ids"]) & set(sample["roles"]["archivist"])
            sent = {e["evidence_id"] for m in s["messages"] if m["kind"] == "EVIDENCE" and m["from_agent"] == s["actor"] for e in m["evidence"]}
            unsent = sorted(needed - sent)
            if unsent:
                if not set(unsent) <= set(role["visible"]):
                    raise ValueError("教师通信标签引用了角色不可见证据")
                return _action("SEND_EVIDENCE", {"evidence_ids": unsent, "purpose": "共享当前可见证据，供同伴核对评价范围", "request_id": None})
            if s["actor"] == "archivist":
                return _action("YIELD", {})
            if not set(sample["gold"]["required_evidence_ids"]) <= set(role["visible"]):
                return _action("REQUEST_PEER", {"question": "请共享与你持有材料有关的评价依据，供联合回答核对。"})
        key = (s["actor"], s["current_turn"])
        if key not in self.planned:
            self.planned.add(key)
            return _action("PLAN_GAPS", {"gaps": [{"gap_id": None, "kind": "SCOPE", "question": "当前可见证据分别支持哪些评价范围和条件？", "related_claim_ids": [], "status": "OPEN", "resolution_claim_ids": []}]})
        if key not in self.read:
            self.read.add(key)
            return _action("READ", {"evidence_ids": list(role["visible"])})
        desired = self._desired()
        for gold in desired:
            claim = _claim(gold)
            if not set(claim["evidence_ids"]) <= set(role["visible"]):
                raise ValueError(f"{sample['sample_id']}: gold 教师标签引用尚未公开证据")
            existing = next((c for c in role["candidates"].values() if c["fingerprint"] == fingerprint(claim)), None)
            if existing is None:
                return _action("PROPOSE", {"claims": [claim]})
            if existing["predicted_claim_id"] not in role["committed"]:
                return _action("COMMIT", {"candidate_id": existing["predicted_claim_id"]})
        if sample["task"] == "T2" and not role["relations"]:
            return _action("COMPARE", {"claim_ids": list(role["committed"]), "evidence_ids": sorted({e for c in role["committed"].values() for e in c["evidence_ids"]}),
                                       "question": "这些已提交评价的范围和条件是什么关系？"})
        # Resolve the generic scope gap using actual committed IDs.
        unresolved = [g for g in role["gaps"].values() if g["status"] == "OPEN"]
        if unresolved:
            return _action("PLAN_GAPS", {"gaps": [{**g, "status": "RESOLVED", "resolution_claim_ids": list(role["committed"])} for g in unresolved]})
        if sample["task"] == "T4" and any(m["delivered_event_seq"] is None for m in s["messages"]):
            return _action("YIELD", {})
        answer = (sample["events"][s["current_turn"] - 1]["reference_answer"] if sample["task"] == "T3" else sample["gold"]["reference_answer"])
        return _action("FINALIZE", {"claim_ids": list(role["committed"]), "relation_ids": list(role["relations"]), "answer": answer})

    def structured(self, stage, request, parser, schema):
        self.state["usage"]["model_calls"] += 1
        if stage == "SUPPORT_CHECK":
            # A training teacher assumption, NOT an independently executed judge.
            c = request["candidate"]
            return {"candidate_id": c["predicted_claim_id"], "support": "SUPPORTED", "scope_consistent": True,
                    "voice_preserved": True, "modality_preserved": True, "evidence_ids": c["evidence_ids"], "reason": "离线教师沿用输入 gold 标签；未执行独立模型核验"}
        if stage == "SCOPE_COMPARE":
            # The next ACTION observes this result. Never copy the held-out
            # teacher answer here: it would leak a future FINALIZE target into
            # the observation. Only describe already committed visible claims.
            scopes = "；".join(f"{claim['predicted_claim_id']} 的范围为{claim['target_scope']}、立场为{claim['stance']}"
                               for claim in request["claims"])
            explanation = "仅汇总当前已提交主张：" + scopes + "。这些范围之间的语义关系尚未经过独立核验。"
            return {"claim_ids": [c["predicted_claim_id"] for c in request["claims"]], "evidence_ids": [e["evidence_id"] for e in request["evidence"]],
                    "relation_type": "UNRESOLVED", "explanation": explanation}
        if stage != "ACTION":
            raise ValueError(f"未支持教师阶段 {stage}")
        action = self._teacher_action()
        system = action_system_prompt()
        record = {"messages": [{"role": "system", "content": system},
                               {"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
                  "chosen": canonical(action), "rejected": None,
                  "metadata": {"sample_id": self.sample["sample_id"], "doc_id": self.sample["doc_id"], "task": self.sample["task"],
                               "split": self.sample.get("split"), "action": action["action"], "turn": self.state["current_turn"],
                               "role": self.state["actor"], "teacher": "deterministic_gold_trajectory", "human_verified": self.sample.get("human_verified", False)}}
        if action["action"] == "PROPOSE":
            claim = action["arguments"]["claims"][0]
            gold_id = next(c["claim_id"] for c in self._desired() if _claim(c) == claim)
            pair = _negative(self.sample, claim, gold_id)
            if pair:
                negative, field, audit = pair
                rejected = deepcopy(action)
                rejected["arguments"]["claims"][0] = negative
                rejected = parse_action(canonical(rejected))
                record["rejected"] = canonical(rejected)
                record["semantic_span"] = {"field": field, "chosen": _span(action, field), "rejected": _span(rejected, field)}
                record["metadata"]["negative_audit"] = {"claim_id": gold_id, **audit}
        self.records.append(record)
        return action


def _iter_json_array(path, chunk_size=65536):
    """Incrementally decode strict UTF-8 JSON, retaining one array item at a time.

    This avoids an additional dependency and avoids keeping the full source and
    the much larger expanded trajectory corpus simultaneously in RAM. A syntax
    error is fatal even when it occurs after previously yielded valid samples.
    Consumers must only publish prepared caches after exhausting the iterator.
    """
    decoder = json.JSONDecoder(parse_constant=_strict_constant)
    with path.open("r", encoding="utf-8-sig") as source:
        buffer, offset, eof = "", 0, False

        def refill():
            nonlocal buffer, offset, eof
            if eof:
                return False
            buffer = buffer[offset:]
            offset = 0
            chunk = source.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        def skip_space():
            nonlocal offset
            while True:
                while offset < len(buffer) and buffer[offset] in " \t\r\n":
                    offset += 1
                if offset < len(buffer) or not refill():
                    return

        skip_space()
        if offset == len(buffer) or buffer[offset] != "[":
            raise ValueError("训练数据必须是非空 JSON 数组")
        offset += 1
        skip_space()
        count = 0
        while True:
            if offset < len(buffer) and buffer[offset] == "]":
                if count == 0:
                    raise ValueError("训练数据必须是非空 JSON 数组")
                offset += 1
                skip_space()
                if offset != len(buffer):
                    raise ValueError("JSON 数组结束后包含多余内容")
                return
            # ``raw_decode`` reports incomplete objects as JSONDecodeError; read
            # more only until EOF, then preserve that underlying error.
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, offset)
                    offset = end
                    break
                except json.JSONDecodeError:
                    if not refill():
                        raise
            count += 1
            yield item
            # Drop the decoded item's raw text before reading the next item.
            skip_space()
            if offset == len(buffer):
                raise ValueError("JSON 数组未正常结束")
            if buffer[offset] == ",":
                offset += 1
                skip_space()
                if offset < len(buffer) and buffer[offset] == "]":
                    raise ValueError("JSON 数组不允许尾随逗号")
            elif buffer[offset] != "]":
                raise ValueError("JSON 数组条目之间缺少逗号")


def iter_records(path, allow_test_split=False):
    """Yield positive ACTION examples per source sample with optional pairs.

    The explicit path is the training corpus, including rows marked test. A
    warning prevents presenting scores on those same rows as held-out results.
    ``allow_test_split=True`` acknowledges/suppresses that warning; no row is
    silently discarded, relabelled or split into another corpus.
    """
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError("训练数据路径必须是绝对路径")
    seen, docs = set(), {}
    warned = False
    for i, sample in enumerate(_iter_json_array(path)):
        if not isinstance(sample, dict):
            raise ValueError(f"第 {i + 1} 条样本不是 JSON 对象")
        # Metadata absent from uploaded example is not fabricated in the source.
        # Existing runtime validator receives only a compatibility copy.
        validation = deepcopy(sample)
        validation.setdefault("generation_method", "USER_SUPPLIED_TRAINING_LABELS")
        validation.setdefault("human_verified", False)
        validation.setdefault("material_origin", "AUTHENTIC")
        validation.setdefault("task_origin", "CONTROLLED" if sample.get("task") in ("T3", "T4") else "DERIVED")
        errors = _validate_row(validation, True, [], str(sample.get("sample_id", i)))
        # Training can also consume genuinely reviewed annotations. Runtime's
        # historical provenance check requires False, but that metadata does
        # not change ACTION permissions. Preserve True in output metadata.
        if sample.get("human_verified") is True:
            errors = [error for error in errors if error != "human_verified 必须为 false"]
        if errors:
            raise ValueError(f"训练样本 {sample.get('sample_id', i)} 校验失败: " + "; ".join(errors))
        sid = sample["sample_id"]
        if sid in seen:
            raise ValueError(f"重复 sample_id: {sid}")
        seen.add(sid)
        doc_hash = fingerprint({k: sample["document"][k] for k in ("work_title", "text", "evidence")})
        if sample["doc_id"] in docs and docs[sample["doc_id"]] != doc_hash:
            raise ValueError(f"同 doc_id={sample['doc_id']} 的材料不一致")
        docs[sample["doc_id"]] = doc_hash
        if sample.get("split", "").lower() in ("test", "validation", "valid", "eval", "dev") and not allow_test_split and not warned:
            warnings.warn("提供的训练文件含 test/validation 标记；按明确输入路径用于训练，这些样本不得再作为独立测试集报告。", UserWarning, stacklevel=2)
            warned = True
        engine = _TeacherEngine(sample)
        result = engine.run()
        if result["status"] != "COMPLETED" or result["access_violations"]:
            raise ValueError(f"{sid}: 教师轨迹未合法完成 ({result['status']}: {result['termination_reason']})")
        yield from engine.records


def load_records(path, allow_test_split=False):
    """Compatibility wrapper for small corpora; large runs use iter_records."""
    return list(iter_records(path, allow_test_split=allow_test_split))
