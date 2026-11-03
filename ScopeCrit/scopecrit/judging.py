"""独立模型裁判：先做一对一语义对齐，再复用确定性的指标计算。

评分数据只在答题结束后进入此模块，不会回传给 Agent。每个裁判使用独立
请求和独立客户端。调用日志与缓存另存 judges/state.sqlite3，不改变答题状态。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import Field

from .data import fingerprint
from .evaluation import evaluate_predictions
from .judge_config import resolved_base_url
from .judge_providers import make_judge_provider
from .schemas import StrictModel
from .storage import atomic_write, synchronized


class Match(StrictModel):
    predicted_claim_id: str
    gold_claim_id: str
    relation: Literal["equivalent"]
    note: str = Field(default="", max_length=500)


class Alignment(StrictModel):
    sample_id: str
    turn: int | None
    alignment_complete: bool
    matches: list[Match]
    relation_quality: Literal["correct", "partial", "incorrect", "unscored"] | None = None
    dual_dependency: bool | None = None


class JudgeAnswer(StrictModel):
    alignments: list[Alignment]


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _write(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _project_claim(claim, predicted):
    """含义配对不看 aspect/stance/证据标签，避免循环定义标签正确性。"""
    key = "predicted_claim_id" if predicted else "claim_id"
    return {key: claim[key], "text": claim["text"], "target_scope": claim.get("target_scope", "")}


def _relation(relation):
    return {k: deepcopy(relation[k]) for k in
            ("relation_id", "claim_ids", "evidence_ids", "relation_type", "explanation") if k in relation}


def _evidence(evidence):
    return {key: evidence[key] for key in ("evidence_id", "text", "start", "end")}


def build_judge_request(sample, prediction):
    """白名单构造评分请求，不携带答题模型名称、其他裁判意见或隐藏推理。"""
    gold_claims = sample["gold"]["claims"]
    final = prediction.get("prediction", {})
    targets = [{"sample_id": sample["sample_id"], "turn": None,
                "question": sample.get("question", ""),
                "predicted_claims": [_project_claim(c, True) for c in final.get("claims", [])],
                "gold_claims": [_project_claim(c, False) for c in gold_claims],
                "answer": final.get("answer", ""),
                "relations": [_relation(r) for r in final.get("relations", [])]}]
    if sample["task"] == "T3":
        for event in sample["events"]:
            snapshot = next((t for t in prediction.get("turns", []) if t["turn"] == event["turn"]), {})
            ids = set(event["gold_memory_state"])
            targets.append({"sample_id": sample["sample_id"], "turn": event["turn"],
                            "question": event.get("question", ""),
                            "predicted_claims": [_project_claim(c, True) for c in snapshot.get("claims", [])],
                            "gold_claims": [_project_claim(c, False) for c in gold_claims if c["claim_id"] in ids],
                            "answer": snapshot.get("answer", ""), "relations": []})
    document = sample["document"]
    request = {"task": sample["task"], "work_title": document.get("work_title", ""),
               "source": {"text": document["text"], "evidence": [_evidence(e) for e in document["evidence"]]},
               "targets": targets}
    if sample["task"] == "T4":
        request["collaboration"] = {
            "initial_evidence": {role: list(ids) for role, ids in sample["roles"].items()},
            "delivered_messages": [{"from_agent": m["from_agent"], "to_agent": m["to_agent"],
                                     "evidence": [_evidence(e) for e in m.get("evidence", [])]}
                                    for m in prediction.get("messages", [])
                                    if m.get("kind") == "EVIDENCE" and m.get("delivered_event_seq") is not None]}
    return request


def parse_judge_answer(text, sample, prediction, request):
    """严格结构和引用检查；只有合法的一对一配对才能进入指标计算。"""
    parsed = JudgeAnswer.model_validate_json(text).model_dump()
    records = []
    expected = {(t["sample_id"], t["turn"]) for t in request["targets"]}
    actual = [(r["sample_id"], r["turn"]) for r in parsed["alignments"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("JUDGE_TARGET_MISMATCH")
    for record in parsed["alignments"]:
        # 现有离线评分器对缺失标注保留 null；不能把 null 强行转成 false。
        if record["dual_dependency"] is None:
            del record["dual_dependency"]
        if record["relation_quality"] is None:
            del record["relation_quality"]
        if sample["task"] == "T2" and "relation_quality" not in record:
            raise ValueError("JUDGE_RELATION_QUALITY_MISSING")
        if sample["task"] != "T2" and "relation_quality" in record:
            raise ValueError("JUDGE_UNEXPECTED_RELATION_QUALITY")
        if sample["task"] != "T4" and "dual_dependency" in record:
            raise ValueError("JUDGE_UNEXPECTED_DUAL_DEPENDENCY")
        records.append(record)
    # 标签、ID 合法性、重复配对均由同一个确定性评分器检查。
    evaluate_predictions([prediction], [sample], records)
    return records


def _records_complete(records, task):
    return all(r["alignment_complete"]
               and (task != "T2" or r.get("relation_quality") != "unscored")
               and (task != "T4" or type(r.get("dual_dependency")) is bool)
               for r in records)


def _empty_failure_alignment(sample, request):
    """真正失败且没有主张的答卷直接记零，不额外花钱请模型确认空答卷。"""
    records = []
    for target in request["targets"]:
        record = {"sample_id": sample["sample_id"], "turn": target["turn"],
                  "alignment_complete": True, "matches": []}
        if sample["task"] == "T2":
            record["relation_quality"] = "incorrect"
        if sample["task"] == "T4":
            record["dual_dependency"] = False
        records.append(record)
    return records


class JudgeJournal:
    """每次真实尝试独立记录；已知失败可重试，未决调用默认不重复收费。"""
    def __init__(self, path, manifest):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.invocation = uuid4().hex
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY, logical_id TEXT NOT NULL, generation INTEGER NOT NULL,
                judge TEXT NOT NULL, sample_id TEXT NOT NULL, status TEXT NOT NULL,
                invocation TEXT NOT NULL, request_json TEXT NOT NULL, response_json TEXT,
                error_json TEXT, UNIQUE(logical_id,generation));
        """)
        content = _json(manifest)
        old = self.db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if old and old[0] != content:
            self.db.close()
            raise ValueError("JUDGE_FINGERPRINT_MISMATCH: 裁判配置、输入、预测或提示词已改变，请使用新的运行目录")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('manifest',?)", (content,))

    @synchronized
    def bind_endpoint(self, name, endpoint):
        key, value = "endpoint:" + name, fingerprint(endpoint)
        old = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if old and old[0] != value:
            raise ValueError("JUDGE_ENDPOINT_CHANGED: 裁判接口地址已改变，请使用新的评分目录")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)", (key, value))

    @synchronized
    def bind_prediction(self, sample_id, request, prediction):
        """已评分答卷冻结；此前未就绪的样本可以在后续恢复时加入。"""
        key = "prediction:" + sample_id
        value = fingerprint([request, prediction])
        old = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if old and old[0] != value:
            raise ValueError("JUDGE_PREDICTION_CHANGED: 已评分样本的预测已改变，请使用新的评分目录")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)", (key, value))

    @synchronized
    def latest(self, logical_id):
        row = self.db.execute("SELECT * FROM calls WHERE logical_id=? ORDER BY generation DESC LIMIT 1", (logical_id,)).fetchone()
        return dict(row) if row else None

    @synchronized
    def start(self, logical, judge, sample_id, request, retry_uncertain):
        old = self.latest(logical)
        if old and old["status"] == "COMPLETED":
            return old, False
        if old and old["status"] == "STARTED":
            if not retry_uncertain:
                return old, False
            with self.db:
                self.db.execute("UPDATE calls SET status='UNCERTAIN' WHERE call_id=?", (old["call_id"],))
        elif old and old["status"] == "FAILED" and old["invocation"] == self.invocation:
            return old, False
        generation = old["generation"] + 1 if old else 0
        call_id = f"{logical}:{generation}"
        with self.db:
            self.db.execute("INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,NULL,NULL)",
                            (call_id, logical, generation, judge, sample_id, "STARTED", self.invocation, _json(request)))
        return self.latest(logical), True

    @synchronized
    def finish(self, call_id, response=None, error=None):
        with self.db:
            self.db.execute("UPDATE calls SET status=?,response_json=?,error_json=? WHERE call_id=?",
                            ("COMPLETED" if response is not None else "FAILED",
                             _json(response) if response is not None else None,
                             _json(error) if error is not None else None, call_id))

    @synchronized
    def calls(self, name=None):
        if name is None:
            rows = self.db.execute("SELECT * FROM calls ORDER BY rowid").fetchall()
        else:
            rows = self.db.execute("SELECT * FROM calls WHERE judge=? ORDER BY rowid", (name,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("request", "response", "error"):
                content = item.pop(key + "_json")
                item[key] = json.loads(content) if content is not None else None
            result.append(item)
        return result

    @synchronized
    def close(self):
        self.db.close()


class JudgeFailure(Exception):
    def __init__(self, code, fatal=False, uncertain=False):
        self.code, self.fatal, self.uncertain = code, fatal, uncertain


class _LazyProvider:
    """仅在真正需要新请求时检查认证；读取已保存响应不需要重新配置密钥。"""
    def __init__(self, judge, journal):
        self.judge, self.journal, self.client = judge, journal, None

    def prepare(self):
        if self.client is None:
            try:
                client = make_judge_provider(self.judge)
                endpoint = resolved_base_url(self.judge) if self.judge.get("provider") else self.judge.get("base_url")
                self.journal.bind_endpoint(self.judge["name"], endpoint or "provider-default")
            except ValueError:
                raise JudgeFailure("JUDGE_CONFIGURATION_MISSING_OR_INVALID", fatal=True) from None
            self.client = client

    def complete(self, prompt, request, call_id):
        return self.client.complete(prompt, request, call_id)


def _safe_response(response):
    """只保留共同响应协议，不允许 SDK headers/密钥/隐藏推理字段进入日志。"""
    keys = ("content", "request_id", "usage", "latency_ms", "provider_name", "model_name", "finish_reason")
    result = {key: response.get(key) for key in keys}
    usage = result.get("usage")
    result["usage"] = ({key: usage.get(key) if type(usage.get(key)) is int else None
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
                       if isinstance(usage, dict) else None)
    return result


def _call(journal, provider, judge, sample_id, request, prompt, config, repair, log, retry_uncertain):
    for attempt in range(config.get("max_api_retries", 2) + 1):
        logical = fingerprint([judge["name"], sample_id, repair, attempt])
        latest = journal.latest(logical)
        cached = latest and (latest["status"] == "COMPLETED" or
                             (latest["status"] == "STARTED" and not retry_uncertain))
        if not cached:
            provider.prepare()
        row, execute = journal.start(logical, judge["name"], sample_id, request, retry_uncertain)
        if not execute:
            if row["status"] == "COMPLETED":
                return json.loads(row["response_json"])["content"]
            if row["status"] == "STARTED":
                raise JudgeFailure("CALL_UNCERTAIN", uncertain=True)
            error = json.loads(row["error_json"] or "{}")
            if error.get("retryable") and attempt < config.get("max_api_retries", 2):
                continue
            raise JudgeFailure(error.get("code", "JUDGE_API_ERROR"), fatal=error.get("fatal", False))
        log(f'[裁判] {judge["name"]} / {sample_id} / 尝试 {attempt + 1} / 格式修复 {repair}')
        try:
            response = provider.complete(prompt, request, row["call_id"])
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status == 429 or (type(status) is int and 500 <= status < 600)
            retryable |= type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectionError")
            code = f"API_{status or type(exc).__name__}"
            fatal = status in (400, 401, 403, 404, 422)
            journal.finish(row["call_id"], error={"code": code, "retryable": retryable, "fatal": fatal})
            if retryable and attempt < config.get("max_api_retries", 2):
                time.sleep(min(2 ** attempt, 8))
                continue
            raise JudgeFailure(code, fatal=fatal) from None
        response = _safe_response(response)
        journal.finish(row["call_id"], response=response)
        return response["content"]
    raise JudgeFailure("JUDGE_RETRIES_EXHAUSTED")


def _stats(calls):
    complete = all(c["status"] == "COMPLETED" and isinstance(c.get("response", {}).get("usage"), dict)
                   and all(type(c["response"]["usage"].get(k)) is int for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                   for c in calls)
    result = {"model_calls": len(calls), "completed_calls": sum(c["status"] == "COMPLETED" for c in calls),
              "failed_calls": sum(c["status"] == "FAILED" for c in calls),
              "uncertain_calls": sum(c["status"] in ("STARTED", "UNCERTAIN") for c in calls), "usage_complete": complete}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [c["response"]["usage"][key] for c in calls if c.get("response") and c["response"].get("usage")
                  and type(c["response"]["usage"].get(key)) is int]
        result[key] = sum(values) if complete else None
        result["known_" + key] = sum(values) if values else (0 if not calls else None)
    return result


def _export_judge(folder, name, alignments, metrics, failures, journal):
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / "alignment.json", alignments)
    _write(folder / "metrics.json", metrics)
    atomic_write(folder / "failures.jsonl", "".join(_json(item) + "\n" for item in failures))
    calls = journal.calls(name)
    atomic_write(folder / "calls.jsonl", "".join(_json(item) + "\n" for item in calls))
    return _stats(calls)


def run_judges(predictions, gold_rows, run_dir, config, log=print, retry_uncertain=False):
    """分别独立对齐，返回各裁判完整指标；均值由 reporting 层另行计算。"""
    if not config.get("enabled", True):
        raise ValueError("裁判配置 enabled=false，不能执行付费自动评分")
    judges = config.get("judges", [])
    if not judges or len({j["name"] for j in judges}) != len(judges):
        raise ValueError("裁判列表不能为空，名称必须唯一")
    indexed = {p["sample_id"]: p for p in predictions}
    gold = {s["sample_id"]: s for s in gold_rows}
    if len(indexed) != len(predictions) or len(gold) != len(gold_rows) or set(indexed) - set(gold):
        raise ValueError("裁判输入 sample_id 重复或不属于数据集")
    requests = {sid: build_judge_request(sample, indexed[sid]) for sid, sample in gold.items()
                if sid in indexed and indexed[sid].get("status") != "INPUT_INVALID"}
    base_prompt = (Path(__file__).parent / "prompts" / "judge.txt").read_text(encoding="utf-8")
    prompt = base_prompt + "\n输出结构（严格遵守，不能增加字段）：\n" + _json(JudgeAnswer.model_json_schema())
    directory = Path(run_dir) / "judges"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "1.0", "config_fingerprint": fingerprint(config),
                "gold_fingerprint": fingerprint(gold_rows), "prompt_fingerprint": fingerprint(prompt)}
    journal = JudgeJournal(directory / "state.sqlite3", manifest)
    result = {"per_judge": {}, "complete": True, "scoring_source": "llm_judged", "judge_count": len(judges)}
    try:
        for sid, prediction in indexed.items():
            if prediction.get("status") not in ("INPUT_INVALID", "PENDING", "RUNNING", "PAUSED_UNCERTAIN"):
                journal.bind_prediction(sid, requests[sid], prediction)
        def evaluate_judge(judge):
            name = judge["name"]
            folder = directory / hashlib.sha256(name.encode()).hexdigest()[:20]
            records, failures = [], []
            provider = _LazyProvider(judge, journal)
            blocked = False
            eligible, completed = 0, 0
            for sid, sample in gold.items():
                prediction = indexed.get(sid)
                if prediction is None or prediction.get("status") in ("PENDING", "RUNNING", "PAUSED_UNCERTAIN"):
                    failures.append({"sample_id": sid, "code": "PREDICTION_NOT_READY"})
                    continue
                if prediction.get("status") == "INPUT_INVALID":
                    continue
                request = requests[sid]
                eligible += 1
                if prediction.get("status") in ("FAILED", "PARTIAL") and not any(t["predicted_claims"] for t in request["targets"]):
                    records.extend(_empty_failure_alignment(sample, request))
                    completed += 1
                    continue
                if blocked:
                    failures.append({"sample_id": sid, "code": "JUDGE_BLOCKED_BY_PRIOR_ERROR"})
                    continue
                if len(prompt) + len(_json(request)) > config.get("max_input_chars", 120000):
                    failures.append({"sample_id": sid, "code": "JUDGE_CONTEXT_LIMIT"})
                    continue
                current_request = request
                for repair in range(config.get("max_format_repairs", 1) + 1):
                    try:
                        text = _call(journal, provider, judge, sid, current_request, prompt, config, repair, log, retry_uncertain)
                        accepted = parse_judge_answer(text, sample, prediction, request)
                    except JudgeFailure as exc:
                        failures.append({"sample_id": sid, "code": exc.code})
                        blocked |= exc.fatal
                        break
                    except (ValueError, TypeError):
                        if repair < config.get("max_format_repairs", 1):
                            current_request = {**request, "format_repair": "上一响应结构或引用不合法，请按给定 schema 重新输出；不要省略任何 target。"}
                            if len(prompt) + len(_json(current_request)) > config.get("max_input_chars", 120000):
                                failures.append({"sample_id": sid, "code": "JUDGE_CONTEXT_LIMIT"})
                                break
                            continue
                        failures.append({"sample_id": sid, "code": "INVALID_JUDGE_RESPONSE"})
                        break
                    records.extend(accepted)
                    if _records_complete(accepted, sample["task"]):
                        completed += 1
                    else:
                        failures.append({"sample_id": sid, "code": "JUDGE_UNSCORED"})
                    break
            metrics = evaluate_predictions(predictions, gold_rows, records)
            stats = _export_judge(folder, name, records, metrics, failures, journal)
            if any(f["code"] == "CALL_UNCERTAIN" for f in failures):
                status = "PAUSED_UNCERTAIN"
            elif not failures and eligible > 0 and completed == eligible:
                status = "COMPLETED"
            else:
                status = "PARTIAL" if completed else "FAILED"
            item = {"status": status, "model": judge["model"], "alignment": records, "metrics": metrics,
                    "failures": failures, "eligible_sample_count": eligible, "completed_sample_count": completed,
                    "output_dir": str(folder), "request_stats": stats}
            log(f'[裁判结果] {name}: {status}，完整评分 {completed}/{eligible}')
            return name, item

        workers = config.get("workers", 1)
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("裁判 workers 必须为 1–16 的整数")
        if workers == 1:
            items = map(evaluate_judge, judges)
            for name, item in items:
                result["per_judge"][name] = item
                result["complete"] &= item["status"] == "COMPLETED"
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(judges)), thread_name_prefix="judge") as pool:
                for name, item in pool.map(evaluate_judge, judges):
                    result["per_judge"][name] = item
                    result["complete"] &= item["status"] == "COMPLETED"
        result["request_stats"] = _stats(journal.calls())
        _write(directory / "results.json", result)
        return result
    finally:
        journal.close()
