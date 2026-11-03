"""单进程 SQLite 状态库：状态、动作与消息一起提交，导出文件只是只读视图。

远端调用分为 STARTED 和 COMPLETED 两次提交。若进程在两者之间退出，
调用是否已收费无法确定；调用者必须暂停或明确选择重试，不能默默重发。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from functools import wraps
from threading import RLock
from collections import Counter
from pathlib import Path
from typing import Any


def _safe(value: Any) -> Any:
    """额外保险：不保存凭证、请求头和模型的隐藏推理字段。"""
    private = {"api_key", "authorization", "proxy_authorization", "headers",
               "reasoning_content", "secret", "password", "access_token"}
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()
                if key.lower().replace("-", "_") not in private}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(_safe(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def atomic_write(path: Path | str, text: str) -> None:
    """同目录临时文件加原子替换，避免中断留下半份 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def synchronized(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return guarded


class Store:
    """所有样本都存入一份数据库；sample_id 不参与磁盘路径拼接。"""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = sqlite3.connect(self.run_dir / "state.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_key INTEGER PRIMARY KEY CHECK(run_key = 1),
                manifest_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes (
                sample_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                state_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL REFERENCES episodes(sample_id),
                event_seq INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                UNIQUE(sample_id, event_seq)
            );
            CREATE TABLE IF NOT EXISTS model_calls (
                call_id TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL REFERENCES episodes(sample_id),
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                request_fingerprint TEXT NOT NULL,
                response_json TEXT,
                error_json TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL REFERENCES episodes(sample_id),
                message_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                sample_id TEXT PRIMARY KEY REFERENCES episodes(sample_id),
                prediction_json TEXT NOT NULL
            );
        """)

    @synchronized
    def initialize(self, manifest: dict) -> None:
        """初始化新运行；已有运行只允许完全相同的输入、配置及版本指纹。"""
        content = _json(manifest)
        existing = self.db.execute("SELECT manifest_json FROM runs WHERE run_key=1").fetchone()
        if existing is not None and existing[0] != content:
            raise ValueError("RUN_FINGERPRINT_MISMATCH: 输入、配置、提示词或版本改变，请新建 run-dir")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO runs VALUES (1, ?)", (content,))
        atomic_write(self.run_dir / "manifest.json",
                     json.dumps(_safe(manifest), ensure_ascii=False, indent=2) + "\n")

    @synchronized
    def get_state(self, sample_id: str) -> dict | None:
        row = self.db.execute("SELECT state_json FROM episodes WHERE sample_id=?", (sample_id,)).fetchone()
        return json.loads(row[0]) if row else None

    @synchronized
    def save_state(self, sample_id: str, state: dict, event: dict | None = None,
                   messages: list[dict] | None = None, prediction: dict | None = None) -> None:
        """动作事件、完整 checkpoint、消息 outbox、预测在一个事务中写入。

        同一动作 ID 的相同重放直接返回，不能把旧 checkpoint 覆盖到新状态上。
        消息从 outbox 转为实际投递时只允许增加 delivered_event_seq。
        """
        with self.db:
            if event is not None:
                seq = event.get("event_seq", event.get("seq"))
                if type(seq) is not int or seq < 0:
                    raise ValueError("event.event_seq 必须是非负整数")
                event_id = event.get("event_id", f"{sample_id}:{seq}")
                content = _json(event)
                rows = self.db.execute(
                    "SELECT event_id,sample_id,event_seq,event_json FROM events "
                    "WHERE event_id=? OR (sample_id=? AND event_seq=?)",
                    (event_id, sample_id, seq),
                ).fetchall()
                if rows:
                    if len(rows) != 1 or tuple(rows[0]) != (event_id, sample_id, seq, content):
                        raise ValueError("IDEMPOTENCY_CONFLICT: 动作 ID 或序号对应不同内容")
                    return
            self.db.execute(
                "INSERT INTO episodes VALUES(?,?,?) ON CONFLICT(sample_id) "
                "DO UPDATE SET status=excluded.status,state_json=excluded.state_json",
                (sample_id, state.get("status", "PENDING"), _json(state)),
            )
            if event is not None:
                self.db.execute("INSERT INTO events VALUES(?,?,?,?)",
                                (event_id, sample_id, seq, content))
            for message in messages or []:
                self._save_message(sample_id, message)
            if prediction is not None:
                self.db.execute(
                    "INSERT INTO predictions VALUES(?,?) ON CONFLICT(sample_id) "
                    "DO UPDATE SET prediction_json=excluded.prediction_json",
                    (sample_id, _json(prediction)),
                )

    def _save_message(self, sample_id: str, message: dict) -> None:
        message_id = message["message_id"]
        old = self.db.execute("SELECT sample_id,message_json FROM messages WHERE message_id=?",
                              (message_id,)).fetchone()
        content = _json(message)
        if old:
            previous = json.loads(old["message_json"])
            updated = _safe(message)
            # 实际投递是已有消息的一次合法状态迁移，消息正文不可更改。
            before = previous.pop("delivered_event_seq", None)
            after = updated.pop("delivered_event_seq", None)
            if old["sample_id"] != sample_id or previous != updated:
                raise ValueError("IDEMPOTENCY_CONFLICT: 消息 ID 对应不同内容")
            if before is not None and after != before:
                raise ValueError("IDEMPOTENCY_CONFLICT: 已投递消息不能撤回或再次投递")
            self.db.execute("UPDATE messages SET message_json=? WHERE message_id=?", (content, message_id))
        else:
            self.db.execute("INSERT INTO messages VALUES(?,?,?)", (message_id, sample_id, content))

    @synchronized
    def get_call(self, call_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM model_calls WHERE call_id=?", (call_id,)).fetchone()
        return self._call_dict(row) if row else None

    @staticmethod
    def _call_dict(row: sqlite3.Row) -> dict:
        return {"call_id": row["call_id"], "sample_id": row["sample_id"],
                "stage": row["stage"], "status": row["status"],
                "request_fingerprint": row["request_fingerprint"],
                "request": json.loads(row["request_json"]),
                "response": json.loads(row["response_json"]) if row["response_json"] else None,
                "error": json.loads(row["error_json"]) if row["error_json"] else None}

    @synchronized
    def start_call(self, call_id: str, sample_id: str, stage: str, request: dict) -> None:
        content = _json(request)
        old = self.get_call(call_id)
        if old:
            if old["sample_id"] != sample_id or old["stage"] != stage or _json(old["request"]) != content:
                raise ValueError("IDEMPOTENCY_CONFLICT: 调用 ID 对应不同请求")
            return
        with self.db:
            self.db.execute("INSERT INTO model_calls VALUES(?,?,?,?,?,?,NULL,NULL)",
                            (call_id, sample_id, stage, "STARTED", content,
                             hashlib.sha256(content.encode("utf-8")).hexdigest()))

    @synchronized
    def complete_call(self, call_id: str, response: dict) -> None:
        old = self.get_call(call_id)
        if old is None:
            raise ValueError("UNKNOWN_CALL: 必须先记录 CALL_STARTED")
        content = _json(response)
        if old["status"] == "COMPLETED":
            if _json(old["response"]) != content:
                raise ValueError("IDEMPOTENCY_CONFLICT: 已完成调用的响应不可更改")
            return
        with self.db:
            self.db.execute("UPDATE model_calls SET status='COMPLETED',response_json=?,error_json=NULL "
                            "WHERE call_id=?", (content, call_id))

    @synchronized
    def fail_call(self, call_id: str, error: dict) -> None:
        old = self.get_call(call_id)
        if old is None:
            raise ValueError("UNKNOWN_CALL: 必须先记录 CALL_STARTED")
        if old["status"] == "COMPLETED":
            raise ValueError("IDEMPOTENCY_CONFLICT: 已完成调用不能变为失败")
        with self.db:
            self.db.execute("UPDATE model_calls SET status='FAILED',error_json=? WHERE call_id=?",
                            (_json(error), call_id))

    @synchronized
    def uncertain_calls(self, sample_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM model_calls WHERE status='STARTED'"
        parameters = ()
        if sample_id is not None:
            query += " AND sample_id=?"
            parameters = (sample_id,)
        return [self._call_dict(row) for row in self.db.execute(query + " ORDER BY rowid", parameters)]

    @synchronized
    def list_states(self) -> list[dict]:
        return self._list_json("episodes", "state_json")

    @synchronized
    def list_predictions(self) -> list[dict]:
        return self._list_json("predictions", "prediction_json")

    @synchronized
    def list_events(self) -> list[dict]:
        return self._list_json("events", "event_json")

    @synchronized
    def list_messages(self) -> list[dict]:
        return self._list_json("messages", "message_json")

    @synchronized
    def list_calls(self) -> list[dict]:
        return [self._call_dict(row) for row in self.db.execute("SELECT * FROM model_calls ORDER BY rowid")]

    def _list_json(self, table: str, column: str) -> list[dict]:
        # table/column 仅由上面的固定方法传入，用户输入不用于 SQL 标识符。
        return [json.loads(row[0]) for row in self.db.execute(f"SELECT {column} FROM {table} ORDER BY rowid")]

    @synchronized
    def export(self) -> dict:
        """从已提交事务导出；API 未报告的 token 不会被当作零。"""
        predictions = self.list_predictions()
        states = self.list_states()
        calls = self.list_calls()
        events = [{"record_type": "event", **event} for event in self.list_events()]
        traces = events + [{"record_type": "model_call", **call} for call in calls]
        failures = [p for p in predictions if p.get("status") not in ("COMPLETED", "PENDING", "RUNNING")]
        for name, items in (("predictions", predictions), ("traces", traces),
                            ("messages", self.list_messages()), ("failures", failures)):
            atomic_write(self.run_dir / f"{name}.jsonl", "".join(_json(item) + "\n" for item in items))
        progress = {"total": len(states),
                    "statuses": dict(Counter(state.get("status", "PENDING") for state in states)),
                    "model_calls": len(calls), "uncertain_calls": len(self.uncertain_calls())}
        atomic_write(self.run_dir / "progress.json", json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
        return progress

    @synchronized
    def close(self) -> None:
        self.db.close()
