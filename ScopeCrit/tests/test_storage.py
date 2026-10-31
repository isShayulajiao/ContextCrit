"""持久化测试只用临时 SQLite；不发出任何模型请求。"""

import json

import pytest

from scopecrit.storage import Store


def test_wal_fingerprints_and_reopen(tmp_path):
    store = Store(tmp_path)
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    manifest = {"input_fingerprint": "abc", "config": {"provider": "mock"}}
    store.initialize(manifest)
    store.initialize(manifest)
    with pytest.raises(ValueError, match="FINGERPRINT_MISMATCH"):
        store.initialize({**manifest, "input_fingerprint": "changed"})
    store.save_state("../a", {"sample_id": "../a", "status": "RUNNING", "memory": ["P001"]})
    store.close()
    reopened = Store(tmp_path)
    assert reopened.get_state("../a")["memory"] == ["P001"]
    assert not (tmp_path.parent / "a").exists()
    reopened.close()


def test_action_atomicity_and_old_replay_does_not_rewind(tmp_path):
    store = Store(tmp_path)
    first = {"status": "RUNNING", "memory": ["P001"]}
    event = {"event_seq": 1, "action": "COMMIT"}
    store.save_state("s", first, event)
    store.save_state("s", {"status": "COMPLETED", "memory": ["P001", "P002"]},
                     {"event_seq": 2, "action": "FINALIZE"})
    store.save_state("s", first, event)
    assert store.get_state("s")["memory"] == ["P001", "P002"]
    with pytest.raises(ValueError, match="IDEMPOTENCY_CONFLICT"):
        store.save_state("s", {"status": "FAILED"}, {"event_seq": 1, "action": "DIFFERENT"})
    assert store.get_state("s")["status"] == "COMPLETED"
    assert len(store.list_events()) == 2


def test_message_delivery_and_transaction_rollback(tmp_path):
    store = Store(tmp_path)
    pending = {"message_id": "s:MSG1", "evidence": [{"text": "准确原文"}], "delivered_event_seq": None}
    store.save_state("s", {"status": "RUNNING"}, {"event_seq": 1}, [pending])
    delivered = {**pending, "delivered_event_seq": 2}
    store.save_state("s", {"status": "COMPLETED"}, {"event_seq": 2}, [delivered])
    assert store.list_messages() == [delivered]
    with pytest.raises(ValueError, match="IDEMPOTENCY_CONFLICT"):
        store.save_state("s", {"status": "FAILED"}, {"event_seq": 3},
                         [{**delivered, "evidence": [{"text": "改写"}]}])
    assert store.get_state("s")["status"] == "COMPLETED"
    assert len(store.list_events()) == 2
    assert store.list_messages()[0]["evidence"][0]["text"] == "准确原文"


def test_call_reuse_uncertainty_and_credentials_excluded(tmp_path):
    store = Store(tmp_path)
    store.save_state("s", {"status": "RUNNING"})
    request = {"visible_evidence": [], "api_key": "never-store-this", "headers": {"Authorization": "no"}}
    store.start_call("s:0001", "s", "ACTION", request)
    assert len(store.uncertain_calls("s")) == 1
    response = {"content": "{}", "usage": None, "reasoning_content": "hidden"}
    store.complete_call("s:0001", response)
    assert store.uncertain_calls() == []
    assert store.get_call("s:0001")["response"] == {"content": "{}", "usage": None}
    store.complete_call("s:0001", response)
    with pytest.raises(ValueError, match="IDEMPOTENCY_CONFLICT"):
        store.complete_call("s:0001", {"content": "different"})
    with pytest.raises(ValueError, match="IDEMPOTENCY_CONFLICT"):
        store.start_call("s:0001", "s", "ACTION", {"visible_evidence": ["different"]})
    store.export()
    trace = (tmp_path / "traces.jsonl").read_text()
    assert "never-store-this" not in trace and "reasoning_content" not in trace
    assert store.get_call("s:0001")["response"]["usage"] is None


def test_export_prediction_and_failure(tmp_path):
    store = Store(tmp_path)
    prediction = {"sample_id": "s", "status": "PARTIAL", "termination_reason": "BUDGET_EXHAUSTED"}
    store.save_state("s", {"status": "PARTIAL"}, {"event_seq": 1}, prediction=prediction)
    progress = store.export()
    assert progress["statuses"] == {"PARTIAL": 1}
    assert json.loads((tmp_path / "predictions.jsonl").read_text()) == prediction
    assert json.loads((tmp_path / "failures.jsonl").read_text()) == prediction
