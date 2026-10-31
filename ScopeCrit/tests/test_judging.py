"""裁判测试全部使用人工 fixture 和假客户端，不访问付费服务。"""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from scopecrit import judging
from scopecrit.judging import JudgeJournal, build_judge_request, parse_judge_answer, run_judges


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset():
    samples = json.loads((ROOT / "examples/merged_demo.json").read_text())
    annotations = json.loads((ROOT / "examples/alignment_demo.json").read_text())
    predictions = []
    for sample in samples:
        prefix = "I:" if sample["task"] == "T4" else ""
        claims = []
        for index, gold in enumerate(sample["gold"]["claims"], 1):
            claim = deepcopy(gold)
            del claim["claim_id"]
            claim["predicted_claim_id"] = f"{prefix}P{index:03d}"
            claims.append(claim)
        prediction = {"sample_id": sample["sample_id"], "task": sample["task"], "status": "COMPLETED",
                      "model_name": "INFERENCE_MODEL_NAME_CANARY", "prediction": {"claims": claims,
                      "answer": sample["gold"]["reference_answer"], "relations": []}, "turns": [], "messages": []}
        for event in sample["events"]:
            ids = set(event["gold_memory_state"])
            selected = [claim for claim, gold in zip(claims, sample["gold"]["claims"]) if gold["claim_id"] in ids]
            prediction["turns"].append({"turn": event["turn"], "claims": selected, "answer": event["reference_answer"]})
        predictions.append(prediction)
    return samples, predictions, annotations


def config(names=("one", "two", "three", "four")):
    return {"enabled": True, "fixture": True,
            "judges": [{"name": n, "model": "fake-model", "base_url": "https://fake.invalid"} for n in names],
            "max_api_retries": 0, "max_format_repairs": 1, "max_input_chars": 120000}


class FakeProvider:
    def __init__(self, records, seen):
        self.records, self.seen = records, seen

    def complete(self, prompt, request, call_id):
        self.seen.append((prompt, deepcopy(request), call_id))
        sid = request["targets"][0]["sample_id"]
        result = {"alignments": [r for r in self.records if r["sample_id"] == sid]}
        return {"content": json.dumps(result, ensure_ascii=False), "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "reasoning_content": "PRIVATE_REASONING_CANARY", "headers": {"Authorization": "PRIVATE_KEY_CANARY"}}


def fake_factory(monkeypatch, annotations, seen):
    monkeypatch.setattr(judging, "make_judge_provider", lambda _: FakeProvider(annotations, seen))


def test_four_independent_judges_all_tasks_and_cache(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    seen = []
    fake_factory(monkeypatch, annotations, seen)
    report = run_judges(predictions, samples, tmp_path, config(), log=lambda _: None)
    assert report["complete"]
    assert len(seen) == 16
    assert report["request_stats"]["model_calls"] == 16
    assert report["request_stats"]["total_tokens"] == 80
    for judge in report["per_judge"].values():
        assert judge["metrics"]["per_task"]["T1"]["aspect_micro_f1"] == 1
        assert judge["metrics"]["per_task"]["T3"]["memory_state_f1"] == 1
        assert Path(judge["output_dir"], "alignment.json").exists()
    repeated = run_judges(predictions, samples, tmp_path, config(), log=lambda _: None)
    assert repeated["complete"] and len(seen) == 16
    for path in (tmp_path / "judges").rglob("*.json*"):
        text = path.read_text()
        assert "PRIVATE_REASONING_CANARY" not in text
        assert "PRIVATE_KEY_CANARY" not in text


def test_requests_are_blind_and_drop_label_and_extra_metadata(dataset):
    samples, predictions, _ = dataset
    samples[0]["document"]["evidence"][0]["secret_extra"] = "EVIDENCE_EXTRA_CANARY"
    request = build_judge_request(samples[0], predictions[0])
    text = json.dumps(request)
    assert "INFERENCE_MODEL_NAME_CANARY" not in text
    assert "EVIDENCE_EXTRA_CANARY" not in text
    assert '"aspect"' not in text and '"stance"' not in text
    assert "target_scope" in text


def test_t3_requires_final_and_each_turn(dataset):
    samples, predictions, annotations = dataset
    sample, prediction = samples[2], predictions[2]
    request = build_judge_request(sample, prediction)
    assert {t["turn"] for t in request["targets"]} == {None, 1, 2}
    records = [r for r in annotations if r["sample_id"] == sample["sample_id"]]
    with pytest.raises(ValueError, match="JUDGE_TARGET_MISMATCH"):
        parse_judge_answer(json.dumps({"alignments": records[:-1]}), sample, prediction, request)


@pytest.mark.parametrize("bad", ["duplicate", "unknown", "extra"])
def test_illegal_alignment_rejected(dataset, bad):
    samples, predictions, annotations = dataset
    sample, prediction = samples[0], predictions[0]
    records = deepcopy([annotations[0]])
    if bad == "duplicate":
        records[0]["matches"].append(deepcopy(records[0]["matches"][0]))
    elif bad == "unknown":
        records[0]["matches"][0]["gold_claim_id"] = "C404"
    else:
        records[0]["extra"] = "not allowed"
    with pytest.raises(ValueError):
        parse_judge_answer(json.dumps({"alignments": records}), sample, prediction, build_judge_request(sample, prediction))


def test_null_dual_dependency_remains_unscored(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    records = deepcopy(annotations)
    records[-1]["dual_dependency"] = None
    fake_factory(monkeypatch, records, [])
    report = run_judges(predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    judge = report["per_judge"]["one"]
    assert judge["status"] == "PARTIAL"
    assert judge["metrics"]["per_task"]["T4"]["task_success_rate"] is None
    assert "dual_dependency" not in judge["alignment"][-1]


def test_format_repair_is_separate_recorded_call(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    seen = []
    provider = FakeProvider(annotations, seen)
    complete = provider.complete
    counter = 0

    def malformed_once(prompt, request, call_id):
        nonlocal counter
        counter += 1
        if counter == 1:
            return {"content": "{broken", "usage": None}
        assert "format_repair" in request
        return complete(prompt, request, call_id)

    provider.complete = malformed_once
    monkeypatch.setattr(judging, "make_judge_provider", lambda _: provider)
    report = run_judges(predictions[:1], samples[:1], tmp_path, config(("one",)), log=lambda _: None)
    assert report["complete"]
    assert report["request_stats"]["model_calls"] == 2
    assert report["request_stats"]["total_tokens"] is None


def test_auth_failure_stops_one_judge_others_continue(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    seen = []

    class Unauthorized(Exception):
        status_code = 401

    class FailedProvider:
        def complete(self, *args):
            seen.append("failed")
            raise Unauthorized("PRIVATE_KEY_DO_NOT_LOG")

    monkeypatch.setattr(judging, "make_judge_provider", lambda judge: FailedProvider() if judge["name"] == "bad" else FakeProvider(annotations, seen))
    report = run_judges(predictions, samples, tmp_path, config(("bad", "good")), log=lambda _: None)
    assert report["per_judge"]["bad"]["status"] == "FAILED"
    assert report["per_judge"]["good"]["status"] == "COMPLETED"
    assert seen.count("failed") == 1
    assert report["request_stats"]["total_tokens"] is None
    assert "PRIVATE_KEY_DO_NOT_LOG" not in (tmp_path / "judges/results.json").read_text()


def test_missing_credentials_can_be_supplied_on_later_run(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    monkeypatch.setattr(judging, "make_judge_provider", lambda _: (_ for _ in ()).throw(ValueError("PRIVATE_KEY")))
    first = run_judges(predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    assert first["request_stats"]["model_calls"] == 0 and not first["complete"]
    fake_factory(monkeypatch, annotations, [])
    second = run_judges(predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    assert second["complete"] and second["request_stats"]["model_calls"] == 4


def test_known_failed_api_call_can_retry_on_later_invocation(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset

    class Failing:
        def complete(self, *args):
            raise TimeoutError("private")

    monkeypatch.setattr(judging, "make_judge_provider", lambda _: Failing())
    first = run_judges(predictions[:1], samples[:1], tmp_path, config(("one",)), log=lambda _: None)
    assert first["request_stats"]["model_calls"] == 1
    fake_factory(monkeypatch, annotations, [])
    second = run_judges(predictions[:1], samples[:1], tmp_path, config(("one",)), log=lambda _: None)
    assert second["complete"] and second["request_stats"]["model_calls"] == 2


def test_uncertain_call_pauses_without_resending_and_explicit_retry(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    seen = []
    fake_factory(monkeypatch, annotations, seen)
    cfg = config(("one",))
    # 先正常生成 manifest，再将已存响应模拟为只留下 STARTED 的崩溃。
    run_judges(predictions[:1], samples[:1], tmp_path, cfg, log=lambda _: None)
    with sqlite3.connect(tmp_path / "judges/state.sqlite3") as db:
        db.execute("UPDATE calls SET status='STARTED',response_json=NULL")
    seen.clear()
    paused = run_judges(predictions[:1], samples[:1], tmp_path, cfg, log=lambda _: None)
    assert paused["per_judge"]["one"]["status"] == "PAUSED_UNCERTAIN" and not seen
    retried = run_judges(predictions[:1], samples[:1], tmp_path, cfg, log=lambda _: None, retry_uncertain=True)
    assert retried["complete"] and len(seen) == 1
    assert retried["request_stats"]["model_calls"] == 2
    assert retried["request_stats"]["total_tokens"] is None
    assert retried["request_stats"]["uncertain_calls"] == 1


def test_pending_invalid_and_empty_failure_do_not_call_judges(dataset, tmp_path, monkeypatch):
    samples, predictions, _ = dataset
    predictions[0]["status"] = "INPUT_INVALID"
    predictions[1]["status"] = "PENDING"
    for prediction in predictions[2:]:
        prediction["status"] = "FAILED"
        prediction["prediction"] = {"claims": [], "answer": ""}
        prediction["turns"] = []
    monkeypatch.setattr(judging, "make_judge_provider", lambda _: (_ for _ in ()).throw(AssertionError("不要调用")))
    result = run_judges(predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    assert result["request_stats"]["model_calls"] == 0
    metrics = result["per_judge"]["one"]["metrics"]
    assert metrics["per_task"]["T4"]["task_success_rate"] == 0
    assert metrics["per_task"]["T3"]["memory_state_f1"] == 0


def test_changed_input_or_judge_config_refuses_cached_scores(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    fake_factory(monkeypatch, annotations, [])
    run_judges(predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    changed = config(("one",))
    changed["judges"][0]["model"] = "different"
    with pytest.raises(ValueError, match="JUDGE_FINGERPRINT_MISMATCH"):
        run_judges(predictions, samples, tmp_path, changed, log=lambda _: None)


def test_newly_ready_prediction_reuses_existing_judge_results(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    seen = []
    fake_factory(monkeypatch, annotations, seen)
    first_predictions = deepcopy(predictions)
    first_predictions[1]["status"] = "PAUSED_UNCERTAIN"
    first_predictions[1]["prediction"] = {"claims": [], "answer": ""}
    first = run_judges(first_predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    assert not first["complete"] and len(seen) == 3
    second = run_judges(predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    assert second["complete"] and len(seen) == 4


def test_changed_already_judged_prediction_is_rejected(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    fake_factory(monkeypatch, annotations, [])
    run_judges(predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    changed = deepcopy(predictions)
    changed[0]["prediction"]["answer"] += "改变了答案"
    with pytest.raises(ValueError, match="JUDGE_PREDICTION_CHANGED"):
        run_judges(changed, samples, tmp_path, config(("one",)), log=lambda _: None)


def test_complete_cached_responses_do_not_require_credentials_again(dataset, tmp_path, monkeypatch):
    samples, predictions, annotations = dataset
    seen = []
    fake_factory(monkeypatch, annotations, seen)
    run_judges(predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    monkeypatch.setattr(judging, "make_judge_provider", lambda _: (_ for _ in ()).throw(ValueError("missing key")))
    cached = run_judges(predictions, samples, tmp_path, config(("one",)), log=lambda _: None)
    assert cached["complete"] and len(seen) == 4
