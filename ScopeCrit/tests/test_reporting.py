"""综合报告只平均同一指标，缺失裁判、未运行与无效输入不伪装为满分。"""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from scopecrit.config import load_config
from scopecrit.engine import Engine
from scopecrit.evaluation import evaluate_predictions
from scopecrit.providers import make_provider
from scopecrit.reporting import build_report, format_console, write_report
from scopecrit.storage import Store


ROOT = Path(__file__).resolve().parents[1]
NAMES = ["deepseek-flash", "GLM-4.5-Air", "qwen3.8-flash", "grok-build-0.1"]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    rows = json.loads((ROOT / "examples/merged_demo.json").read_text(encoding="utf-8"))
    annotations = json.loads((ROOT / "examples/alignment_demo.json").read_text(encoding="utf-8"))
    config = load_config(ROOT / "configs/mock.yaml")
    store = Store(tmp_path_factory.mktemp("report_demo"))
    store.initialize({"test": "reporting-fixture"})
    provider = make_provider(config)
    predictions = [Engine(row, config, provider, store, log=lambda _: None).run() for row in rows]
    store.close()
    return predictions, rows, annotations


def reports(demo):
    predictions, rows, annotations = deepcopy(demo)
    complete = evaluate_predictions(predictions, rows, annotations)
    structural = evaluate_predictions(predictions, rows)
    judges = {name: {"status": "completed", "metrics": deepcopy(complete), "failures": []} for name in NAMES}
    return judges, structural


def test_all_four_judges_and_all_tasks_yield_expected_summary(demo):
    judges, structural = reports(demo)
    report = build_report(judges, structural, "合成演示.json", NAMES)
    assert report["overall"]["macro_task_score"] == 1
    for task in ("T1", "T2", "T3", "T4"):
        assert report["per_task"][task]["composite"]["mean"] == 1
        assert report["per_task"][task]["composite"]["valid_judges"] == 4
    assert report["per_task"]["T3"]["metrics"]["false_overwrite"]["mean"] == 0
    assert report["per_task"]["T3"]["composite"]["weights"] == {"memory_turn_mean_f1": 1}
    assert report["per_task"]["T4"]["composite"]["weights"] == {"task_success_rate": 1}


def test_same_metric_is_averaged_after_independent_alignments(demo):
    predictions, rows, annotations = deepcopy(demo)
    judges, structural = reports(demo)
    target = next(record for record in annotations if record["sample_id"] == "Demo:T1")
    target["matches"] = target["matches"][:1]
    judges[NAMES[-1]]["metrics"] = evaluate_predictions(predictions, rows, annotations)
    report = build_report(judges, structural, "demo", NAMES)
    stats = report["per_task"]["T1"]["metrics"]["aspect_micro_f1"]
    assert stats["mean"] == pytest.approx(0.875)
    assert stats["min"] == 0.5
    assert stats["max"] == 1
    assert stats["std"] > 0
    assert stats["per_judge"][NAMES[-1]]["value"] == 0.5


def test_deterministic_evidence_is_computed_once_not_voted(demo):
    judges, structural = reports(demo)
    for result in judges.values():
        result["metrics"]["per_task"]["T1"]["evidence_f1"] = 0
    stats = build_report(judges, structural, "demo", NAMES)["per_task"]["T1"]["metrics"]["evidence_f1"]
    assert stats["source"] == "program"
    assert stats["mean"] == 1
    assert stats["valid_judges"] == stats["expected_judges"] == 0


def test_missing_judge_does_not_silently_become_three_judge_mean(demo):
    judges, structural = reports(demo)
    del judges[NAMES[-1]]
    report = build_report(judges, structural, "demo", NAMES)
    stats = report["per_task"]["T1"]["metrics"]["aspect_micro_f1"]
    assert stats["mean"] is None
    assert stats["available_mean"] == 1
    assert stats["valid_judges"] == 3
    assert stats["expected_judges"] == 4
    assert report["overall"]["macro_task_score"] is None
    assert report["per_task"]["T1"]["metrics"]["evidence_f1"]["mean"] == 1


def test_partial_target_coverage_cannot_supply_full_judge_score(demo):
    predictions, rows, annotations = deepcopy(demo)
    judges, structural = reports(demo)
    # 最后一轮虽然有完整对齐，第一轮缺失仍应使完整 T3 评分不可用。
    annotations = [r for r in annotations if not (r["sample_id"] == "Demo:T3" and r["turn"] == 1)]
    judges[NAMES[-1]]["metrics"] = evaluate_predictions(predictions, rows, annotations)
    report = build_report(judges, structural, "demo", NAMES)
    assert report["per_task"]["T3"]["metrics"]["memory_state_f1"]["mean"] is None
    assert report["per_task"]["T3"]["composite"]["valid_judges"] == 3
    assert report["per_task"]["T3"]["metrics"]["memory_turn_mean_f1"]["mean"] is None
    assert report["per_task"]["T1"]["composite"]["mean"] == 1


def test_t2_uses_claim_f1_not_coverage_that_rewards_overprediction(demo):
    judges, structural = reports(demo)
    for result in judges.values():
        sample = next(s for s in result["metrics"]["per_sample"] if s["task"] == "T2")
        sample["semantic_counts"]["predicted"] = 3
    report = build_report(judges, structural, "demo", NAMES)
    section = report["per_task"]["T2"]
    assert section["metrics"]["claim_f1"]["mean"] == pytest.approx(0.8)
    assert section["metrics"]["claim_coverage"]["mean"] == 1
    assert section["composite"]["mean"] == pytest.approx((0.8 + 1 + 1) / 3)


def test_relation_partial_has_fixed_mapping_and_unscored_is_not_zero(demo):
    judges, structural = reports(demo)
    for result in judges.values():
        sample = next(s for s in result["metrics"]["per_sample"] if s["task"] == "T2")
        sample["relation_quality"] = "partial"
    report = build_report(judges, structural, "demo", NAMES)
    assert report["per_task"]["T2"]["metrics"]["relation_quality_score"]["mean"] == 0.5
    assert report["per_task"]["T2"]["composite"]["mean"] == pytest.approx(2.5 / 3)
    sample["relation_quality"] = "unscored"
    report = build_report(judges, structural, "demo", NAMES)
    assert report["per_task"]["T2"]["composite"]["mean"] is None


def test_absent_task_keeps_four_task_macro_null(demo):
    judges, structural = reports(demo)
    structural["per_sample"] = [s for s in structural["per_sample"] if s["task"] != "T4"]
    for result in judges.values():
        result["metrics"]["per_sample"] = [s for s in result["metrics"]["per_sample"] if s["task"] != "T4"]
    report = build_report(judges, structural, "demo", NAMES)
    assert report["overall"]["macro_task_score"] is None
    assert report["overall"]["available_task_mean"] == 1
    assert report["overall"]["valid_tasks"] == 3


def test_actual_leak_is_explicit_diagnostic_without_overwriting_mean(demo):
    judges, structural = reports(demo)
    sample = next(s for s in structural["per_sample"] if s["task"] == "T3")
    sample["future_or_private_citation_ids"] = ["E999"]
    report = build_report(judges, structural, "demo", NAMES)
    assert report["per_task"]["T3"]["metrics"]["memory_state_f1"]["mean"] == 1
    assert report["per_task"]["T3"]["validity_gate"]["passed"] is False
    assert report["per_task"]["T3"]["composite"]["mean"] == 1
    assert report["overall"]["macro_task_score"] == 1
    del judges[NAMES[-1]]
    report = build_report(judges, structural, "demo", NAMES)
    assert report["per_task"]["T3"]["composite"]["mean"] is None


def test_blocked_attempt_does_not_equal_actual_leak(demo):
    judges, structural = reports(demo)
    structural["per_sample"][0]["access_denied_attempts"] = 1
    assert build_report(judges, structural, "demo", NAMES)["per_task"]["T1"]["composite"]["mean"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_quality_values_are_rejected(demo, value):
    judges, structural = reports(demo)
    judges[NAMES[0]]["metrics"]["per_task"]["T1"]["aspect_micro_f1"] = value
    with pytest.raises(ValueError, match="0–1"):
        build_report(judges, structural, "demo", NAMES)


def test_writer_preserves_original_metrics_and_emits_all_levels(demo, tmp_path):
    judges, structural = reports(demo)
    report = build_report(judges, structural, "中文数据集.json", NAMES)
    original = tmp_path / "metrics.json"
    original.write_text('{"original": true}')
    write_report(report, tmp_path)
    assert json.loads(original.read_text()) == {"original": True}
    saved = json.loads((tmp_path / "judge_summary.json").read_text())
    assert saved["overall"]["macro_task_score"] == 1
    with (tmp_path / "metrics_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["task"] for r in rows if r["scope"] == "task"} == {"T1", "T2", "T3", "T4"}
    assert any(r["metric"] == "macro_task_score" for r in rows)
    with (tmp_path / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["turn"] for r in rows if r["scope"] == "turn"} == {"1", "2"}
    assert "四任务宏平均：1.0000" in format_console(report)
    assert "程序一次计算" in (tmp_path / "report.md").read_text()


def test_build_report_does_not_mutate_inputs(demo):
    judges, structural = reports(demo)
    before = deepcopy((judges, structural))
    build_report(judges, structural, "demo", NAMES)
    assert (judges, structural) == before
