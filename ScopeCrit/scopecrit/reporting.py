"""四裁判的同指标平均与报告。

程序可核验的指标只计算一次。语义指标必须覆盖本次所选目标，且所有
预定裁判都有有效结果才给出正式均值；不足时只给明确标注的可用均值。
综合分采用预先写明的固定权重，不把成本、权限错误次数和缺失值混进平均。
"""
from __future__ import annotations

from copy import deepcopy
import csv
import io
import math
from pathlib import Path
from statistics import mean, pstdev

from .storage import atomic_write


TASKS = ("T1", "T2", "T3", "T4")
PROGRAM_METRICS = (
    "evidence_f1", "evidence_precision", "evidence_recall",
    "citation_validity_mean", "answer_citations_match_claims_rate",
    "expected_exchange_recall",
)
SEMANTIC_METRICS = (
    "claim_coverage", "claim_precision", "claim_f1", "aspect_micro_f1",
    "stance_macro_f1", "stance_f1_POSITIVE", "stance_f1_NEGATIVE",
    "stance_f1_MIXED", "stance_f1_NEUTRAL", "relation_quality_score",
    "memory_state_f1", "memory_turn_mean_f1", "false_overwrite", "task_success_rate",
)
COMPOSITES = {
    "T1": {"aspect_micro_f1": 1 / 3, "stance_macro_f1": 1 / 3, "evidence_f1": 1 / 3},
    "T2": {"claim_f1": 1 / 3, "relation_quality_score": 1 / 3, "evidence_f1": 1 / 3},
    "T3": {"memory_turn_mean_f1": 1.0},
    "T4": {"task_success_rate": 1.0},
}
RELATION_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def _number(value):
    """bool 是 Python 整数的子类，但此处不把它当作未经定义的分数。"""
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("报告中的质量指标必须在 0–1 之间，未知值使用 null")
    return float(value)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _valid_samples(metrics, task=None):
    return [s for s in (metrics or {}).get("per_sample", [])
            if s.get("status") != "INPUT_INVALID" and (task is None or s.get("task") == task)]


def _complete_semantic(sample):
    return bool(sample.get("semantic_counts", {}).get("scored")) and not sample.get("needs_alignment", False)


def _relation_score(sample):
    score = RELATION_SCORES.get(sample.get("relation_quality"))
    # 真实失败且没有任何预测，关系解释也没有完成；尚未运行不按零分处理。
    if score is None and sample.get("status") in ("FAILED", "PARTIAL"):
        if sample.get("semantic_counts", {}).get("predicted") == 0:
            return 0.0
    return score


def _sample_value(sample, metric):
    """返回 (值, 是否具有该指标所需的完整标注)。"""
    if not sample or sample.get("status") == "INPUT_INVALID":
        return None, False
    semantic_complete = _complete_semantic(sample)
    if metric == "claim_precision":
        counts = sample.get("semantic_counts", {})
        value = _ratio(counts.get("matched", 0) or 0, counts.get("predicted", 0)) if counts.get("scored") else None
        return value, semantic_complete
    if metric.startswith("stance_f1_"):
        label = metric.removeprefix("stance_f1_")
        value = (sample.get("semantic_counts", {}).get("stance") or {}).get(label, {}).get("f1")
        return value, semantic_complete
    if metric == "relation_quality_score":
        value = _relation_score(sample) if sample.get("task") == "T2" else None
        return value, semantic_complete and value is not None
    if metric == "memory_turn_mean_f1":
        turns = sample.get("turns", [])
        values = [t.get("memory_f1") for t in turns]
        complete = bool(turns) and all(t.get("scored") and t.get("memory_f1") is not None for t in turns)
        available = [v for v in values if v is not None]
        return mean(available) if available else None, complete
    if metric == "task_success_rate":
        value = sample.get("task_success")
        return float(value) if type(value) is bool else None, semantic_complete and type(value) is bool
    if metric == "citation_validity_mean":
        value = sample.get("citation_validity")
        return value, value is not None
    if metric == "answer_citations_match_claims_rate":
        value = sample.get("answer_citations_match_claims")
        return float(value) if type(value) is bool else None, type(value) is bool
    if metric in PROGRAM_METRICS:
        value = sample.get(metric)
        return value, value is not None
    value = sample.get(metric)
    if metric == "false_overwrite":
        complete = semantic_complete and sample.get("false_overwrite_counts", {}).get("complete", False)
        return value, complete
    return value, semantic_complete


def _aggregate_value(metrics, metric, task=None):
    """先保留既有 micro/macro 定义，再合成综合分；不平均逐样本 F1 冒充 micro。"""
    if not metrics:
        return None, False
    samples = _valid_samples(metrics, task)
    if not samples:
        return None, False
    section = metrics.get("overall", {}) if task is None else metrics.get("per_task", {}).get(task, {})
    if metric in ("claim_precision", "claim_f1"):
        scored = [s["semantic_counts"] for s in samples if s.get("semantic_counts", {}).get("scored")]
        matched = sum(c["matched"] for c in scored)
        predicted = sum(c["predicted"] for c in scored)
        expected = sum(c["expected"] for c in scored)
        value = _ratio(matched, predicted) if metric == "claim_precision" else _ratio(2 * matched, predicted + expected)
        return value, all(_complete_semantic(s) for s in samples)
    if metric in ("evidence_precision", "evidence_recall"):
        counts = [s["evidence_counts"] for s in samples if s.get("evidence_counts") is not None]
        denominator = "predicted" if metric == "evidence_precision" else "expected"
        value = _ratio(sum(c["tp"] for c in counts), sum(c[denominator] for c in counts))
        return value, len(counts) == len(samples)
    if metric in ("relation_quality_score", "memory_turn_mean_f1", "citation_validity_mean", "answer_citations_match_claims_rate"):
        applicable = samples
        if metric == "relation_quality_score":
            applicable = [s for s in samples if s.get("task") == "T2"]
        elif metric == "memory_turn_mean_f1":
            applicable = [s for s in samples if s.get("task") == "T3"]
        if not applicable:
            return None, False
        if metric == "memory_turn_mean_f1":
            turns = [t for s in applicable for t in s.get("turns", [])]
            values = [t.get("memory_f1") for t in turns if t.get("memory_f1") is not None]
            complete = bool(turns) and all(_sample_value(s, metric)[1] for s in applicable)
            return mean(values) if values else None, complete
        values_and_flags = [_sample_value(s, metric) for s in applicable]
        values = [v for v, _ in values_and_flags if v is not None]
        return mean(values) if values else None, all(c and v is not None for v, c in values_and_flags)
    if metric.startswith("stance_f1_"):
        label = metric.removeprefix("stance_f1_")
        return section.get("stance_per_class_f1", {}).get(label), all(_complete_semantic(s) for s in samples)
    value = section.get(metric)
    if metric == "evidence_f1":
        return value, all(s.get("evidence_counts") is not None for s in samples)
    if metric == "expected_exchange_recall":
        applicable = [s for s in samples if s.get("task") == "T4"]
        return value, bool(applicable) and all(s.get(metric) is not None for s in applicable)
    if metric in ("memory_state_f1", "false_overwrite"):
        applicable = [s for s in samples if s.get("task") == "T3"]
        return value, bool(applicable) and all(_sample_value(s, metric)[1] for s in applicable)
    if metric == "task_success_rate":
        applicable = [s for s in samples if s.get("task") == "T4"]
        return value, bool(applicable) and all(_sample_value(s, metric)[1] for s in applicable)
    return value, all(_complete_semantic(s) for s in samples)


def _stats(values, expected_judges):
    """正式均值要求预定裁判全部完整；诊断均值只使用完整裁判并标出数量。"""
    checked = {name: {"value": _number(value), "complete": bool(complete)}
               for name, (value, complete) in values.items()}
    available = [entry["value"] for entry in checked.values() if entry["complete"] and entry["value"] is not None]
    complete = bool(expected_judges) and len(available) == len(expected_judges)
    return {"source": "judges", "mean": mean(available) if complete else None,
            "available_mean": mean(available) if available else None,
            "valid_judges": len(available), "expected_judges": len(expected_judges),
            "complete": complete, "std": pstdev(available) if available else None,
            "min": min(available) if available else None, "max": max(available) if available else None,
            "per_judge": checked}


def _program_stats(value, complete):
    value = _number(value)
    complete = bool(complete and value is not None)
    return {"source": "program", "mean": value if complete else None,
            "available_mean": value, "complete": complete, "valid_judges": 0, "expected_judges": 0,
            "std": None, "min": None, "max": None, "per_judge": {}}


def _composite(task, program_values, judge_values):
    components = COMPOSITES[task]
    values, complete = [], True
    for metric, weight in components.items():
        value, valid = (program_values if metric in PROGRAM_METRICS else judge_values).get(metric, (None, False))
        value = _number(value)
        if value is None or not valid:
            complete = False
        else:
            values.append(weight * value)
    return sum(values) if complete else None, complete


def _validity_gate(structural, task=None, sample_id=None):
    """权限诊断独立呈现，不用单条异常覆盖整批质量均值。"""
    samples = _valid_samples(structural, task)
    if sample_id is not None:
        samples = [sample for sample in samples if sample.get("sample_id") == sample_id]
    violations = {
        "actual_unauthorized_deliveries": sum(s.get("actual_unauthorized_deliveries", 0) for s in samples),
        "future_or_private_citations": sum(len(s.get("future_or_private_citation_ids", [])) for s in samples),
        "nonexistent_citations": sum(len(s.get("nonexistent_citation_ids", [])) for s in samples),
    }
    return {"passed": not any(violations.values()) if samples else None, "violations": violations,
            "rule": "diagnostic_only_original_metrics_preserved"}


def _build_scope(structural, judge_metrics, expected_judges, task=None, sample_id=None):
    if sample_id is None:
        get_value = lambda report, metric: _aggregate_value(report, metric, task)
    else:
        def get_value(report, metric):
            sample = next((s for s in (report or {}).get("per_sample", []) if s.get("sample_id") == sample_id), None)
            return _sample_value(sample, metric)
    program_values = {metric: get_value(structural, metric) for metric in PROGRAM_METRICS}
    values_by_judge = {name: {metric: get_value(judge_metrics.get(name), metric) for metric in SEMANTIC_METRICS}
                       for name in expected_judges}
    metrics = {metric: _program_stats(*value) for metric, value in program_values.items()}
    for metric in SEMANTIC_METRICS:
        metrics[metric] = _stats({name: values[metric] for name, values in values_by_judge.items()}, expected_judges)
    composite = None
    gate = _validity_gate(structural, task, sample_id)
    if task in TASKS:
        composite_values = {name: _composite(task, program_values, values_by_judge[name]) for name in expected_judges}
        composite = _stats(composite_values, expected_judges)
        composite["weights"] = deepcopy(COMPOSITES[task])
    return {"metrics": metrics, "composite": composite, "validity_gate": gate}


def build_report(per_judge: dict, structural: dict, dataset_name: str, expected_judges: list[str]) -> dict:
    """构造报告，不改写原始 evaluator 指标，不平均 ID 配对或费用。"""
    if not expected_judges or len(expected_judges) != len(set(expected_judges)):
        raise ValueError("expected_judges 必须是非空且无重复的裁判名列表")
    if set(per_judge) - set(expected_judges):
        raise ValueError("报告中存在不属于预定裁判集合的结果")
    judge_metrics = {name: (per_judge.get(name) or {}).get("metrics") for name in expected_judges}
    per_task = {task: _build_scope(structural, judge_metrics, expected_judges, task) for task in TASKS}
    for task, section in per_task.items():
        section["program_summary"] = deepcopy(structural.get("per_task", {}).get(task, {}))
    overall = _build_scope(structural, judge_metrics, expected_judges)
    task_scores = [per_task[task]["composite"]["mean"] for task in TASKS]
    available_tasks = [score for score in task_scores if score is not None]
    overall["macro_task_score"] = mean(task_scores) if len(available_tasks) == len(TASKS) else None
    overall["available_task_mean"] = mean(available_tasks) if available_tasks else None
    overall["valid_tasks"] = len(available_tasks)
    overall["expected_tasks"] = len(TASKS)
    overall["program_summary"] = deepcopy(structural.get("overall", {}))
    per_sample = []
    for sample in structural.get("per_sample", []):
        sid, task = sample["sample_id"], sample["task"]
        section = _build_scope(structural, judge_metrics, expected_judges, task, sid)
        section.update({"sample_id": sid, "task": task, "status": sample.get("status"), "program": deepcopy(sample)})
        if task == "T3":
            judge_samples = {name: next((s for s in (judge_metrics[name] or {}).get("per_sample", []) if s["sample_id"] == sid), {}) for name in expected_judges}
            turns = sorted({t["turn"] for s in judge_samples.values() for t in s.get("turns", [])})
            section["turns"] = []
            for turn in turns:
                targets = {name: next((t for t in s.get("turns", []) if t["turn"] == turn), {}) for name, s in judge_samples.items()}
                turn_metrics = {}
                for metric, field in (("memory_state_f1", "memory_f1"), ("claim_coverage", "claim_coverage"), ("false_overwrite", "false_overwrite")):
                    turn_metrics[metric] = _stats({name: (target.get(field), bool(target.get("scored"))) for name, target in targets.items()}, expected_judges)
                section["turns"].append({"turn": turn, "metrics": turn_metrics})
        per_sample.append(section)
    return {
        "schema_version": "1.0", "dataset_name": dataset_name, "expected_judges": list(expected_judges),
        "methodology": {
            "aggregation": "independent_alignment_then_same_metric_mean",
            "strict_mean_requires_all_judges": True, "normalization": "already_0_to_1_no_minmax",
            "std_definition": "population_standard_deviation_of_complete_judges",
            "composite_version": "scopecrit_quality_v1", "composite_weights": deepcopy(COMPOSITES),
            "relation_quality_mapping": RELATION_SCORES,
            "overall_macro": "equal_weight_T1_T2_T3_T4_requires_all_four_tasks",
            "notes": ["缺失或不完整裁判不记零分；正式均值为 null，可用均值单列。",
                      "T3 综合分使用所有预期轮次的 Memory F1 平均，False Overwrite 不参与综合分。",
                      "T4 综合分使用 Task Success，避免重复计算其主张、证据和权限组成条件。",
                      "费用、token、耗时和评分覆盖率不参与质量平均。",
                      "实际权限与引用错误作为独立诊断，不将整批次质量均值强制归零；T4 成功判定保留原有逐样本权限条件。",
                      "证据与权限指标由程序计算一次；自动模型评测不是人工金标准。"],
        },
        "overall": overall, "per_task": per_task, "per_sample": per_sample,
        "per_judge": {name: {"status": (per_judge.get(name) or {}).get("status", "missing"),
                             "failures": deepcopy((per_judge.get(name) or {}).get("failures", [])),
                             "metrics": deepcopy(judge_metrics[name])} for name in expected_judges},
    }


def _csv_text(rows, expected_judges):
    stream = io.StringIO(newline="")
    fields = ["dataset", "scope", "sample_id", "task", "turn", "metric", "source", "mean", "available_mean", "valid_judges", "expected_judges", "complete", "std", "min", "max"]
    fields += [field for name in expected_judges for field in (f"judge:{name}", f"complete:{name}")]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _metric_rows(report, section, scope, task="", sample_id="", turn=""):
    values = dict(section.get("metrics", {}))
    if section.get("composite"):
        values["composite"] = section["composite"]
    for metric, stats in values.items():
        yield {"dataset": report["dataset_name"], "scope": scope, "sample_id": sample_id, "task": task, "turn": turn,
               "metric": metric, **{key: stats.get(key) for key in ("source", "mean", "available_mean", "valid_judges", "expected_judges", "complete", "std", "min", "max")},
               **{f"judge:{name}": stats.get("per_judge", {}).get(name, {}).get("value") for name in report["expected_judges"]},
               **{f"complete:{name}": stats.get("per_judge", {}).get(name, {}).get("complete") for name in report["expected_judges"]}}


def format_console(report):
    lines = [f'[评测汇总] 数据集：{report["dataset_name"]}（质量分范围 0–1）', "任务 | 综合分 | 完整裁判数"]
    for task in TASKS:
        stats = report["per_task"][task]["composite"]
        value = "null" if stats["mean"] is None else f'{stats["mean"]:.4f}'
        lines.append(f'{task} | {value} | {stats["valid_judges"]}/{stats["expected_judges"]}')
    overall = report["overall"]["macro_task_score"]
    lines.append("四任务宏平均：" + ("null（四任务尚未全部评分）" if overall is None else f"{overall:.4f}"))
    lines.extend(["", "[逐指标结果] 程序指标只计算一次；null 表示无有效分母或尚未完整评分。",
                  "任务 | 指标 | " + " | ".join(report["expected_judges"]) + " | 正式均值 | 完整裁判数"])
    def display(value):
        return "null" if value is None else f"{value:.4f}"
    for task in TASKS:
        for metric, stats in report["per_task"][task]["metrics"].items():
            entries = [display(stats.get("per_judge", {}).get(name, {}).get("value"))
                       if stats["source"] == "judges" else "程序计算" for name in report["expected_judges"]]
            count = f'{stats["valid_judges"]}/{stats["expected_judges"]}' if stats["source"] == "judges" else "—"
            lines.append(f'{task} | {metric} | ' + " | ".join(entries) + f' | {display(stats["mean"])} | {count}')
    return "\n".join(lines)


def write_report(report, run_dir):
    """写独立的裁判汇总文件，保留已有 metrics.json 的原始格式。"""
    import json

    directory = Path(run_dir)
    atomic_write(directory / "judge_summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    metric_rows = list(_metric_rows(report, report["overall"], "overall"))
    for task in TASKS:
        metric_rows.extend(_metric_rows(report, report["per_task"][task], "task", task))
    metric_rows.append({"dataset": report["dataset_name"], "scope": "overall", "sample_id": "", "task": "", "turn": "",
                        "metric": "macro_task_score", "source": "composite", "mean": report["overall"]["macro_task_score"],
                        "available_mean": report["overall"]["available_task_mean"], "complete": report["overall"]["valid_tasks"] == 4})
    atomic_write(directory / "metrics_summary.csv", _csv_text(metric_rows, report["expected_judges"]))
    sample_rows = []
    for sample in report["per_sample"]:
        sample_rows.extend(_metric_rows(report, sample, "sample", sample["task"], sample["sample_id"]))
        for turn in sample.get("turns", []):
            sample_rows.extend(_metric_rows(report, turn, "turn", sample["task"], sample["sample_id"], turn["turn"]))
    atomic_write(directory / "per_sample_metrics.csv", _csv_text(sample_rows, report["expected_judges"]))
    lines = ["# 自动评测报告", "", "```text", format_console(report), "```", "", "正式均值仅在全部预定裁判完整评分时给出；null 表示未完整评分，不等于零分。", "", "| 范围 | 指标 | 正式均值 | 可用均值 | 裁判数 |", "|---|---|---:|---:|---:|"]
    for row in metric_rows:
        number = lambda value: "null" if value is None else f"{value:.6f}"
        judges = "程序一次计算" if row["source"] == "program" else f'{row.get("valid_judges", "")}/{row.get("expected_judges", "")}'
        lines.append(f'| {row["task"] or "overall"} | {row["metric"]} | {number(row.get("mean"))} | {number(row.get("available_mean"))} | {judges} |')
    lines.extend(["", "## 固定综合分定义", "", "- T1：方面 Micro-F1、立场 Macro-F1、证据 F1 等权。", "- T2：主张 Micro-F1、关系解释质量、证据 F1 等权；关系质量 correct=1、partial=0.5、incorrect=0。", "- T3：所有预期轮次 Memory F1 等权；False Overwrite 只作诊断。", "- T4：语义 Task Success；交换召回率作为诊断指标单列。", "- 权限和引用错误作为独立诊断，不覆盖原始指标均值；T4 成功仍包含逐样本权限条件。", "- 四任务综合分等权宏平均，缺少任务时正式整体分为 null。", "", "各裁判原始指标、程序计数与逐样本指标保存在 judge_summary.json；费用与失败记录不混入质量分。", "模型评分不是人工金标准；演示分数不代表真实模型能力。", ""])
    atomic_write(directory / "report.md", "\n".join(lines))
