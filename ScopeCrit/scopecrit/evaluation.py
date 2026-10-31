"""独立离线评分：gold 只在这里使用，绝不回传给运行中的 Agent。

证据集合可以直接比较；主张含义不能靠字符串或 ID 判断。涉及含义的
指标只接受人工提供的一对一对齐，缺少对齐时明确输出 None（JSON null）。
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

STANCES = ("POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL")
CITATION = re.compile(r"\[(E\d+)\]")
# PAUSED_UNCERTAIN 是等待恢复的状态，不能直接当成已经失败的答卷。
FAILURES = {"FAILED", "PARTIAL"}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, predicted: int, expected: int) -> float | None:
    return _ratio(2 * tp, predicted + expected)


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _claims(prediction: dict) -> list[dict]:
    return prediction.get("prediction", {}).get("claims", [])


def _claim_index(claims: list[dict], key: str) -> dict[str, dict]:
    result = {c[key]: c for c in claims}
    if len(result) != len(claims):
        raise ValueError(f"评分遇到重复主张 ID: {key}")
    return result


def _gold_for_turn(sample: dict, turn: int | None) -> list[dict]:
    if turn is None:
        return sample["gold"]["claims"]
    event = next((e for e in sample.get("events", []) if e["turn"] == turn), None)
    if event is None:
        raise ValueError(f"未知轮次: {sample['sample_id']} turn={turn}")
    ids = set(event["gold_memory_state"])
    return [c for c in sample["gold"]["claims"] if c["claim_id"] in ids]


def _turn_prediction(prediction: dict, turn: int) -> list[dict]:
    snapshot = next((t for t in prediction.get("turns", []) if t["turn"] == turn), None)
    return snapshot.get("claims", []) if snapshot else []


def _validate_alignment(
    alignment: list[dict] | None, predictions: dict[str, dict], gold: dict[str, dict]
) -> dict[tuple[str, int | None], dict]:
    """拒绝无效对齐，不能让重复配对或不存在的主张抬高分数。"""
    if alignment is None:
        return {}
    if not isinstance(alignment, list):
        raise ValueError("alignment 必须是 JSON 数组")
    result = {}
    for record in alignment:
        if not isinstance(record, dict):
            raise ValueError("alignment 中每项必须是对象")
        sid, turn = record.get("sample_id"), record.get("turn")
        if sid not in gold or sid not in predictions:
            raise ValueError(f"对齐包含未知 sample_id: {sid}")
        if turn is not None and (type(turn) is not int or gold[sid]["task"] != "T3"):
            raise ValueError(f"只有 T3 可以指定整数 turn: {sid}")
        key = sid, turn
        if key in result:
            raise ValueError(f"重复对齐记录: {sid} turn={turn}")
        if type(record.get("alignment_complete")) is not bool:
            raise ValueError("alignment_complete 必须明确为 true 或 false")
        pc = _claims(predictions[sid]) if turn is None else _turn_prediction(predictions[sid], turn)
        gc = _gold_for_turn(gold[sid], turn)
        pids = set(_claim_index(pc, "predicted_claim_id"))
        gids = set(_claim_index(gc, "claim_id"))
        matches = record.get("matches")
        if not isinstance(matches, list):
            raise ValueError("matches 必须是数组")
        seen_p, seen_g = set(), set()
        for match in matches:
            if not isinstance(match, dict):
                raise ValueError("matches 中每项必须是对象")
            pid, gid = match.get("predicted_claim_id"), match.get("gold_claim_id")
            if pid not in pids or gid not in gids:
                raise ValueError(f"对齐引用不存在的主张: {sid} {pid} -> {gid}")
            if pid in seen_p or gid in seen_g:
                raise ValueError(f"对齐必须一对一: {sid}")
            if match.get("relation") != "equivalent":
                raise ValueError("matches 仅列出 relation=equivalent 的等价配对")
            seen_p.add(pid)
            seen_g.add(gid)
        if record.get("relation_quality", "unscored") not in {"correct", "partial", "incorrect", "unscored"}:
            raise ValueError("relation_quality 不合法")
        if "dual_dependency" in record and type(record["dual_dependency"]) is not bool:
            raise ValueError("dual_dependency 必须为布尔值")
        result[key] = record
    return result


def _semantic_counts(predicted: list[dict], expected: list[dict], annotation: dict | None, failed: bool) -> dict:
    """标签不参与含义匹配；配对确定后，再检查标签是否正确。"""
    complete = bool(annotation and annotation["alignment_complete"])
    # 真正失败且没有任何预测，可以记为覆盖 0；缺对齐的非空预测不能判错。
    scored = complete or (failed and not predicted)
    empty = {
        "scored": False, "matched": None, "predicted": len(predicted), "expected": len(expected),
        "aspect_tp": None, "stance": None, "matched_gold_ids": None,
        "claim_coverage": None, "claim_f1": None, "aspect_micro_f1": None,
        "stance_macro_f1": None, "stance_participating_classes": 0,
    }
    if not scored:
        return empty
    pi, gi = _claim_index(predicted, "predicted_claim_id"), _claim_index(expected, "claim_id")
    matches = annotation["matches"] if complete else []
    paired = [(pi[m["predicted_claim_id"]], gi[m["gold_claim_id"]]) for m in matches]
    aspect_tp = sum(p["aspect"] == g["aspect"] for p, g in paired)
    stance_counts = {}
    for label in STANCES:
        tp = sum(p["stance"] == g["stance"] == label for p, g in paired)
        pn = sum(c["stance"] == label for c in predicted)
        gn = sum(c["stance"] == label for c in expected)
        stance_counts[label] = {"tp": tp, "predicted": pn, "expected": gn, "f1": _f1(tp, pn, gn)}
    return {
        "scored": True, "matched": len(matches), "predicted": len(predicted), "expected": len(expected),
        "aspect_tp": aspect_tp, "stance": stance_counts,
        "matched_gold_ids": sorted(m["gold_claim_id"] for m in matches),
        "claim_coverage": _ratio(len(matches), len(expected)),
        "claim_f1": _f1(len(matches), len(predicted), len(expected)),
        "aspect_micro_f1": _f1(aspect_tp, len(predicted), len(expected)),
        "stance_macro_f1": _mean([c["f1"] for c in stance_counts.values()]),
        "stance_participating_classes": sum(c["f1"] is not None for c in stance_counts.values()),
    }


def _delivered_evidence(prediction: dict, sample: dict) -> tuple[set[tuple], set[str], int]:
    """按真实发送/投递顺序重放权限；outbox 中的消息不增加可见集合。"""
    bank = {e["evidence_id"]: e for e in sample["document"]["evidence"]}
    visible = {role: set(ids) for role, ids in sample.get("roles", {}).items()}
    visible.setdefault("archivist", set())
    visible.setdefault("interpreter", set())
    delivered, invalid = set(), 0
    messages = [m for m in prediction.get("messages", []) if m.get("kind") == "EVIDENCE"]
    timeline = []
    seen = {}
    for message in messages:
        mid = message.get("message_id")
        if mid in seen:
            if seen[mid] != message and message.get("delivered_event_seq") is not None:
                invalid += 1
            continue
        seen[mid] = message
        sent, received = message.get("sent_event_seq"), message.get("delivered_event_seq")
        if type(sent) is not int:
            if received is not None:
                invalid += 1
            continue
        timeline.append((sent, 0, message))
        if received is not None:
            if type(received) is not int or received < sent:
                invalid += 1
            else:
                timeline.append((received, 1, message))
    legal = {}
    for _, event_type, message in sorted(timeline, key=lambda x: (x[0], x[1])):
        mid, sender, receiver = message.get("message_id"), message.get("from_agent"), message.get("to_agent")
        evidence = message.get("evidence", [])
        if event_type == 0:
            legal[mid] = (
                sender in visible and receiver in visible and sender != receiver
                and bool(evidence) and all(
                    isinstance(e, dict) and e.get("evidence_id") in visible[sender]
                    and all(e.get(k) == bank[e["evidence_id"]].get(k) for k in ("evidence_id", "text", "start", "end"))
                    for e in evidence
                )
            )
        elif legal.get(mid):
            for e in evidence:
                eid = e["evidence_id"]
                visible[receiver].add(eid)
                delivered.add((sender, receiver, eid))
        else:
            invalid += 1
    return delivered, visible["interpreter"], invalid


def _structural(prediction: dict, sample: dict) -> dict:
    claims = _claims(prediction)
    evidence = {eid for c in claims for eid in c.get("evidence_ids", [])}
    required = set(sample["gold"]["required_evidence_ids"])
    bank = {e["evidence_id"] for e in sample["document"]["evidence"]}
    answer = prediction.get("prediction", {}).get("answer", "")
    citations = set(CITATION.findall(answer))
    unknown = (evidence | citations) - bank
    delivered, invalid_delivery = set(), 0
    if sample["task"] == "T4":
        delivered, visible, invalid_delivery = _delivered_evidence(prediction, sample)
    elif sample["task"] == "T3":
        # 只使用已执行轮次，不能把未来轮次的材料当作合法证据。
        actual_turns = {t["turn"] for t in prediction.get("turns", [])}
        visible = {eid for event in sample["events"] if event["turn"] in actual_turns for eid in event["visible_evidence_ids"]}
    else:
        visible = bank
    future = (evidence | citations) & bank - visible
    # 最后一轮合法，并不能掩盖较早轮次已经引用未来材料的泄漏。
    if sample["task"] == "T3":
        for snapshot in prediction.get("turns", []):
            turn_visible = {
                eid for event in sample["events"] if event["turn"] <= snapshot["turn"]
                for eid in event["visible_evidence_ids"]
            }
            used = {eid for claim in snapshot.get("claims", []) for eid in claim.get("evidence_ids", [])}
            used.update(CITATION.findall(snapshot.get("answer", "")))
            unknown.update(used - bank)
            future.update(used & bank - turn_visible)
    violations = Counter(v.get("kind") for v in prediction.get("access_violations", []))
    expected = {(e["from_agent"], e["to_agent"], e["evidence_id"]) for e in sample.get("required_exchanges", [])}
    return {
        "evidence_f1": _f1(len(evidence & required), len(evidence), len(required)),
        "evidence_precision": _ratio(len(evidence & required), len(evidence)),
        "evidence_recall": _ratio(len(evidence & required), len(required)),
        "evidence_counts": {"tp": len(evidence & required), "predicted": len(evidence), "expected": len(required)},
        "citation_validity": _ratio(len(citations & visible), len(citations)),
        "answer_citations_match_claims": citations == evidence,
        "access_denied_attempts": violations["access_denied"],
        "actual_unauthorized_deliveries": max(invalid_delivery, violations["actual_delivery"]),
        "future_or_private_citation_ids": sorted(future),
        "nonexistent_citation_ids": sorted(unknown),
        "expected_exchange_recall": _ratio(len(delivered & expected), len(expected)) if sample["task"] == "T4" else None,
        "exchange_counts": {"matched": len(delivered & expected), "expected": len(expected), "delivered_unique": len(delivered)},
    }


def evaluate_predictions(predictions: list[dict], gold_rows: list[dict], alignment: list[dict] | None = None) -> dict:
    """按 T1–T4 汇总。所有有效目标进入分母；失败不会被悄悄删除。"""
    gold = {s["sample_id"]: s for s in gold_rows if isinstance(s, dict) and isinstance(s.get("sample_id"), str)}
    if len(gold) != len(gold_rows):
        raise ValueError("gold 含重复或缺失 sample_id，请先执行数据校验")
    indexed = {p["sample_id"]: p for p in predictions}
    if len(indexed) != len(predictions):
        raise ValueError("predictions 含重复 sample_id")
    if set(indexed) - set(gold):
        raise ValueError("predictions 包含不属于输入数据的样本")
    for sid, sample in gold.items():
        indexed.setdefault(sid, {"sample_id": sid, "task": sample["task"], "status": "PENDING", "prediction": {"claims": []}})
    annotations = _validate_alignment(alignment, indexed, gold)
    results = []
    for sid, sample in gold.items():
        prediction = indexed[sid]
        task = sample["task"]
        status = prediction.get("status", "PENDING")
        item: dict[str, Any] = {"sample_id": sid, "task": task, "status": status}
        if status == "INPUT_INVALID":
            item.update({"scoring_status": "input_invalid", "needs_alignment": False})
            results.append(item)
            continue
        try:
            item.update(_structural(prediction, sample))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"评分输入结构不合法: {sid}: {exc}") from exc
        has_output = bool(_claims(prediction) or prediction.get("prediction", {}).get("answer"))
        has_output = has_output or any(t.get("claims") or t.get("answer") for t in prediction.get("turns", []))
        output_evaluable = has_output or status in FAILURES or status == "COMPLETED"
        if not output_evaluable:
            # 尚未运行/等待恢复的空输出与真实失败不同，不记作零分答卷。
            for name in ("evidence_f1", "evidence_precision", "evidence_recall", "evidence_counts", "answer_citations_match_claims"):
                item[name] = None
            if not prediction.get("messages"):
                item["expected_exchange_recall"] = None
                item["exchange_counts"] = None
        annotation = annotations.get((sid, None))
        if task == "T3" and annotation is None and sample["events"]:
            annotation = annotations.get((sid, sample["events"][-1]["turn"]))
        counts = _semantic_counts(
            _claims(prediction), sample["gold"]["claims"],
            annotation if output_evaluable else None, status in FAILURES,
        )
        item.update({k: counts[k] for k in ("claim_coverage", "claim_f1", "aspect_micro_f1", "stance_macro_f1", "stance_participating_classes")})
        item["semantic_counts"] = counts
        item["needs_alignment"] = output_evaluable and not counts["scored"]
        item["scoring_status"] = "scored" if counts["scored"] else ("needs_alignment" if output_evaluable else "not_run")
        item["relation_quality"] = (annotation or {}).get("relation_quality", "unscored") if task == "T2" else None
        item["task_success"] = None
        if task == "T4":
            if status in FAILURES and not _claims(prediction):
                item["task_success"] = False
            elif counts["scored"] and annotation and "dual_dependency" in annotation:
                item["task_success"] = (
                    counts["matched"] == counts["expected"]
                    and item["evidence_recall"] == 1
                    and not item["future_or_private_citation_ids"]
                    and not item["nonexistent_citation_ids"]
                    and not item["actual_unauthorized_deliveries"]
                    and item["answer_citations_match_claims"]
                    and annotation["dual_dependency"]
                )
        if task == "T3":
            turns, previous_matched = [], None
            overwrite_lost, overwrite_opportunities, overwrite_complete = 0, 0, True
            for event in sample["events"]:
                turn = event["turn"]
                snapshot = next((t for t in prediction.get("turns", []) if t["turn"] == turn), None)
                turn_counts = _semantic_counts(
                    snapshot.get("claims", []) if snapshot else [], _gold_for_turn(sample, turn),
                    annotations.get((sid, turn)) if snapshot is not None or status in FAILURES else None,
                    status in FAILURES,
                )
                current_expected = set(event["gold_memory_state"])
                current_matched = set(turn_counts["matched_gold_ids"]) if turn_counts["scored"] else None
                lost, opportunities = None, None
                if previous_matched is not None and current_matched is not None:
                    retained = previous_matched & current_expected
                    opportunities, lost = len(retained), len(retained - current_matched)
                    overwrite_lost += lost
                    overwrite_opportunities += opportunities
                elif turns:
                    overwrite_complete = False
                turns.append({"turn": turn, "memory_f1": turn_counts["claim_f1"], "claim_coverage": turn_counts["claim_coverage"],
                              "scored": turn_counts["scored"], "false_overwrite_lost": lost,
                              "false_overwrite_opportunities": opportunities, "false_overwrite": _ratio(lost, opportunities) if opportunities is not None else None})
                previous_matched = current_matched
            item["turns"] = turns
            item["memory_state_f1"] = turns[-1]["memory_f1"] if turns else None
            item["false_overwrite"] = _ratio(overwrite_lost, overwrite_opportunities) if overwrite_complete else None
            item["false_overwrite_counts"] = {"lost": overwrite_lost, "opportunities": overwrite_opportunities, "complete": overwrite_complete}
            item["needs_alignment"] = output_evaluable and any(not t["scored"] for t in turns)
            item["scoring_status"] = ("needs_alignment" if item["needs_alignment"] else "scored") if output_evaluable else "not_run"
        item["usage"] = prediction.get("usage", {})
        results.append(item)
    per_task = {task: _aggregate([r for r in results if r["task"] == task]) for task in ("T1", "T2", "T3", "T4")}
    return {"overall": _aggregate(results), "per_task": per_task, "per_sample": results,
            "notes": ["语义指标依赖人工一对一对齐；未评分项为 null。", "Evidence F1 只检查证据集合，不证明语义支持。", "无 bootstrap 置信区间；应在研究分析阶段按 doc_id 分组重采样。"]}


def _aggregate(items: list[dict]) -> dict:
    valid = [r for r in items if r["status"] != "INPUT_INVALID"]
    semantic = [r for r in valid if r.get("semantic_counts", {}).get("scored")]
    ec = [r["evidence_counts"] for r in valid if r.get("evidence_counts") is not None]
    sc = [r["semantic_counts"] for r in semantic]
    stance = {}
    for label in STANCES:
        rows = [s["stance"][label] for s in sc]
        stance[label] = _f1(sum(x["tp"] for x in rows), sum(x["predicted"] for x in rows), sum(x["expected"] for x in rows))
    exchanges = [r["exchange_counts"] for r in valid if r["task"] == "T4" and r.get("exchange_counts") is not None]
    overwrite = [r["false_overwrite_counts"] for r in valid if r["task"] == "T3"]
    usage_complete = bool(valid) and all(
        r.get("usage", {}).get("usage_complete", False)
        and type(r.get("usage", {}).get("total_tokens")) is int
        for r in valid
    )
    known_tokens = [r.get("usage", {}).get("total_tokens") for r in valid
                    if type(r.get("usage", {}).get("total_tokens")) is int]
    return {
        "count": len(items), "valid_input_count": len(valid), "input_invalid_count": len(items) - len(valid),
        "status_counts": dict(Counter(r["status"] for r in items)),
        "semantic_scored_count": len(semantic), "semantic_scoring_coverage": _ratio(len(semantic), len(valid)),
        "needs_alignment_count": sum(r.get("needs_alignment", False) for r in valid),
        "not_run_count": sum(r.get("scoring_status") == "not_run" for r in valid),
        "evidence_scored_count": len(ec), "evidence_scoring_coverage": _ratio(len(ec), len(valid)),
        "evidence_f1": _f1(sum(x["tp"] for x in ec), sum(x["predicted"] for x in ec), sum(x["expected"] for x in ec)),
        "claim_coverage": _ratio(sum(x["matched"] for x in sc), sum(x["expected"] for x in sc)),
        "aspect_micro_f1": _f1(sum(x["aspect_tp"] for x in sc), sum(x["predicted"] for x in sc), sum(x["expected"] for x in sc)),
        "stance_macro_f1": _mean(list(stance.values())), "stance_per_class_f1": stance,
        "stance_participating_classes": sum(v is not None for v in stance.values()),
        "memory_state_f1": _mean([r.get("memory_state_f1") for r in valid]),
        "false_overwrite": _ratio(sum(x["lost"] for x in overwrite), sum(x["opportunities"] for x in overwrite)) if overwrite and all(x["complete"] for x in overwrite) else None,
        "expected_exchange_recall": _ratio(sum(x["matched"] for x in exchanges), sum(x["expected"] for x in exchanges)),
        "task_success_rate": _mean([float(r["task_success"]) if r.get("task_success") is not None else None for r in valid]),
        "task_success_scored_count": sum(r.get("task_success") is not None for r in valid),
        "relation_quality_counts": dict(Counter(r["relation_quality"] for r in valid if r.get("relation_quality") is not None)),
        "access_denied_attempts": sum(r.get("access_denied_attempts", 0) for r in valid),
        "actual_unauthorized_deliveries": sum(r.get("actual_unauthorized_deliveries", 0) for r in valid),
        "future_or_private_citation_count": sum(len(r.get("future_or_private_citation_ids", [])) for r in valid),
        "nonexistent_citation_count": sum(len(r.get("nonexistent_citation_ids", [])) for r in valid),
        "model_calls": sum(r.get("usage", {}).get("model_calls", 0) or 0 for r in valid),
        "reported_total_tokens": sum(known_tokens) if usage_complete else None,
        "known_reported_total_tokens": sum(known_tokens) if known_tokens else None,
        "usage_complete": usage_complete,
    }


def alignment_template(predictions: list[dict]) -> list[dict]:
    """生成待人工填写的模板，不自动推断任何语义匹配。"""
    result = []
    for prediction in predictions:
        if prediction.get("status") == "INPUT_INVALID":
            continue
        targets = [(None, _claims(prediction))]
        if prediction.get("task") == "T3":
            targets = [(t["turn"], t.get("claims", [])) for t in prediction.get("turns", [])]
        for turn, claims in targets:
            result.append({"sample_id": prediction["sample_id"], "turn": turn, "alignment_complete": False,
                           "matches": [], "predicted_claim_ids": [c["predicted_claim_id"] for c in claims]})
    return result
