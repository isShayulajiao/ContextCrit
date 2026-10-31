"""评分测试使用明确标注的合成材料，不评估文学理解能力。"""

from copy import deepcopy

import pytest

from scopecrit.evaluation import alignment_template, evaluate_predictions


def gold(task="T1"):
    return {
        "sample_id": f"SYNTHETIC:{task}", "task": task, "doc_id": "SYNTHETIC",
        "document": {"text": "整体好结尾差", "evidence": [
            {"evidence_id": "E001", "text": "整体好", "start": 0, "end": 3},
            {"evidence_id": "E002", "text": "结尾差", "start": 3, "end": 6},
        ]},
        "gold": {"claims": [
            {"claim_id": "C1", "text": "整体好", "aspect": "OVERALL", "stance": "POSITIVE", "evidence_ids": ["E001"]},
            {"claim_id": "C2", "text": "结尾差", "aspect": "STRUCTURE", "stance": "NEGATIVE", "evidence_ids": ["E002"]},
        ], "required_evidence_ids": ["E001", "E002"]},
        "events": [], "roles": {}, "required_exchanges": [],
    }


def predicted(sample):
    claims = []
    for index, claim in enumerate(sample["gold"]["claims"], 1):
        copy = deepcopy(claim)
        del copy["claim_id"]
        copy["predicted_claim_id"] = f"P{index}"
        claims.append(copy)
    return {
        "sample_id": sample["sample_id"], "task": sample["task"], "status": "COMPLETED",
        "prediction": {"claims": claims, "answer": "整体好[E001]，结尾差[E002]。"},
        "turns": [], "messages": [], "usage": {"model_calls": 5, "total_tokens": 50, "usage_complete": True},
    }


def aligned(sample, turn=None, ids=("P1", "P2")):
    return {"sample_id": sample["sample_id"], "turn": turn, "alignment_complete": True,
            "matches": [{"predicted_claim_id": pid, "gold_claim_id": f"C{i}", "relation": "equivalent"} for i, pid in enumerate(ids, 1)]}


def test_no_alignment_does_not_use_matching_text_or_ids():
    sample = gold()
    report = evaluate_predictions([predicted(sample)], [sample])
    result = report["per_sample"][0]
    assert result["evidence_f1"] == 1
    assert result["aspect_micro_f1"] is None
    assert result["claim_coverage"] is None
    assert report["overall"]["semantic_scoring_coverage"] == 0
    assert result["needs_alignment"]


def test_renamed_ids_have_identical_scores_after_alignment():
    sample, record = gold(), predicted(gold())
    original = evaluate_predictions([record], [sample], [aligned(sample)])
    record["prediction"]["claims"][0]["predicted_claim_id"] = "a completely arbitrary ID"
    renamed = evaluate_predictions([record], [sample], [aligned(sample, ids=("a completely arbitrary ID", "P2"))])
    assert original["per_task"] == renamed["per_task"]
    assert renamed["per_task"]["T1"]["aspect_micro_f1"] == 1
    assert renamed["per_task"]["T1"]["stance_macro_f1"] == 1


def test_wrong_labels_and_duplicate_predictions_are_penalized():
    sample, record = gold(), predicted(gold())
    record["prediction"]["claims"][0]["aspect"] = "OTHER"
    record["prediction"]["claims"][0]["stance"] = "NEGATIVE"
    duplicate = deepcopy(record["prediction"]["claims"][1])
    duplicate["predicted_claim_id"] = "P3"
    record["prediction"]["claims"].append(duplicate)
    result = evaluate_predictions([record], [sample], [aligned(sample)])["per_task"]["T1"]
    assert result["aspect_micro_f1"] == pytest.approx(2 / 5)
    assert result["stance_macro_f1"] == pytest.approx(0.25)


def test_incomplete_alignment_leaves_nonempty_prediction_unscored():
    sample, record = gold(), predicted(gold())
    annotation = aligned(sample)
    annotation["alignment_complete"] = False
    result = evaluate_predictions([record], [sample], [annotation])["per_sample"][0]
    assert result["claim_coverage"] is None


@pytest.mark.parametrize("change", ["duplicate", "unknown_predicted", "unknown_gold", "unknown_turn", "unknown_sample"])
def test_invalid_alignment_is_rejected(change):
    sample, record = gold(), predicted(gold())
    annotation = aligned(sample)
    if change == "duplicate":
        annotation["matches"].append(deepcopy(annotation["matches"][0]))
    elif change == "unknown_predicted":
        annotation["matches"][0]["predicted_claim_id"] = "P404"
    elif change == "unknown_gold":
        annotation["matches"][0]["gold_claim_id"] = "C404"
    elif change == "unknown_turn":
        annotation["turn"] = 1
    else:
        annotation["sample_id"] = "missing"
    with pytest.raises(ValueError):
        evaluate_predictions([record], [sample], [annotation])


def test_failures_remain_in_denominator_and_pending_is_not_failure():
    sample = gold()
    failed = {"sample_id": sample["sample_id"], "task": "T1", "status": "FAILED", "prediction": {"claims": [], "answer": ""}}
    report = evaluate_predictions([failed], [sample])
    assert report["overall"]["count"] == 1
    assert report["overall"]["claim_coverage"] == 0
    assert report["overall"]["evidence_f1"] == 0
    pending = evaluate_predictions([], [sample])
    assert pending["overall"]["claim_coverage"] is None
    assert pending["overall"]["evidence_f1"] is None
    assert pending["overall"]["not_run_count"] == 1
    assert pending["overall"]["evidence_scoring_coverage"] == 0
    assert not pending["per_sample"][0]["needs_alignment"]


@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "PAUSED_UNCERTAIN"])
def test_empty_nonterminal_outputs_are_unscored(status):
    sample = gold("T4")
    record = {"sample_id": sample["sample_id"], "task": "T4", "status": status, "prediction": {"claims": []}}
    report = evaluate_predictions([record], [sample])
    assert report["overall"]["claim_coverage"] is None
    assert report["overall"]["evidence_f1"] is None
    assert report["overall"]["expected_exchange_recall"] is None
    assert report["overall"]["task_success_rate"] is None


def test_unknown_usage_is_null_instead_of_zero():
    sample, record = gold(), predicted(gold())
    record["usage"] = {"model_calls": 2, "total_tokens": None, "usage_complete": False}
    result = evaluate_predictions([record], [sample])["overall"]
    assert result["model_calls"] == 2
    assert result["reported_total_tokens"] is None
    assert result["known_reported_total_tokens"] is None
    assert not result["usage_complete"]


def test_partial_usage_preserves_known_sum_without_claiming_complete_total():
    sample1, sample2 = gold(), gold("T2")
    record1, record2 = predicted(sample1), predicted(sample2)
    record2["usage"] = {"model_calls": 2, "total_tokens": None, "usage_complete": False}
    result = evaluate_predictions([record1, record2], [sample1, sample2])["overall"]
    assert result["reported_total_tokens"] is None
    assert result["known_reported_total_tokens"] == 50
    assert not result["usage_complete"]


def t3_pair():
    sample = gold("T3")
    sample["events"] = [
        {"turn": 1, "gold_memory_state": ["C1"], "visible_evidence_ids": ["E001"]},
        {"turn": 2, "gold_memory_state": ["C1", "C2"], "visible_evidence_ids": ["E002"]},
    ]
    record = predicted(sample)
    claims = record["prediction"]["claims"]
    record["turns"] = [
        {"turn": 1, "claims": deepcopy(claims[:1]), "answer": "整体好[E001]"},
        {"turn": 2, "claims": deepcopy(claims), "answer": "整体好[E001]，结尾差[E002]"},
    ]
    annotations = [aligned(sample, 1, ("P1",)), aligned(sample, 2)]
    return sample, record, annotations


def test_t3_memory_and_false_overwrite_use_per_turn_semantic_matches():
    sample, record, annotations = t3_pair()
    result = evaluate_predictions([record], [sample], annotations)["per_sample"][0]
    assert result["memory_state_f1"] == 1
    assert result["false_overwrite"] == 0
    record["turns"][1]["claims"] = record["turns"][1]["claims"][1:]
    record["prediction"]["claims"] = deepcopy(record["turns"][1]["claims"])
    annotations[1]["matches"] = annotations[1]["matches"][1:]
    result = evaluate_predictions([record], [sample], annotations)["per_sample"][0]
    assert result["memory_state_f1"] == pytest.approx(2 / 3)
    assert result["false_overwrite"] == 1


def test_t3_missing_alignment_is_null_even_if_no_overwrite_in_data():
    sample, record, annotations = t3_pair()
    result = evaluate_predictions([record], [sample], annotations[1:])["per_sample"][0]
    assert result["false_overwrite"] is None
    assert result["turns"][0]["memory_f1"] is None


def test_t3_final_visibility_does_not_hide_earlier_future_citation():
    sample, record, _ = t3_pair()
    record["turns"][0]["answer"] += "结尾差[E002]"
    result = evaluate_predictions([record], [sample])["per_sample"][0]
    assert result["future_or_private_citation_ids"] == ["E002"]


def t4_pair():
    sample = gold("T4")
    sample["roles"] = {"archivist": ["E001"], "interpreter": ["E002"]}
    sample["required_exchanges"] = [{"from_agent": "archivist", "to_agent": "interpreter", "evidence_id": "E001"}]
    record = predicted(sample)
    record["messages"] = [{"message_id": "M1", "kind": "EVIDENCE", "from_agent": "archivist", "to_agent": "interpreter",
                           "evidence": [deepcopy(sample["document"]["evidence"][0])], "sent_event_seq": 2, "delivered_event_seq": None}]
    return sample, record


def test_t4_outbox_is_not_delivery_and_duplicate_delivery_does_not_raise_recall():
    sample, record = t4_pair()
    result = evaluate_predictions([record], [sample])["per_sample"][0]
    assert result["expected_exchange_recall"] == 0
    assert result["future_or_private_citation_ids"] == ["E001"]
    record["messages"][0]["delivered_event_seq"] = 3
    record["messages"].append(deepcopy(record["messages"][0]))
    result = evaluate_predictions([record], [sample])["per_sample"][0]
    assert result["expected_exchange_recall"] == 1
    assert result["exchange_counts"]["delivered_unique"] == 1
    assert result["future_or_private_citation_ids"] == []
    assert result["task_success"] is None


def test_t4_altered_evidence_is_not_legal_exchange():
    sample, record = t4_pair()
    record["messages"][0]["delivered_event_seq"] = 3
    record["messages"][0]["evidence"][0]["text"] = "伪造原文"
    result = evaluate_predictions([record], [sample])["per_sample"][0]
    assert result["expected_exchange_recall"] == 0
    assert result["actual_unauthorized_deliveries"] == 1


def test_t4_task_success_needs_explicit_dual_dependency_annotation():
    sample, record = t4_pair()
    record["messages"][0]["delivered_event_seq"] = 3
    annotation = aligned(sample)
    assert evaluate_predictions([record], [sample], [annotation])["per_sample"][0]["task_success"] is None
    annotation["dual_dependency"] = True
    assert evaluate_predictions([record], [sample], [annotation])["per_sample"][0]["task_success"] is True


def test_alignment_template_never_invents_matches():
    sample, record, _ = t3_pair()
    result = alignment_template([record])
    assert [r["turn"] for r in result] == [1, 2]
    assert all(not r["alignment_complete"] and not r["matches"] for r in result)
