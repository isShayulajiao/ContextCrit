"""人工 SYNTHETIC fixture；不修改仓库中真实评论。"""

import copy
import json

import pytest

from scopecrit.data import fingerprint, load_dataset, sample_key


def sample(task="T1", doc_id="demo"):
    return {
        "sample_id": f"{doc_id}:{task}", "doc_id": doc_id, "task": task,
        "question": "评价？", "events": [], "roles": {}, "required_exchanges": [],
        "document": {"doc_id": doc_id, "work_title": "作品", "text": "甲😀乙好。",
                     "evidence": [{"evidence_id": "E001", "text": "甲😀", "start": 0, "end": 2},
                                  {"evidence_id": "E002", "text": "乙好。", "start": 2, "end": 5}]},
        "gold": {"claims": [{"claim_id": "C1", "text": "正面评价", "aspect": "OVERALL",
                              "stance": "POSITIVE", "target_scope": "整体", "evidence_ids": ["E001"]}],
                 "required_evidence_ids": ["E001"], "reference_answer": "评价[E001]"},
        "material_origin": "SYNTHETIC", "task_origin": "DERIVED", "human_verified": False,
        "generation_method": "HANDWRITTEN_TEST_FIXTURE",
    }


def write_load(tmp_path, rows, fixture=True):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    before = path.read_bytes()
    result = load_dataset(path, fixture=fixture)
    assert path.read_bytes() == before
    return result


def test_unicode_offsets_and_input_unchanged(tmp_path):
    rows = [sample()]
    original = copy.deepcopy(rows)
    loaded, report = write_load(tmp_path, rows)
    assert loaded == original
    assert report["valid"]


def test_synthetic_only_allowed_with_fixture_flag(tmp_path):
    _, report = write_load(tmp_path, [sample()], fixture=False)
    assert "demo:T1" in report["invalid"]


def test_individual_error_does_not_quarantine_other_task(tmp_path):
    good, bad = sample(), sample("T2")
    bad["gold"]["claims"][0]["stance"] = "UNKNOWN"
    _, report = write_load(tmp_path, [good, bad])
    assert set(report["invalid"]) == {"demo:T2"}


def test_bad_offsets_are_document_wide_if_material_conflicts(tmp_path):
    good, bad = sample(), sample("T2")
    bad["document"]["evidence"][0]["start"] = False
    _, report = write_load(tmp_path, [good, bad])
    assert set(report["invalid"]) == {"demo:T1", "demo:T2"}
    assert any("严格整数" in x for x in report["invalid"]["demo:T2"])


def test_duplicate_ids_abort_run(tmp_path):
    with pytest.raises(ValueError, match="sample_id 重复"):
        write_load(tmp_path, [sample(), sample()])


@pytest.mark.parametrize("text", ["{broken", "[NaN]", "[Infinity]"])
def test_invalid_json_rejected(tmp_path, text):
    path = tmp_path / "broken.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="JSON"):
        load_dataset(path)


def test_overlapping_evidence_warns_without_rejection(tmp_path):
    row = sample()
    row["document"]["evidence"][1].update(start=1, end=3, text="😀乙")
    _, report = write_load(tmp_path, [row])
    assert report["valid"]
    assert any("重叠" in warning for warning in report["warnings"])


@pytest.mark.parametrize("rows", [{"task": "T1"}, "text", None])
def test_top_level_must_be_array(tmp_path, rows):
    with pytest.raises(ValueError, match="顶层"):
        write_load(tmp_path, rows)


def test_malformed_row_is_isolated(tmp_path):
    _, report = write_load(tmp_path, [None, sample(), {"sample_id": []}])
    assert set(report["invalid"]) == {"__invalid_0", "__invalid_2"}
    assert sample_key(None, 8) == "__invalid_8"


def test_equal_text_with_conflicting_titles_quarantines_both_docs(tmp_path):
    first, second = sample(), sample(doc_id="second")
    second["document"]["work_title"] = "另外的作品"
    _, report = write_load(tmp_path, [first, second])
    assert set(report["invalid"]) == {"demo:T1", "second:T1"}


def test_same_text_same_title_different_doc_is_warning(tmp_path):
    _, report = write_load(tmp_path, [sample(), sample(doc_id="second")])
    assert report["valid"] and report["warnings"]


def t3_sample():
    row = sample("T3")
    row["task_origin"] = "CONTROLLED"
    row["gold"]["claims"].append({"claim_id": "C2", "text": "局部肯定", "aspect": "LANGUAGE",
                                    "stance": "POSITIVE", "target_scope": "局部", "evidence_ids": ["E002"]})
    row["gold"]["required_evidence_ids"] = ["E001", "E002"]
    row["gold"]["reference_answer"] = "完整评价[E001][E002]"
    row["events"] = [
        {"turn": 1, "question": "先评价", "visible_evidence_ids": ["E001"], "expected_operation": "ADD",
         "gold_memory_state": ["C1"], "reference_answer": "评价[E001]"},
        {"turn": 2, "question": "再评价", "visible_evidence_ids": ["E002"], "expected_operation": "EXPAND",
         "gold_memory_state": ["C1", "C2"], "reference_answer": "完整评价[E001][E002]"},
    ]
    return row


def test_t3_cumulative_memory_valid(tmp_path):
    _, report = write_load(tmp_path, [t3_sample()])
    assert report["valid"]


def test_t3_future_evidence_reference_rejected(tmp_path):
    row = t3_sample()
    row["events"][0]["gold_memory_state"] = ["C1", "C2"]
    _, report = write_load(tmp_path, [row])
    assert any("尚未公开" in x for x in report["invalid"]["demo:T3"])


def test_t3_overwrite_rejected(tmp_path):
    row = t3_sample()
    row["events"][1]["gold_memory_state"] = ["C2"]
    _, report = write_load(tmp_path, [row])
    assert any("删除" in x for x in report["invalid"]["demo:T3"])


def test_t4_allocation_and_exchange_ownership(tmp_path):
    row = t3_sample()
    row.update(sample_id="demo:T4", task="T4", events=[],
               roles={"archivist": ["E001"], "interpreter": ["E002"]},
               required_exchanges=[{"from_agent": "archivist", "to_agent": "interpreter",
                                    "evidence_id": "E001", "reason": "补充必要证据"}])
    assert write_load(tmp_path, [row])[1]["valid"]
    row["required_exchanges"][0]["evidence_id"] = "E002"
    _, report = write_load(tmp_path, [row])
    assert any("不属于发送方" in x for x in report["invalid"]["demo:T4"])


def test_fingerprint_only_sorts_object_keys():
    assert fingerprint({"a": "😀", "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": "😀"})
    assert fingerprint("甲 乙") != fingerprint("甲乙")
    assert fingerprint([1, 2]) != fingerprint([2, 1])


def test_prefix_limit_is_applied_before_validation_and_duplicate_detection(tmp_path):
    path = tmp_path / "prefix.json"
    # 尾部有重复 ID 和非法条目，选第一条时均不属于本次处理范围。
    path.write_text(json.dumps([sample(), sample(), None]), encoding="utf-8")
    rows, report = load_dataset(path, fixture=True, max_samples=1)
    assert len(rows) == report["total"] == 1
    assert report["source_total"] == 3 and report["max_samples"] == 1
    assert report["valid"]
    with pytest.raises(ValueError, match="sample_id 重复"):
        load_dataset(path, fixture=True, max_samples=None)


def test_first_521_positions_selected_without_replacing_invalid_rows(tmp_path):
    rows = [sample(doc_id=f"demo{i}") for i in range(522)]
    rows[10] = None
    path = tmp_path / "large.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    selected, report = load_dataset(path, fixture=True, max_samples=521)
    assert len(selected) == report["total"] == 521
    assert report["source_total"] == 522
    assert selected[-1]["sample_id"] == "demo520:T1"
    assert set(report["invalid"]) == {"__invalid_10"}
    # 超过文件长度时自然处理全部，不填充或重复记录。
    assert len(load_dataset(path, fixture=True, max_samples=999)[0]) == 522


@pytest.mark.parametrize("limit", [0, -1, True, False, "521", 1.5])
def test_dataset_rejects_nonpositive_and_noninteger_limits(tmp_path, limit):
    path = tmp_path / "input.json"
    path.write_text(json.dumps([sample()]))
    with pytest.raises(ValueError, match="max_samples"):
        load_dataset(path, fixture=True, max_samples=limit)
