import copy
import unittest

from dataset_validation import validate_dataset


def fixtures():
    text = "😀整体结构严谨。结尾收束仓促。"
    boundary = text.index("结尾")
    evidence = [
        {"evidence_id": "E001", "text": text[:boundary], "start": 0, "end": boundary},
        {"evidence_id": "E002", "text": text[boundary:], "start": boundary, "end": len(text)},
    ]
    claims = [
        {"claim_id": "C1", "text": "评论者肯定整体结构。", "aspect": "STRUCTURE",
         "target_scope": "整体", "stance": "POSITIVE", "evidence_ids": ["E001"]},
        {"claim_id": "C2", "text": "评论者批评结尾。", "aspect": "STRUCTURE",
         "target_scope": "结尾", "stance": "NEGATIVE", "evidence_ids": ["E002"]},
    ]
    base = {"sample_id": "Fixture:T1", "doc_id": "Fixture", "task": "T1",
            "question": "评论者如何评价整体与结尾？",
            "document": {"doc_id": "Fixture", "work_title": "《合成示例》", "text": text, "evidence": evidence},
            "gold": {"claims": claims, "required_evidence_ids": ["E001", "E002"],
                     "reference_answer": "整体严谨[E001]，结尾仓促[E002]。"},
            "events": [], "roles": {}, "required_exchanges": [],
            "material_origin": "SYNTHETIC", "task_origin": "DERIVED", "human_verified": False,
            "generation_method": "HAND_WRITTEN_TEST_FIXTURE"}
    rows = []
    for task in ("T1", "T2", "T3", "T4"):
        row = copy.deepcopy(base)
        row["sample_id"] = "Fixture:" + task
        row["task"] = task
        if task in ("T3", "T4"):
            row["task_origin"] = "CONTROLLED"
        if task == "T3":
            row["events"] = [
                {"turn": 1, "question": "先有何评价？", "visible_evidence_ids": ["E001"],
                 "expected_operation": "ADD", "gold_memory_state": ["C1"], "reference_answer": "整体严谨[E001]。"},
                {"turn": 2, "question": "新增结尾评价？", "visible_evidence_ids": ["E002"],
                 "expected_operation": "EXPAND", "gold_memory_state": ["C1", "C2"],
                 "reference_answer": "整体严谨[E001]，结尾仓促[E002]。"},
            ]
        if task == "T4":
            row["roles"] = {"archivist": ["E001"], "interpreter": ["E002"]}
            row["required_exchanges"] = [
                {"from_agent": "archivist", "to_agent": "interpreter", "evidence_id": "E001", "reason": "共享整体评价。"},
                {"from_agent": "interpreter", "to_agent": "archivist", "evidence_id": "E002", "reason": "共享结尾评价。"},
            ]
        rows.append(row)
    return rows


class DatasetValidationTests(unittest.TestCase):
    def test_valid_t1_through_t4_and_unicode_offsets(self):
        rows = fixtures()
        before = copy.deepcopy(rows)
        self.assertEqual(validate_dataset(rows), {row["sample_id"]: [] for row in rows})
        self.assertEqual(rows, before)

    def test_run_level_bad_shape_ids_and_duplicate(self):
        for rows in ({}, [None], [{}], [{"sample_id": " "}], [{"sample_id": []}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_dataset(rows)
        row = fixtures()[0]
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_dataset([row, copy.deepcopy(row)])

    def test_empty_dataset(self):
        self.assertEqual(validate_dataset([]), {})

    def test_boolean_offset_is_invalid(self):
        row = fixtures()[0]
        row["document"]["evidence"][0]["start"] = False
        self.assertTrue(validate_dataset([row])[row["sample_id"]])

    def test_evidence_mismatch_and_sequence(self):
        row = fixtures()[0]
        row["document"]["evidence"][0]["text"] = "不同原文"
        row["document"]["evidence"][1]["evidence_id"] = "E004"
        errors = validate_dataset([row])[row["sample_id"]]
        self.assertTrue(any("偏移不一致" in e for e in errors))
        self.assertTrue(any("连续编号" in e for e in errors))

    def test_local_error_does_not_isolate_other_tasks(self):
        rows = fixtures()
        rows[0]["gold"]["claims"][0]["stance"] = "UNKNOWN"
        errors = validate_dataset(rows)
        self.assertTrue(errors["Fixture:T1"])
        self.assertFalse(errors["Fixture:T2"])
        self.assertFalse(errors["Fixture:T3"])

    def test_shared_document_conflict_isolates_all_its_tasks(self):
        rows = fixtures()
        rows[0]["document"]["text"] += "新增文字"
        errors = validate_dataset(rows)
        self.assertTrue(all(any("共享 doc_id" in e for e in errs) for errs in errors.values()))

    def test_same_text_different_titles_isolates_related_docs(self):
        rows = fixtures()[:2]
        rows[1]["doc_id"] = "Other"
        rows[1]["sample_id"] = "Other:T2"
        rows[1]["document"]["doc_id"] = "Other"
        rows[1]["document"]["work_title"] = "《另一作品》"
        self.assertTrue(all(validate_dataset(rows).values()))

    def test_gold_reference_union_and_illegal_labels(self):
        row = fixtures()[0]
        row["gold"]["required_evidence_ids"] = ["E001"]
        row["gold"]["reference_answer"] = "错误[E999]"
        errors = validate_dataset([row])[row["sample_id"]]
        self.assertTrue(any("并集" in e for e in errors))

    def test_t3_future_evidence_and_memory_loss_rejected(self):
        row = fixtures()[2]
        row["events"][0]["gold_memory_state"] = ["C1", "C2"]
        row["events"][1]["gold_memory_state"] = ["C2"]
        errors = validate_dataset([row])[row["sample_id"]]
        self.assertTrue(any("尚未公开" in e for e in errors))
        self.assertTrue(any("删除了旧有效" in e for e in errors))

    def test_t3_repeated_evidence_and_bad_defer_rejected(self):
        row = fixtures()[2]
        row["events"][1]["visible_evidence_ids"] = ["E001", "E002"]
        row["events"][1]["expected_operation"] = "DEFER"
        errors = validate_dataset([row])[row["sample_id"]]
        self.assertTrue(any("重复公开" in e for e in errors))
        self.assertTrue(any("DEFER" in e for e in errors))

    def test_t4_overlap_and_exchange_ownership_rejected(self):
        row = fixtures()[3]
        row["roles"]["interpreter"].append("E001")
        row["required_exchanges"][0]["evidence_id"] = "E002"
        errors = validate_dataset([row])[row["sample_id"]]
        self.assertTrue(any("互斥" in e for e in errors))
        self.assertTrue(any("不属于发送方" in e for e in errors))

    def test_arbitrarily_wrong_nested_types_are_isolated(self):
        fields = ["task", "doc_id", "question", "gold", "events", "roles", "required_exchanges"]
        for field in fields:
            for bad in (None, [], {}, True, 0):
                row = fixtures()[0]
                if row[field] == bad:
                    continue
                row[field] = bad
                with self.subTest(field=field, bad=bad):
                    self.assertTrue(validate_dataset([row])[row["sample_id"]])


if __name__ == "__main__":
    unittest.main()
