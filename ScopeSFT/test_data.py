"""Offline tests for training labels, legal rollout and restricted observations."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ScopeSFT.data import _iter_json_array, canonical, iter_records, load_records

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ScopeCrit/examples/merged_demo.json"


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, rows):
        p = Path(self.directory.name) / "train.json"
        p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def fixture(self):
        return json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_all_four_tasks_reach_finalization_and_targets_are_actions(self):
        records = load_records(str(FIXTURE))
        by_task = {task: [] for task in ("T1", "T2", "T3", "T4")}
        for record in records:
            by_task[record["metadata"]["task"]].append(record)
            target = json.loads(record["chosen"])
            self.assertIn("arguments", target)
            self.assertNotIn("gold", json.loads(record["messages"][1]["content"]))
        for task, selected in by_task.items():
            self.assertEqual(json.loads(selected[-1]["chosen"])["action"], "FINALIZE", task)
        self.assertTrue(any(r["metadata"]["action"] == "COMPARE" for r in by_task["T2"]))
        self.assertTrue(any(r["metadata"]["action"] == "SEND_EVIDENCE" for r in by_task["T4"]))

    def test_t2_reference_answer_never_leaks_through_comparison_observation(self):
        sample = self.fixture()[1]
        sentinel = "仅限最终教师目标的唯一答案哨兵"
        sample["gold"]["reference_answer"] = sentinel + "[E001][E002]。"
        records = load_records(self.write([sample]))
        self.assertTrue(any(r["metadata"]["action"] == "COMPARE" for r in records))
        for record in records:
            self.assertNotIn(sentinel, record["messages"][1]["content"])
        final = [r for r in records if r["metadata"]["action"] == "FINALIZE"]
        self.assertEqual(len(final), 1)
        self.assertIn(sentinel, final[0]["chosen"])
        view = json.loads(final[0]["messages"][1]["content"])
        self.assertEqual(view["relations"][0]["relation_type"], "UNRESOLVED")
        self.assertIn("尚未经过独立核验", view["relations"][0]["explanation"])

    def test_t3_future_evidence_and_future_claims_absent_from_observations(self):
        rows = self.fixture()
        selected = [r for r in rows if r["task"] == "T3"]
        # Unique sentinel for the future source, so it cannot hide in another field.
        future = "未来私有证据哨兵"
        old = selected[0]["document"]["evidence"][1]["text"]
        selected[0]["document"]["text"] = selected[0]["document"]["text"].replace(old, future)
        selected[0]["document"]["evidence"][1]["text"] = future
        selected[0]["document"]["evidence"][1]["end"] = len(selected[0]["document"]["text"])
        records = load_records(self.write(selected))
        first = [r for r in records if r["metadata"]["turn"] == 1]
        self.assertTrue(first)
        for record in first:
            view = json.loads(record["messages"][1]["content"])
            self.assertNotIn("document_text", view)
            self.assertNotIn(future, record["messages"][1]["content"])
            self.assertTrue(all(e["evidence_id"] != "E002" for e in view["visible_evidence"]))

    def test_t4_private_material_arrives_only_via_real_delivery(self):
        records = load_records(self.write([self.fixture()[3]]))
        first_view = json.loads(records[0]["messages"][1]["content"])
        self.assertEqual(first_view["role"], "archivist")
        self.assertEqual({e["evidence_id"] for e in first_view["visible_evidence"]}, {"E001"})
        self.assertNotIn("document_text", first_view)
        interpreter = [json.loads(r["messages"][1]["content"]) for r in records if r["metadata"]["role"] == "interpreter"]
        self.assertTrue(interpreter[0]["delivered_messages"])
        self.assertEqual({e["evidence_id"] for e in interpreter[0]["visible_evidence"]}, {"E001", "E002"})
        self.assertTrue(any(e["access_origin"] == "received" and e["received_message_id"] for e in interpreter[0]["visible_evidence"]))

    def test_source_anchored_pair_has_semantic_span_and_preserves_ids(self):
        sample = self.fixture()[0]
        # Synthetic source explicitly restricts the assessment to two parts.
        text = "《示例小说》只登了两段，第一第二段的描写法很好。"
        sample["document"]["text"] = text
        sample["document"]["evidence"] = [{"evidence_id": "E001", "text": text, "start": 0, "end": len(text)}]
        sample["gold"] = {"claims": [{"claim_id": "C1", "text": "评论者肯定第一、二段的描写法。", "aspect": "NARRATION", "stance": "POSITIVE", "target_scope": "第一、二段", "evidence_ids": ["E001"]}],
                          "required_evidence_ids": ["E001"], "reference_answer": "评论者肯定前两段描写[E001]。"}
        records = load_records(self.write([sample]))
        pairs = [r for r in records if r["rejected"]]
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        span = pair["semantic_span"]
        self.assertIn("第一、二段", pair["chosen"][slice(*span["chosen"])])
        self.assertIn("全书", pair["rejected"][slice(*span["rejected"])])
        positive, negative = (json.loads(pair[key])["arguments"]["claims"][0] for key in ("chosen", "rejected"))
        self.assertEqual(positive["evidence_ids"], negative["evidence_ids"])
        self.assertEqual(positive["stance"], negative["stance"])
        self.assertEqual(pair["metadata"]["negative_audit"]["kind"], "rule_generated")
        self.assertFalse(pair["metadata"]["negative_audit"]["human_verified_by_code"])

    def voice_fixture(self, introduction="有狄克先生的文章为证——", quotation="“小说整体是一首史诗，但局部不够真实。", disclaimer=True):
        sample = self.fixture()[0]
        text = introduction + "\n" + quotation
        sample["document"]["text"] = text
        sample["document"]["evidence"] = [
            {"evidence_id": "E001", "text": introduction, "start": 0, "end": len(introduction)},
            {"evidence_id": "E002", "text": quotation, "start": len(introduction) + 1, "end": len(text)}]
        positive = "文中转引狄克的文章称小说整体是一首史诗，但认为局部不够真实"
        positive += "；这不是本文叙述者直接给出的小说总评。" if disclaimer else "。"
        sample["gold"] = {"claims": [{"claim_id": "C1", "text": positive, "aspect": "OVERALL", "stance": "MIXED",
                                      "target_scope": "被引评论的总体肯定与局部真实性保留", "evidence_ids": ["E001", "E002"]}],
                          "required_evidence_ids": ["E001", "E002"], "reference_answer": "文中转引狄克评价小说[E001][E002]。"}
        return sample

    def test_voice_negative_requires_matching_source_and_changes_only_voice(self):
        records = load_records(self.write([self.voice_fixture()]))
        pairs = [r for r in records if r["rejected"]]
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        positive, negative = (json.loads(pair[key])["arguments"]["claims"][0] for key in ("chosen", "rejected"))
        self.assertEqual(negative["text"], "本文叙述者直接断言小说整体是一首史诗，但认为局部不够真实。")
        self.assertEqual(negative["target_scope"], "本文叙述者的总体肯定与局部真实性保留")
        self.assertNotIn("这不是", negative["text"])
        for field in ("aspect", "stance", "evidence_ids"):
            self.assertEqual(positive[field], negative[field])
        audit = pair["metadata"]["negative_audit"]
        self.assertEqual(audit["attributed_speaker"], "狄克")
        self.assertEqual(audit["evidence_ids"], ["E001", "E002"])
        self.assertEqual(audit["rule"], "explicit_quoted_voice_to_narrator")
        self.assertFalse(audit["human_verified_by_code"])
        a, b = pair["semantic_span"]["rejected"]
        self.assertEqual(pair["rejected"][a:b], negative["text"])

    def test_voice_rule_refuses_missing_attribution_quote_or_disclaimer(self):
        candidates = [self.voice_fixture(introduction="这里讨论小说。"),
                      self.voice_fixture(introduction="有另一先生的文章为证——"),
                      self.voice_fixture(quotation="小说整体是一首史诗，但局部不够真实。"),
                      self.voice_fixture(disclaimer=False)]
        for sample in candidates:
            with self.subTest(sample=sample["document"]["text"]):
                self.assertFalse(any(r["rejected"] for r in load_records(self.write([sample]))))

    def test_unsupported_negatives_are_not_fabricated(self):
        # Overall/ending contrast alone is not the narrow two-parts proof rule.
        self.assertFalse(any(r["rejected"] for r in load_records(str(FIXTURE))))

    def test_test_split_is_warned_used_and_original_file_unchanged(self):
        rows = [self.fixture()[0]]
        rows[0]["split"] = "test"
        del rows[0]["generation_method"]
        path = self.write(rows)
        before = Path(path).read_bytes()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = load_records(path)
        self.assertTrue(caught)
        self.assertTrue(records)
        self.assertTrue(all(r["metadata"]["split"] == "test" for r in records))
        self.assertEqual(before, Path(path).read_bytes())

    def test_bad_gold_citation_and_duplicate_samples_fail(self):
        rows = self.fixture()
        rows[0]["gold"]["claims"][0]["evidence_ids"] = ["E999"]
        with self.assertRaisesRegex(ValueError, "校验失败"):
            load_records(self.write(rows[:1]))
        with self.assertRaisesRegex(ValueError, "重复 sample_id"):
            load_records(self.write([self.fixture()[0]] * 2))

    def test_incremental_decoder_crosses_unicode_and_escape_boundaries(self):
        rows = [{"text": "范围\\字符\"与换行\n" * 100, "nested": [{"id": "E001"}]}]
        path = self.write(rows)
        self.assertEqual(list(_iter_json_array(Path(path), chunk_size=3)), rows)
        # Expansion is a generator; requesting the first record need not parse
        # or build the rest of the corpus.
        source = self.fixture()[0]
        malformed = Path(self.directory.name) / "late-invalid.json"
        malformed.write_text("[" + canonical(source) + ", {bad]", encoding="utf-8")
        records = iter_records(str(malformed))
        self.assertEqual(next(records)["metadata"]["sample_id"], source["sample_id"])
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            list(records)

    def test_incremental_decoder_rejects_bad_array_termination(self):
        path = Path(self.directory.name) / "bad-array.json"
        for content in ("[]", "[{} ,]", "[{}] trailing", "[{} {}]", "[{}", "[{},,{}]"):
            path.write_text(content, encoding="utf-8")
            with self.subTest(content=content):
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    list(_iter_json_array(path, chunk_size=2))

    def test_reviewed_annotation_metadata_does_not_block_training(self):
        rows = [self.fixture()[0]]
        rows[0]["human_verified"] = True
        records = load_records(self.write(rows))
        self.assertTrue(all(r["metadata"]["human_verified"] is True for r in records))

    def test_t3_retracting_existing_memory_is_rejected_by_runtime_contract(self):
        rows = [self.fixture()[2]]
        rows[0]["events"][1]["gold_memory_state"] = ["C2"]
        rows[0]["events"][1]["reference_answer"] = "仅保留新主张[E002]。"
        with self.assertRaisesRegex(ValueError, "删除了旧有效主张"):
            load_records(self.write(rows))

    def test_absolute_path_and_strict_json(self):
        with self.assertRaisesRegex(ValueError, "绝对路径"):
            load_records("relative.json")
        p = Path(self.directory.name) / "invalid.json"
        p.write_text("[NaN]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "非法 JSON"):
            load_records(str(p))


if __name__ == "__main__":
    unittest.main()
