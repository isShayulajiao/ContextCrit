"""Source-row limits retain complete trajectories and auditable selection counts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ScopeSFT.data import canonical, iter_records, load_records, read_samples


FIXTURE = Path(__file__).resolve().parents[1] / "ScopeCrit/examples/merged_demo.json"


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.rows = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def write(self, rows):
        path = Path(self.directory.name) / "source.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def test_first_original_rows_are_selected_before_action_expansion(self):
        path = self.write(self.rows)
        selection = {}
        limited = load_records(path, max_samples=2, selection=selection)
        expected = load_records(self.write(self.rows[:2]))
        self.assertEqual(limited, expected)
        self.assertGreater(len(limited), 2)
        self.assertEqual({r["metadata"]["sample_id"] for r in limited},
                         {s["sample_id"] for s in self.rows[:2]})
        self.assertEqual(selection, {"max_samples": 2, "source_total": 4, "selected_samples": 2,
                                     "selected_docs": 1, "selected_tasks": {"T1": 1, "T2": 1}})

    def test_t3_and_t4_keep_all_rounds_roles_and_finalization(self):
        rows = [self.rows[2], self.rows[3], self.rows[0], self.rows[1]]
        selected = load_records(self.write(rows), max_samples=2)
        complete = load_records(self.write(rows[:2]))
        self.assertEqual(selected, complete)
        t3 = [r for r in selected if r["metadata"]["task"] == "T3"]
        self.assertEqual({r["metadata"]["turn"] for r in t3}, {1, 2})
        self.assertEqual(sum(r["metadata"]["action"] == "FINALIZE" for r in t3), 2)
        t4 = [r for r in selected if r["metadata"]["task"] == "T4"]
        self.assertEqual({r["metadata"]["role"] for r in t4}, {"archivist", "interpreter"})
        self.assertEqual(t4[-1]["metadata"]["action"], "FINALIZE")

    def test_none_and_oversized_limits_use_all_rows(self):
        path = self.write(self.rows)
        unlimited = load_records(path)
        for limit in (None, 100):
            selection = {}
            with self.subTest(limit=limit):
                self.assertEqual(load_records(path, max_samples=limit, selection=selection), unlimited)
                self.assertEqual(selection["source_total"], 4)
                self.assertEqual(selection["selected_samples"], 4)
                self.assertEqual(selection["selected_tasks"], {"T1": 1, "T2": 1, "T3": 1, "T4": 1})

    def test_invalid_selected_row_is_fatal_instead_of_replaced(self):
        rows = deepcopy(self.rows)
        rows[0]["gold"]["claims"][0]["evidence_ids"] = ["E999"]
        with self.assertRaisesRegex(ValueError, "校验失败"):
            load_records(self.write(rows), max_samples=1)
        with self.assertRaisesRegex(ValueError, "不是 JSON 对象"):
            read_samples(self.write([None, *self.rows]), max_samples=1)

    def test_unselected_schema_invalid_tail_is_counted_without_becoming_training(self):
        selection = {}
        rows = read_samples(self.write([self.rows[0], None, {"not": "a training row"}]),
                            max_samples=1, selection=selection)
        self.assertEqual(rows, self.rows[:1])
        self.assertEqual(selection["source_total"], 3)
        self.assertEqual(selection["selected_samples"], 1)

    def test_duplicate_selected_ids_fail_and_source_is_not_changed(self):
        path = self.write([self.rows[0], self.rows[0], self.rows[1]])
        before = Path(path).read_bytes()
        with self.assertRaisesRegex(ValueError, "重复 sample_id"):
            load_records(path, max_samples=2)
        self.assertEqual(Path(path).read_bytes(), before)

    def test_malformed_json_tail_still_fails_after_selected_rows(self):
        path = Path(self.directory.name) / "malformed.json"
        path.write_text("[" + canonical(self.rows[0]) + ", {bad]", encoding="utf-8")
        records = iter_records(str(path), max_samples=1)
        self.assertEqual(next(records)["metadata"]["sample_id"], self.rows[0]["sample_id"])
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            list(records)

    def test_python_limit_api_rejects_invalid_values(self):
        path = self.write(self.rows)
        for limit in (0, -1, 1.5, True, "2"):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "max-samples"):
                read_samples(path, max_samples=limit)


if __name__ == "__main__":
    unittest.main()
