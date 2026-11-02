"""Offline HTTP/CLI regression tests; all HTTP requests target localhost only."""
from __future__ import annotations

import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

import consistency_check as app

HERE = Path(__file__).resolve().parent


def review(score=5, *, evidence_errors=None, rationale="合成离线接口返回，不能作为实际质量结论。"):
    ids = evidence_errors or []
    scores = {name: score for name in app.DIMENSIONS}
    return {"scores": scores, "pass_fail_review": app.decision(scores, ids),
            "evidence_error_ids": ids, "short_rationale": rationale}


@contextmanager
def mock_api(responder=None):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            call = {"path": self.path, "payload": payload,
                    "authorization": self.headers.get("Authorization", "")}
            calls.append(call)
            if responder:
                status, body = responder(call, len(calls))
            else:
                sample = json.loads(payload["messages"][1]["content"])
                score = {"T1": 5, "T2": 4, "T3": 3, "T4": 2}[sample["task"]]
                result = review(score, rationale="LEAK_PREVIOUS_JUDGE_RATIONALE")
                body = {"id": "offline-mock", "model": payload["model"],
                        "choices": [{"finish_reason": "stop", "message": {
                            "content": json.dumps(result, ensure_ascii=False),
                            "reasoning_content": "PRIVATE_REASONING_MUST_NOT_BE_SAVED"}}],
                        "usage": {"prompt_tokens": 123, "completion_tokens": 50, "total_tokens": 173}}
                status = 200
            encoded = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}):
            yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def ok_response(call, result=None):
    return 200, {"id": "mock", "model": call["payload"]["model"],
                 "choices": [{"finish_reason": "stop", "message": {
                     "content": json.dumps(result or review(), ensure_ascii=False)}}]}


class ReviewParsingTests(unittest.TestCase):
    def test_single_fenced_json_is_accepted(self):
        value = review()
        self.assertEqual(app.parse_review("```json\n" + json.dumps(value) + "\n```", {"E001"}), value)
        with self.assertRaises(ValueError):
            app.parse_review("Some prose " + json.dumps(value), {"E001"})

    def test_boolean_float_out_of_range_and_missing_dimensions_rejected(self):
        for bad in (True, False, 5.0, "5", 0, 6, None):
            result = review()
            result["scores"]["Accuracy"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                app.validate_review(result)
        result = review()
        del result["scores"]["Naturalness"]
        with self.assertRaises(ValueError):
            app.validate_review(result)

    def test_frozen_label_thresholds(self):
        for score, errors, expected in ((5, [], "PASS"), (4, [], "PASS"),
                                        (3, [], "REVIEW"), (2, [], "FAIL"),
                                        (1, [], "FAIL"), (5, ["E001"], "REVIEW")):
            result = review(score, evidence_errors=errors)
            self.assertEqual(app.validate_review(result, {"E001"})["pass_fail_review"], expected)
            result["pass_fail_review"] = "FAIL" if expected != "FAIL" else "PASS"
            with self.assertRaises(ValueError):
                app.validate_review(result, {"E001"})

    def test_unknown_ids_duplicate_json_keys_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            app.validate_review(review(5, evidence_errors=["E999"]), {"E001"})
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                app.strict_json(raw)

    def test_nonstring_labels_are_validation_errors(self):
        for label in ([], {}, None, True, 5):
            result = review()
            result["pass_fail_review"] = label
            with self.subTest(label=label), self.assertRaises(ValueError):
                app.validate_review(result)


class EndToEndOfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="consistency-offline-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = json.loads((HERE / "example_dataset.json").read_text(encoding="utf-8"))
        for row in rows:
            row["deepseek_review"] = {"score": "DEEPSEEK_REVIEW_CANARY", "reason": "不要传给独立裁判"}
            row["generator"] = "GPT_GENERATOR_CANARY"
        self.original = rows
        self.inputs = [self.root / "train.json", self.root / "test.json"]
        for path, chunk in zip(self.inputs, (rows[:2], rows[2:])):
            app.write_json(path, chunk)
        self.out = self.root / "results"
        self.config_path = self.root / "config.json"
        self.cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        self.cfg.pop("input_file", None)
        self.cfg.update(input_files=[str(path) for path in self.inputs], output_dir=str(self.out),
                        limit=0, timeout_seconds=5, max_attempts=1, retry_delay_seconds=0)
        for index, judge in enumerate(self.cfg["judges"]):
            judge["api_key"] = f"UNIT_TEST_FAKE_KEY_{index}_DO_NOT_SAVE"
            judge["api_key_env"] = f"CONSISTENCY_TEST_EMPTY_ENV_{index}"

    def configure(self, address):
        for judge, segment in zip(self.cfg["judges"], ("qwen", "glm", "grok")):
            judge["base_url"] = address + "/" + segment
        app.write_json(self.config_path, self.cfg)

    def run_app(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = app.main(["--config", str(self.config_path), *map(str, args)])
        return code, stdout.getvalue() + stderr.getvalue()

    def ratings(self):
        return app.read_json(self.out / "ratings.json")

    def call_count(self):
        with sqlite3.connect(self.out / "checkpoint.sqlite3") as db:
            return db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def test_two_files_cli_limit_independence_resume_and_input_unchanged(self):
        before = [path.read_bytes() for path in self.inputs]
        with mock_api() as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--input", *self.inputs, "--limit", 1)
            self.assertEqual(code, 2)  # The chosen batch completed, entire dataset has not.
            self.assertEqual(len(calls), 3)
            self.assertEqual(sum(len(r["ratings"]) == 3 for r in self.ratings()), 1)
            code, _ = self.run_app("--limit", 1)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 6)
            code, _ = self.run_app("--limit", 0)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 12)
            code, _ = self.run_app()
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 12)
            for call in calls:
                self.assertEqual(call["path"].count("/chat/completions"), 1)
                messages = call["payload"]["messages"]
                self.assertEqual(len(messages), 2)
                self.assertEqual(messages[0]["content"], app.SYSTEM_PROMPT)
                sample = json.loads(messages[1]["content"])
                self.assertEqual(set(sample), set(app.ALLOWED_SAMPLE_FIELDS))
                for canary in ("DEEPSEEK_REVIEW_CANARY", "GPT_GENERATOR_CANARY", "LEAK_PREVIOUS_JUDGE_RATIONALE"):
                    self.assertNotIn(canary, messages[1]["content"])
                self.assertNotIn("ratings", sample)
            by_sample = {}
            for call in calls:
                sample = json.loads(call["payload"]["messages"][1]["content"])
                by_sample.setdefault(sample["sample_id"], []).append(sample)
            for values in by_sample.values():
                self.assertEqual(len(values), 3)
                self.assertEqual(values[0], values[1])
                self.assertEqual(values[1], values[2])
        self.assertEqual([path.read_bytes() for path in self.inputs], before)
        summary = app.read_json(self.out / "summary.json")
        self.assertEqual(summary["run"]["fully_rated_samples"], 4)
        self.assertEqual(self.call_count(), 12)
        self.assertNotIn("PRIVATE_REASONING_MUST_NOT_BE_SAVED", (self.out / "calls.jsonl").read_text())
        task_rows = app.read_json(self.out / "task_scores.json")
        self.assertEqual(len(task_rows), 12)
        self.assertEqual({row["scope"] for row in task_rows}, {"COMBINED", *map(str, self.inputs)})
        combined = {row["task"]: row for row in task_rows if row["scope"] == "COMBINED"}
        self.assertEqual(set(combined), {"T1", "T2", "T3", "T4"})
        self.assertTrue(all(row["valid_samples"] == row["complete_three_judge_samples"] == 1
                            for row in combined.values()))
        # One task per synthetic sample has a constant category/score, so chance-
        # corrected agreement is undefined rather than manufactured as 1.0.
        self.assertTrue(all(row["kappa_"+dim] is None and row["kappa_"+dim+"_reason"]
                            for row in combined.values() for dim in app.DIMENSIONS))
        for row in task_rows:
            if row["scope"] == str(self.inputs[0]) and row["task"] in {"T3", "T4"}:
                self.assertEqual(row["valid_samples"], 0)
                self.assertTrue(all(row["kappa_"+dim] is None for dim in app.DIMENSIONS))
        report = (self.out / "report.md").read_text(encoding="utf-8")
        self.assertEqual(sum(line.startswith("| T") for line in report.splitlines()), 60)

    def test_retry_errors_only_repeats_failed_judge(self):
        failed_once = False

        def responder(call, number):
            nonlocal failed_once
            if call["path"].startswith("/glm/") and not failed_once:
                failed_once = True
                return 503, {"error": {"message": "temporary offline test failure"}}
            return ok_response(call)

        with mock_api(responder) as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--limit", 1)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 3)
            first = self.ratings()[0]
            self.assertEqual(first["judge_status"]["GLM"]["status"], "error")
            self.assertEqual(set(first["ratings"]), {"Qwen", "Grok"})
            code, _ = self.run_app("--retry-errors", "--limit", 1)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 4)
            self.assertTrue(calls[-1]["path"].startswith("/glm/"))
            self.assertEqual(len(self.ratings()[0]["ratings"]), 3)

    def test_fatal_401_stops_batch_and_does_not_leak_credentials(self):
        def responder(call, number):
            return 401, {"error": {"message": "bad key " + call["authorization"]}}

        with mock_api(responder) as (address, calls):
            self.configure(address)
            code, printed = self.run_app("--limit", 0)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(self.call_count(), 1)
            self.assertEqual(self.ratings()[0]["judge_status"]["Qwen"]["status"], "error")
            for judge in self.cfg["judges"]:
                secret = judge["api_key"]
                self.assertNotIn(secret, printed)
                for path in self.out.iterdir():
                    if path.is_file():
                        self.assertNotIn(secret.encode(), path.read_bytes(), msg=path.name)

    def test_echoed_credentials_redacted_from_final_reply_and_metadata(self):
        def responder(call, number):
            secret = call["authorization"].removeprefix("Bearer ")
            status, body = ok_response(call, review(rationale="echo " + secret))
            body["id"] = "id_" + secret
            return status, body

        with mock_api(responder) as (address, calls):
            self.configure(address)
            _, printed = self.run_app("--limit", 1)
            self.assertEqual(len(calls), 3)
        for judge in self.cfg["judges"]:
            secret = judge["api_key"]
            self.assertNotIn(secret, printed)
            for path in self.out.iterdir():
                if path.is_file():
                    self.assertNotIn(secret.encode(), path.read_bytes(), msg=path.name)

    def test_long_metadata_survives_and_truncated_reply_is_missing_not_a_score(self):
        def responder(call, number):
            status, body = ok_response(call)
            body["id"] = "long_" + "x" * 3000
            body["usage"] = {"prompt_tokens": 100, "completion_tokens": 15, "total_tokens": 115}
            if call["path"].startswith("/glm/"):
                body["choices"][0]["finish_reason"] = "length"
            return status, body

        with mock_api(responder) as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--limit", 1)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 3)
        first = self.ratings()[0]
        self.assertEqual(set(first["ratings"]), {"Qwen", "Grok"})
        self.assertEqual(first["judge_status"]["GLM"]["status"], "error")
        exported = [json.loads(line) for line in (self.out / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(exported[0]["metadata"]["response_id"]), 3005)
        self.assertEqual(exported[1]["metadata"]["usage"]["total_tokens"], 115)

    def test_invalid_boolean_score_retries_without_creating_a_rating(self):
        def responder(call, number):
            result = review()
            if number == 1:
                result["scores"]["Accuracy"] = True
            return ok_response(call, result)

        with mock_api(responder) as (address, calls):
            self.configure(address)
            self.cfg["max_attempts"] = 2
            app.write_json(self.config_path, self.cfg)
            code, _ = self.run_app("--limit", 1)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0]["path"], calls[1]["path"])
        self.assertEqual(len(self.ratings()[0]["ratings"]), 3)
        exported = [json.loads(line) for line in (self.out / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(exported[0]["status"], "error")
        self.assertEqual(exported[1]["status"], "ok")

    def test_report_only_requires_no_keys_and_calls_no_api(self):
        with mock_api() as (address, calls):
            self.configure(address)
            self.run_app("--limit", 1)
            self.assertEqual(len(calls), 3)
            for judge in self.cfg["judges"]:
                judge["api_key"] = ""
            app.write_json(self.config_path, self.cfg)
            code, _ = self.run_app("--report-only")
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 3)
            self.assertTrue((self.out / "report.md").is_file())

    def test_judge_configuration_change_refuses_checkpoint_reuse(self):
        with mock_api() as (address, calls):
            self.configure(address)
            self.run_app("--limit", 1)
            self.cfg["judges"][0]["temperature"] = 0.2
            app.write_json(self.config_path, self.cfg)
            with self.assertRaisesRegex(ValueError, "变化"):
                self.run_app("--limit", 1)
            self.assertEqual(len(calls), 3)

    def test_two_files_duplicate_id_rejected_before_any_request(self):
        second = app.read_json(self.inputs[1])
        second.append(copy.deepcopy(self.original[0]))
        app.write_json(self.inputs[1], second)
        with mock_api() as (address, calls):
            self.configure(address)
            with self.assertRaisesRegex(ValueError, "重复"):
                self.run_app()
            self.assertFalse(calls)

    def test_check_only_needs_no_keys(self):
        with mock_api() as (address, calls):
            self.configure(address)
            for judge in self.cfg["judges"]:
                judge["api_key"] = ""
            app.write_json(self.config_path, self.cfg)
            code, _ = self.run_app("--check-only")
            self.assertEqual(code, 0)
            self.assertFalse(calls)
            self.assertFalse((self.out / "checkpoint.sqlite3").exists())

    def test_empty_second_file_still_has_twelve_task_rows(self):
        app.write_json(self.inputs[1], [])
        with mock_api() as (address, calls):
            self.configure(address)
            code, _ = self.run_app()
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 6)
        rows = app.read_json(self.out / "task_scores.json")
        self.assertEqual(len(rows), 12)
        empty = [r for r in rows if r["scope"] == str(self.inputs[1])]
        self.assertEqual(len(empty), 4)
        self.assertTrue(all(r["valid_samples"] == 0 and r["kappa_"+dim] is None
                            for r in empty for dim in app.DIMENSIONS))

    def test_five_score_kappas_and_alphas_are_distinct_from_label_kappa(self):
        first = copy.deepcopy(self.original[0])
        second = copy.deepcopy(first)
        second["doc_id"] = "SecondSyntheticDoc"
        second["sample_id"] = "SecondSyntheticDoc:T1"
        second["document"]["doc_id"] = second["doc_id"]
        app.write_json(self.inputs[0], [first])
        app.write_json(self.inputs[1], [second])
        def responder(call, number):
            sample = json.loads(call["payload"]["messages"][1]["content"])
            score = 1 if sample["sample_id"] == first["sample_id"] else (4 if call["path"].startswith("/grok/") else 5)
            return ok_response(call, review(score))
        with mock_api(responder) as (address, calls):
            self.configure(address)
            code, _ = self.run_app()
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 6)
            # Re-export the same checkpoint: no rerating and no API request.
            code, _ = self.run_app("--report-only")
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 6)
        rows = app.read_json(self.out / "task_scores.json")
        row = next(r for r in rows if r["scope"] == "COMBINED" and r["task"] == "T1")
        for dim in app.DIMENSIONS:
            self.assertAlmostEqual(row["kappa_"+dim], 5/11)
            self.assertAlmostEqual(row["alpha_"+dim], 7/8)
            self.assertEqual(row["kappa_"+dim+"_included_samples"], 2)
        self.assertEqual(row["label_kappa"], 1.0)

    def test_single_file_cli_completes_and_report_only_reuses_every_rating(self):
        single = self.root / "single.json"
        app.write_json(single, self.original)
        self.cfg["input_files"] = ["", ""]
        with mock_api() as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--input", single, "--limit", 0)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 12)
            self.assertEqual(len(self.ratings()), 4)
            self.assertEqual({r["source_file"] for r in self.ratings()}, {str(single)})
            for judge in self.cfg["judges"]:
                judge["api_key"] = ""
            app.write_json(self.config_path, self.cfg)
            code, _ = self.run_app("--input", single, "--report-only")
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 12)
        rows = app.read_json(self.out / "task_scores.json")
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["scope"] for row in rows}, {"COMBINED", str(single)})

    def test_single_file_check_only_needs_no_keys(self):
        with mock_api() as (address, calls):
            self.configure(address)
            self.cfg["input_files"] = [str(self.inputs[0])]
            for judge in self.cfg["judges"]:
                judge["api_key"] = ""
            app.write_json(self.config_path, self.cfg)
            code, _ = self.run_app("--check-only")
            self.assertEqual(code, 0)
            self.assertEqual(app.read_json(self.out / "input_validation.json")["total"], 2)
            self.assertFalse(calls)
            self.assertFalse((self.out / "checkpoint.sqlite3").exists())

    def test_config_accepts_single_string_single_element_and_trailing_blank(self):
        with mock_api() as (address, calls):
            self.configure(address)
            for setting in (str(self.inputs[0]), [str(self.inputs[0])], [str(self.inputs[0]), ""]):
                with self.subTest(setting=setting):
                    self.cfg["input_files"] = setting
                    app.write_json(self.config_path, self.cfg)
                    code, _ = self.run_app("--check-only")
                    self.assertEqual(code, 0)
                    self.assertEqual(app.read_json(self.out / "input_validation.json")["total"], 2)
            self.assertFalse(calls)

    def test_legacy_input_file_alias(self):
        with mock_api() as (address, calls):
            self.configure(address)
            self.cfg["input_file"] = str(self.inputs[0])
            for placeholder in (None, [], ["", ""], ""):
                with self.subTest(placeholder=placeholder):
                    if placeholder is None:
                        self.cfg.pop("input_files", None)
                    else:
                        self.cfg["input_files"] = placeholder
                    app.write_json(self.config_path, self.cfg)
                    code, _ = self.run_app("--check-only")
                    self.assertEqual(code, 0)
                    self.assertEqual(app.read_json(self.out / "input_validation.json")["total"], 2)
            self.assertFalse(calls)

    def test_previous_two_file_checkpoint_contract_is_still_accepted(self):
        with mock_api() as (address, calls):
            self.configure(address)
            cfg = app.load_config(self.config_path)
            datasets = [app.read_json(path) for path in self.inputs]
            # Freeze the former two-file protocol rather than constructing it
            # from a new manifest, so accidental schema/order changes are caught.
            previous_protocol = {
                "version": app.VERSION,
                "dataset_hash": app.fingerprint(datasets[0] + datasets[1]),
                "source_files": [{"path": str(path), "dataset_hash": app.fingerprint(rows)}
                                 for path, rows in zip(self.inputs, datasets)],
                "rubric_hash": app.fingerprint(app.SYSTEM_PROMPT),
                "judges": app.public_judges(cfg),
                "order": "file1_then_file2_then_configured_judge_order",
                "unit": "one_complete_task_sample",
                "decision_rule": "min<=2:FAIL; otherwise min==3 or evidence_errors:REVIEW; otherwise PASS",
            }
            previous_manifest = {**previous_protocol, "run_fingerprint": app.fingerprint(previous_protocol),
                                 "created_at_utc": "2026-11-02T00:00:00+00:00"}
            self.out.mkdir()
            db = app.open_store(self.out / "checkpoint.sqlite3", previous_manifest)
            db.execute("INSERT INTO results VALUES (?,?,?,?,?,?)",
                       (self.original[0]["sample_id"], "Qwen", "ok", app.encode(review()), None, app.now()))
            db.commit()
            db.close()
            code, _ = self.run_app("--limit", 1)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(not call["path"].startswith("/qwen/") for call in calls))
            self.assertEqual(set(self.ratings()[0]["ratings"]), {"Qwen", "GLM", "Grok"})

    def test_three_cli_files_flatten_in_order_without_losing_samples(self):
        third = self.root / "third.json"
        app.write_json(self.inputs[0], self.original[:1])
        app.write_json(self.inputs[1], self.original[1:2])
        app.write_json(third, self.original[2:])
        self.cfg["input_files"] = []  # CLI must override an unfilled config.
        with mock_api() as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--input", *self.inputs, third, "--limit", 0)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 12)
            actual = [json.loads(call["payload"]["messages"][1]["content"])["sample_id"]
                      for call in calls]
            expected = [row["sample_id"] for row in self.original for _ in range(3)]
            self.assertEqual(actual, expected)
            self.assertEqual(len(self.ratings()), 4)
        task_rows = app.read_json(self.out / "task_scores.json")
        self.assertEqual(len(task_rows), 16)
        self.assertEqual({r["scope"] for r in task_rows}, {"COMBINED", *map(str, self.inputs), str(third)})

    def test_repeated_cli_input_flags_extend_all_paths(self):
        third = self.root / "third.json"
        app.write_json(self.inputs[0], self.original[:1])
        app.write_json(self.inputs[1], self.original[1:2])
        app.write_json(third, self.original[2:])
        with mock_api() as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--input", self.inputs[0], "--input", self.inputs[1], third,
                                   "--check-only")
            self.assertEqual(code, 0)
            self.assertEqual(app.read_json(self.out / "input_validation.json")["total"], 4)
            self.assertFalse(calls)

    def test_missing_configured_inputs_fail_before_any_request(self):
        with mock_api() as (address, calls):
            self.configure(address)
            self.cfg["input_files"] = ["", ""]
            app.write_json(self.config_path, self.cfg)
            with self.assertRaisesRegex(ValueError, "输入|路径|input|JSON"):
                self.run_app("--check-only")
            self.assertFalse(calls)

    def test_nonexistent_input_has_friendly_cli_error_without_traceback(self):
        import subprocess
        import sys

        with mock_api() as (address, calls):
            self.configure(address)
            missing = self.root / "does-not-exist.json"
            result = subprocess.run([sys.executable, str(HERE / "consistency_check.py"),
                                     "--config", str(self.config_path), "--input", str(missing),
                                     "--check-only"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stdout + result.stderr)
            self.assertIn("does-not-exist.json", result.stdout + result.stderr)
            self.assertFalse(calls)

    def test_malformed_choice_or_message_is_recorded_as_missing(self):
        def responder(call, number):
            if call["path"].startswith("/qwen/"):
                return 200, {"choices": ["invalid choice"]}
            if call["path"].startswith("/glm/"):
                return 200, {"choices": [{"message": "invalid message"}]}
            return ok_response(call)
        with mock_api(responder) as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--limit", 1)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 3)
        self.assertEqual(set(self.ratings()[0]["ratings"]), {"Grok"})


if __name__ == "__main__":
    unittest.main()
