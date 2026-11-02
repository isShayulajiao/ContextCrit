"""Concurrency regressions against localhost; no external or paid API access."""
from __future__ import annotations

from collections import Counter
import json
import sqlite3
import threading
import time
import unittest
from unittest.mock import patch

import consistency_check as app
import test_consistency_check as helpers


def job_id(call):
    sample = json.loads(call["payload"]["messages"][1]["content"])
    return sample["sample_id"], call["path"].split("/")[1]


def distinct_review(call):
    sample = json.loads(call["payload"]["messages"][1]["content"])
    task_number = int(sample["task"][1:])
    judge_number = {"qwen": 0, "glm": 1, "grok": 2}[job_id(call)[1]]
    score = (task_number + judge_number - 1) % 5 + 1
    return helpers.review(score, rationale="RESULT_" + "_".join(job_id(call)))


class Activity:
    """Track true simultaneous HTTP handlers, independent of completion order."""
    def __init__(self, target=1, delay=0.03):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.target = target
        self.delay = delay
        self.overlap = threading.Event()
        self.started = threading.Event()

    def responder(self, call, number):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.started.set()
            if self.active >= self.target:
                self.overlap.set()
        try:
            # A bounded barrier proves overlap without comparing wall-clock runtimes.
            self.overlap.wait(timeout=2)
            task = int(json.loads(call["payload"]["messages"][1]["content"])["task"][1:])
            time.sleep(self.delay * (5 - task))
            return helpers.ok_response(call, distinct_review(call))
        finally:
            with self.lock:
                self.active -= 1


class ConcurrentExecutionTests(unittest.TestCase):
    # Reuse only fixture/helper methods, avoiding inherited test duplication.
    setUp = helpers.EndToEndOfflineTests.setUp
    configure = helpers.EndToEndOfflineTests.configure
    run_app = helpers.EndToEndOfflineTests.run_app
    ratings = helpers.EndToEndOfflineTests.ratings
    call_count = helpers.EndToEndOfflineTests.call_count

    def assert_settled_database(self):
        with sqlite3.connect(self.out / "checkpoint.sqlite3") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM calls WHERE status='running'").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results WHERE status='running'").fetchone()[0], 0)

    def test_parallel_overlap_limit_two_samples_and_blind_material(self):
        tracker = Activity(target=3)
        with helpers.mock_api(tracker.responder) as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--workers", 3, "--limit", 2)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 6)
            self.assertEqual(tracker.peak, 3)
            self.assertTrue(tracker.overlap.is_set())
            expected = {row["sample_id"] for row in self.original[:2]}
            self.assertEqual({job_id(call)[0] for call in calls}, expected)
            self.assertTrue(all(count == 1 for count in Counter(job_id(call) for call in calls).values()))
            materials = {}
            for call in calls:
                messages = call["payload"]["messages"]
                self.assertEqual(len(messages), 2)
                self.assertEqual(messages[0]["content"], app.SYSTEM_PROMPT)
                for marker in ("DEEPSEEK_REVIEW_CANARY", "GPT_GENERATOR_CANARY", "RESULT_"):
                    self.assertNotIn(marker, messages[1]["content"])
                materials.setdefault(job_id(call)[0], []).append(messages[1]["content"])
            self.assertTrue(all(len(values) == 3 and len(set(values)) == 1 for values in materials.values()))
        self.assertEqual(sum(len(row["ratings"]) == 3 for row in self.ratings()), 2)
        self.assert_settled_database()

    def test_out_of_order_results_preserve_sample_judge_identity(self):
        tracker = Activity(target=4, delay=0.01)
        with helpers.mock_api(tracker.responder) as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--workers", 4, "--limit", 0)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 12)
            self.assertEqual(tracker.peak, 4)
            expected = {job_id(call): distinct_review(call) for call in calls}
        judge_paths = {"Qwen": "qwen", "GLM": "glm", "Grok": "grok"}
        for record in self.ratings():
            self.assertEqual(len(record["ratings"]), 3)
            for judge, value in record["ratings"].items():
                self.assertEqual(value, expected[(record["sample_id"], judge_paths[judge])])
        self.assertEqual(self.call_count(), 12)
        self.assert_settled_database()

    def test_parallel_retry_errors_only_supplements_the_failed_judge(self):
        counts = Counter()
        lock = threading.Lock()
        failing = (self.original[0]["sample_id"], "glm")

        def responder(call, number):
            key = job_id(call)
            with lock:
                counts[key] += 1
                attempt = counts[key]
            time.sleep(0.015)
            if key == failing and attempt == 1:
                return 503, {"error": {"message": "temporary mocked outage"}}
            return helpers.ok_response(call, distinct_review(call))

        with helpers.mock_api(responder) as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--workers", 3, "--limit", 2)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 6)
            previous = self.ratings()[0]
            self.assertEqual(set(previous["ratings"]), {"Qwen", "Grok"})
            code, _ = self.run_app("--workers", 4, "--retry-errors", "--limit", 1)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 7)
            self.assertEqual(job_id(calls[-1]), failing)
            current = self.ratings()[0]
            self.assertEqual(len(current["ratings"]), 3)
            for judge in ("Qwen", "Grok"):
                self.assertEqual(current["ratings"][judge], previous["ratings"][judge])
        self.assert_settled_database()

    def test_fatal_authentication_error_stops_new_dispatch_and_saves_inflight(self):
        def responder(call, number):
            if call["path"].startswith("/qwen/"):
                return 401, {"error": {"message": "mock invalid key"}}
            time.sleep(0.15)
            return helpers.ok_response(call, distinct_review(call))

        with helpers.mock_api(responder) as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--workers", 3, "--limit", 0)
            self.assertEqual(code, 2)
            self.assertGreaterEqual(len(calls), 1)
            self.assertLessEqual(len(calls), 3)
            self.assertLess(len(calls), 12)
        self.assertEqual(self.call_count(), len(calls))
        self.assert_settled_database()
        recorded = sum(len(row["ratings"]) for row in self.ratings())
        self.assertEqual(recorded, sum(not call["path"].startswith("/qwen/") for call in calls))

    def test_429_retries_only_its_sample_and_judge(self):
        counts = Counter()
        lock = threading.Lock()
        throttled = (self.original[0]["sample_id"], "glm")

        def responder(call, number):
            key = job_id(call)
            with lock:
                counts[key] += 1
                attempt = counts[key]
            if key == throttled and attempt == 1:
                return 429, {"error": {"message": "mock rate limited"}}
            time.sleep(0.01)
            return helpers.ok_response(call, distinct_review(call))

        with helpers.mock_api(responder) as (address, calls):
            self.configure(address)
            self.cfg.update(max_attempts=2, retry_delay_seconds=0.01)
            app.write_json(self.config_path, self.cfg)
            code, _ = self.run_app("--workers", 4, "--limit", 0)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 13)
            self.assertEqual(counts[throttled], 2)
            self.assertTrue(all(count == 1 for key, count in counts.items() if key != throttled))
        self.assertEqual(self.call_count(), 13)
        self.assertTrue(all(len(row["ratings"]) == 3 for row in self.ratings()))
        self.assert_settled_database()

    def test_config_workers_and_cli_override_apply_actual_http_limits(self):
        for configured, override, expected in ((3, None, 3), (4, 2, 2)):
            with self.subTest(configured=configured, override=override):
                tracker = Activity(target=expected, delay=0.01)
                with helpers.mock_api(tracker.responder) as (address, calls):
                    self.cfg["output_dir"] = str(self.root / f"workers-{configured}-{override}")
                    self.cfg["workers"] = configured
                    self.configure(address)
                    args = ["--limit", 1]
                    if override is not None:
                        args += ["--workers", override]
                    code, _ = self.run_app(*args)
                    self.assertEqual(code, 2)
                    self.assertEqual(len(calls), 3)
                    self.assertEqual(tracker.peak, expected)

    def test_invalid_worker_settings_are_rejected_before_http(self):
        with helpers.mock_api() as (address, calls):
            self.configure(address)
            for bad in (0, -1, True, 1.5, "3", None):
                with self.subTest(config_workers=bad):
                    self.cfg["workers"] = bad
                    app.write_json(self.config_path, self.cfg)
                    with self.assertRaises(ValueError):
                        self.run_app("--check-only")
            self.cfg["workers"] = 1
            app.write_json(self.config_path, self.cfg)
            for bad in (0, -1):
                with self.subTest(cli_workers=bad), self.assertRaises(ValueError):
                    self.run_app("--workers", bad, "--check-only")
            self.assertFalse(calls)

    def test_worker_changes_do_not_invalidate_checkpoint_or_require_report_keys(self):
        tracker = Activity(target=3, delay=0.01)
        with helpers.mock_api(tracker.responder) as (address, calls):
            self.configure(address)
            self.run_app("--workers", 3, "--limit", 1)
            self.assertEqual(len(calls), 3)
            fingerprint_before = app.read_json(self.out / "run_manifest.json")["run_fingerprint"]
            self.cfg["workers"] = 6
            for judge in self.cfg["judges"]:
                judge["api_key"] = ""
            app.write_json(self.config_path, self.cfg)
            code, _ = self.run_app("--workers", 4, "--report-only")
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 3)
            self.assertEqual(app.read_json(self.out / "run_manifest.json")["run_fingerprint"], fingerprint_before)

    def test_per_model_cap_preserves_cross_model_parallelism(self):
        lock = threading.Lock()
        active = Counter()
        peak = Counter()
        overall_peak = 0
        ready = threading.Event()

        def responder(call, number):
            nonlocal overall_peak
            model = job_id(call)[1]
            with lock:
                active[model] += 1
                peak[model] = max(peak[model], active[model])
                overall_peak = max(overall_peak, sum(active.values()))
                if sum(active.values()) == 3:
                    ready.set()
            try:
                ready.wait(timeout=2)
                time.sleep(0.02)
                return helpers.ok_response(call, distinct_review(call))
            finally:
                with lock:
                    active[model] -= 1

        with helpers.mock_api(responder) as (address, calls):
            self.configure(address)
            code, _ = self.run_app("--workers", 4, "--workers-per-model", 1, "--limit", 0)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 12)
            self.assertEqual(overall_peak, 3)
            self.assertEqual(dict(peak), {"qwen": 1, "glm": 1, "grok": 1})
        self.assert_settled_database()

    def test_keyboard_interrupt_waits_for_and_saves_inflight_responses(self):
        tracker = Activity(target=3, delay=0.02)
        real_wait = app.wait
        interrupted = False

        def interrupt_once(*args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                self.assertTrue(tracker.overlap.wait(timeout=2))
                raise KeyboardInterrupt()
            return real_wait(*args, **kwargs)

        with helpers.mock_api(tracker.responder) as (address, calls):
            self.configure(address)
            with patch.object(app, "wait", side_effect=interrupt_once):
                code, _ = self.run_app("--workers", 3, "--limit", 0)
            self.assertEqual(code, 130)
            self.assertEqual(len(calls), 3)
            self.assertEqual(sum(len(row["ratings"]) for row in self.ratings()), 3)
        self.assert_settled_database()

    def test_second_submit_failure_stops_dispatch_drains_first_and_marks_uncertain(self):
        release_first = threading.Event()
        real_submit = app.ThreadPoolExecutor.submit
        submitted = 0

        def responder(call, number):
            self.assertTrue(release_first.wait(timeout=2))
            return helpers.ok_response(call, distinct_review(call))

        def fail_second_submit(pool, *args, **kwargs):
            nonlocal submitted
            submitted += 1
            if submitted == 2:
                release_first.set()
                raise RuntimeError("mock broken executor on second submission")
            return real_submit(pool, *args, **kwargs)

        with helpers.mock_api(responder) as (address, calls):
            self.configure(address)
            with patch.object(app.ThreadPoolExecutor, "submit", new=fail_second_submit):
                code, _ = self.run_app("--workers", 3, "--limit", 0)
            self.assertEqual(code, 2)
            self.assertEqual(submitted, 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(job_id(calls[0]), (self.original[0]["sample_id"], "qwen"))
        self.assertEqual(self.call_count(), 2)
        first = self.ratings()[0]
        self.assertEqual(first["judge_status"]["Qwen"]["status"], "ok")
        self.assertEqual(first["judge_status"]["GLM"]["status"], "uncertain")
        self.assertEqual(first["judge_status"]["Grok"]["status"], "pending")
        with sqlite3.connect(self.out / "checkpoint.sqlite3") as db:
            self.assertEqual(db.execute("SELECT status FROM calls ORDER BY id").fetchall(),
                             [("ok",), ("uncertain",)])
        self.assert_settled_database()

    def test_report_only_recovers_orphan_running_execution_batch_as_uncertain(self):
        with helpers.mock_api() as (address, calls):
            self.configure(address)
            self.run_app("--workers", 2, "--limit", 1)
            self.assertEqual(len(calls), 3)
            with sqlite3.connect(self.out / "checkpoint.sqlite3") as db:
                cursor = db.execute(
                    "INSERT INTO execution_batches(started_at,workers,workers_per_model,requested_limit,selected_samples,status) "
                    "VALUES (?,?,?,?,?,'running')",
                    (app.now(), 3, 0, 1, 1))
                orphan_id = cursor.lastrowid
                db.commit()
            for judge in self.cfg["judges"]:
                judge["api_key"] = ""
            app.write_json(self.config_path, self.cfg)
            code, _ = self.run_app("--report-only", "--workers", 4)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 3)
        with sqlite3.connect(self.out / "checkpoint.sqlite3") as db:
            self.assertEqual(db.execute("SELECT status FROM execution_batches WHERE id=?", (orphan_id,)).fetchone()[0],
                             "uncertain")
        summary = app.read_json(self.out / "summary.json")
        self.assertEqual(next(batch for batch in summary["execution_batches"] if batch["id"] == orphan_id)["status"],
                         "uncertain")
        self.assert_settled_database()

    def test_interrupt_after_begin_rating_marks_orphan_uncertain_before_http(self):
        real_begin = app.begin_rating
        interrupted = False

        def interrupt_after_commit(*args, **kwargs):
            nonlocal interrupted
            result = real_begin(*args, **kwargs)
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt()
            return result

        with helpers.mock_api() as (address, calls):
            self.configure(address)
            with patch.object(app, "begin_rating", side_effect=interrupt_after_commit):
                code, _ = self.run_app("--workers", 3, "--limit", 1)
            self.assertEqual(code, 130)
            self.assertFalse(calls)
            self.assertEqual(self.call_count(), 1)
            self.assertEqual(self.ratings()[0]["judge_status"]["Qwen"]["status"], "uncertain")
            with sqlite3.connect(self.out / "checkpoint.sqlite3") as db:
                self.assertEqual(db.execute("SELECT status FROM calls").fetchall(), [("uncertain",)])
            self.assert_settled_database()
            code, _ = self.run_app("--workers", 3, "--report-only")
            self.assertEqual(code, 2)
            self.assertFalse(calls)
            code, _ = self.run_app("--workers", 3, "--retry-errors", "--limit", 1)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(self.ratings()[0]["ratings"]), 3)
        self.assert_settled_database()


if __name__ == "__main__":
    unittest.main()
