"""Offline tests: synthetic records and a fake client, with no paid API calls."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import review_t1234 as review


def sample(sid="s1"):
    text = "这是证据。"
    return {
        "sample_id": sid, "doc_id": "d1", "task": "T1", "split": "test",
        "question": "原问题", "gold": {"required_evidence_ids": ["e1"]},
        "events": [], "roles": {}, "required_exchanges": [],
        "document": {
            "doc_id": "d1", "text": text,
            "evidence": [{"evidence_id": "e1", "start": 0,
                          "end": len(text), "text": text}],
        },
        "human_verified": True, "material_origin": "fixture",
    }


def response(decision="pass", replacement=None, *, finish="stop", content=None):
    if content is None:
        content = json.dumps({"decision": decision, "reason": "模拟审核理由",
                              "replacement": replacement or {}}, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content),
    )])


class FakeAPIError(Exception):
    def __init__(self, status):
        super().__init__("模拟 API 错误")
        self.status_code = status


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "merged.json"
        self.output = self.root / "optimized.json"
        self.checkpoint = self.root / "optimized.checkpoint.json"

    def write_input(self, rows):
        self.input.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def run_review(self, replies=(), extra=(), *, key="fake-key-for-tests",
                   available_models=("deepseek-chat",)):
        create = Mock(side_effect=list(replies))
        list_models = Mock(return_value=SimpleNamespace(
            data=[SimpleNamespace(id=name) for name in available_models],
        ))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            models=SimpleNamespace(list=list_models),
        )
        factory = Mock(return_value=client)
        argv = ["--input", str(self.input), "--output", str(self.output),
                "--model", "fixture-model", *extra]
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": key}), patch.dict(
            sys.modules, {"openai": SimpleNamespace(OpenAI=factory)}
        ), contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            code = review.main(argv)
        return code, create, factory

    def checkpoint_data(self):
        return review.read_json(self.checkpoint)

    def test_pass_revise_abstain_keep_order_and_immutable_fields(self):
        rows = [sample("s1"), sample("s2"), sample("s3")]
        self.write_input(rows)
        code, calls, _ = self.run_review([
            response(), response("revise", {"question": "更清楚的问题"}), response("abstain"),
        ])
        self.assertEqual(code, 0)
        output = review.read_json(self.output)
        self.assertEqual([s["sample_id"] for s in output], ["s1", "s2"])
        self.assertEqual(output[1]["question"], "更清楚的问题")
        for original, accepted in zip(rows, output):
            self.assertEqual(accepted["split"], original["split"])
            self.assertFalse(accepted["human_verified"])
            self.assertEqual(original["document"], accepted["document"])
            self.assertEqual(original["doc_id"], accepted["doc_id"])
        self.assertEqual(review.read_json(self.input), rows)
        for call in calls.call_args_list:
            self.assertEqual(call.kwargs["model"], "fixture-model")
            payload = json.loads(call.kwargs["messages"][1]["content"])
            self.assertNotIn("split", payload)
        progress = self.checkpoint_data()["progress"]
        self.assertTrue(progress["review_complete"])
        self.assertFalse(progress["all_accepted"])

    def test_resume_rebuilds_output_without_key_or_new_calls(self):
        self.write_input([sample()])
        self.run_review([response()])
        self.output.unlink()  # Simulate interruption after checkpoint replacement.
        code, calls, factory = self.run_review(key="")
        self.assertEqual(code, 0)
        calls.assert_not_called()
        factory.assert_not_called()
        self.assertEqual(len(review.read_json(self.output)), 1)

    def test_retry_only_errors_and_structure_failures(self):
        self.write_input([sample("s1"), sample("s2"), sample("s3")])
        code, _, _ = self.run_review([
            response(), response(content="not json"),
            response("revise", {"gold": {"required_evidence_ids": ["missing"]}}),
        ])
        self.assertEqual(code, 2)
        self.assertEqual(self.checkpoint_data()["results"]["s3"]["status"], "structure_failed")
        code, calls, _ = self.run_review([response(), response()], ["--retry-errors"])
        self.assertEqual(code, 0)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(len(review.read_json(self.output)), 3)

    def test_abstain_is_retried_only_when_explicitly_requested(self):
        self.write_input([sample()])
        self.run_review([response("abstain")])
        _, calls, _ = self.run_review(extra=["--retry-errors"])
        calls.assert_not_called()
        code, calls, _ = self.run_review([response()], ["--retry-abstain"])
        self.assertEqual(code, 0)
        self.assertEqual(calls.call_count, 1)

    def test_rejects_forbidden_fields_unknown_decisions_and_truncation(self):
        cases = [response("revise", {"document": {}}), response("unknown"),
                 response(finish="length"),
                 response(content='{"decision":"revise","reason":"x",'
                                  '"replacement":{"gold":{"score":NaN}}}')]
        for reply in cases:
            with self.subTest(reply=reply):
                self.write_input([sample()])
                if self.checkpoint.exists():
                    self.checkpoint.unlink()
                    self.output.unlink()
                code, _, _ = self.run_review([reply])
                self.assertEqual(code, 2)
                self.assertEqual(self.checkpoint_data()["results"]["s1"]["status"], "error")
                self.assertEqual(review.read_json(self.output), [])

    def test_global_api_errors_stop_batch_without_marking_remaining_samples(self):
        self.write_input([sample("s1"), sample("s2")])
        code, calls, _ = self.run_review([FakeAPIError(402)])
        self.assertEqual(code, 2)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(self.checkpoint_data()["progress"]["remaining"], 1)
        self.assertFalse(self.checkpoint_data()["progress"]["attempted_all"])

    def test_changed_input_or_model_refuses_to_mix_results(self):
        self.write_input([sample()])
        self.run_review([response()])
        before = self.checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError, "已改变"):
            self.run_review(extra=["--model", "another-model"])
        changed = sample()
        changed["question"] = "已修改"
        self.write_input([changed])
        with self.assertRaisesRegex(ValueError, "已改变"):
            self.run_review()
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_invalid_source_does_not_call_model(self):
        row = sample()
        row["document"]["evidence"][0]["end"] = 1000
        self.write_input([row])
        code, calls, _ = self.run_review()
        self.assertEqual(code, 2)
        calls.assert_not_called()
        self.assertEqual(self.checkpoint_data()["results"]["s1"]["status"], "input_invalid")

    def test_interrupt_preserves_completed_results(self):
        self.write_input([sample("s1"), sample("s2")])
        code, _, _ = self.run_review([response(), KeyboardInterrupt()])
        self.assertEqual(code, 130)
        self.assertEqual(set(self.checkpoint_data()["results"]), {"s1"})
        code, calls, _ = self.run_review([response()])
        self.assertEqual(code, 0)
        self.assertEqual(calls.call_count, 1)

    def test_path_collision_does_not_overwrite_input(self):
        self.input = self.checkpoint
        self.write_input([sample()])
        before = self.input.read_bytes()
        with self.assertRaisesRegex(ValueError, "互不相同"):
            self.run_review()
        self.assertEqual(self.input.read_bytes(), before)

    def test_preexisting_unrelated_output_is_not_overwritten(self):
        self.write_input([sample()])
        self.output.write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "输出已存在"):
            self.run_review()
        self.assertEqual(self.output.read_text(), "preserve me")

    def test_atomic_write_failure_preserves_existing_file(self):
        self.output.write_text("original", encoding="utf-8")
        with patch.object(review.os, "replace", side_effect=OSError("simulated failure")):
            with self.assertRaises(OSError):
                review.write_json(self.output, {"new": "value"})
        self.assertEqual(self.output.read_text(), "original")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_errors_do_not_save_the_api_key(self):
        key = "fake-sensitive-key"
        details = review.error_details(ValueError(f"Bearer {key}; key={key}"), key)
        self.assertNotIn(key, json.dumps(details))

    def test_mutable_fields_are_deeply_copied(self):
        original = sample()
        report = review.validate_report({
            "decision": "revise", "reason": "修复", "replacement": {"roles": {"a": {}}},
        })
        snapshot = copy.deepcopy(original)
        accepted, errors, _ = review.accepted_sample(original, report)
        accepted["roles"]["a"]["changed"] = True
        self.assertEqual(original, snapshot)
        self.assertEqual(report["replacement"]["roles"], {"a": {}})
        self.assertEqual(errors, [])

    def test_limit_smaller_than_dataset_and_resume_next_batches(self):
        self.write_input([sample("s1"), sample("s2"), sample("s3")])
        for index, expected_code in [(1, 2), (2, 2), (3, 0)]:
            code, calls, _ = self.run_review([response()], ["--limit", "1"])
            self.assertEqual(code, expected_code)
            self.assertEqual(calls.call_count, 1)
            payload = json.loads(calls.call_args.kwargs["messages"][1]["content"])
            self.assertEqual(payload["sample_id"], f"s{index}")
            self.assertEqual(len(review.read_json(self.output)), index)
            progress = self.checkpoint_data()["progress"]
            self.assertEqual(progress["remaining"], 3 - index)
            self.assertEqual(progress["review_complete"], index == 3)
        code, calls, factory = self.run_review(extra=["--limit", "1"], key="")
        self.assertEqual(code, 0)
        calls.assert_not_called()
        factory.assert_not_called()

    def test_limit_larger_than_dataset_processes_all(self):
        self.write_input([sample("s1"), sample("s2")])
        code, calls, _ = self.run_review([response(), response()], ["--limit", "10"])
        self.assertEqual(code, 0)
        self.assertEqual(calls.call_count, 2)
        self.assertTrue(self.checkpoint_data()["progress"]["review_complete"])

    def test_zero_limit_means_all_pending(self):
        self.write_input([sample("s1"), sample("s2")])
        code, calls, _ = self.run_review([response(), response()], ["--limit", "0"])
        self.assertEqual(code, 0)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(len(review.read_json(self.output)), 2)

    def test_negative_limit_is_rejected_before_any_request(self):
        self.write_input([sample()])
        with self.assertRaises((ValueError, SystemExit)):
            self.run_review(extra=["--limit", "-1"])
        self.assertFalse(self.checkpoint.exists())
        self.assertFalse(self.output.exists())

    def test_limit_counts_attempts_including_failures(self):
        self.write_input([sample("s1"), sample("s2"), sample("s3")])
        code, calls, _ = self.run_review(
            [response(content="invalid JSON"), response()], ["--limit", "1"],
        )
        self.assertEqual(code, 2)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(set(self.checkpoint_data()["results"]), {"s1"})
        self.assertEqual(self.checkpoint_data()["progress"]["remaining"], 2)

    def test_limit_applies_to_retryable_results_without_repeating_passes(self):
        self.write_input([sample("s1"), sample("s2"), sample("s3"), sample("s4")])
        self.run_review([
            response(), response(content="invalid JSON"),
            response("revise", {"gold": {"required_evidence_ids": ["missing"]}}),
            response("abstain"),
        ])
        code, calls, _ = self.run_review([response()], ["--retry-errors", "--limit", "1"])
        self.assertEqual(code, 2)
        self.assertEqual(calls.call_count, 1)
        payload = json.loads(calls.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["sample_id"], "s2")
        self.assertEqual(self.checkpoint_data()["results"]["s3"]["status"], "structure_failed")
        code, calls, _ = self.run_review([response()], ["--retry-errors", "--limit", "1"])
        self.assertEqual(code, 0)
        payload = json.loads(calls.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["sample_id"], "s3")
        self.assertEqual(self.checkpoint_data()["results"]["s4"]["status"], "abstain")

    def test_split_summary_counts_missing_and_unknown_values(self):
        rows = [sample(f"s{i}") for i in range(5)]
        for row, split, task in zip(rows, ["train", "train", "test", None, "validation"],
                                    ["T1", "T2", "T1", "T3", "T4"]):
            row["task"] = task
            if split is None:
                row.pop("split")
            else:
                row["split"] = split
        summary = review.dataset_summary(rows)
        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["splits"]["train"]["total"], 2)
        self.assertEqual(summary["splits"]["train"]["T1"], 1)
        self.assertEqual(summary["splits"]["train"]["T2"], 1)
        self.assertEqual(summary["splits"]["test"]["total"], 1)
        self.assertEqual(summary["splits"]["unknown"]["total"], 2)
        self.assertEqual(summary["splits"]["unknown"]["T3"], 1)
        self.assertEqual(summary["splits"]["unknown"]["T4"], 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            review.print_dataset_summary("审核前", rows)
        for label in ("审核前", "train", "test", "未标注", "T1", "T2", "T3", "T4"):
            self.assertIn(label, output.getvalue())

    def test_split_is_preserved_in_output_and_missing_split_stays_missing(self):
        train, test, unlabeled = sample("s1"), sample("s2"), sample("s3")
        train["split"] = "train"
        unlabeled.pop("split")
        self.write_input([train, test, unlabeled])
        code, _, _ = self.run_review([response(), response(), response()])
        self.assertEqual(code, 0)
        output = review.read_json(self.output)
        self.assertEqual(output[0]["split"], "train")
        self.assertEqual(output[1]["split"], "test")
        self.assertNotIn("split", output[2])

    def test_progress_displays_global_and_current_batch_percentages(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            review.print_progress(5, 10, 1, 2)
        text = output.getvalue()
        self.assertIn("全库", text)
        self.assertIn("本次", text)
        self.assertIn("5/10", text)
        self.assertIn("1/2", text)
        self.assertIn("50.0%", text)

    def test_auto_model_prefers_available_deepseek_chat(self):
        self.write_input([sample()])
        code, calls, factory = self.run_review(
            [response()], ["--model", "auto"],
            available_models=("deepseek-reasoner", "deepseek-flash", "deepseek-chat"),
        )
        self.assertEqual(code, 0)
        factory.return_value.models.list.assert_called_once_with()
        self.assertEqual(calls.call_args.kwargs["model"], "deepseek-chat")
        self.assertEqual(self.checkpoint_data()["identity"]["model"], "deepseek-chat")

    def test_auto_model_falls_back_to_actual_available_flash_id(self):
        self.write_input([sample()])
        code, calls, _ = self.run_review(
            [response()], ["--model", "auto"],
            available_models=("deepseek-reasoner", "deepseek-v9-flash"),
        )
        self.assertEqual(code, 0)
        self.assertEqual(calls.call_args.kwargs["model"], "deepseek-v9-flash")

    def test_auto_model_rejects_reasoning_only_list_without_chat_request(self):
        self.write_input([sample()])
        with self.assertRaises(ValueError):
            self.run_review(extra=["--model", "auto"],
                            available_models=("deepseek-reasoner", "deepseek-pro"))
        self.assertFalse(self.checkpoint.exists())

    def test_auto_resume_reuses_checkpoint_model_without_discovery(self):
        self.write_input([sample("s1"), sample("s2")])
        self.run_review([response()], ["--model", "auto", "--limit", "1"])
        code, calls, factory = self.run_review(
            [response()], ["--model", "auto", "--limit", "1"], available_models=(),
        )
        self.assertEqual(code, 0)
        factory.return_value.models.list.assert_not_called()
        self.assertEqual(calls.call_args.kwargs["model"], "deepseek-chat")

    def test_explicit_model_does_not_query_model_list(self):
        self.write_input([sample()])
        code, _, factory = self.run_review([response()])
        self.assertEqual(code, 0)
        factory.return_value.models.list.assert_not_called()

    def test_command_line_api_key_takes_precedence_over_environment(self):
        self.write_input([sample()])
        code, _, factory = self.run_review([response()], ["--api-key", "fake-cli-key"])
        self.assertEqual(code, 0)
        self.assertEqual(factory.call_args.kwargs["api_key"], "fake-cli-key")
        for path in (self.output, self.checkpoint):
            self.assertNotIn("fake-cli-key", path.read_text(encoding="utf-8"))

    def test_no_arguments_prompts_for_input_path_and_hidden_key(self):
        self.write_input([sample()])
        create = Mock(return_value=response())
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            models=SimpleNamespace(list=Mock(return_value=SimpleNamespace(
                data=[SimpleNamespace(id="deepseek-chat")],
            ))),
        )
        factory = Mock(return_value=client)
        stdout = io.StringIO()
        with contextlib.chdir(self.root), patch.dict(os.environ, {}, clear=True), patch.dict(
            sys.modules, {"openai": SimpleNamespace(OpenAI=factory)}
        ), patch("builtins.input", return_value=str(self.input)) as input_prompt, patch.object(
            review.getpass, "getpass", return_value="fake-prompt-key"
        ) as key_prompt, contextlib.redirect_stdout(stdout):
            code = review.main([])
        self.assertEqual(code, 0)
        input_prompt.assert_called_once()
        key_prompt.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["api_key"], "fake-prompt-key")
        self.assertNotIn("fake-prompt-key", stdout.getvalue())
        result = review.read_json(self.input.with_name(f"{self.input.stem}_optimized.json"))
        self.assertEqual(result[0]["sample_id"], "s1")


if __name__ == "__main__":
    unittest.main()
