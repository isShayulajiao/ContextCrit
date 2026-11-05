"""Check CLI validation and exact torchrun forwarding without loading torch."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from ScopeSFT import launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = Path(self.directory.name) / "data.json"
        self.data.write_text("[]", encoding="utf-8")
        self.output = Path(self.directory.name) / "new-output"

    def invoke(self, flags, mode="sft", returncode=0):
        argv = [str(Path(__file__).with_name("train.py")), "--data", str(self.data),
                "--output", str(self.output), *flags]
        with patch.object(sys, "argv", argv), patch.object(launcher.subprocess, "call", return_value=returncode) as call:
            with self.assertRaises(SystemExit) as raised:
                launcher.launch(mode)
        return raised.exception.code, call

    def test_new_defaults_preserve_qwen3_and_select_every_source_row(self):
        args = launcher.parser("sft").parse_args([])
        self.assertEqual(args.model, "Qwen/Qwen3-1.7B")
        self.assertIsNone(args.max_samples)
        self.assertEqual(args.loss_chunk_size, 128)
        self.assertEqual(args.attention_chunk_size, 128)
        self.assertEqual(args.linear_chunk_size, 512)
        self.assertEqual(args.activation_offload, "auto")
        self.assertEqual(args.low_memory, "auto")
        self.assertEqual(args.dtype, "auto")
        self.assertEqual(args.gradient_checkpointing, "on")

    def test_max_samples_accepts_positive_integer_and_explicit_none(self):
        for text, expected in (("1", 1), ("125", 125), ("None", None), ("NONE", None), ("none", None)):
            with self.subTest(text=text):
                self.assertEqual(launcher.parser("ct").parse_args(["--max-samples", text]).max_samples, expected)
        for text in ("0", "-1", "1.5", "NaN", "null"):
            with self.subTest(text=text), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                launcher.parser("ct").parse_args(["--max-samples", text])

    def test_both_methods_forward_sample_and_memory_options_exactly(self):
        flags = ["--model", "/models/Qwen3.5-2B", "--max-length", "65536", "--max-samples", "73",
                 "--gpus", "0,2", "--epochs", "1.5", "--max-steps", "3", "--batch-size", "2",
                 "--grad-accum", "7", "--learning-rate", "0.001", "--seed", "9", "--lora-r", "8",
                 "--ct-weight", "0.3", "--margin", "0.5", "--attention", "eager",
                 "--loss-chunk-size", "17", "--attention-chunk-size", "19", "--low-memory", "off",
                 "--linear-chunk-size", "256", "--activation-offload", "cpu",
                 "--dtype", "fp32", "--gradient-checkpointing", "off", "--allow-sft-fallback"]
        for mode in ("sft", "ct"):
            with self.subTest(mode=mode):
                code, call = self.invoke(flags, mode, returncode=7)
                self.assertEqual(code, 7)
                command = call.call_args.args[0]
                self.assertEqual(command[:4], [sys.executable, "-m", "torch.distributed.run", "--standalone"])
                self.assertIn("--nproc_per_node=2", command)
                worker = launcher.parser(mode).parse_args(command[command.index("--worker"):])
                expected = launcher.parser(mode).parse_args(flags)
                for key in vars(expected):
                    if key not in ("data", "output", "worker"):
                        self.assertEqual(getattr(worker, key), getattr(expected, key), key)
                self.assertEqual(worker.data, str(self.data))
                self.assertEqual(worker.output, str(self.output))
                self.assertTrue(worker.worker)

    def test_explicit_none_reaches_worker_as_default_none(self):
        _, call = self.invoke(["--max-samples", "None", "--gpus", "0123"])
        command = call.call_args.args[0]
        worker = launcher.parser("sft").parse_args(command[command.index("--worker"):])
        self.assertIsNone(worker.max_samples)
        self.assertIn("--nproc_per_node=4", command)

    def test_nonpositive_chunk_sizes_fail_before_launch(self):
        for name in ("--loss-chunk-size", "--attention-chunk-size"):
            for value in ("0", "-1"):
                with self.subTest(name=name, value=value), redirect_stderr(StringIO()):
                    code, call = self.invoke([name, value])
                    self.assertEqual(code, 2)
                    call.assert_not_called()

    def test_invalid_linear_segments_fail_before_launch(self):
        for value in ("0", "32", "63", "65", "-64"):
            with self.subTest(value=value), redirect_stderr(StringIO()):
                code, call = self.invoke(["--linear-chunk-size", value])
                self.assertEqual(code, 2)
                call.assert_not_called()

    def test_old_environment_is_rejected_before_torchrun(self):
        from ScopeSFT.preflight import TrainingEnvironmentError
        with patch("ScopeSFT.preflight.check_training_environment",
                   side_effect=TrainingEnvironmentError("transformers=4.51.3; requires 5.3.0")), \
                redirect_stderr(StringIO()) as error:
            code, call = self.invoke(["--gpus", "0123"])
        self.assertEqual(code, 1)
        call.assert_not_called()
        self.assertIn("transformers=4.51.3", error.getvalue())

    def test_worker_receives_options_without_relaunch(self):
        runtime = ModuleType("ScopeSFT.runtime")
        runtime.run = Mock()
        argv = ["train.py", "--worker", "--data", str(self.data), "--output", str(self.output),
                "--max-samples", "2", "--low-memory", "on", "--dtype", "fp16"]
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"ScopeSFT.runtime": runtime}), \
                patch.object(launcher.subprocess, "call") as call:
            launcher.launch("ct")
        call.assert_not_called()
        args, mode = runtime.run.call_args.args
        self.assertEqual(mode, "ct")
        self.assertEqual(args.max_samples, 2)
        self.assertEqual(args.low_memory, "on")
        self.assertEqual(args.dtype, "fp16")


if __name__ == "__main__":
    unittest.main()
