"""CPU/mocked CUDA checks for reports, without claiming real GPU validation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ScopeSFT.diagnostics import MemoryDiagnostics, cuda_snapshot, is_cuda_oom


class MemoryDiagnosticTests(unittest.TestCase):
    def test_mocked_allocator_snapshot_keeps_free_reserved_and_peaks_distinct(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.current_device", return_value=2), \
                patch("torch.cuda.get_device_name", return_value="mock GPU"), \
                patch("torch.cuda.memory_allocated", return_value=100), \
                patch("torch.cuda.memory_reserved", return_value=180), \
                patch("torch.cuda.max_memory_allocated", return_value=220), \
                patch("torch.cuda.max_memory_reserved", return_value=260), \
                patch("torch.cuda.mem_get_info", return_value=(300, 1000)):
            actual = cuda_snapshot()
        self.assertTrue(actual["cuda_available"])
        self.assertEqual(actual["allocated_bytes"], 100)
        self.assertEqual(actual["reserved_bytes"], 180)
        self.assertEqual(actual["peak_allocated_bytes"], 220)
        self.assertEqual(actual["peak_reserved_bytes"], 260)
        self.assertEqual(actual["device_free_bytes"], 300)
        self.assertEqual(actual["device_total_bytes"], 1000)
        self.assertEqual(actual["device_index"], 2)

    def test_oom_report_records_sizes_without_input_or_label_content(self):
        with tempfile.TemporaryDirectory() as folder:
            diagnostic = MemoryDiagnostics(folder, rank=3)
            diagnostic.record_batch({"input_ids": torch.tensor([[11, 12, 0], [13, 14, 15]]),
                                     "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                                     "labels": torch.tensor([[-100, 12, -100], [-100, 14, 15]]),
                                     "raw_document": "must never appear"})
            error = RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaises(RuntimeError) as caught:
                    with diagnostic.watch("training.backward_or_optimizer"):
                        raise error
            self.assertIs(caught.exception, error)
            files = list(Path(folder).glob("*.json"))
            self.assertEqual([p.name for p in files], ["oom_report.rank3.json"])
            report = json.loads(files[0].read_text())
            self.assertEqual(report["phase"], "training.backward_or_optimizer")
            self.assertEqual(report["batch"]["input_shape"], [2, 3])
            self.assertEqual(report["batch"]["nonpadding_tokens"], 5)
            self.assertEqual(report["batch"]["max_actual_tokens"], 3)
            self.assertEqual(report["batch"]["supervised_tokens"], 3)
            self.assertNotIn("must never appear", files[0].read_text())
            self.assertNotIn("input_ids", report["batch"])
            self.assertNotIn("labels", report["batch"])

    def test_unrelated_exception_is_unchanged_and_does_not_create_oom_report(self):
        with tempfile.TemporaryDirectory() as folder:
            diagnostic = MemoryDiagnostics(folder)
            error = ModuleNotFoundError("No module named 'transformers.masking_utils'")
            with self.assertRaises(ModuleNotFoundError) as caught:
                with diagnostic.watch("loading"):
                    raise error
            self.assertIs(caught.exception, error)
            self.assertFalse(list(Path(folder).iterdir()))
            self.assertFalse(is_cuda_oom(error))

    def test_cuda_query_failure_does_not_mask_original_oom(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.current_device", side_effect=RuntimeError("context lost")):
            result = cuda_snapshot()
        self.assertEqual(result["snapshot_error_type"], "RuntimeError")
        self.assertNotIn("allocated_bytes", result)


if __name__ == "__main__":
    unittest.main()
