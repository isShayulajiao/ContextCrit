"""Small per-process memory reports; never record document or answer contents.

CUDA calls are lazy, so offline data validation can import this module without
loading torch. Reports are local run artifacts, not model inputs or outputs.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import warnings


def cuda_snapshot():
    """Best-effort allocator/device measurements, without clearing the cache."""
    import torch

    result = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": bool(torch.cuda.is_available())}
    if not result["cuda_available"]:
        return result
    try:
        device = torch.cuda.current_device()
        result.update(
            device_index=device, device_name=torch.cuda.get_device_name(device),
            allocated_bytes=int(torch.cuda.memory_allocated(device)),
            reserved_bytes=int(torch.cuda.memory_reserved(device)),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
        free, total = torch.cuda.mem_get_info(device)
        result.update(device_free_bytes=int(free), device_total_bytes=int(total))
    except (RuntimeError, AssertionError) as error:
        # A poisoned CUDA context must not replace the original training error.
        result["snapshot_error_type"] = type(error).__name__
    return result


def is_cuda_oom(error):
    message = str(error).lower()
    return ("cuda out of memory" in message or "cuda error: out of memory" in message
            or (type(error).__name__ == "OutOfMemoryError" and "cuda" in message))


class MemoryDiagnostics:
    """Track phase and batch sizes; emit one rank-specific report on CUDA OOM.

    Recommended integration: watch the whole training run, set_phase before
    loading, forward, backward/optimizer and export, and record_batch at each
    compute_loss entry. ``watch`` always re-raises the original exception.
    """
    def __init__(self, output_dir, *, rank=None):
        self.output_dir = Path(output_dir)
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
        self.phase = "initialization"
        self.batch = None

    def set_phase(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("Memory diagnostic phase must be a nonempty string")
        self.phase = name

    def record_batch(self, inputs):
        ids = inputs.get("input_ids")
        if ids is None:
            self.batch = None
            return
        result = {"input_shape": list(ids.shape), "dtype": str(ids.dtype),
                  "device": str(ids.device)}
        mask, labels = inputs.get("attention_mask"), inputs.get("labels")
        if mask is not None:
            result["nonpadding_tokens"] = int(mask.sum().item())
            if mask.ndim == 2:
                result["max_actual_tokens"] = int(mask.sum(dim=1).max().item())
        if labels is not None:
            result["supervised_tokens"] = int((labels != -100).sum().item())
        self.batch = result

    def report(self, error=None):
        result = {"time_utc": datetime.now(timezone.utc).isoformat(), "rank": self.rank,
                  "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
                  "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                  "python_executable": sys.executable, "phase": self.phase,
                  "batch": self.batch, "memory": cuda_snapshot()}
        if error is not None:
            result.update(error_type=type(error).__name__, error_message=str(error)[:8000])
        return result

    def write_report(self, error=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = "oom_report" if error is not None else "memory_report"
        path = self.output_dir / f"{name}.rank{self.rank}.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.report(error), ensure_ascii=False, indent=2,
                                        allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        return path

    @contextmanager
    def watch(self, phase=None):
        if phase is not None:
            self.set_phase(phase)
        try:
            yield self
        except BaseException as error:
            if is_cuda_oom(error):
                try:
                    path = self.write_report(error)
                    print(f"[显存诊断] rank={self.rank} phase={self.phase}；报告：{path}",
                          file=sys.stderr, flush=True)
                except Exception as reporting_error:
                    warnings.warn("无法保存显存诊断，保留原始异常：" + type(reporting_error).__name__)
            raise
