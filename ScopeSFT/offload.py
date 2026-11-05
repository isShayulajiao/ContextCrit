"""Optional CPU storage for saved activations, including checkpoint inputs.

Only autograd's saved copies move: the live forward tensor stays on its original
device, the computation/gradient graph is unchanged, and no history is detached.
Parameters and their views stay resident because the model already needs their
GPU storage. The context must enclose the normal PEFT/DDP ``model(...)`` call.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


def resolve_activation_offload(requested, *, cpu, max_tokens):
    """Choose saved-activation storage from actual data length, not its cap.

    The automatic policy is deterministic across DDP ranks. It does not change
    samples, losses, batches or recurrent-state boundaries. A long CUDA run uses
    host memory by default; the selected policy is printed and saved by runtime.
    """
    if requested not in ("auto", "none", "cpu"):
        raise ValueError("activation offload must be auto, none or cpu")
    if not isinstance(cpu, bool) or not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("cpu must be bool and max_tokens a nonnegative integer")
    return requested == "cpu" or (requested == "auto" and not cpu and max_tokens >= 16384)


def _storage_key(tensor):
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        return None
    return tensor.device, tensor.untyped_storage().data_ptr()


@dataclass
class _PackedActivation:
    device: torch.device
    value: torch.Tensor
    offloaded: bool


class _SavedActivationOffload(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, model=None, *, pin_memory=True, min_bytes=0):
        self.pin_memory = pin_memory
        self.min_bytes = min_bytes
        self.stats = {"saved_tensors": 0, "offloaded_tensors": 0,
                      "offloaded_bytes": 0, "retained_parameter_tensors": 0}
        # Include buffers and storage aliases: weight.T and tied embedding/head
        # tensors must not cause extra CPU copies of an already resident weight.
        resident = [] if model is None else [*model.parameters(), *model.buffers()]
        self.resident_storages = {_storage_key(tensor) for tensor in resident}
        self.resident_storages.discard(None)
        super().__init__(self._pack, self._unpack)

    def __enter__(self):
        super().__enter__()
        return self

    def _pack(self, tensor):
        self.stats["saved_tensors"] += 1
        resident = _storage_key(tensor) in self.resident_storages
        if resident:
            self.stats["retained_parameter_tensors"] += 1
        size = tensor.numel() * tensor.element_size()
        if (resident or tensor.device.type != "cuda" or tensor.layout != torch.strided
                or size < self.min_bytes or size == 0):
            # detach() here is only autograd's packed saved value, like the
            # built-in saved_tensors_hooks example. The live forward and its
            # differentiable graph are not detached, and backward uses the
            # original operation's grad_fn. Avoid retaining a graph cycle.
            return _PackedActivation(tensor.device, tensor.detach(), False)
        packed = torch.empty(tensor.size(), dtype=tensor.dtype, device="cpu",
                             pin_memory=self.pin_memory)
        # Complete D2H before exposing host storage. H2D can then be asynchronous
        # on the requesting autograd stream when the storage is pinned.
        packed.copy_(tensor, non_blocking=False)
        self.stats["offloaded_tensors"] += 1
        self.stats["offloaded_bytes"] += size
        return _PackedActivation(tensor.device, packed, True)

    def _unpack(self, packed):
        if not packed.offloaded:
            return packed.value
        return packed.value.to(packed.device, non_blocking=self.pin_memory)


def saved_activation_context(enabled, model=None, *, pin_memory=True, min_bytes=0):
    """Return a context storing saved CUDA activations on CPU when enabled.

    Example::

        with saved_activation_context(offload == "cpu", model):
            outputs = model(**inputs)  # keep the outer DDP/PEFT call intact

    CPU tensors are never copied. Normal model parameters and buffers (including
    transposed/tied views) are excluded when ``model`` is supplied. Nested
    nonreentrant checkpointing keeps its own inner hooks; their input saves use
    this outer context. Intermediate recomputation and transient workspaces
    during backward still need GPU memory. Host RAM/PCIe transfer is the cost;
    this does not shard weights or guarantee that a long sample fits.

    The enabled context exposes ``stats`` for diagnostics. Its byte counter sums
    saves rather than measuring peak host/GPU usage. Disabled mode returns an
    ordinary no-op context and does not replace any existing autograd hooks.
    """
    if not isinstance(enabled, bool) or not isinstance(pin_memory, bool):
        raise ValueError("enabled and pin_memory must be bool")
    if not isinstance(min_bytes, int) or isinstance(min_bytes, bool) or min_bytes < 0:
        raise ValueError("min_bytes must be a nonnegative integer")
    if not enabled:
        return nullcontext()
    return _SavedActivationOffload(model, pin_memory=pin_memory, min_bytes=min_bytes)
