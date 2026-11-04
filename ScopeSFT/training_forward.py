"""DDP-safe, training-only replacement for the large vocabulary forward.

The call still enters PeftModel/DistributedDataParallel normally. Only the
native causal LM's internal training forward is replaced; inference uses its
original method, and state_dict/model classes remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from transformers.utils import ModelOutput

from ScopeSFT.memory import chunked_scope_loss
from ScopeSFT.model_support import text_decoder


@dataclass
class ScopeTrainingOutput(ModelOutput):
    loss: torch.Tensor | None = None
    scope_metrics: dict | None = None


def install_training_forward(model, *, ct_weight=0.0, margin=1.0, chunk_size=128):
    """Install before PEFT/DDP wrapping; never bypass their outer forward.

    The flag ``scope_compute_loss`` is private to our Trainer. Without that
    flag, delegate to the original forward including ordinary KV-cache use.
    """
    if hasattr(model, "_scope_original_forward"):
        raise ValueError("Scope training forward is already installed")
    decoder = text_decoder(model)
    original = model.forward

    def forward(self, *args, scope_compute_loss=False, **kwargs):
        if not scope_compute_loss:
            return original(*args, **kwargs)
        if args:
            raise ValueError("Scope training arguments must be supplied by keyword")
        labels = kwargs.pop("labels", None)
        critical_positions = kwargs.pop("critical_positions", None)
        positive_ids = kwargs.pop("positive_ids", None)
        negative_ids = kwargs.pop("negative_ids", None)
        if labels is None:
            raise ValueError("Scope training forward requires full original labels")
        if kwargs.pop("use_cache", False):
            raise ValueError("Training with KV cache is unsupported")
        if "logits_to_keep" in kwargs:
            raise ValueError("Scope loss needs full hidden states, not sliced logits")
        kwargs.pop("return_dict", None)
        outputs = decoder(use_cache=False, return_dict=True, **kwargs)
        loss, metrics = chunked_scope_loss(
            outputs.last_hidden_state, self.lm_head, labels,
            critical_positions, positive_ids, negative_ids,
            ct_weight=ct_weight, margin=margin, chunk_size=chunk_size,
        )
        return ScopeTrainingOutput(loss=loss, scope_metrics=metrics)

    model._scope_original_forward = original
    model.forward = MethodType(forward, model)
    return model


def remove_training_forward(model):
    """Restore a native forward before export; no training data enters weights."""
    if hasattr(model, "_scope_original_forward"):
        model.forward = model._scope_original_forward
        del model._scope_original_forward
