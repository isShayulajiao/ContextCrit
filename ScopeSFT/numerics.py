"""Stable precision for Qwen3.5's native PyTorch GatedDeltaNet fallbacks.

Transformers 5.3.0 normalizes Q/K before promoting them to float32. FP16
normalization can produce nonfinite gradients for small or zero Q/K vectors.
Its later float32 matrix multiplies can also be downcast by AMP. Keep the
existing fallback formulas in float32 and return activations in their input
dtype. Optimized third-party kernels are left untouched.
"""
from __future__ import annotations

from functools import wraps
from inspect import signature

import torch


def _fp32_fallback(function):
    parameters = signature(function)

    @wraps(function)
    def forward(query, key, value, *args, **kwargs):
        original_dtype = query.dtype
        arguments = parameters.bind(query, key, value, *args, **kwargs)
        for name in ("query", "key", "value", "g", "beta", "initial_state"):
            tensor = arguments.arguments.get(name)
            if isinstance(tensor, torch.Tensor):
                arguments.arguments[name] = tensor.float()
        with torch.autocast(device_type=query.device.type, enabled=False):
            output, state = function(*arguments.args, **arguments.kwargs)
        return output.to(original_dtype), state

    return forward


def install_fp32_native_fallback(model):
    """Wrap only identity-matching native chunk/recurrent fallback functions.

    This is idempotent, changes no model weights, and reports the layers whose
    native fallbacks use float32. An installed FLA kernel keeps its own policy.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    native = {
        "chunk_gated_delta_rule": torch_chunk_gated_delta_rule,
        "recurrent_gated_delta_rule": torch_recurrent_gated_delta_rule,
    }
    counts = {"gated_delta_layers": 0, "chunk_fallback_layers": 0,
              "recurrent_fallback_layers": 0, "native_only": True, "precision": "float32"}
    for module in model.modules():
        if not isinstance(module, Qwen3_5GatedDeltaNet):
            continue
        counts["gated_delta_layers"] += 1
        originals = getattr(module, "_scope_fp32_native_fallbacks", {})
        for name, function in native.items():
            if name in originals:
                counts["chunk_fallback_layers" if name.startswith("chunk") else "recurrent_fallback_layers"] += 1
            elif getattr(module, name) is function:
                originals[name] = function
                setattr(module, name, _fp32_fallback(function))
                counts["chunk_fallback_layers" if name.startswith("chunk") else "recurrent_fallback_layers"] += 1
        if originals:
            module._scope_fp32_native_fallbacks = originals
    return counts


def remove_fp32_native_fallback(model):
    """Restore native callables before exporting an ordinary model."""
    for module in model.modules():
        originals = getattr(module, "_scope_fp32_native_fallbacks", None)
        if originals is not None:
            for name, function in originals.items():
                setattr(module, name, function)
            del module._scope_fp32_native_fallbacks
