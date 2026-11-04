"""Exact causal attention with bounded score-memory on pre-Ampere GPUs.

This is query chunking, not local attention: each query still attends to every
unmasked earlier key.  Even when SDPA falls back to its math implementation, at
most ``query_chunk_size`` query rows are scored together.  Each block (including
mask creation) is checkpointed, so backward recomputes its temporary score map
instead of retaining the maps of every block, which would again be quadratic.

Compute remains quadratic in sequence length.  This backend deliberately accepts
only the full-sequence causal, optional 2-D padding masks used by this trainer.
KV-cache decoding, packed sequences and custom/sliding masks fail explicitly.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _positive_chunk_size(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("query_chunk_size must be a positive integer")


def chunked_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    query_chunk_size: int = 128,
    scaling: float | None = None,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Return [B,Hq,L,Dv] full causal attention from [B,H,L,D] inputs.

    ``attention_mask`` is [B,L], with True/1 for real keys.  Grouped query
    attention is supported without changing which key head serves each query.
    ``dropout`` is the effective probability (pass zero during evaluation).
    Checkpoint RNG preservation reuses each forward dropout mask in backward.
    """
    _positive_chunk_size(query_chunk_size)
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("query, key and value must have shape [B,H,L,D]")
    batch, heads, length, dimension = query.shape
    if batch < 1 or length < 1 or dimension < 1 or heads < 1:
        raise ValueError("Attention dimensions must be nonempty")
    if (key.shape[:1] != (batch,) or value.shape[:1] != (batch,)
            or key.shape[2] != length or value.shape[2] != length
            or key.shape[1] != value.shape[1] or key.shape[1] < 1
            or key.shape[3] != dimension or heads % key.shape[1]):
        raise ValueError("Expected full-sequence matching Q/K/V and divisible GQA heads; caches are unsupported")
    if any(x.device != query.device or x.dtype != query.dtype for x in (key, value)):
        raise ValueError("query, key and value must have the same device and dtype")
    if not all(x.is_floating_point() for x in (query, key, value)):
        raise ValueError("query, key and value must be floating point")
    if not math.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
        raise ValueError("dropout must be finite and in [0,1]")
    if scaling is not None and (not math.isfinite(scaling) or scaling <= 0.0):
        raise ValueError("scaling must be finite and positive")
    if attention_mask is not None:
        if attention_mask.shape != (batch, length) or attention_mask.device != query.device:
            raise ValueError("attention_mask must be [B,L] on the query device; dense masks are unsupported")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must contain only 0/1 padding indicators")
        attention_mask = attention_mask.bool()

    blocks = []
    use_checkpoint = torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value))
    for start in range(0, length, query_chunk_size):
        end = min(start + query_chunk_size, length)

        # Bind offsets by value: checkpoint calls this closure later in backward.
        def block(q, k, v, padding, query_start=start, query_end=end):
            query_positions = torch.arange(query_start, query_end, device=q.device)
            key_positions = torch.arange(query_end, device=q.device)
            allowed = (key_positions[None, :] <= query_positions[:, None])[None, None]
            if padding is not None:
                allowed = allowed & padding[:, None, None, :query_end]
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, dropout_p=dropout,
                is_causal=False, scale=scaling, enable_gqa=heads != key.shape[1],
            )

        arguments = (query[:, :, start:end], key[:, :, :end], value[:, :, :end], attention_mask)
        if use_checkpoint:
            output = checkpoint(block, *arguments, use_reentrant=False, preserve_rng_state=True)
        else:
            output = block(*arguments)
        blocks.append(output)
    return torch.cat(blocks, dim=2)


def _padding_only_mask(
    batch_size, cache_position, kv_length, kv_offset=0,
    attention_mask=None, mask_function=None, use_vmap=False, **kwargs,
):
    """HF mask hook: keep padding linear; create each causal block on demand."""
    from transformers.masking_utils import causal_mask_function

    if (kv_offset != 0 or kv_length != cache_position.numel()
            or not torch.equal(cache_position, torch.arange(kv_length, device=cache_position.device))):
        raise ValueError("scope_chunked requires an uncached full-sequence forward (use_cache=False)")
    if mask_function is not causal_mask_function or use_vmap:
        raise ValueError("scope_chunked supports ordinary full causal masks; packed/custom/sliding masks are unsupported")
    if attention_mask is not None and attention_mask.shape != (batch_size, kv_length):
        raise ValueError("scope_chunked requires a 2-D padding mask")
    return attention_mask


def register_chunked_attention(query_chunk_size: int = 128) -> str:
    """Register both HF interfaces before constructing a model; return its name.

    Use the result as ``attn_implementation`` with Transformers 5.3.0.  The mask
    registration is essential: an eager/SDPA mask hook can allocate [L,L] before
    the attention function is called.  Call once per training process.
    """
    _positive_chunk_size(query_chunk_size)
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    def attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                          scaling=None, is_causal=None, **kwargs):
        if not (getattr(module, "is_causal", True) if is_causal is None else is_causal):
            raise ValueError("scope_chunked supports causal text attention only")
        if kwargs.get("output_attentions", False):
            raise ValueError("scope_chunked cannot return quadratic attention weights")
        if kwargs.get("sliding_window") is not None:
            raise ValueError("scope_chunked does not change sliding-window semantics")
        result = chunked_causal_attention(
            query, key, value, attention_mask, query_chunk_size=query_chunk_size,
            scaling=scaling, dropout=dropout if module.training else 0.0,
        )
        return result.transpose(1, 2).contiguous(), None

    name = "scope_chunked"
    AttentionInterface.register(name, attention_forward)
    AttentionMaskInterface.register(name, _padding_only_mask)
    return name
