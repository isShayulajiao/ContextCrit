"""Bounded workspace for the native Qwen3.5 GatedDeltaNet recurrence.

Sequence segments share the full differentiable recurrent state. This is exact
recomputation of the native recurrence, not a context window or truncated BPTT.
Only a segment is promoted to float32, including its Q/K normalization, and each
native call is checkpointed so its internal triangular workspaces are not kept
for the whole sequence. The native 64-token chunk partition stays unchanged.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def chunked_native_delta_rule(
    query, key, value, g, beta, chunk_size=64, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=False,
    *, sequence_chunk_size=512,
):
    """Run the original native formula on segments with a live state boundary.

    Q/K/V have shape [batch,sequence,heads,dimension], while g/beta have
    [batch,sequence,heads]. The returned activations keep query.dtype and the
    final state, when requested, is float32 as in the native implementation.

    ``sequence_chunk_size`` must be a multiple of the native ``chunk_size``;
    otherwise additional padding would change where its chunk computation is
    grouped. Final padding has zero keys, beta and g and does not change state.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    for name, size in (("chunk_size", chunk_size), ("sequence_chunk_size", sequence_chunk_size)):
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"{name} must be a positive integer")
    if sequence_chunk_size % chunk_size:
        raise ValueError("sequence_chunk_size must be a multiple of native chunk_size")
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("Q/K/V must have shape [B,L,H,D]")
    if query.shape[:3] != key.shape[:3] or query.shape[:3] != value.shape[:3]:
        raise ValueError("Q/K/V batch, sequence and head dimensions must agree")
    if query.shape[-1] != key.shape[-1] or any(size < 1 for size in query.shape) or value.shape[-1] < 1:
        raise ValueError("Q/K dimensions must agree and dimensions must be nonempty")
    if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError("g and beta must have shape [B,L,H]")
    tensors = (query, key, value, g, beta)
    if any(t.device != query.device or not t.is_floating_point() for t in tensors):
        raise ValueError("All inputs must be floating point on one device")
    expected_state = (query.shape[0], query.shape[2], query.shape[-1], value.shape[-1])
    if initial_state is not None:
        if (initial_state.shape != expected_state or initial_state.device != query.device
                or not initial_state.is_floating_point()):
            raise ValueError("initial_state must be floating point [B,H,Dk,Dv] on the input device")

    original_dtype = query.dtype

    def segment(q, k, v, decay, update, state):
        # Keep casts inside checkpoint, so no full-sequence FP32 Q/K/V tensors
        # survive the forward. AMP must not downcast native FP32 matrix math.
        with torch.autocast(device_type=q.device.type, enabled=False):
            output, next_state = torch_chunk_gated_delta_rule(
                q.float(), k.float(), v.float(), decay.float(), update.float(),
                chunk_size=chunk_size,
                initial_state=None if state is None else state.float(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            return output.to(original_dtype), next_state

    outputs = []
    state = initial_state
    # Non-reentrant checkpoint supports None and state tensors, and preserves
    # the recurrent-state gradient across segment boundaries without detach.
    use_checkpoint = torch.is_grad_enabled() and any(
        t.requires_grad for t in tensors + (() if initial_state is None else (initial_state,))
    )
    for start in range(0, query.shape[1], sequence_chunk_size):
        end = min(start + sequence_chunk_size, query.shape[1])
        arguments = tuple(t[:, start:end] for t in tensors) + (state,)
        if use_checkpoint:
            output, state = checkpoint(segment, *arguments, use_reentrant=False)
        else:
            output, state = segment(*arguments)
        outputs.append(output)
    return torch.cat(outputs, dim=1), state if output_final_state else None
