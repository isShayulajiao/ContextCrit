"""Exact response loss without retaining sequence-by-vocabulary activations.

This changes when the language-model head is evaluated, not which tokens are
trained. The causal backbone must still process the complete input sequence.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_scope_loss(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    critical_positions: torch.Tensor | None = None,
    positive_ids: torch.Tensor | None = None,
    negative_ids: torch.Tensor | None = None,
    ct_weight: float = 0.2,
    margin: float = 1.0,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the same per-example NLL and ScopeCT objective as ``scope_loss``.

    ``hidden_states`` and ``labels`` cover the whole positive trajectory, with
    shapes [B,L,H] and [B,L]. Target j is scored from hidden state j-1. Every
    nonignored target contributes, including EOS, and missing pairs contribute
    zero CT loss to the batch mean. Pair validation matches ``scope_loss``.

    Each head/CE block is checkpointed, so neither forward nor backward retains
    an [L,V] logits/probability tensor. Peak head workspace is O(chunk_size*V),
    apart from the existing head parameters and their gradients. Checkpointing
    also recomputes autocast and RNG state for heads with stochastic layers.
    ``lm_head`` must expose a [vocabulary,hidden] ``weight`` tensor, as the
    supported Qwen causal LM heads do; its normal forward is always used.
    """
    if hidden_states.ndim != 3 or labels.ndim != 2 or hidden_states.shape[:2] != labels.shape:
        raise ValueError("Expected hidden states [B,L,H] and matching labels [B,L]")
    batch, length, hidden = hidden_states.shape
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[1] != hidden:
        raise ValueError("lm_head must expose a matching [vocabulary,hidden] weight")
    vocab = weight.shape[0]
    if batch < 1 or length < 2 or vocab < 2:
        raise ValueError("Need a nonempty batch, at least two tokens and two vocabulary entries")
    if labels.dtype != torch.long or labels.device != hidden_states.device:
        raise ValueError("labels must be torch.long on the hidden states device")
    if not hidden_states.is_floating_point():
        raise ValueError("hidden states must be floating point")
    if not math.isfinite(ct_weight) or ct_weight < 0:
        raise ValueError("ct_weight must be finite and nonnegative")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if bool(((labels != -100) & ((labels < 0) | (labels >= vocab))).any()):
        raise ValueError("labels contain out-of-vocabulary IDs")
    counts = (labels[:, 1:] != -100).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every example needs at least one supervised next-token target")

    supplied = [x is not None for x in (critical_positions, positive_ids, negative_ids)]
    if any(supplied) and not all(supplied):
        raise ValueError("Provide all three critical-token metadata tensors, or none")
    if not any(supplied):
        critical_positions = torch.full((batch,), -1, dtype=torch.long, device=hidden_states.device)
        positive_ids = torch.full_like(critical_positions, -1)
        negative_ids = torch.full_like(critical_positions, -1)
    for item in (critical_positions, positive_ids, negative_ids):
        if item.shape != (batch,) or item.dtype != torch.long or item.device != hidden_states.device:
            raise ValueError("Critical-token metadata must be torch.long [B] on the hidden states device")
    active = critical_positions != -1
    inactive = ~active
    if bool(((positive_ids[inactive] != -1) | (negative_ids[inactive] != -1)).any()):
        raise ValueError("Absent pairs must use -1 for position and both token IDs")
    positions = critical_positions[active]
    pos_ids, neg_ids = positive_ids[active], negative_ids[active]
    if bool(((positions < 1) | (positions >= length)).any()):
        raise ValueError("Critical target positions must be in [1,L)")
    if bool(((pos_ids < 0) | (pos_ids >= vocab) | (neg_ids < 0) | (neg_ids >= vocab)).any()):
        raise ValueError("Critical token ID is outside the vocabulary")
    if bool((pos_ids == neg_ids).any()):
        raise ValueError("Critical tokens must differ")
    rows = torch.arange(batch, device=hidden_states.device)[active]
    if bool((labels[rows, positions] != pos_ids).any()):
        raise ValueError("Critical positive ID must match the supervised label at its target position")

    def cross_entropy(block, target):
        return F.cross_entropy(lm_head(block).float(), target, reduction="sum")

    example_losses = []
    for row in range(batch):
        target_positions = (labels[row, 1:] != -100).nonzero(as_tuple=True)[0] + 1
        total = hidden_states[row, 0, 0].float() * 0.0
        for start in range(0, target_positions.numel(), chunk_size):
            indices = target_positions[start:start + chunk_size]
            total = total + checkpoint(
                cross_entropy, hidden_states[row, indices - 1], labels[row, indices],
                use_reentrant=False,
            )
        example_losses.append(total / counts[row])
    nll = torch.stack(example_losses).mean()

    if bool(active.any()):
        def critical_gaps(block, positives, negatives):
            # Scoring through the actual head also supports biased or adapted
            # heads. Only the two needed logits survive this checkpoint.
            logits = lm_head(block)
            positive = logits.gather(1, positives[:, None]).squeeze(1).float()
            negative = logits.gather(1, negatives[:, None]).squeeze(1).float()
            return positive - negative

        gaps = torch.cat([
            checkpoint(critical_gaps,
                       hidden_states[rows[start:start + chunk_size], positions[start:start + chunk_size] - 1],
                       pos_ids[start:start + chunk_size], neg_ids[start:start + chunk_size],
                       use_reentrant=False)
            for start in range(0, len(rows), chunk_size)
        ])
        ct = F.softplus(margin - gaps).sum() / batch
        gap_mean = gaps.detach().mean()
    else:
        ct = nll * 0.0
        gap_mean = nll.detach() * 0.0
    return nll + ct_weight * ct, {
        "nll_loss": nll.detach(),
        "ct_loss": ct.detach(),
        "ct_pairs": active.sum().detach(),
        "ct_gap_mean": gap_mean,
        "supervised_tokens": counts.sum().detach(),
    }
