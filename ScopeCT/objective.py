"""Single-positive-forward Scope-CT loss; no reference or negative forward.

The caller proves that the pair has an identical token prefix and that its first
divergence belongs to a source-verified semantic field. ``critical_positions``
index the target tokens in ``labels``: token j is predicted by logits at j - 1.
Both losses use an example mean, including zero CT loss for examples without a
valid pair. This keeps gradient accumulation and DDP denominators consistent.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def scope_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    critical_positions: torch.Tensor | None = None,
    positive_ids: torch.Tensor | None = None,
    negative_ids: torch.Tensor | None = None,
    ct_weight: float = 0.2,
    margin: float = 1.0,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return NLL + weight * softplus(margin + z_negative - z_positive).

    ``labels`` are [batch, length] with -100 for prompt/padding. Metadata tensors
    have shape [batch]; all three use -1 for absent pairs. NLL is the mean over
    supervised next-token targets *per example*, followed by an example mean.
    CT is a batch mean, with absent pairs contributing zero. The full negative
    sequence must never be scored using positive-prefix logits after divergence.

    CE is computed in position chunks and float32 for FP16 stability, without a
    full extra float32 copy of the vocabulary logits. The model's original
    logits allocation is still necessary; chunking is not attention chunking.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [B,L,V] and matching labels [B,L]")
    batch, length, vocab = logits.shape
    if batch < 1 or length < 2 or vocab < 2:
        raise ValueError("Need a nonempty batch, at least two tokens and two vocabulary entries")
    if labels.dtype != torch.long or labels.device != logits.device:
        raise ValueError("labels must be torch.long on the logits device")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    if not math.isfinite(ct_weight) or ct_weight < 0:
        raise ValueError("ct_weight must be finite and nonnegative")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    invalid_labels = (labels != -100) & ((labels < 0) | (labels >= vocab))
    if bool(invalid_labels.any()):
        raise ValueError("labels contain out-of-vocabulary IDs")

    counts = (labels[:, 1:] != -100).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every example needs at least one supervised next-token target")

    supplied = [x is not None for x in (critical_positions, positive_ids, negative_ids)]
    if any(supplied) and not all(supplied):
        raise ValueError("Provide all three critical-token metadata tensors, or none")
    if not any(supplied):
        critical_positions = torch.full((batch,), -1, dtype=torch.long, device=logits.device)
        positive_ids = torch.full_like(critical_positions, -1)
        negative_ids = torch.full_like(critical_positions, -1)
    for item in (critical_positions, positive_ids, negative_ids):
        if item.shape != (batch,) or item.dtype != torch.long or item.device != logits.device:
            raise ValueError("Critical-token metadata must be torch.long [B] on the logits device")
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
    rows = torch.arange(batch, device=logits.device)[active]
    if bool((labels[rows, positions] != pos_ids).any()):
        raise ValueError("Critical positive ID must match the supervised label at its target position")

    example_losses = []
    for row in range(batch):
        # Ignore the final logits row: no target exists at position L.
        total = logits[row, 0, 0].float() * 0.0
        for start in range(0, length - 1, chunk_size):
            end = min(start + chunk_size, length - 1)
            target = labels[row, start + 1:end + 1]
            selected = target != -100
            if bool(selected.any()):
                total = total + F.cross_entropy(
                    logits[row, start:end][selected].float(), target[selected], reduction="sum"
                )
        example_losses.append(total / counts[row])
    nll = torch.stack(example_losses).mean()

    if bool(active.any()):
        critical_logits = logits[rows, positions - 1].float()
        indices = torch.arange(len(rows), device=logits.device)
        gaps = critical_logits[indices, pos_ids] - critical_logits[indices, neg_ids]
        ct = F.softplus(margin - gaps).sum() / batch
        gap_mean = gaps.detach().mean()
    else:
        ct = nll * 0.0
        gap_mean = nll.detach() * 0.0
    loss = nll + ct_weight * ct
    return loss, {
        "nll_loss": nll.detach(),
        "ct_loss": ct.detach(),
        "ct_pairs": active.sum().detach(),
        "ct_gap_mean": gap_mean,
        "supervised_tokens": counts.sum().detach(),
    }
