"""Numerically stable per-token generation uncertainty helpers."""

from __future__ import annotations

from typing import Iterable


def mean_entropy_from_scores(scores: Iterable) -> float:
    """Compute mean Shannon entropy (nats) over autoregressive score rows."""
    score_rows = list(scores)
    if not score_rows:
        return 0.0
    import torch

    values = []
    for score in score_rows:
        logits = score.float()
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
        values.append(entropy.mean())
    return float(torch.stack(values).mean().item())
