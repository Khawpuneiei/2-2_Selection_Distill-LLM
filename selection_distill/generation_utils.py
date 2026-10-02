"""Attention-mask-safe prompt handling for causal model generation."""

from __future__ import annotations


def truncate_generation_prompt(input_ids, attention_mask, *, max_length: int):
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if getattr(input_ids, "ndim", None) != 2:
        raise ValueError("input_ids must be a two-dimensional batch")
    if attention_mask is None:
        import torch

        attention_mask = torch.ones_like(input_ids)
    if getattr(attention_mask, "shape", None) != getattr(input_ids, "shape", None):
        raise ValueError("attention_mask shape must match input_ids")
    if input_ids.shape[1] <= max_length:
        return input_ids, attention_mask
    return input_ids[:, -max_length:], attention_mask[:, -max_length:]


STOP_STRINGS = ("\nProblem:",)


def trim_generation(text: str) -> str:
    """Cut a continuation where the model starts a new few-shot style problem."""
    value = str(text)
    cut = len(value)
    for marker in STOP_STRINGS:
        index = value.find(marker)
        if index >= 0:
            cut = min(cut, index)
    return value[:cut].strip()


def left_pad(sequences: list[list[int]], pad_id: int, *, max_length: int):
    """Left-truncate each prompt to max_length, then left-pad into a batch."""
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    import torch

    kept = [list(sequence)[-max_length:] for sequence in sequences]
    if not kept or any(not sequence for sequence in kept):
        raise ValueError("each prompt needs at least one token")
    width = max(len(sequence) for sequence in kept)
    ids = torch.full((len(kept), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(kept), width), dtype=torch.long)
    for row, sequence in enumerate(kept):
        ids[row, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long)
        mask[row, width - len(sequence):] = 1
    return ids, mask


def generated_lengths(continuations, stop_ids) -> list[int]:
    """Length of each row through its first stop/pad token (inclusive)."""
    stops = set(int(token) for token in stop_ids)
    lengths = []
    for row in continuations.tolist():
        length = len(row)
        for index, token in enumerate(row):
            if token in stops:
                length = index + 1
                break
        lengths.append(length)
    return lengths


def masked_row_means(step_values, lengths: list[int]) -> list[float]:
    """Mean of per-step values (steps × batch) over each row's valid prefix."""
    if not step_values:
        return [0.0 for _ in lengths]
    import torch

    stacked = torch.stack([value.float().cpu() for value in step_values], dim=1)
    means = []
    for row, length in enumerate(lengths):
        valid = stacked[row, : max(0, min(length, stacked.shape[1]))]
        means.append(float(valid.mean().item()) if valid.numel() else 0.0)
    return means
