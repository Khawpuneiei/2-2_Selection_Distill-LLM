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
