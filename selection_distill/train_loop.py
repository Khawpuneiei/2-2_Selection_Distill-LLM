"""Testable token-accurate optimization loop for one LoRA SFT arm."""

from __future__ import annotations

import math
import random
from typing import Any

from .training_utils import encode_supervised_example


def _normalize_gradients(parameters, denominator: int) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


def run_training_loop(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    device,
    max_length: int,
    tokens_per_update: int,
    learning_rate: float,
    seed: int,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """Train the supplied model over one arm and return a token-bound receipt."""
    if not rows:
        raise ValueError("the selected SFT arm is empty")
    if max_length <= 0 or tokens_per_update <= 0 or learning_rate <= 0:
        raise ValueError("max_length, tokens_per_update, and learning_rate must be positive")
    if max_updates is not None and max_updates <= 0:
        raise ValueError("max_updates must be positive when supplied")

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if getattr(device, "type", str(device)) == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    model.train()

    expected_tokens = sum(min(max_length, int(row["allocated_tokens"])) for row in rows)
    tokens_seen = 0
    supervised_seen = 0
    accumulated_supervised = 0
    optimizer_steps = 0
    weighted_loss_sum = 0.0
    consumed_rows = 0
    early_stopped = False

    def take_optimizer_step() -> None:
        nonlocal optimizer_steps, accumulated_supervised
        if accumulated_supervised <= 0:
            raise ValueError("optimizer step has no supervised answer tokens")
        _normalize_gradients(parameters, accumulated_supervised)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        accumulated_supervised = 0

    for row in rows:
        try:
            allocated = int(row["allocated_tokens"])
            sample_id = str(row["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("selected rows need id and allocated_tokens") from exc
        if allocated <= 0:
            raise ValueError(f"allocated token count must be positive for {sample_id}")
        encoded = encode_supervised_example(row, tokenizer, max_length=max_length)
        expected_row_tokens = min(max_length, allocated)
        if encoded["active_tokens"] != expected_row_tokens:
            raise ValueError(
                f"tokenizer length changed for {sample_id}: expected {expected_row_tokens}, "
                f"got {encoded['active_tokens']}"
            )
        supervised = encoded["supervised_tokens"]
        try:
            weight = float(row.get("loss_weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid confidence loss weight for {sample_id}") from exc
        if supervised <= 0 or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"sample {sample_id} has no supervised tokens or invalid loss weight")

        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
        result = model(input_ids=input_ids, labels=labels, use_cache=False)
        (result.loss * supervised * weight).backward()
        weighted_loss_sum += float(result.loss.detach().float().item()) * supervised * weight
        tokens_seen += encoded["active_tokens"]
        supervised_seen += supervised
        accumulated_supervised += supervised
        consumed_rows += 1

        if accumulated_supervised >= tokens_per_update:
            take_optimizer_step()
            if max_updates is not None and optimizer_steps >= max_updates:
                early_stopped = consumed_rows < len(rows)
                break

    if accumulated_supervised > 0:
        take_optimizer_step()

    return {
        "selection_rows": len(rows),
        "selection_rows_consumed": consumed_rows,
        "unique_examples_consumed": len({str(row.get("id")) for row in rows[:consumed_rows]}),
        "training_tokens_budgeted": expected_tokens,
        "training_tokens_seen": tokens_seen,
        "supervised_tokens_seen": supervised_seen,
        "optimizer_steps": optimizer_steps,
        "weighted_mean_loss": weighted_loss_sum / supervised_seen if supervised_seen else None,
        "completed_token_budget": tokens_seen == expected_tokens,
        "early_stopped": early_stopped,
    }
