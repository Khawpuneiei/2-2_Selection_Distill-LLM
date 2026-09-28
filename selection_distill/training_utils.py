"""Model-agnostic formatting and token-budget preparation for causal SFT."""

from __future__ import annotations

from typing import Any


def format_prompt(question: str) -> str:
    return f"Problem: {str(question).strip()}\nSolution:"


def encode_supervised_example(
    row: dict[str, Any], tokenizer: Any, *, max_length: int
) -> dict[str, Any]:
    """Tokenize a causal SFT row and mask prompt tokens from the loss.

    If an arm's final row is capped part-way, retain the token suffix so at
    least the answer remains supervised while total active tokens stay exact.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    try:
        question = str(row["question"]).strip()
        solution = str(row["solution"]).strip()
        requested_cap = row["allocated_tokens"]
    except (KeyError, TypeError) as exc:
        raise ValueError("training row needs question, solution, and allocated_tokens") from exc
    if not question or not solution:
        raise ValueError("training question and solution must be non-empty")
    if (
        isinstance(requested_cap, bool)
        or not isinstance(requested_cap, int)
        or requested_cap <= 0
    ):
        raise ValueError("allocated_tokens must be a positive integer")

    prompt_ids = tokenizer.encode(format_prompt(question), add_special_tokens=True)
    eos = getattr(tokenizer, "eos_token", None)
    target = solution if not eos else f"{solution} {eos}"
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    full_ids = list(prompt_ids) + list(target_ids)
    full_labels = [-100] * len(prompt_ids) + list(target_ids)
    cap = min(max_length, requested_cap)
    if len(full_ids) > cap:
        full_ids = full_ids[-cap:]
        full_labels = full_labels[-cap:]
    if not full_ids or all(label == -100 for label in full_labels):
        raise ValueError("allocated token cap leaves no supervised answer token")
    return {
        "input_ids": full_ids,
        "labels": full_labels,
        "active_tokens": len(full_ids),
        "supervised_tokens": sum(label != -100 for label in full_labels),
    }
