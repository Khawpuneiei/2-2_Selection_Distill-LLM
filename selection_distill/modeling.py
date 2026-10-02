"""Shared Hugging Face model loading and per-question evaluation."""

from __future__ import annotations

from typing import Any, Iterable

from .generation_utils import (
    STOP_STRINGS,
    generated_lengths,
    left_pad,
    masked_row_means,
    trim_generation,
)
from .scoring import answers_match
from .training_utils import encode_supervised_example, format_prompt


def load_causal_lm(model_name: str, adapter: str | None = None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Model commands need compatible PyTorch and Transformers installs; "
            "use the project environment setup instructions."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install PEFT to load a LoRA adapter") from exc
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _question_loss(row: dict[str, Any], model, tokenizer, device, max_length: int) -> tuple[float, int]:
    encoded = encode_supervised_example(
        {**row, "allocated_tokens": max_length}, tokenizer, max_length=max_length
    )
    torch = _torch()
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
    return float(output.loss.detach().float().item()), encoded["supervised_tokens"]


def _torch():
    import torch

    return torch


class _StepEntropy:
    """Logits processor that records raw next-token entropy (nats) per step.

    Custom processors run before temperature / top-p warpers, so the entropy is
    that of the model's unwarped distribution.
    """

    def __init__(self):
        self.steps = []

    def __call__(self, input_ids, scores):
        torch = _torch()
        log_probs = torch.log_softmax(scores.float(), dim=-1)
        self.steps.append(-(log_probs.exp() * log_probs).sum(dim=-1))
        return scores


def _generate_batch(
    questions: list[str],
    model,
    tokenizer,
    device,
    *,
    max_new_tokens: int,
    max_length: int,
    sample: bool,
    temperature: float,
    top_p: float,
) -> list[tuple[str, float, bool]]:
    """Generate one continuation per question; return (text, entropy, hit_cap)."""
    torch = _torch()
    from transformers import LogitsProcessorList

    prompts = [tokenizer.encode(format_prompt(question), add_special_tokens=True) for question in questions]
    input_ids, attention_mask = left_pad(prompts, tokenizer.pad_token_id, max_length=max_length)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    recorder = _StepEntropy()
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "stop_strings": list(STOP_STRINGS),
        "tokenizer": tokenizer,
        "logits_processor": LogitsProcessorList([recorder]),
    }
    if sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    else:
        kwargs["temperature"] = None
        kwargs["top_p"] = None
        kwargs["top_k"] = None
    with torch.no_grad():
        sequences = model.generate(**kwargs)
    continuations = sequences[:, input_ids.shape[1]:]
    stop_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    lengths = generated_lengths(continuations, stop_ids)
    entropies = masked_row_means(recorder.steps, lengths)
    results = []
    for row, length in enumerate(lengths):
        tokens = continuations[row, :length].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        hit_cap = length >= max_new_tokens and not any(token in stop_ids for token in tokens)
        trimmed = trim_generation(text)
        results.append((trimmed, entropies[row], hit_cap and trimmed == text.strip()))
    return results


def evaluate_rows(
    rows: Iterable[dict[str, Any]],
    model,
    tokenizer,
    device,
    *,
    attempts: int = 1,
    max_new_tokens: int = 256,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 16,
    progress: bool = False,
) -> list[dict[str, Any]]:
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = list(rows)
    for row in rows:
        if not row.get("id") or not row.get("concept"):
            raise ValueError("evaluation rows require id and concept")
        if not row.get("question") or not row.get("solution") or not row.get("answer"):
            raise ValueError(f"evaluation row {row.get('id')} requires question, solution, and answer")
    tokenizer.padding_side = "left"
    results: list[dict[str, Any]] = []
    # Sort by prompt length so batches pad little; output order is restored below.
    order = sorted(range(len(rows)), key=lambda index: len(rows[index]["question"]))
    by_index: dict[int, list[dict[str, Any]]] = {}
    for start in range(0, len(order), batch_size):
        batch_indices = order[start:start + batch_size]
        batch = [rows[index] for index in batch_indices]
        losses = [_question_loss(row, model, tokenizer, device, max_length) for row in batch]
        per_row: list[list[dict[str, Any]]] = [[] for _ in batch]
        for attempt in range(1, attempts + 1):
            generations = _generate_batch(
                [row["question"] for row in batch], model, tokenizer, device,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                sample=attempts > 1,
                temperature=temperature,
                top_p=top_p,
            )
            for position, (row, (predicted, entropy, hit_cap)) in enumerate(zip(batch, generations)):
                loss, supervised_tokens = losses[position]
                per_row[position].append(
                    {
                        "id": str(row["id"]),
                        "concept": str(row["concept"]),
                        "dataset": str(row.get("dataset", "")),
                        "attempt": attempt,
                        "correct": answers_match(predicted, str(row["answer"])),
                        "prediction": predicted,
                        "reference": str(row["answer"]),
                        "entropy": entropy,
                        "loss": loss,
                        "supervised_tokens": supervised_tokens,
                        "hit_token_cap": hit_cap,
                    }
                )
        for index, records in zip(batch_indices, per_row):
            by_index[index] = records
        if progress:
            done = min(start + batch_size, len(order))
            print(f"  generated {done}/{len(order)} questions", flush=True)
    for index in range(len(rows)):
        results.extend(by_index[index])
    return results
