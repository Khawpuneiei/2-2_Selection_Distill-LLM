"""Shared Hugging Face model loading and per-question evaluation."""

from __future__ import annotations

from typing import Any, Iterable

from .metrics import mean_entropy_from_scores
from .generation_utils import truncate_generation_prompt
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


def _generate(
    question: str,
    model,
    tokenizer,
    device,
    *,
    max_new_tokens: int,
    max_length: int,
    sample: bool,
    temperature: float,
    top_p: float,
) -> tuple[str, float]:
    torch = _torch()
    prompt = tokenizer(format_prompt(question), return_tensors="pt", add_special_tokens=True)
    raw_attention_mask = prompt.get("attention_mask")
    input_ids, attention_mask = truncate_generation_prompt(
        prompt["input_ids"].to(device),
        raw_attention_mask.to(device) if raw_attention_mask is not None else None,
        max_length=max_length,
    )
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    with torch.no_grad():
        generated = model.generate(**kwargs)
    continuation = generated.sequences[0, input_ids.shape[1] :]
    answer = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    entropy = mean_entropy_from_scores(generated.scores or ())
    return answer, entropy


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
) -> list[dict[str, Any]]:
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    results: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("id") or not row.get("concept"):
            raise ValueError("evaluation rows require id and concept")
        if not row.get("question") or not row.get("solution") or not row.get("answer"):
            raise ValueError(f"evaluation row {row.get('id')} requires question, solution, and answer")
        loss, supervised_tokens = _question_loss(row, model, tokenizer, device, max_length)
        for attempt in range(1, attempts + 1):
            predicted, entropy = _generate(
                row["question"], model, tokenizer, device,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                sample=attempts > 1,
                temperature=temperature,
                top_p=top_p,
            )
            results.append(
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
                }
            )
    return results
