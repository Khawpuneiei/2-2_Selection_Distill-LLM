"""Hugging Face / PEFT setup around the separately tested training loop."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .records import write_json
from .train_loop import run_training_loop


def train_lora_arm(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    output_dir: str | Path,
    max_length: int = 1024,
    tokens_per_update: int = 2048,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    seed: int = 17,
    max_updates: int | None = None,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
) -> dict[str, Any]:
    if not rows:
        raise ValueError("the selected SFT arm is empty")
    if lora_rank <= 0 or lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Training needs compatible PyTorch, Transformers, and PEFT installs; "
            "use the project environment setup instructions."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    base_revision = getattr(model.config, "_commit_hash", None)
    model.config.use_cache = False
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, config)
    # Keep the frozen base in half precision but train the adapter in fp32.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_parameters == 0:
        raise RuntimeError("PEFT created no trainable adapter parameters")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    loop_receipt = run_training_loop(
        model,
        tokenizer,
        rows,
        device=device,
        max_length=max_length,
        tokens_per_update=tokens_per_update,
        learning_rate=learning_rate,
        seed=seed,
        max_updates=max_updates,
    )
    model.save_pretrained(output_path)
    elapsed = time.perf_counter() - started
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    gpu_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    try:
        peft_version = importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError:
        peft_version = None
    summary = {
        "arm": rows[0].get("selection_arm", "unknown"),
        "model": model_name,
        "model_revision": base_revision,
        "adapter_dir": str(output_path),
        "device": str(device),
        "base_dtype": str(dtype),
        "gpu_name": gpu_name,
        "gpu_peak_allocated_bytes": gpu_memory_bytes,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": peft_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "lora": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "target_modules": list(target_modules),
            "adapter_dtype": "float32",
            "trainable_parameters": trainable_parameters,
        },
        "learning_rate": learning_rate,
        "tokens_per_optimizer_update": tokens_per_update,
        "max_length": max_length,
        "elapsed_seconds": elapsed,
        "reused_rows_in_full_selection": sum(bool(row.get("reused")) for row in rows),
        "truncated_rows_in_full_selection": sum(bool(row.get("truncated")) for row in rows),
        **loop_receipt,
    }
    write_json(output_path / "training_summary.json", summary)
    return summary
