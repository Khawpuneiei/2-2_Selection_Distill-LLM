from __future__ import annotations

import argparse
from pathlib import Path

from selection_distill.modeling import evaluate_rows, load_causal_lm
from selection_distill.records import read_jsonl, write_json, write_jsonl
from selection_distill.selection import summarize_student_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base model or adapter on benchmark splits")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", help="Optional LoRA adapter directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Development-only limit per benchmark")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.attempts <= 0 or args.max_length <= 0 or args.max_new_tokens <= 0:
        parser.error("attempts, max-length, and max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, device = load_causal_lm(args.model, args.adapter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {}
    for input_path in args.inputs:
        rows = read_jsonl(input_path)
        if args.limit is not None:
            rows = rows[: args.limit]
        metrics = evaluate_rows(
            rows, model, tokenizer, device,
            attempts=args.attempts,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
        )
        concept_metrics = summarize_student_metrics(metrics)
        name = input_path.stem
        write_jsonl(args.output_dir / f"{name}.metrics.jsonl", metrics)
        correct = sum(bool(row["correct"]) for row in metrics)
        token_counts = [row["supervised_tokens"] for row in metrics]
        split_summaries[name] = {
            "examples": len({row["id"] for row in metrics}),
            "attempts": len(metrics),
            "accuracy": correct / len(metrics) if metrics else None,
            "mean_entropy": sum(row["entropy"] for row in metrics) / len(metrics) if metrics else None,
            "mean_loss": sum(row["loss"] for row in metrics) / len(metrics) if metrics else None,
            "supervised_tokens_per_attempt_mean": sum(token_counts) / len(token_counts) if token_counts else None,
            "repeated_failures": sum(row["repeated_failures"] for row in concept_metrics.values()),
            "concepts": concept_metrics,
        }
        print(f"{name}: accuracy={split_summaries[name]['accuracy']} ({len(rows)} questions)")

    receipt = {
        "model": args.model,
        "adapter": args.adapter,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": str(device),
        "attempts_per_question": args.attempts,
        "splits": split_summaries,
    }
    write_json(args.output_dir / "evaluation_summary.json", receipt)
    print(f"Wrote evaluation receipt to {args.output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
