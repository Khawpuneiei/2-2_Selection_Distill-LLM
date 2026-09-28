from __future__ import annotations

import argparse
from pathlib import Path

from selection_distill.modeling import evaluate_rows, load_causal_lm
from selection_distill.records import read_jsonl, write_json, write_jsonl
from selection_distill.selection import summarize_student_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a student's concept-level weaknesses")
    parser.add_argument("--input", type=Path, default=Path("data/processed/profile.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", help="Optional initial-student LoRA adapter")
    parser.add_argument("--output", type=Path, default=Path("outputs/student_profile_metrics.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/student_profile_summary.json"))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, help="Development-only limit on profile questions")
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
    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    model, tokenizer, device = load_causal_lm(args.model, args.adapter)
    metrics = evaluate_rows(
        rows, model, tokenizer, device,
        attempts=args.attempts,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": str(device),
        "attempts_per_question": args.attempts,
        "profile_questions": len(rows),
        "metrics": summarize_student_metrics(metrics),
    }
    write_jsonl(args.output, metrics)
    write_json(args.summary, summary)
    print(f"Wrote {len(metrics)} attempt records to {args.output}")
    print(f"Wrote concept profile to {args.summary}")


if __name__ == "__main__":
    main()
