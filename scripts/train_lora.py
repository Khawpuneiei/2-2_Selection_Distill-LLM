from __future__ import annotations

import argparse
from pathlib import Path

from selection_distill.records import read_json, read_jsonl
from selection_distill.training import train_lora_arm


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA-train one selected SFT arm")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, default=Path("data/processed/selection/selection_manifest.json"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--tokens-per-update", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--target-modules", nargs="+", default=["q_proj", "v_proj"],
        help="LoRA target projections, e.g. q_proj k_proj v_proj o_proj gate_proj up_proj down_proj",
    )
    parser.add_argument("--max-updates", type=int, help="Development smoke cap; marks incomplete runs")
    args = parser.parse_args()
    if args.max_length <= 0 or args.tokens_per_update <= 0:
        parser.error("max-length and tokens-per-update must be positive")

    rows = read_jsonl(args.input)
    arm = rows[0].get("selection_arm", "") if rows else ""
    if args.selection_manifest.exists():
        manifest = read_json(args.selection_manifest)
        if manifest.get("max_length") != args.max_length:
            parser.error(
                f"--max-length {args.max_length} differs from selection manifest "
                f"({manifest.get('max_length')}); rebuild or pass the matching value"
            )
        stats = manifest.get("arms", {}).get(arm, {})
        actual_tokens = sum(int(row["allocated_tokens"]) for row in rows)
        if actual_tokens != stats.get("training_tokens"):
            parser.error("selected rows do not match the arm token count in selection manifest")
    summary = train_lora_arm(
        rows,
        model_name=args.model,
        output_dir=args.output_dir,
        max_length=args.max_length,
        tokens_per_update=args.tokens_per_update,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        max_updates=args.max_updates,
        target_modules=tuple(args.target_modules),
    )
    print(
        f"Saved {summary['arm']} adapter to {summary['adapter_dir']}; "
        f"tokens={summary['training_tokens_seen']}/{summary['training_tokens_budgeted']}, "
        f"complete={summary['completed_token_budget']}"
    )


if __name__ == "__main__":
    main()
