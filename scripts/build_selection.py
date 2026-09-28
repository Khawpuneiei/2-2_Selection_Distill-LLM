from __future__ import annotations

import argparse
from pathlib import Path

from selection_distill.preparation import attach_teacher_scores
from selection_distill.records import read_csv, read_jsonl, write_csv, write_json, write_jsonl
from selection_distill.selection import build_training_arms, classify_quadrant, summarize_student_metrics
from selection_distill.training_utils import encode_supervised_example


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the three matched-token SFT arms")
    parser.add_argument("--train-pool", type=Path, default=Path("data/processed/train_pool.jsonl"))
    parser.add_argument("--profile-metrics", type=Path, default=Path("outputs/student_profile_metrics.jsonl"))
    parser.add_argument("--teacher-scores", type=Path, required=True, help="Apply 1 CSV: id or concept,confidence,teacher_tokens")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/selection"))
    parser.add_argument("--token-budget", type=int, default=200000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--poor-accuracy", type=float, default=0.50)
    parser.add_argument("--teacher-confidence", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.token_budget <= 0 or args.max_length <= 0:
        parser.error("token-budget and max-length must be positive")

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Install the project environment before tokenizing the selection pool") from exc

    pool = read_jsonl(args.train_pool)
    profile_attempts = read_jsonl(args.profile_metrics)
    profile = summarize_student_metrics(profile_attempts)
    teacher_rows = read_csv(args.teacher_scores)
    attached, confidence_by_concept = attach_teacher_scores(pool, teacher_rows)
    concepts = sorted({str(row["concept"]) for row in attached})
    missing_profiles = sorted(set(concepts) - set(profile))
    if missing_profiles:
        raise ValueError(f"student profile is missing concepts: {', '.join(missing_profiles)}")
    concept_quadrants = {
        concept: classify_quadrant(
            profile[concept]["accuracy"],
            confidence_by_concept[concept],
            poor_accuracy_threshold=args.poor_accuracy,
            teacher_confidence_threshold=args.teacher_confidence,
        )
        for concept in concepts
    }

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenized_samples = []
    by_id = {}
    for row in attached:
        if row["id"] in by_id:
            raise ValueError(f"duplicate training sample id: {row['id']}")
        encoded = encode_supervised_example(
            {**row, "allocated_tokens": args.max_length}, tokenizer, max_length=args.max_length
        )
        token_count = encoded["active_tokens"]
        tokenized_samples.append(
            {
                "id": row["id"],
                "concept": row["concept"],
                "tokens": token_count,
                "teacher_confidence": row["teacher_confidence"],
            }
        )
        by_id[row["id"]] = row

    arms = build_training_arms(
        tokenized_samples, concept_quadrants,
        token_budget=args.token_budget,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_tokens_by_key = {
        str(row["teacher_score_key"]): row["teacher_tokens"] for row in attached
    }
    arm_stats = {}
    for arm_name, selections in arms.items():
        output_rows = []
        for selection in selections:
            source = by_id[selection["id"]]
            output_rows.append(
                {
                    **source,
                    **selection,
                    "selection_arm": arm_name,
                }
            )
        write_jsonl(args.output_dir / f"{arm_name}.jsonl", output_rows)
        unique_score_keys = {str(row["teacher_score_key"]) for row in output_rows}
        score_token_values = [teacher_tokens_by_key.get(key) for key in unique_score_keys]
        token_count_complete = bool(score_token_values) and all(
            value is not None for value in score_token_values
        )
        arm_stats[arm_name] = {
            "training_tokens": sum(row["allocated_tokens"] for row in output_rows),
            "selected_rows": len(output_rows),
            "unique_examples": len({row["id"] for row in output_rows}),
            "reused_rows": sum(bool(row["reused"]) for row in output_rows),
            "truncated_rows": sum(bool(row["truncated"]) for row in output_rows),
            "teacher_tokens_used": sum(score_token_values) if token_count_complete else None,
            "teacher_token_receipt_complete": token_count_complete,
        }

    quadrant_rows = []
    for concept in concepts:
        quadrant_rows.append(
            {
                "concept": concept,
                "student_accuracy": profile[concept]["accuracy"],
                "student_mean_entropy": profile[concept]["mean_entropy"],
                "student_mean_loss": profile[concept]["mean_loss"],
                "student_examples": profile[concept]["examples"],
                "student_attempts": profile[concept]["attempts"],
                "student_repeated_failures": profile[concept]["repeated_failures"],
                "teacher_confidence": confidence_by_concept[concept],
                "quadrant": concept_quadrants[concept],
            }
        )
    write_csv(
        args.output_dir / "concept_quadrants.csv",
        quadrant_rows,
        [
            "concept", "student_accuracy", "student_mean_entropy", "student_mean_loss",
            "student_examples", "student_attempts", "student_repeated_failures",
            "teacher_confidence", "quadrant",
        ],
    )
    distinct_pool_score_keys = {str(row["teacher_score_key"]) for row in attached}
    pool_score_token_values = [teacher_tokens_by_key.get(key) for key in distinct_pool_score_keys]
    pool_token_receipt_complete = bool(pool_score_token_values) and all(
        value is not None for value in pool_score_token_values
    )
    manifest = {
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "token_budget": args.token_budget,
        "seed": args.seed,
        "thresholds": {
            "poor_student_accuracy_at_or_below": args.poor_accuracy,
            "teacher_confidence_at_or_above": args.teacher_confidence,
        },
        "inputs": {
            "train_pool": str(args.train_pool),
            "student_profile_metrics": str(args.profile_metrics),
            "apply1_teacher_scores": str(args.teacher_scores),
        },
        "arms": arm_stats,
        "teacher_scoring_pool": {
            "unique_confidence_records_for_training_pool": len(distinct_pool_score_keys),
            "teacher_tokens_used": sum(pool_score_token_values) if pool_token_receipt_complete else None,
            "token_receipt_complete": pool_token_receipt_complete,
            "apply2_additional_teacher_tokens": 0,
        },
        "notes": [
            "Uniform and confidence-weighted arms use the same deterministic sample draw.",
            "Quadrant-prioritized uses only poor-student, confident-teacher examples.",
            "Confidence weights are teacher confidence divided by the pool mean confidence.",
            "Repeated examples and the final partial example are marked in the arm JSONL files.",
            "Teacher token cost is null unless every distinct score record used has a teacher_tokens receipt.",
            "The shared teacher-scoring-pool total is reported once; Apply 2 reuses these saved records and adds zero teacher queries.",
        ],
    }
    write_json(args.output_dir / "selection_manifest.json", manifest)
    print(f"Wrote quadrant map and selection arms under {args.output_dir}")
    for arm, stats in arm_stats.items():
        print(
            f"  {arm}: {stats['training_tokens']} tokens, "
            f"{stats['unique_examples']} unique examples, "
            f"teacher tokens={stats['teacher_tokens_used']}"
        )


if __name__ == "__main__":
    main()
