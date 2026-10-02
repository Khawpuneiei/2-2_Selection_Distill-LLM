from __future__ import annotations

import argparse
from pathlib import Path

from selection_distill.apply1 import SIGNALS, concept_confidence_rows
from selection_distill.records import read_jsonl, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn Apply 1 predictions.jsonl into the Apply 2 concept-level confidence CSV"
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Apply 1 run predictions.jsonl")
    parser.add_argument("--output", type=Path, default=Path("data/processed/apply1_confidence.csv"))
    parser.add_argument("--signal", choices=SIGNALS, default="greedy_agreement_share")
    args = parser.parse_args()

    rows = concept_confidence_rows(read_jsonl(args.predictions), signal=args.signal)
    write_csv(
        args.output, rows,
        ["id", "concept", "confidence", "teacher_tokens", "apply1_questions", "apply1_greedy_accuracy"],
    )
    print(f"Wrote {len(rows)} concept-level confidence rows to {args.output}")
    for row in rows:
        print(
            f"  {row['concept']}: confidence={row['confidence']:.3f} "
            f"(n={row['apply1_questions']}, teacher tokens={row['teacher_tokens']})"
        )


if __name__ == "__main__":
    main()
