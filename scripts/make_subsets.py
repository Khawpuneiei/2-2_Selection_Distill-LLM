from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from selection_distill.records import read_jsonl, write_json, write_jsonl


def stratified_sample(rows: list[dict], per_concept: int | None, total: int | None, seed: int) -> list[dict]:
    """Deterministic sample: up to `per_concept` per concept, or `total` overall."""
    rng = random.Random(seed)
    if per_concept is not None:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["concept"]].append(row)
        chosen = []
        for concept in sorted(grouped):
            items = sorted(grouped[concept], key=lambda row: row["id"])
            rng.shuffle(items)
            chosen.extend(items[:per_concept])
        return chosen
    items = sorted(rows, key=lambda row: row["id"])
    rng.shuffle(items)
    return items[:total] if total is not None else items


def main() -> None:
    parser = argparse.ArgumentParser(description="Write fixed, seeded profile and evaluation subsets")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/subsets"))
    parser.add_argument("--profile-per-concept", type=int, default=100)
    parser.add_argument("--gsm8k-test", type=int, default=500)
    parser.add_argument("--math-test-per-subject", type=int, default=50)
    parser.add_argument("--gsm-plus-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = args.processed_dir
    subsets = {
        "profile": stratified_sample(read_jsonl(source / "profile.jsonl"), args.profile_per_concept, None, args.seed),
        "gsm8k_test": stratified_sample(read_jsonl(source / "gsm8k_test.jsonl"), None, args.gsm8k_test, args.seed),
        "math_test": stratified_sample(read_jsonl(source / "math_test.jsonl"), args.math_test_per_subject, None, args.seed),
        "gsm_plus_test": stratified_sample(read_jsonl(source / "gsm_plus_test.jsonl"), None, args.gsm_plus_test, args.seed),
    }
    manifest = {"seed": args.seed, "source_dir": str(source), "subsets": {}}
    for name, rows in subsets.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
        manifest["subsets"][name] = {
            "rows": len(rows),
            "concepts": dict(sorted(Counter(row["concept"] for row in rows).items())),
        }
        if name == "gsm_plus_test":
            manifest["subsets"][name]["perturbation_types"] = dict(
                sorted(Counter(row.get("subconcept", "") for row in rows).items())
            )
        print(f"  {name}: {len(rows)} rows")
    write_json(args.output_dir / "subsets_manifest.json", manifest)


if __name__ == "__main__":
    main()
