from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from selection_distill.data import normalize_example, stable_id
from selection_distill.preparation import split_profile_rows
from selection_distill.records import read_csv, write_csv, write_json, write_jsonl


MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _load_concept_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    assignments: dict[str, str] = {}
    for row in read_csv(path):
        sample_id = row.get("id", "").strip()
        concept = row.get("concept", "").strip()
        if not sample_id or not concept:
            raise ValueError("concept map CSV rows need non-empty id and concept columns")
        if sample_id in assignments:
            raise ValueError(f"duplicate concept assignment for {sample_id}")
        assignments[sample_id] = concept
    return assignments


def _dataset_rows(dataset, limit: int | None):
    if limit is None:
        yield from dataset
    else:
        for index in range(min(limit, len(dataset))):
            yield dataset[index]


def prepare(output_dir: Path, concept_map_path: str | None, limit_per_config: int | None) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before downloading datasets") from exc

    assignments = _load_concept_map(concept_map_path)
    gsm_train_raw = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test_raw = load_dataset("openai/gsm8k", "main", split="test")
    gsm_train = [
        normalize_example(
            "gsm8k", "train", raw,
            concept=assignments.get(stable_id("gsm8k", "train", raw["question"])),
        )
        for raw in _dataset_rows(gsm_train_raw, limit_per_config)
    ]
    gsm_test = [
        normalize_example(
            "gsm8k", "test", raw,
            concept=assignments.get(stable_id("gsm8k", "test", raw["question"])),
        )
        for raw in _dataset_rows(gsm_test_raw, limit_per_config)
    ]

    math_train: list[dict] = []
    math_test: list[dict] = []
    for subject in MATH_SUBJECTS:
        train_dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        test_dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        math_train.extend(
            normalize_example("math", "train", raw, subject=subject)
            for raw in _dataset_rows(train_dataset, limit_per_config)
        )
        math_test.extend(
            normalize_example("math", "test", raw, subject=subject)
            for raw in _dataset_rows(test_dataset, limit_per_config)
        )

    gsm_plus_raw = load_dataset("qintongli/GSM-Plus", split="test")
    gsm_plus = []
    for raw in _dataset_rows(gsm_plus_raw, limit_per_config):
        row = normalize_example("gsm_plus", "test", raw)
        row["concept"] = assignments.get(row["seed_id"], row["concept"])
        gsm_plus.append(row)

    profile, training_pool = split_profile_rows(gsm_train + math_train, fraction=0.10, seed=17)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train_pool.jsonl", training_pool)
    write_jsonl(output_dir / "profile.jsonl", profile)
    write_jsonl(output_dir / "gsm8k_test.jsonl", gsm_test)
    write_jsonl(output_dir / "math_test.jsonl", math_test)
    write_jsonl(output_dir / "gsm_plus_test.jsonl", gsm_plus)
    write_csv(
        output_dir / "gsm8k_concept_template.csv",
        (
            {"id": row["id"], "question": row["question"], "concept": ""}
            for row in gsm_train + gsm_test
        ),
        ["id", "question", "concept"],
    )

    def counts(rows: list[dict]) -> dict[str, int]:
        return dict(sorted(Counter(row["concept"] for row in rows).items()))

    manifest = {
        "sources": {
            "gsm8k": "openai/gsm8k:main",
            "math": "EleutherAI/hendrycks_math",
            "gsm_plus": "qintongli/GSM-Plus (evaluation only)",
        },
        "split_policy": "10% stable per-concept holdout from source train splits for initial student profiling; test splits remain evaluation-only",
        "concept_policy": {
            "math": "native subject config; native type retained as subconcept",
            "gsm8k": "gsm8k_word_problem unless overridden by the supplied id,concept map",
            "gsm_plus": "inherits concept assignment from the GSM8K seed question when available; otherwise gsm8k_word_problem",
        },
        "concept_map": str(Path(concept_map_path).name) if concept_map_path else None,
        "limit_per_config": limit_per_config,
        "counts": {
            "train_pool": len(training_pool),
            "profile": len(profile),
            "gsm8k_test": len(gsm_test),
            "math_test": len(math_test),
            "gsm_plus_test": len(gsm_plus),
        },
        "concept_counts": {
            "train_pool": counts(training_pool),
            "profile": counts(profile),
            "gsm8k_test": counts(gsm_test),
            "math_test": counts(math_test),
            "gsm_plus_test": counts(gsm_plus),
        },
    }
    try:
        import datasets

        manifest["datasets_library_version"] = datasets.__version__
    except Exception:
        pass
    write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize Apply 2 datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--concept-map", help="CSV with id,concept columns, including GSM8K IDs")
    parser.add_argument(
        "--limit-per-config", type=int,
        help="development-only cap per source/config; omit for complete datasets",
    )
    args = parser.parse_args()
    if args.limit_per_config is not None and args.limit_per_config <= 0:
        parser.error("--limit-per-config must be positive")
    manifest = prepare(args.output_dir, args.concept_map, args.limit_per_config)
    print(f"Prepared records under {args.output_dir}")
    for name, count in manifest["counts"].items():
        print(f"  {name}: {count}")
    print("GSM-Plus was downloaded only as an evaluation set.")


if __name__ == "__main__":
    main()
