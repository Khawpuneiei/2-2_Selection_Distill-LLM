"""Join selection and benchmark receipts into transparent ablation rows."""

from __future__ import annotations

from typing import Any


ARM_ORDER = ("base", "uniform", "quadrant_prioritized", "confidence_weighted")


def build_ablation_rows(
    selection_manifest: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    arm_stats = selection_manifest.get("arms", {})
    rows: list[dict[str, Any]] = []
    dataset_names = sorted(
        {
            name
            for receipt in evaluations.values()
            for name in receipt.get("splits", {})
        }
    )
    for dataset in dataset_names:
        baseline = evaluations.get("uniform", {}).get("splits", {}).get(dataset, {}).get("accuracy")
        for arm in ARM_ORDER:
            split = evaluations.get(arm, {}).get("splits", {}).get(dataset)
            if split is None:
                continue
            selection = arm_stats.get(
                arm,
                {"training_tokens": 0, "teacher_tokens_used": 0, "teacher_token_receipt_complete": True}
                if arm == "base"
                else {},
            )
            accuracy = split.get("accuracy")
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "accuracy": accuracy,
                    "accuracy_delta_vs_uniform": (
                        accuracy - baseline
                        if accuracy is not None and baseline is not None
                        else None
                    ),
                    "mean_entropy": split.get("mean_entropy"),
                    "mean_loss": split.get("mean_loss"),
                    "examples": split.get("examples"),
                    "training_tokens": selection.get("training_tokens"),
                    "teacher_tokens_used": selection.get("teacher_tokens_used"),
                    "teacher_token_receipt_complete": selection.get(
                        "teacher_token_receipt_complete", False
                    ),
                }
            )
    return rows
