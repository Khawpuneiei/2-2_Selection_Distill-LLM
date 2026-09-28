"""Pure selection and profiling logic used by the experiment pipeline."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


QUADRANTS = {"high_priority", "delay", "low_priority", "exclude"}


def _probability(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be a number between 0 and 1")
    return result


def classify_quadrant(
    student_accuracy: float,
    teacher_confidence: float,
    *,
    poor_accuracy_threshold: float = 0.50,
    teacher_confidence_threshold: float = 0.70,
) -> str:
    """Classify a concept; threshold ties count as poor or confident."""
    accuracy = _probability(student_accuracy, "student_accuracy")
    confidence = _probability(teacher_confidence, "teacher_confidence")
    poor_threshold = _probability(poor_accuracy_threshold, "poor_accuracy_threshold")
    confident_threshold = _probability(
        teacher_confidence_threshold, "teacher_confidence_threshold"
    )
    student_is_weak = accuracy <= poor_threshold
    teacher_is_confident = confidence >= confident_threshold

    if student_is_weak and teacher_is_confident:
        return "high_priority"
    if student_is_weak:
        return "delay"
    if teacher_is_confident:
        return "low_priority"
    return "exclude"


def summarize_student_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate repeated per-question evaluations into concept profiles.

    Accuracy, entropy, and loss are averaged across attempts. A repeated failure
    is a question answered incorrectly on every recorded attempt.
    """
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_attempts: set[tuple[str, str, Any]] = set()

    for row in rows:
        try:
            sample_id = str(row["id"])
            concept = str(row["concept"])
            attempt = row["attempt"]
            correct = row["correct"]
            entropy = float(row["entropy"])
            loss = float(row["loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "each student metric row needs id, concept, attempt, correct, entropy, and loss"
            ) from exc

        if not sample_id or not concept:
            raise ValueError("student metric id and concept must be non-empty")
        if not isinstance(correct, bool):
            raise ValueError(f"correct must be a boolean for sample {sample_id}")
        if not math.isfinite(entropy) or entropy < 0 or not math.isfinite(loss) or loss < 0:
            raise ValueError(f"entropy and loss must be finite and non-negative for {sample_id}")
        attempt_key = (concept, sample_id, attempt)
        if attempt_key in seen_attempts:
            raise ValueError(f"duplicate attempt {attempt!r} for sample {sample_id}")
        seen_attempts.add(attempt_key)
        grouped[concept][sample_id].append(
            {"correct": correct, "entropy": entropy, "loss": loss}
        )

    if not grouped:
        raise ValueError("at least one student metric row is required")

    result: dict[str, dict[str, Any]] = {}
    for concept in sorted(grouped):
        examples = grouped[concept]
        attempts = [attempt for sample_rows in examples.values() for attempt in sample_rows]
        result[concept] = {
            "examples": len(examples),
            "attempts": len(attempts),
            "accuracy": sum(row["correct"] for row in attempts) / len(attempts),
            "mean_entropy": sum(row["entropy"] for row in attempts) / len(attempts),
            "mean_loss": sum(row["loss"] for row in attempts) / len(attempts),
            "repeated_failures": sum(
                all(not row["correct"] for row in sample_rows)
                for sample_rows in examples.values()
            ),
        }
    return result


def _draw_exact_budget(
    samples: list[dict[str, Any]], token_budget: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Sample complete rows, then trim the last row to meet an exact token cap."""
    candidates = list(samples)
    rng.shuffle(candidates)
    next_index = 0
    used_counts: dict[str, int] = defaultdict(int)
    remaining = token_budget
    chosen: list[dict[str, Any]] = []

    while remaining:
        if next_index == len(candidates):
            rng.shuffle(candidates)
            next_index = 0
        sample = candidates[next_index]
        next_index += 1
        allocated = min(sample["tokens"], remaining)
        chosen.append(
            {
                "id": sample["id"],
                "concept": sample["concept"],
                "tokens": sample["tokens"],
                "allocated_tokens": allocated,
                "teacher_confidence": sample["teacher_confidence"],
                "loss_weight": 1.0,
                "truncated": allocated < sample["tokens"],
                "reused": used_counts[sample["id"]] > 0,
            }
        )
        used_counts[sample["id"]] += 1
        remaining -= allocated
    return chosen


def build_training_arms(
    samples: Iterable[dict[str, Any]],
    concept_quadrants: dict[str, str],
    *,
    token_budget: int,
    seed: int = 17,
) -> dict[str, list[dict[str, Any]]]:
    """Build matched-budget uniform, quadrant, and confidence-loss arms.

    The uniform and confidence-weighted arms use the same sampled examples.
    The last example may be token-truncated; repeated draws are marked so the
    training manifest makes oversampling visible.
    """
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget <= 0:
        raise ValueError("token_budget must be a positive integer")

    pool = list(samples)
    if not pool:
        raise ValueError("at least one training sample is required")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in pool:
        try:
            sample_id = str(row["id"])
            concept = str(row["concept"])
            tokens = row["tokens"]
            confidence = _probability(row["teacher_confidence"], "teacher_confidence")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "each training sample needs id, concept, positive token count, and teacher_confidence"
            ) from exc
        if not sample_id or not concept:
            raise ValueError("training sample id and concept must be non-empty")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate training sample id: {sample_id}")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(f"tokens must be a positive integer for sample {sample_id}")
        if concept not in concept_quadrants:
            raise ValueError(f"missing quadrant for concept {concept}")
        quadrant = concept_quadrants[concept]
        if quadrant not in QUADRANTS:
            raise ValueError(f"unknown quadrant {quadrant!r} for concept {concept}")
        seen_ids.add(sample_id)
        normalized.append(
            {
                "id": sample_id,
                "concept": concept,
                "tokens": tokens,
                "teacher_confidence": confidence,
            }
        )

    priority_pool = [
        row for row in normalized if concept_quadrants[row["concept"]] == "high_priority"
    ]
    if not priority_pool:
        raise ValueError("quadrant_prioritized arm requires at least one high_priority sample")

    rng = random.Random(seed)
    uniform_rows = _draw_exact_budget(normalized, token_budget, rng)
    weighted_rows = [dict(row) for row in uniform_rows]
    average_confidence = sum(row["teacher_confidence"] for row in normalized) / len(normalized)
    if average_confidence <= 0:
        raise ValueError("confidence-weighted arm requires positive mean teacher confidence")
    for row in weighted_rows:
        row["loss_weight"] = row["teacher_confidence"] / average_confidence

    prioritized_rows = _draw_exact_budget(priority_pool, token_budget, random.Random(seed + 1))
    return {
        "uniform": uniform_rows,
        "quadrant_prioritized": prioritized_rows,
        "confidence_weighted": weighted_rows,
    }
