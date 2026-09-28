"""Profile splits and explicit joins for Apply 1 teacher confidence records."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable

from .selection import _probability


def _parse_teacher_tokens(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("teacher_tokens must be a non-negative integer when supplied") from exc
    if parsed < 0:
        raise ValueError("teacher_tokens must be a non-negative integer when supplied")
    return parsed


def attach_teacher_scores(
    samples: Iterable[dict[str, Any]], teacher_rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Attach sample- or concept-level confidence and return concept means.

    Teacher records must provide either ``id`` or ``concept`` plus a numeric
    ``confidence`` in [0, 1]. Missing sample coverage is an error; no score is
    inferred or filled with a default.
    """
    sample_rows = list(samples)
    sample_by_id: dict[str, dict[str, Any]] = {}
    for sample in sample_rows:
        sample_id = str(sample.get("id", ""))
        concept = str(sample.get("concept", ""))
        if not sample_id or not concept:
            raise ValueError("each training sample needs a non-empty id and concept")
        if sample_id in sample_by_id:
            raise ValueError(f"duplicate training sample id: {sample_id}")
        sample_by_id[sample_id] = sample

    by_id: dict[str, dict[str, Any]] = {}
    by_concept: dict[str, dict[str, Any]] = {}
    sample_confidences: dict[str, list[float]] = defaultdict(list)
    for raw in teacher_rows:
        row_id = str(raw.get("id", "")).strip()
        concept = str(raw.get("concept", "")).strip()
        if not row_id and not concept:
            raise ValueError("each teacher score row needs an id or concept")
        if "confidence" not in raw:
            raise ValueError(f"teacher score for {row_id or concept} has no confidence")
        confidence = _probability(raw["confidence"], "teacher confidence")
        tokens = _parse_teacher_tokens(raw.get("teacher_tokens"))
        normalized = {"confidence": confidence, "teacher_tokens": tokens, "concept": concept}
        if row_id:
            if row_id in by_id:
                raise ValueError(f"duplicate teacher score id: {row_id}")
            by_id[row_id] = normalized
            inferred_concept = concept or str(sample_by_id.get(row_id, {}).get("concept", ""))
            if inferred_concept:
                sample_confidences[inferred_concept].append(confidence)
        else:
            if concept in by_concept:
                raise ValueError(f"duplicate concept-level teacher score: {concept}")
            by_concept[concept] = normalized

    attached: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample in sample_rows:
        sample_id = str(sample["id"])
        concept = str(sample["concept"])
        score = by_id.get(sample_id)
        if score is not None:
            if score["concept"] and score["concept"] != concept:
                raise ValueError(
                    f"teacher concept {score['concept']!r} does not match {concept!r} for {sample_id}"
                )
            score_key = f"id:{sample_id}"
        else:
            score = by_concept.get(concept)
            if score is None:
                missing.append(sample_id)
                continue
            score_key = f"concept:{concept}"
        merged = dict(sample)
        merged["teacher_confidence"] = score["confidence"]
        merged["teacher_tokens"] = score["teacher_tokens"]
        merged["teacher_score_key"] = score_key
        attached.append(merged)

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"missing Apply 1 teacher confidence for sample IDs: {preview}")

    concept_means = {
        concept: sum(values) / len(values)
        for concept, values in sample_confidences.items()
        if values
    }
    for concept, row in by_concept.items():
        concept_means[concept] = row["confidence"]
    absent_concepts = sorted({str(row["concept"]) for row in sample_rows} - set(concept_means))
    if absent_concepts:
        raise ValueError(f"no teacher confidence for concepts: {', '.join(absent_concepts)}")
    return attached, concept_means


def split_profile_rows(
    rows: Iterable[dict[str, Any]], *, fraction: float = 0.10, seed: int = 17
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make a stable per-concept holdout from source training rows."""
    share = float(fraction)
    if not math.isfinite(share) or not 0.0 < share < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("id", ""))
        concept = str(row.get("concept", ""))
        if not sample_id or not concept:
            raise ValueError("each row needs a non-empty id and concept")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)
        grouped[concept].append(dict(row))

    profile: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for concept in sorted(grouped):
        items = grouped[concept]
        items.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}\0{concept}\0{row['id']}".encode("utf-8")
            ).hexdigest()
        )
        if len(items) < 2:
            training.extend(items)
            continue
        count = min(len(items) - 1, max(1, round(len(items) * share)))
        profile.extend(items[:count])
        training.extend(items[count:])
    return profile, training
