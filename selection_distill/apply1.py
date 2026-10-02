"""Convert Apply 1 teacher-reliability predictions into Apply 2 confidence rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

GSM8K_CONCEPT = "gsm8k_word_problem"
SIGNALS = ("greedy_agreement_share", "majority_vote_share")


def apply1_concept(row: dict[str, Any]) -> str | None:
    """Map an Apply 1 source record to the Apply 2 concept, or None if eval-only."""
    source = row.get("source") or {}
    dataset = str(source.get("dataset", "")).lower()
    if dataset == "gsm8k":
        return GSM8K_CONCEPT
    if dataset == "math":
        subject = str(source.get("subject") or "").strip().lower().replace(" ", "_")
        if not subject:
            raise ValueError(f"Apply 1 MATH row {row.get('example_id')} has no subject")
        return subject
    return None


def _completion_tokens(row: dict[str, Any]) -> int:
    greedy = row.get("greedy") or {}
    total = len(greedy.get("token_ids") or [])
    for sample in row.get("samples") or []:
        total += len(sample.get("token_ids") or [])
    return total


def concept_confidence_rows(
    predictions: Iterable[dict[str, Any]], *, signal: str = "greedy_agreement_share"
) -> list[dict[str, Any]]:
    """Average one Apply 1 self-consistency signal per concept.

    Only completed GSM8K and MATH rows are used; GSM-Plus stays evaluation-only.
    ``teacher_tokens`` is the generated (completion) token count of the greedy
    answer plus all samples for the rows behind that concept score.
    """
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {', '.join(SIGNALS)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        if row.get("status") != "complete":
            continue
        concept = apply1_concept(row)
        if concept is None:
            continue
        value = (row.get("self_consistency") or {}).get(signal)
        if value is None:
            raise ValueError(f"Apply 1 row {row.get('example_id')} has no {signal}")
        grouped[concept].append(
            {
                "confidence": float(value),
                "tokens": _completion_tokens(row),
                "correct": bool((row.get("greedy") or {}).get("correct")),
            }
        )
    if not grouped:
        raise ValueError("no completed GSM8K or MATH rows in the Apply 1 predictions")
    rows = []
    for concept in sorted(grouped):
        items = grouped[concept]
        rows.append(
            {
                "id": "",
                "concept": concept,
                "confidence": sum(item["confidence"] for item in items) / len(items),
                "teacher_tokens": sum(item["tokens"] for item in items),
                "apply1_questions": len(items),
                "apply1_greedy_accuracy": sum(item["correct"] for item in items) / len(items),
            }
        )
    return rows
