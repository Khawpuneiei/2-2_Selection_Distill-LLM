"""Dataset adapters that keep source labels and provenance explicit."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .scoring import extract_final_answer


def stable_id(dataset: str, split: str, question: str) -> str:
    """Return a stable source ID derived from dataset, split, and question text."""
    canonical_question = " ".join(str(question).split())
    payload = "\0".join((dataset.strip().lower(), split.strip().lower(), canonical_question))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{dataset.lower()}-{split.lower()}-{digest}"


def normalize_example(
    dataset: str,
    split: str,
    raw: dict[str, Any],
    *,
    subject: str | None = None,
    concept: str | None = None,
) -> dict[str, Any]:
    """Normalize GSM8K, MATH, or GSM-Plus rows without fabricating fine tags."""
    name = dataset.strip().lower().replace("-", "_")
    if name == "gsm8k":
        question = str(raw.get("question", "")).strip()
        raw_answer = str(raw.get("answer", "")).strip()
        if not question or not raw_answer:
            raise ValueError("GSM8K rows require non-empty question and answer fields")
        answer_match = re.search(r"####\s*(.+?)\s*$", raw_answer, flags=re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else extract_final_answer(raw_answer)
        solution = raw_answer[: answer_match.start()].strip() if answer_match else raw_answer
        default_concept = "gsm8k_word_problem"
        result = {
            "id": stable_id("gsm8k", split, question),
            "dataset": "gsm8k",
            "split": split,
            "question": question,
            "solution": solution,
            "answer": answer,
            "concept": concept or default_concept,
            "subconcept": "",
            "evaluation_only": split.lower() in {"test", "validation", "valid"},
        }
        return result

    if name == "math":
        question = str(raw.get("problem", "")).strip()
        solution = str(raw.get("solution", "")).strip()
        subject_name = (subject or "").strip().lower().replace(" ", "_")
        if not question or not solution or not subject_name:
            raise ValueError("MATH rows require problem, solution, and a subject config")
        result = {
            "id": stable_id(f"math-{subject_name}", split, question),
            "dataset": "math",
            "split": split,
            "question": question,
            "solution": solution,
            "answer": extract_final_answer(solution),
            "concept": concept or subject_name,
            "subconcept": str(raw.get("type", "")).strip().lower().replace(" ", "_"),
            "level": str(raw.get("level", "")).strip(),
            "evaluation_only": split.lower() in {"test", "validation", "valid"},
        }
        return result

    if name == "gsm_plus":
        question = str(raw.get("question", "")).strip()
        solution = str(raw.get("solution", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError("GSM-Plus rows require non-empty question and answer fields")
        seed_question = str(raw.get("seed_question", "")).strip()
        variant = str(raw.get("perturbation_type", "")).strip()
        return {
            "id": stable_id("gsm_plus", split, f"{seed_question}\0{variant}\0{question}"),
            "seed_id": stable_id("gsm8k", "test", seed_question) if seed_question else "",
            "dataset": "gsm_plus",
            "split": split,
            "question": question,
            "solution": solution,
            "answer": answer,
            "concept": concept or "gsm8k_word_problem",
            "subconcept": variant,
            "evaluation_only": True,
        }

    raise ValueError(f"unsupported dataset: {dataset!r}")
