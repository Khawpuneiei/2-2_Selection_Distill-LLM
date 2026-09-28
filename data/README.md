# Local data area

Downloaded datasets, concept assignments, Apply 1 confidence scores, and
prepared JSONL files stay local and are excluded from Git. See
[`docs/data-contracts.md`](../docs/data-contracts.md) for the complete schema.

Templates:

- [`templates/concept_map.csv`](templates/concept_map.csv): optional GSM8K
  question IDs mapped to a human-reviewed concept.
- [`templates/teacher_scores.csv`](templates/teacher_scores.csv): Apply 1
  confidence values and optional teacher-token receipts.

Run `python -m scripts.prepare_data` to download and normalize GSM8K, MATH, and
GSM-Plus into `data/processed/`. GSM-Plus is used only for evaluation.
