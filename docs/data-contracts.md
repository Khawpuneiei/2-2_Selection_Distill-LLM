# Data contracts

## Prepared records

`python -m scripts.prepare_data` writes JSONL files under `data/processed/`.
Every row contains:

```json
{"id":"gsm8k-train-…","dataset":"gsm8k","split":"train","question":"…","solution":"…","answer":"…","concept":"gsm8k_word_problem","subconcept":"","evaluation_only":false}
```

IDs are stable hashes of dataset, split, and normalized question text. MATH
records preserve the source subject as `concept`, native `type` as
`subconcept`, and `level`. GSM8K has no native concept column; it receives the
coarse `gsm8k_word_problem` label until a reviewed mapping overrides it.
GSM-Plus rows use `seed_id` to inherit a supplied GSM8K test concept tag when
available. All GSM-Plus rows remain `evaluation_only: true`.

## Concept map

CSV columns: `id,concept`. IDs can be copied from the generated
`gsm8k_concept_template.csv`. Leave out unreviewed rows; the pipeline keeps
their coarse label. Duplicate IDs or empty concept names fail preparation.
MATH's source labels are not overwritten unless a row is explicitly mapped.

## Apply 1 teacher confidence

CSV columns: `id,concept,confidence,teacher_tokens`.

- Per-example format: set `id`; `concept` may be supplied for cross-checking;
  `confidence` must be in `[0,1]`; `teacher_tokens` is the prompt plus
  completion token count used to produce that saved score, when available.
- Per-concept format: leave `id` blank and set `concept`, `confidence`, and
  optional `teacher_tokens`.
- Every training-pool sample must match a sample-level score or a score for
  its concept. Missing scores, duplicate score IDs, and concept mismatches are
  errors. The pipeline does not call the teacher or invent a score.
- When both formats are present, a sample-level score takes precedence for
  that row; the explicit concept-level value is used as the concept's
  quadrant confidence.
- A blank `teacher_tokens` field is an unknown receipt, not zero.

`python -m scripts.export_apply1_confidence --predictions <run>/predictions.jsonl`
builds this CSV from an Apply 1 teacher-reliability run. It writes one
concept-level row per GSM8K/MATH concept: `confidence` is the mean
self-consistency agreement (`greedy_agreement_share`, or
`--signal majority_vote_share`), and `teacher_tokens` counts the generated
tokens (greedy + samples) behind that score. GSM-Plus rows are skipped. Extra
columns `apply1_questions` and `apply1_greedy_accuracy` are informational.

A blank header template is tracked at
[`data/templates/teacher_scores.csv`](../data/templates/teacher_scores.csv).

## Student profile metrics

`profile_student.py` writes one JSONL row per generated attempt with
`id,concept,attempt,correct,entropy,loss,supervised_tokens`. The loss is
repeated for attempts of a question because it is a teacher-forced property of
the known solution, not of the sampled generation. `summarize_student_metrics`
computes concept means and counts questions wrong on every attempt.

## Selection arm records

Each selected row preserves the source question and solution and adds:

- `tokens`: full training sequence length capped to `max_length`.
- `allocated_tokens`: exact active-token allocation for this exposure.
- `loss_weight`: `1.0` for Uniform and quadrant-prioritized; normalized
  teacher confidence for the weighted arm.
- `reused` and `truncated`: flags for oversampling and final partial exposure.
- `teacher_score_key` and `teacher_tokens`: provenance for the confidence
  receipt attributed to this example.

The sum of `allocated_tokens` is exactly the same configured token budget for
all three arms. The training loop checks that tokenization still agrees with
the selection manifest before each row is trained.
