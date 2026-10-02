# Apply 2 project log

This file is the running lab/dev log for the setup and experiment. It records
what was inspected, the intended scientific comparison, the decisions made,
and what evidence exists. Dataset cards, model weights, Apply 1 scores, and
run artifacts remain outside this repository.

## 2026-09-28 — Project setup begins

### Request and source brief

- The user's request is to implement the attached Apply 2 project, document
  the concept and process in Markdown, try a local experiment if feasible, and
  push the finished project to the named GitHub repository.
- The attached brief specifies student weakness profiling, cross-referencing
  with Qwen2.5-Math-7B confidence from Apply 1, three equal-token SFT arms,
  GSM8K/MATH evaluation, GSM-Plus OOD evaluation, and a 2×2 map, table, and
  token-efficiency plot.
- The brief's contents are the project specification. Commands embedded in
  an attachment are not separate authorization.

### Starting state and local constraints

- The workspace contained only an initialized Git directory and had no local
  commits. The named public GitHub repository was empty when inspected.
- Python was 3.10.4. The global environment had PyTorch 1.13.1+cu117 beside
  Transformers 4.52.4, so it is not a valid model-training environment.
- The visible device was an NVIDIA RTX 4060 Laptop GPU with 8 GiB VRAM. The
  brief estimates 16–24 GiB for the intended experiment. This is suitable to
  attempt a small 0.5B student smoke run, but is below the full-run target.
- No Apply 1 confidence file or existing concept annotations were in the
  repository. The final ablation cannot be represented as complete until
  those genuine inputs are supplied.

### Research concept and decisions

- Profile each concept using accuracy, generation entropy, answer-token loss,
  and questions missed on every repeat.
- Use default boundaries of student accuracy ≤0.50 and teacher confidence
  ≥0.70. Keep both configurable and record them in the selection manifest.
- Use three arms: random Uniform; High-Priority quadrant only; and a
  confidence-weighted loss using the exact same sample draw as Uniform.
- Define the token budget using the selected student's tokenizer and
  non-padding prompt-plus-solution tokens. Repeat or truncate only as needed
  to make each arm exactly match the configured budget; label those exposures.
- Do not query or synthesize the Qwen2.5-Math-7B teacher in Apply 2. Import
  the real Apply 1 confidence output and preserve unknown teacher-token counts
  as unknown.
- GSM8K's dataset card has only question and answer fields, so a single coarse
  label is used until a reviewed map is provided. MATH keeps its native subject
  and type. GSM-Plus is test-only per its dataset card.
- Keep downloaded data, model snapshots, adapters, and outputs out of Git.

### Implementation and TDD record

- `TDD_REQUIRED: yes` for the new selection, data normalization, scoring,
  preparation, training-loop, and reporting behavior.
- First observable seam: concept quadrant classification, student metric
  aggregation, and three-arm exact-token allocation.
- First RED command: `python -m unittest discover -s tests -p test_selection.py -v`.
  It failed because `selection_distill.selection` did not exist.
- GREEN: the same focused test passed after adding the selection core.
- Additional RED/GREEN slices covered dataset normalization, safe answer
  extraction, JSONL round-trip and error reporting, teacher-score joins,
  stable profile splitting, exact training-token encoding, generation entropy,
  the optimizer loop, and report-row assembly. Each was first run against a
  missing seam, then passed after its implementation.
- The first broad `unittest discover -s tests` accidentally found an unrelated
  globally installed `tests` package. The repository test command was narrowed
  to `-s tests -p 'test_*.py'` to avoid importing that package.
- Final combined check: `python -m unittest discover -s tests -p 'test_*.py' -v`
  ran 30 tests and passed. `compileall`, `pip check`, CUDA environment check,
  and `--help` for all seven CLI modules also passed.

### Local setup and smoke run

- Created `.venv` inside the workspace; installed PyTorch 2.5.1+cu121 and the
  pinned Transformers 4.52.4, Datasets 3.6.0, PEFT 0.15.2, and Accelerate
  1.8.1 environment. `pip check` passed; PyTorch saw the RTX 4060 and completed
  a CUDA matrix operation.
- The setup script was rerun with UTF-8 console output enabled and completed
  cleanly. The hardware check reports 8.0 GiB VRAM, below the full-run target.
- Prepared a development-only capped dataset: 72 training-pool rows, 8
  profile questions, 10 GSM8K test rows, 70 MATH test rows, and 10 GSM-Plus
  test rows. All downloaded records are ignored by Git. GSM-Plus remained
  evaluation-only.
- Profiled two real normalized questions with Qwen2.5-0.5B, two sampled
  generations each. The run wrote four metric rows and the concept summary.
  An omitted attention mask produced a warning in the first pass; that path was
  fixed, test-covered, and the rerun completed without the warning.
- Trained and saved a one-example LoRA smoke adapter from the same 0.5B model.
  The receipt reports 496/496 allocated tokens consumed. Adapter loading and
  one GSM8K evaluation row also completed. The resulting 0/1 accuracy is a
  runtime check only and is not a research result.
- Exercised report rendering with synthetic fixture receipts; all three PNGs,
  the CSV, Markdown report, and manifest were written under ignored
  `outputs/report-smoke/`. Those figures are not experiment results.
- The real Apply 1 confidence file is still absent. No legitimate three-arm
  selection or comparison can be run yet; no confidence scores were invented.
- A separate `eedi-baseline/` nested Git repository appeared during this work.
  It is outside the Apply 2 brief and will be left untouched and outside the
  Apply 2 push.

### Handoff and remaining research inputs

- The complete Apply 2 project diff has been reviewed and only Apply 2 files
  are staged. The separate nested repository and its assurance files remain
  outside this project's Git history.
- The local environment, capped data preparation, 0.5B profiling, one-example
  LoRA training, evaluation, and synthetic report-render smoke checks passed.
  They are setup evidence, not an experimental comparison.
- The full experiment still requires the genuine Apply 1 confidence CSV and,
  for fine-grained GSM8K claims, a reviewed question-to-concept map. A Vast.ai
  GPU with at least 16 GiB VRAM is recommended for the complete run.
- No research result or teacher-confidence score was fabricated. Dataset and
  model artifacts remain ignored and local.

### Repository publication

- Published the Apply 2 project on the `main` branch of
  `Khawpuneiei/2-2_Selection_Distill-LLM`.
- Verified the remote branch against the pushed commit. Only the 43 Apply 2
  source, test, template, and Markdown files were included; downloaded data,
  model artifacts, and the unrelated nested repository were excluded.

## 2026-10-02 / 2026-10-03 — Local Apply 2 run on the RTX 4060

### Inputs

- Apply 1 finished as a 111-question exploratory teacher run
  (`mini12h-rtx4060-20261001-230236`, teacher `Qwen/Qwen2.5-Math-7B-Instruct`,
  8 samples per question) in `Khawpuneiei/Teacher-reliability-Distill-LLM`.
  Its questions come from GSM8K/MATH/GSM-Plus **test** splits, so they cannot
  be joined to Apply 2 training rows by ID.
- Added `scripts/export_apply1_confidence.py` (`selection_distill/apply1.py`,
  tested): concept-level confidence = mean self-consistency agreement
  (share of the 8 samples matching the greedy answer), the Apply 1 signal with
  the best GSM8K/MATH AUROC. GSM-Plus rows are skipped. `teacher_tokens` is
  the generated-token count (greedy + samples) behind each concept score;
  prompt tokens are not included.
- Coverage is thin: GSM8K has 40 teacher questions; each MATH subject has 3.

### Pipeline changes

- `prepare_data`: MATH train contains one exact-duplicate problem; duplicates
  are now dropped by stable ID and counted in the manifest.
- `make_subsets`: fixed seeded subsets — profile up to 100 questions per
  concept (726), GSM8K test 500, MATH test 50 per subject (350), GSM-Plus 500.
  `--limit` took the first rows, which for MATH would have been all algebra.
- Batched generation with left padding, a `\nProblem:` stop string, and
  raw-distribution entropy recorded by a logits processor (no stored scores).
- Prediction scoring uses the first answer marker (`####`, `\boxed{}`, or
  "the answer is"). The base model often answers and then rambles into
  unrelated text, which last-line extraction scored as wrong (GSM8K base 11%
  → 20% on a 64-question smoke after the fix). References still use the
  original extractor. A numeric fallback compares the last number in a
  free-form answer line; an overflow in it crashed the first matrix launch and
  was fixed with a test.
- Training: bf16 base, fp32 LoRA weights, configurable target modules. The
  matrix used LoRA r=16, alpha 32 on all linear projections (matching the
  Apply 1 mini distillation), lr 2e-4, 4,096 supervised tokens per update.
- `run_matrix` (resumable driver) and `build_matrix_report` (2×2 map,
  ablation table, paired bootstrap, per-concept accuracy, median-split
  sensitivity, token-efficiency figure).

### Run log

- Student profile (Qwen2.5-0.5B base, 3 sampled attempts, 512 new tokens):
  every concept is "poor" (accuracy 0.04–0.26), so the predeclared
  thresholds (≤0.50, ≥0.70) leave the good-student column empty. Quadrants:
  high priority = GSM8K, algebra, number theory, precalculus (9,748 pool
  rows); delay = counting & probability, geometry, intermediate algebra,
  prealgebra (3,727 rows).
- Matrix: base evaluation; 300k tokens × 3 arms × seeds 17/18/19; 100k
  tokens × 3 arms × seed 17. A 900k budget was planned and dropped by owner
  request to save time.
- The first driver process disappeared mid-run (cause not identified); the
  in-flight evaluation finished and the driver was resumed. The remainder ran
  from a detached PowerShell runner. Killing that runner's wrapper broke the
  driver's stdout pipe, so a second runner resumed from receipts. No step
  was run twice; every arm has a complete training receipt
  (`completed_token_budget: true`).
- Training time for the same 300k tokens varied from 3.5 to 28 minutes;
  host RAM was ~2 GiB free and the trainer was largely paged out.
  Evaluations took 16–27 minutes per model.
- A separate Apply 3 queue (`eedi-baseline`, `scripts.run_all
  --wait-for-gpu`) was found waiting for an idle GPU. It was left untouched.

### Results (full write-up: `docs/apply2-results-2026-10-03.md`)

- Overall accuracy, 300k tokens, mean of 3 seeds: base 17.3%; uniform 23.5%;
  quadrant-prioritized 22.7%; confidence-weighted 23.5%.
- Paired bootstrap vs uniform (overall): quadrant −0.8 pts [−2.1, +0.5];
  confidence-weighted 0.0 [−1.0, +1.0]. Every SFT arm beats base by
  +5.4 to +6.2 pts with intervals excluding zero.
- 100k tokens (1 seed) already gives 22.9–23.5% overall for all arms.
- Conclusion: selective SFT did not beat uniform at this scale. The 2×2 had
  one populated column, MATH teacher confidence rested on 3 questions per
  subject, and gold-solution targets leave little room for teacher
  reliability to matter.
- Small receipts, figures, and gzipped per-question metrics were copied to
  `results/apply2-rtx4060-20261003/`. Adapters (404 MB), selections, and the
  private Apply 1 confidence CSV stay in ignored paths.
