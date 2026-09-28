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
