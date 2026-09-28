# Experiment design: where to teach

## Research question

Does selecting examples from concepts that the student handles poorly and the
teacher handles confidently improve student accuracy over uniform supervised
fine-tuning (SFT), when each training arm consumes the same number of student
tokens?

The proposal calls this **Apply 2: Student Weakness Profiling + 2×2 Data
Selection**. It is the data-selection half of the larger Where + When study.

## Models and datasets

- **Student:** base `Qwen/Qwen2.5-0.5B` or `Qwen/Qwen2.5-1.5B`, adapted with
  LoRA. The Qwen2.5 model card lists the 0.5B base checkpoint and Apache-2.0
  license: [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B).
- **Teacher:** `Qwen/Qwen2.5-Math-7B` from Apply 1. Apply 2 reads the saved
  confidence output; it does not load or query the teacher again. The teacher
  model card describes its math-reasoning focus and requires Transformers
  4.37.0 or later: [model card](https://huggingface.co/Qwen/Qwen2.5-Math-7B).
- **Training pool:** GSM8K train plus MATH train. MATH provides subject and
  subtype fields; the dataset card lists subject configurations and
  `problem`, `level`, `type`, and `solution` fields:
  [MATH card](https://huggingface.co/datasets/EleutherAI/hendrycks_math).
- **Evaluation:** GSM8K test, MATH test, and GSM-Plus test. The GSM8K card
  exposes only `question` and `answer`, so fine-grained GSM8K concepts require
  an explicit concept map or another reviewed annotation source:
  [GSM8K card](https://huggingface.co/datasets/openai/gsm8k).
- **OOD:** GSM-Plus is downloaded only as an evaluation set. Its dataset card
  says the dataset was designed as a test set and that training on it is
  prohibited: [GSM-Plus card](https://huggingface.co/datasets/qintongli/GSM-Plus).

Raw datasets and model weights are not committed to this repository.

## Profiling and 2×2 map

The initial student is evaluated on a stable 10% per-concept holdout taken
from the source training splits. Benchmark test questions never enter this
profile holdout or the SFT pool.

For each concept, the profile records:

- **Accuracy:** correct generated answers divided by all recorded attempts.
- **Entropy:** mean next-token Shannon entropy in nats over generated tokens.
- **Loss:** mean causal-language-model cross-entropy over answer tokens, with
  question tokens masked from the loss.
- **Repeated failures:** unique questions answered incorrectly on every
  recorded attempt.

The default quadrant boundaries are predeclared as student accuracy at or
below `0.50` = poor and teacher confidence at or above `0.70` = confident.
They are CLI-configurable; any final paper should report sensitivity analyses
around them. Ties belong to the poor/confident side.

| Student need | Teacher confidence | Quadrant | Apply 2 treatment |
|---|---|---|---|
| Poor | Confident | High priority | Candidate concepts for selective SFT |
| Poor | Uncertain | Delay | Leave out until the teacher is more reliable |
| Good | Confident | Low priority | Exclude from student-aware selection |
| Good | Uncertain | Exclude | No current teaching need and weak teacher signal |

The profiling student accuracy is computed over a held-out subset, not over
training examples. Confidence is either an Apply 1 per-concept score or the
mean of the supplied Apply 1 scores for examples in that concept.

## Three matched-token SFT arms

1. **Uniform:** deterministic random sampling from the full training pool.
2. **Quadrant-prioritized:** sampling only examples whose concept is poor for
   the student and confident for the teacher.
3. **Confidence-weighted loss:** the same example draw as Uniform, with each
   example's SFT loss multiplied by its teacher confidence divided by the
   training-pool mean confidence.

All three arms receive the same number of active student tokens. If an arm has
fewer unique examples than needed, its examples repeat; the output marks every
reused row. The final exposure is truncated at a token boundary when needed.
These choices are recorded in the selection manifest so oversampling and
truncation are visible.

All arms use the same base student, tokenizer, seed, maximum sequence length,
LoRA configuration, learning rate, and token-based gradient accumulation.
The current implementation uses ordinary FP16 LoRA on CUDA. It does not claim
QLoRA or run a 7B teacher locally. PEFT documents LoRA as frozen base weights
plus small trainable low-rank matrices:
[PEFT LoRA guide](https://huggingface.co/docs/peft/en/package_reference/lora).

## Metrics and artifacts

The evaluation command produces per-question generations, exact/scalar answer
matches, token entropy, answer-token loss, concept summaries, and per-dataset
accuracy. The report command generates:

- `quadrant_heatmap.png`: concept counts and mean student accuracy for each
  student-need × teacher-confidence cell.
- `ablation_accuracy.png`: GSM8K, MATH, and GSM-Plus accuracy across the base
  model and three SFT arms.
- `token_efficiency.png`: accuracy against recorded teacher-token cost.
- `ablation_results.csv` and `experiment_report.md`.

Teacher confidence scores are imported from Apply 1. Apply 2 adds no teacher
inference calls. The manifest records a shared teacher-scoring-pool cost once
and per-arm attributed score-token counts when `teacher_tokens` receipts are
present. Missing token receipts stay unknown; they are never silently changed
to zero. Statistical uncertainty and multiple seeds are still required for a
publication-quality conclusion.

## Source references

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
- [GSM-Plus](https://huggingface.co/datasets/qintongli/GSM-Plus)
- [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B)
- [Hugging Face PEFT integration](https://huggingface.co/docs/transformers/peft)
- [PyTorch previous-version install commands](https://docs.pytorch.org/get-started/previous-versions/)
