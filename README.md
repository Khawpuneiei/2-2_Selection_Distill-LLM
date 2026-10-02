# 2×2 Selection Distill LLM

An Apply 2 research pipeline for profiling a small math student by concept and
testing whether **poor student performance + high teacher confidence** is the
best place to spend SFT tokens.

The proposal is to compare three equal-token SFT arms—Uniform,
Quadrant-Prioritized, and Confidence-Weighted Loss—using Qwen2.5-0.5B or
Qwen2.5-1.5B as the student. Apply 2 reuses teacher-confidence scores produced
by Apply 1; it does not run the 7B teacher itself. See
[`docs/experiment-design.md`](docs/experiment-design.md) for the concept and
[`DEVLOG.md`](DEVLOG.md) for the setup and decision record.

## Status

**Completed local run (2026-10-03).** The 0.5B student was profiled, mapped
against Apply 1 teacher confidence, and fine-tuned under all three arms
(300k tokens × 3 seeds, plus 100k × 1 seed) on an RTX 4060 Laptop GPU.
Selective SFT did **not** beat uniform SFT at this scale: overall accuracy
was 23.5% (uniform), 22.7% (quadrant-prioritized), and 23.5%
(confidence-weighted), against 17.3% for the base student. See
[`docs/apply2-results-2026-10-03.md`](docs/apply2-results-2026-10-03.md) for
the 2×2 map, ablation table, confidence intervals, token-efficiency plot,
and limitations. Receipts and figures are in
[`results/apply2-rtx4060-20261003/`](results/apply2-rtx4060-20261003/).

The local run used the 0.5B student. The 1.5B student and per-example
teacher scores remain follow-up work; the
[Vast.ai runbook](docs/vast-ai-runbook.md) covers larger GPUs.

## Results (local run, 2026-10-03)

Student `Qwen/Qwen2.5-0.5B` + LoRA; teacher confidence from the Apply 1 run
(`Qwen2.5-Math-7B-Instruct`, self-consistency agreement). Test subsets: GSM8K
500, MATH 350 (50 per subject), GSM-Plus 500 (out-of-distribution, never
trained on). Every arm gets exactly the same number of student training tokens.

### 2×2 map: student need × teacher confidence

![2×2 map of concepts by student accuracy and teacher confidence](results/apply2-rtx4060-20261003/report/quadrant_heatmap.png)

The student is weak on every concept (4–26% profile accuracy), so under the
predeclared thresholds (student ≤ 0.50, teacher ≥ 0.70) only the
poor-student column is filled. **High priority:** GSM8K, algebra, number
theory, precalculus. **Delay:** counting & probability, geometry,
intermediate algebra, prealgebra. MATH teacher confidence rests on only 3
Apply 1 questions per subject.

### Ablation (300k tokens per arm, mean ± sd over 3 seeds)

| Arm | GSM8K | MATH | GSM-Plus (OOD) | Overall |
| --- | ---: | ---: | ---: | ---: |
| Base (no SFT) | 22.6 | 15.7 | 13.2 | 17.3 |
| Uniform | 33.3 ± 0.9 | 13.1 ± 0.5 | **21.1 ± 1.1** | **23.5 ± 0.7** |
| Quadrant-prioritized | 32.5 ± 2.0 | 12.9 ± 0.5 | 19.9 ± 1.1 | 22.7 ± 1.0 |
| Confidence-weighted loss | **34.2 ± 0.9** | 12.5 ± 1.3 | 20.6 ± 2.1 | **23.5 ± 1.4** |

![Accuracy by arm and benchmark](results/apply2-rtx4060-20261003/report/ablation_accuracy.png)

Paired bootstrap, overall accuracy (95% CI):

| Comparison | Δ (points) |
| --- | ---: |
| Uniform − Base | +6.2 [+4.0, +8.3] |
| Quadrant-prioritized − Uniform | −0.8 [−2.1, +0.5] |
| Confidence-weighted − Uniform | 0.0 [−1.0, +1.0] |

### Token efficiency

![Accuracy vs student training tokens and vs teacher tokens](results/apply2-rtx4060-20261003/report/token_efficiency.png)

All arms reach 22.9–23.5% overall within 100k training tokens (1 seed) and
stay flat to 300k. Apply 2 makes no new teacher calls; the right panel counts
the Apply 1 teacher tokens behind the confidence scores each arm relies on.

### Takeaways

- Fine-tuning helps GSM8K (+10 to +12 points) and transfers to GSM-Plus
  (+7 to +8); MATH slips slightly (−3 points) for every arm.
- **Selective SFT did not beat uniform SFT here.** The 2×2 collapsed to one
  column, MATH confidence is thin, and gold-solution targets leave little room
  for teacher reliability to matter.
- Full tables, per-concept accuracy, threshold sensitivity, and limitations:
  [`docs/apply2-results-2026-10-03.md`](docs/apply2-results-2026-10-03.md).

## Setup on Windows

```powershell
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -p "test_*.py" -v
```

The setup script creates a repository-local environment, installs PyTorch
2.5.1 with CUDA 12.1, then installs the pinned Python dependencies. The CUDA
12.1 wheel is chosen for the observed local driver; PyTorch's official
Windows install guidance supports Python 3.9–3.12 and recommends matching the
CUDA build to the machine:
[PyTorch install guide](https://docs.pytorch.org/get-started/locally/).

## Run the experiment

1. Download and normalize the datasets. GSM-Plus is evaluation-only.

   ```powershell
   python -m scripts.prepare_data
   ```

   To add reviewed GSM8K concept labels, fill `id,concept` rows in a copy of
   `data/templates/concept_map.csv` using IDs from
   `data/processed/gsm8k_concept_template.csv`, then pass
   `--concept-map path\to\your_map.csv`.

2. Profile the base student on the held-out profile set.

   ```powershell
   python -m scripts.profile_student --model Qwen/Qwen2.5-0.5B --attempts 3
   ```

3. Put the Apply 1 scores in a private CSV with columns
   `id,concept,confidence,teacher_tokens` (or concept-level rows with blank
   `id`). See [`docs/data-contracts.md`](docs/data-contracts.md).

4. Build the 2×2 map and three matched-token sets.

   ```powershell
   python -m scripts.build_selection --teacher-scores data/processed/apply1_confidence.csv
   ```

5. Train each arm independently from the same student checkpoint.

   ```powershell
   foreach ($arm in @("uniform", "quadrant_prioritized", "confidence_weighted")) {
       python -m scripts.train_lora `
         --input "data/processed/selection/$arm.jsonl" `
         --model Qwen/Qwen2.5-0.5B `
         --output-dir "outputs/adapters/$arm"
   }
   ```

6. Evaluate the unadapted base and each adapter on GSM8K, MATH, and GSM-Plus.
   Then create the heatmap, ablation table, and token-efficiency figure.
   Exact commands are in [`docs/vast-ai-runbook.md`](docs/vast-ai-runbook.md).

Generated datasets, scores, model caches, adapters, and run outputs are ignored
by Git. The repository tracks source, protocol, and schema templates only.

## Checks

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m scripts.check_environment
```

The tests use small local fixtures and do not download models or datasets.
