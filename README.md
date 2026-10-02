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
