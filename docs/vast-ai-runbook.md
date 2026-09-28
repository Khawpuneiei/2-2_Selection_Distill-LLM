# Vast.ai runbook

The current local GPU has 8 GiB VRAM. The supplied brief estimates 16–24 GiB
for the experiment runs, so use a Vast.ai GPU in that range or larger for the
full ablation, especially if running the 1.5B student. The teacher is not
loaded in Apply 2 when the Apply 1 scores are available.

## Start the instance

1. Select an Ubuntu image with Python 3.10+ and a CUDA-capable PyTorch build.
2. Attach persistent disk for Hugging Face cache, datasets, and adapters; plan
   for at least 40 GiB free space before downloading models and data.
3. Clone the repository, verify CUDA, then install the pinned project
   dependencies:

```bash
git clone https://github.com/Khawpuneiei/2-2_Selection_Distill-LLM.git
cd 2-2_Selection_Distill-LLM
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
python -m pip install -r requirements.txt
python -m scripts.check_environment --require-cuda --min-vram-gb 16
```

If the image's PyTorch is older than 2.1, install a CUDA-matched PyTorch wheel
from the [official PyTorch selector](https://pytorch.org/get-started/locally/)
before installing the remaining dependencies. Do not install a wheel whose
CUDA runtime exceeds the instance driver's support.

## Data and score inputs

Copy the Apply 1 confidence CSV to a private location such as
`data/processed/apply1_confidence.csv`. Do not commit it. If the GSM8K labels
need fine-grained concepts, review and fill the generated
`gsm8k_concept_template.csv`, then pass it as `--concept-map`.

```bash
python -m scripts.prepare_data --concept-map data/processed/gsm8k_concepts.csv
```

Prepare and review the student's profile and matched token arms:

```bash
python -m scripts.profile_student --model Qwen/Qwen2.5-0.5B --attempts 3
python -m scripts.build_selection --teacher-scores data/processed/apply1_confidence.csv
```

## Train, evaluate, report

Train each arm from the same base checkpoint. The adapters save under
`outputs/adapters/` and should stay on the persistent disk.

```bash
for arm in uniform quadrant_prioritized confidence_weighted; do
  python -m scripts.train_lora \
    --input "data/processed/selection/${arm}.jsonl" \
    --model Qwen/Qwen2.5-0.5B \
    --output-dir "outputs/adapters/${arm}"
done
```

Evaluate the unadapted base and each adapter on all three fixed test sets:

```bash
python -m scripts.evaluate_model \
  --inputs data/processed/gsm8k_test.jsonl data/processed/math_test.jsonl data/processed/gsm_plus_test.jsonl \
  --model Qwen/Qwen2.5-0.5B --output-dir outputs/evaluation/base

for arm in uniform quadrant_prioritized confidence_weighted; do
  python -m scripts.evaluate_model \
    --inputs data/processed/gsm8k_test.jsonl data/processed/math_test.jsonl data/processed/gsm_plus_test.jsonl \
    --model Qwen/Qwen2.5-0.5B --adapter "outputs/adapters/${arm}" \
    --output-dir "outputs/evaluation/${arm}"
done

python -m scripts.build_report
```

Before shutting down, copy only the small result receipts and figures needed
for analysis. Keep model weights, adapters, private confidence files, and
downloaded datasets out of Git. Record the instance GPU, driver, actual run
duration, and any OOM or retry in `DEVLOG.md` and the generated run summaries.
