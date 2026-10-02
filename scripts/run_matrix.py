"""Resumable driver: selection → LoRA training → evaluation for budgets × seeds.

Each step runs in a fresh Python process so GPU memory is released between
arms, and a step is skipped when its receipt already exists.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ARMS = ("uniform", "quadrant_prioritized", "confidence_weighted")


def _run(args: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {' '.join(args[2:])}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {' '.join(args)}\n")
        stream.flush()
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
    if result.returncode != 0:
        raise SystemExit(f"step failed ({result.returncode}); see {log}")
    print(f"    done in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[300000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--teacher-scores", type=Path, default=Path("data/processed/apply1_confidence.csv"))
    parser.add_argument("--profile-metrics", type=Path, default=Path("outputs/student_profile_metrics.jsonl"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/processed/subsets"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--tokens-per-update", type=int, default=4096)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("outputs/matrix"))
    args = parser.parse_args()

    python = sys.executable
    log = args.root / "matrix.log"
    eval_inputs = [str(args.eval_dir / f"{name}.jsonl") for name in ("gsm8k_test", "math_test", "gsm_plus_test")]
    eval_flags = ["--max-new-tokens", str(args.max_new_tokens), "--batch-size", str(args.batch_size),
                  "--max-length", str(args.max_length)]

    if not args.skip_base:
        base_dir = args.root / "evaluation" / "base"
        if not (base_dir / "evaluation_summary.json").exists():
            _run([python, "-m", "scripts.evaluate_model", "--inputs", *eval_inputs, "--model", args.model,
                  "--output-dir", str(base_dir), *eval_flags], log)

    for budget in args.budgets:
        for seed in args.seeds:
            tag = f"b{budget}_s{seed}"
            selection_dir = args.root / "selection" / tag
            manifest = selection_dir / "selection_manifest.json"
            if not manifest.exists():
                _run([python, "-m", "scripts.build_selection", "--teacher-scores", str(args.teacher_scores),
                      "--profile-metrics", str(args.profile_metrics), "--tokenizer", args.model,
                      "--output-dir", str(selection_dir), "--token-budget", str(budget),
                      "--max-length", str(args.max_length), "--seed", str(seed)], log)
            for arm in args.arms:
                adapter = args.root / "adapters" / tag / arm
                if not (adapter / "training_summary.json").exists():
                    _run([python, "-m", "scripts.train_lora", "--input", str(selection_dir / f"{arm}.jsonl"),
                          "--model", args.model, "--output-dir", str(adapter),
                          "--selection-manifest", str(manifest), "--max-length", str(args.max_length),
                          "--tokens-per-update", str(args.tokens_per_update),
                          "--learning-rate", str(args.learning_rate),
                          "--lora-rank", str(args.lora_rank), "--lora-alpha", str(args.lora_alpha),
                          "--target-modules", *args.target_modules, "--seed", str(seed)], log)
                evaluation = args.root / "evaluation" / tag / arm
                if not (evaluation / "evaluation_summary.json").exists():
                    _run([python, "-m", "scripts.evaluate_model", "--inputs", *eval_inputs, "--model", args.model,
                          "--adapter", str(adapter), "--output-dir", str(evaluation), *eval_flags], log)
    print("matrix complete", flush=True)


if __name__ == "__main__":
    main()
