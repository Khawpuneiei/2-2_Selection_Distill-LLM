"""Aggregate run_matrix receipts into the Apply 2 tables, figures, and report."""

from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

from selection_distill.records import read_csv, read_json, read_jsonl, write_csv, write_json

ARMS = ("uniform", "quadrant_prioritized", "confidence_weighted")
SPLITS = ("gsm8k_test", "math_test", "gsm_plus_test")
LABELS = {
    "base": "Base (no SFT)",
    "uniform": "Uniform",
    "quadrant_prioritized": "Quadrant-prioritized",
    "confidence_weighted": "Confidence-weighted loss",
}
COLORS = {"base": "#8a8a8a", "uniform": "#4c72b0", "quadrant_prioritized": "#dd8452",
          "confidence_weighted": "#55a868"}


def _load_correct(evaluation_dir: Path) -> dict[str, dict[str, bool]]:
    """split -> question id -> correct (attempt 1)."""
    result = {}
    for split in SPLITS:
        path = evaluation_dir / f"{split}.metrics.jsonl"
        if path.exists():
            result[split] = {row["id"]: bool(row["correct"]) for row in read_jsonl(path) if row["attempt"] == 1}
    return result


def _per_question(runs: list[dict[str, dict[str, bool]]], split: str | None) -> dict[str, float]:
    """Mean correctness per question across seeds; split None = all splits."""
    totals: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for name, rows in run.items():
            if split is not None and name != split:
                continue
            for qid, correct in rows.items():
                totals[f"{name}:{qid}"].append(float(correct))
    return {qid: sum(values) / len(values) for qid, values in totals.items()}


def _paired_bootstrap(a: dict[str, float], b: dict[str, float], samples: int, seed: int):
    keys = sorted(set(a) & set(b))
    diffs = [a[key] - b[key] for key in keys]
    if not diffs:
        return None, None, None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return sum(diffs) / n, means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def _accuracy(run: dict[str, dict[str, bool]], split: str | None) -> float:
    values = [c for name, rows in run.items() if split in (None, name) for c in rows.values()]
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/matrix"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/matrix/report"))
    parser.add_argument("--teacher-scores", type=Path, default=Path("data/processed/apply1_confidence.csv"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    base_run = _load_correct(args.root / "evaluation" / "base")
    tags = sorted(p.name for p in (args.root / "evaluation").iterdir() if p.is_dir() and p.name != "base")
    runs: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    manifests: dict[str, dict] = {}
    training: dict[tuple[str, str], dict] = {}
    seeds_by_budget: dict[int, list[int]] = defaultdict(list)
    for tag in tags:
        budget = int(tag.split("_")[0][1:])
        seed = int(tag.split("_s")[1])
        complete = all((args.root / "evaluation" / tag / arm / "evaluation_summary.json").exists() for arm in ARMS)
        if not complete:
            print(f"skipping incomplete {tag}")
            continue
        seeds_by_budget[budget].append(seed)
        manifests[tag] = read_json(args.root / "selection" / tag / "selection_manifest.json")
        for arm in ARMS:
            runs[budget][arm].append(_load_correct(args.root / "evaluation" / tag / arm))
            training[(tag, arm)] = read_json(args.root / "adapters" / tag / arm / "training_summary.json")
    if not runs:
        raise SystemExit("no complete budget/seed runs found")
    budgets = sorted(runs)
    main_budget = max(budgets, key=lambda b: (len(seeds_by_budget[b]), b))
    first_tag = next(tag for tag in manifests if tag.startswith(f"b{main_budget}_"))
    quadrants = read_csv(args.root / "selection" / first_tag / "concept_quadrants.csv")
    thresholds = manifests[first_tag]["thresholds"]
    teacher_rows = {row["concept"]: row for row in read_csv(args.teacher_scores)}

    # ---- ablation table -------------------------------------------------
    table = []
    for budget in budgets:
        for arm in ("base",) + ARMS:
            arm_runs = [base_run] if arm == "base" else runs[budget][arm]
            row = {"budget": budget, "arm": arm, "seeds": len(arm_runs)}
            for split in SPLITS + (None,):
                key = split or "overall"
                accs = [_accuracy(run, split) for run in arm_runs]
                row[f"{key}_mean"] = statistics.mean(accs)
                row[f"{key}_sd"] = statistics.stdev(accs) if len(accs) > 1 else None
            macro = [statistics.mean(_accuracy(run, s) for s in SPLITS) for run in arm_runs]
            row["macro_mean"] = statistics.mean(macro)
            if arm != "base":
                tags_b = [t for t in manifests if t.startswith(f"b{budget}_")]
                stats = [manifests[t]["arms"][arm] for t in tags_b]
                row["training_tokens"] = stats[0]["training_tokens"]
                row["unique_examples"] = round(statistics.mean(s["unique_examples"] for s in stats))
                row["reused_rows"] = round(statistics.mean(s["reused_rows"] for s in stats))
                row["teacher_tokens_used"] = stats[0]["teacher_tokens_used"]
                row["train_seconds_mean"] = statistics.mean(training[(t, arm)]["elapsed_seconds"] for t in tags_b)
            else:
                row.update(training_tokens=0, unique_examples=0, reused_rows=0, teacher_tokens_used=0,
                           train_seconds_mean=0)
            table.append(row)
    write_csv(out / "ablation_results.csv", table, list(table[0].keys()))

    # ---- paired bootstrap vs uniform and vs base -------------------------
    comparisons = []
    for budget in budgets:
        for split in SPLITS + (None,):
            uni = _per_question(runs[budget]["uniform"], split)
            base = _per_question([base_run], split)
            pairs = [("quadrant_prioritized", "uniform"), ("confidence_weighted", "uniform"),
                     ("uniform", "base"), ("quadrant_prioritized", "base"), ("confidence_weighted", "base")]
            for a_name, b_name in pairs:
                a = _per_question(runs[budget][a_name], split)
                b = uni if b_name == "uniform" else base
                delta, low, high = _paired_bootstrap(a, b, args.bootstrap, seed=budget)
                comparisons.append({"budget": budget, "split": split or "overall", "comparison": f"{a_name} - {b_name}",
                                    "delta": delta, "ci_low": low, "ci_high": high, "n_questions": len(set(a) & set(b))})
    write_csv(out / "paired_bootstrap.csv", comparisons, list(comparisons[0].keys()))

    # ---- per-concept accuracy (MATH subjects + GSM8K) at main budget ------
    quad_by_concept = {row["concept"]: row["quadrant"] for row in quadrants}
    concept_rows = []

    def concept_acc(run_list, concept):
        values = []
        for run in run_list:
            for split in ("gsm8k_test", "math_test"):
                path_rows = run.get(split, {})
                values.extend(c for qid, c in path_rows.items() if concept_of.get(qid) == concept)
        return sum(values) / len(values) if values else None, len(values) // max(1, len(run_list))

    concept_of = {}
    for split in ("gsm8k_test", "math_test"):
        for row in read_jsonl(Path("data/processed/subsets") / f"{split}.jsonl"):
            concept_of[row["id"]] = row["concept"]
    for concept in sorted(quad_by_concept):
        entry = {"concept": concept, "quadrant": quad_by_concept[concept]}
        base_acc, n = concept_acc([base_run], concept)
        entry["n_test"] = n
        entry["base"] = base_acc
        for arm in ARMS:
            entry[arm] = concept_acc(runs[main_budget][arm], concept)[0]
        entry["quadrant_minus_uniform"] = entry["quadrant_prioritized"] - entry["uniform"]
        concept_rows.append(entry)
    write_csv(out / "per_concept_accuracy.csv", concept_rows, list(concept_rows[0].keys()))

    # ---- threshold sensitivity: median split (relabeling only, no retraining) ---
    from selection_distill.selection import classify_quadrant

    median_acc = statistics.median(float(row["student_accuracy"]) for row in quadrants)
    median_conf = statistics.median(float(row["teacher_confidence"]) for row in quadrants)
    sensitivity = [
        {
            "concept": row["concept"],
            "student_accuracy": float(row["student_accuracy"]),
            "teacher_confidence": float(row["teacher_confidence"]),
            "predeclared_quadrant": row["quadrant"],
            "median_split_quadrant": classify_quadrant(
                float(row["student_accuracy"]), float(row["teacher_confidence"]),
                poor_accuracy_threshold=median_acc, teacher_confidence_threshold=median_conf,
            ),
        }
        for row in quadrants
    ]
    write_csv(out / "threshold_sensitivity.csv", sensitivity, list(sensitivity[0].keys()))

    # ---- figures ----------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    poor = thresholds["poor_student_accuracy_at_or_below"]
    confident = thresholds["teacher_confidence_at_or_above"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.axvspan(0, poor, ymin=confident, ymax=1, color="#dd8452", alpha=0.18)
    ax.axvspan(0, poor, ymin=0, ymax=confident, color="#c4ad66", alpha=0.15)
    ax.axvspan(poor, 1, ymin=confident, ymax=1, color="#4c72b0", alpha=0.10)
    ax.axvspan(poor, 1, ymin=0, ymax=confident, color="#8a8a8a", alpha=0.10)
    ax.axvline(poor, color="black", lw=1, ls="--")
    ax.axhline(confident, color="black", lw=1, ls="--")
    for row in quadrants:
        x, y = float(row["student_accuracy"]), float(row["teacher_confidence"])
        n_teacher = teacher_rows.get(row["concept"], {}).get("apply1_questions", "?")
        ax.scatter(x, y, s=40 + 4 * int(row["student_examples"]) ** 0.5 * 8, color=COLORS["quadrant_prioritized"]
                   if row["quadrant"] == "high_priority" else "#555555", alpha=0.8, edgecolor="black")
        ax.annotate(f"{row['concept']}\n(teacher n={n_teacher})", (x, y), textcoords="offset points",
                    xytext=(8, -18) if row["concept"] == "gsm8k_word_problem" else (8, -4), fontsize=8)
    ax.text(0.01, 0.99, "HIGH PRIORITY\npoor + confident", va="top", fontsize=9, weight="bold")
    ax.text(0.01, 0.02, "DELAY\npoor + uncertain", va="bottom", fontsize=9, weight="bold")
    ax.text(0.99, 0.99, "LOW PRIORITY\ngood + confident", va="top", ha="right", fontsize=9, weight="bold")
    ax.text(0.99, 0.02, "EXCLUDE\ngood + uncertain", va="bottom", ha="right", fontsize=9, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Student accuracy on profile set (Qwen2.5-0.5B base, 3 sampled attempts)")
    ax.set_ylabel("Teacher confidence (Apply 1 self-consistency agreement)")
    ax.set_title("2×2 map: student need × teacher confidence, per concept")

    ax = axes[1]
    cells = {"high_priority": (0, 0), "low_priority": (0, 1), "delay": (1, 0), "exclude": (1, 1)}
    heat = np.zeros((2, 2))
    text = {key: [] for key in cells}
    pool_counts = manifests[first_tag].get("pool_rows_by_quadrant") or {}
    for row in quadrants:
        heat[cells[row["quadrant"]]] += 1
        text[row["quadrant"]].append(row["concept"])
    ax.imshow(heat, cmap="Oranges", vmin=0, vmax=max(1, heat.max()) * 1.3)
    for quadrant, (y, x) in cells.items():
        body = "\n".join(text[quadrant]) or "—"
        extra = f"\npool rows: {pool_counts[quadrant]}" if quadrant in pool_counts else ""
        ax.text(x, y, f"{quadrant.replace('_', ' ').upper()}\n{int(heat[y, x])} concepts{extra}\n\n{body}",
                ha="center", va="center", fontsize=8)
    ax.set_xticks([0, 1], ["Poor student", "Good student"])
    ax.set_yticks([0, 1], ["Confident teacher", "Uncertain teacher"])
    ax.set_title(f"Quadrant cells (poor ≤ {poor:.2f}, confident ≥ {confident:.2f})")
    fig.tight_layout()
    fig.savefig(out / "quadrant_heatmap.png", dpi=170)
    plt.close(fig)

    # Ablation bars at the main budget with seed error bars.
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = list(SPLITS) + ["overall"]
    positions = np.arange(len(groups))
    width = 0.2
    for index, arm in enumerate(("base",) + ARMS):
        row = next(r for r in table if r["budget"] == main_budget and r["arm"] == arm)
        means = [row[f"{g}_mean"] for g in groups]
        errors = [row[f"{g}_sd"] or 0 for g in groups]
        ax.bar(positions + (index - 1.5) * width, means, width, yerr=errors, capsize=3,
               label=LABELS[arm], color=COLORS[arm])
    ax.set_xticks(positions, ["GSM8K", "MATH", "GSM-Plus (OOD)", "Overall"])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Matched-token SFT ablation ({main_budget:,} tokens/arm, "
                 f"{len(seeds_by_budget[main_budget])} seeds, ±1 sd)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ablation_accuracy.png", dpi=170)
    plt.close(fig)

    # Token efficiency: accuracy vs SFT tokens, and vs teacher tokens.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arm in ARMS:
        xs = [0] + budgets
        for split, style in (("overall", "-"), ("gsm_plus_test", ":")):
            ys = [next(r for r in table if r["arm"] == "base")[f"{split}_mean"]]
            ys += [next(r for r in table if r["budget"] == b and r["arm"] == arm)[f"{split}_mean"] for b in budgets]
            axes[0].plot(xs, ys, style, marker="o", color=COLORS[arm],
                         label=f"{LABELS[arm]} · {'overall' if split == 'overall' else 'GSM-Plus'}")
    seed_note = ", ".join(f"{b // 1000}k: {len(seeds_by_budget[b])} seed(s)" for b in budgets)
    axes[0].set_xlabel(f"SFT training tokens per arm (student tokens; {seed_note})")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs student training tokens")
    axes[0].legend(fontsize=7)
    for arm in ("base",) + ARMS:
        row = next(r for r in table if r["budget"] == main_budget and r["arm"] == arm)
        tokens = row["teacher_tokens_used"]
        if tokens is not None:
            if arm == "uniform":  # shares its draw and teacher records with the weighted arm
                axes[1].scatter(tokens, row["overall_mean"], s=260, facecolors="none", linewidths=2,
                                edgecolors=COLORS[arm], label=LABELS[arm], zorder=3)
            else:
                axes[1].scatter(tokens, row["overall_mean"], s=90, color=COLORS[arm], label=LABELS[arm], zorder=2)
    axes[1].set_xlabel("Apply 1 teacher tokens behind the confidence records each arm relies on")
    axes[1].set_ylabel("Overall accuracy")
    axes[1].set_title(f"Accuracy vs teacher tokens ({main_budget:,}-token arms)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "token_efficiency.png", dpi=170)
    plt.close(fig)

    write_json(out / "results.json", {
        "main_budget": main_budget,
        "budgets": budgets,
        "seeds_by_budget": {str(k): sorted(v) for k, v in seeds_by_budget.items()},
        "thresholds": thresholds,
        "quadrants": quadrants,
        "ablation": table,
        "paired_bootstrap": comparisons,
        "per_concept": concept_rows,
        "median_split": {"student_accuracy": median_acc, "teacher_confidence": median_conf,
                         "rows": sensitivity},
        "selection_manifests": manifests,
    })
    print(f"Wrote report tables and figures to {out}")


if __name__ == "__main__":
    main()
