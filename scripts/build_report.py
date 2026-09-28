from __future__ import annotations

import argparse
from pathlib import Path

from selection_distill.records import read_csv, read_json, write_csv, write_json
from selection_distill.reporting import ARM_ORDER, build_ablation_rows


def _load_evaluations(root: Path) -> dict:
    receipts = {}
    for arm in ARM_ORDER:
        path = root / arm / "evaluation_summary.json"
        if path.exists():
            receipts[arm] = read_json(path)
    return receipts


def _format_percent(value) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _write_markdown(path: Path, rows: list[dict], selection: dict, receipts: dict) -> None:
    lines = [
        "# Apply 2 Ablation Results",
        "",
        "Student-aware selection is compared with uniform SFT using matched active-token budgets.",
        "These results describe only the receipts currently present in the output directory.",
        "",
        "## Run coverage",
        "",
    ]
    for arm in ARM_ORDER:
        training = Path("outputs/adapters") / arm / "training_summary.json"
        training_status = "not applicable (unadapted base)" if arm == "base" else f"training receipt path `{training}`"
        lines.append(f"- **{arm}:** evaluation receipt {'present' if arm in receipts else 'missing'}; {training_status}.")
    lines.extend(
        [
            "",
            "## Accuracy and token use",
            "",
            "| Arm | Dataset | Accuracy | Δ vs uniform | Mean loss | Training tokens | Teacher tokens used |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        delta = row["accuracy_delta_vs_uniform"]
        delta_text = "n/a" if delta is None else f"{delta * 100:+.1f} pp"
        loss_text = "n/a" if row["mean_loss"] is None else f"{float(row['mean_loss']):.4f}"
        training_text = "n/a" if row["training_tokens"] is None else str(row["training_tokens"])
        teacher_text = "unrecorded" if row["teacher_tokens_used"] is None else str(row["teacher_tokens_used"])
        lines.append(
            f"| {row['arm']} | {row['dataset']} | {_format_percent(row['accuracy'])} | "
            f"{delta_text} | {loss_text} | {training_text} | {teacher_text} |"
        )
    lines.extend(
        [
            "",
            "Teacher-token values are not treated as zero when the Apply 1 token receipt is missing.",
            "The per-arm teacher-token figure attributes distinct imported Apply 1 score records to selected examples; those scores are reused, and Apply 2 does not query the teacher again.",
            "Shared teacher-scoring cost for the whole candidate pool is recorded once in the selection manifest, not charged repeatedly to every arm.",
            "",
            "## Figures",
            "",
            "- `quadrant_heatmap.png` shows concept counts and average student accuracy in each quadrant.",
            "- `ablation_accuracy.png` compares benchmark accuracy across SFT arms.",
            "- `token_efficiency.png` relates accuracy to recorded teacher-token use where receipts exist.",
            "",
            "## Interpretation guardrails",
            "",
            "- GSM-Plus is evaluation-only.",
            "- The confidence-weighted arm shares the uniform arm's sample draw and changes the per-example loss weight.",
            "- The quadrant-prioritized arm draws only from poor-student, confident-teacher concepts.",
            "- The plots do not establish statistical significance; repeat seeds and report uncertainty for a final study.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(quadrants_path: Path, rows: list[dict], output_dir: Path, thresholds: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    concepts = read_csv(quadrants_path)
    poor_threshold = thresholds["poor_student_accuracy_at_or_below"]
    confident_threshold = thresholds["teacher_confidence_at_or_above"]
    cells = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for row in concepts:
        poor = float(row["student_accuracy"]) <= poor_threshold
        confident = float(row["teacher_confidence"]) >= confident_threshold
        key = (0 if confident else 1, 0 if poor else 1)
        cells[key].append(row)
    heat = np.zeros((2, 2), dtype=float)
    for (y, x), items in cells.items():
        heat[y, x] = len(items)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(heat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks([0, 1], ["Poor student", "Good student"])
    ax.set_yticks([0, 1], ["Confident teacher", "Uncertain teacher"])
    ax.set_xlabel("Student need by concept accuracy")
    ax.set_ylabel("Teacher confidence")
    ax.set_title("Student-need × teacher-confidence map")
    labels = {
        (0, 0): "HIGH PRIORITY\nPoor + confident",
        (1, 0): "LOW PRIORITY\nGood + confident",
        (0, 1): "DELAY\nPoor + uncertain",
        (1, 1): "EXCLUDE\nGood + uncertain",
    }
    for (y, x), items in cells.items():
        mean_accuracy = sum(float(row["student_accuracy"]) for row in items) / len(items) if items else 0
        mean_confidence = sum(float(row["teacher_confidence"]) for row in items) / len(items) if items else 0
        annotation = (
            f"{labels[(y, x)]}\n{len(items)} concepts\n"
            f"mean accuracy {mean_accuracy:.2f}\nmean confidence {mean_confidence:.2f}"
        )
        ax.text(x, y, annotation, ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Number of concepts")
    fig.tight_layout()
    fig.savefig(output_dir / "quadrant_heatmap.png", dpi=180)
    plt.close(fig)

    datasets = sorted({row["dataset"] for row in rows})
    arms_present = [arm for arm in ARM_ORDER if any(row["arm"] == arm for row in rows)]
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(datasets))
    width = 0.8 / max(1, len(arms_present))
    for index, arm in enumerate(arms_present):
        by_dataset = {row["dataset"]: row["accuracy"] for row in rows if row["arm"] == arm}
        values = [by_dataset.get(dataset, float("nan")) for dataset in datasets]
        offset = (index - (len(arms_present) - 1) / 2) * width
        ax.bar(positions + offset, values, width, label=arm)
    ax.set_xticks(positions, datasets, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Matched-token SFT ablation")
    if arms_present:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for dataset in datasets:
        for arm in arms_present:
            row = next((item for item in rows if item["dataset"] == dataset and item["arm"] == arm), None)
            if row and row["teacher_token_receipt_complete"] and row["teacher_tokens_used"] is not None:
                ax.scatter(row["teacher_tokens_used"], row["accuracy"], label=f"{arm} · {dataset}", s=60)
                plotted += 1
    ax.set_xlabel("Teacher tokens used (distinct confidence-score records)")
    ax.set_ylabel("Benchmark accuracy")
    ax.set_title("Accuracy per teacher-token budget")
    ax.set_ylim(0, 1)
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Apply 1 teacher-token receipts are missing; values were not treated as zero.",
                ha="center", va="center", transform=ax.transAxes, wrap=True)
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "token_efficiency.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tables and figures from experiment receipts")
    parser.add_argument("--selection-manifest", type=Path, default=Path("data/processed/selection/selection_manifest.json"))
    parser.add_argument("--quadrants", type=Path, default=Path("data/processed/selection/concept_quadrants.csv"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report"))
    args = parser.parse_args()
    selection = read_json(args.selection_manifest)
    evaluations = _load_evaluations(args.evaluation_dir)
    if not evaluations:
        raise SystemExit(f"No evaluation_summary.json receipts found under {args.evaluation_dir}")
    rows = build_ablation_rows(selection, evaluations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "ablation_results.csv", rows,
        [
            "arm", "dataset", "accuracy", "accuracy_delta_vs_uniform", "mean_entropy",
            "mean_loss", "examples", "training_tokens", "teacher_tokens_used",
            "teacher_token_receipt_complete",
        ],
    )
    _make_figures(args.quadrants, rows, args.output_dir, selection["thresholds"])
    _write_markdown(args.output_dir / "experiment_report.md", rows, selection, evaluations)
    write_json(args.output_dir / "report_manifest.json", {
        "selection_manifest": str(args.selection_manifest),
        "evaluation_dir": str(args.evaluation_dir),
        "arms_evaluated": sorted(evaluations),
        "datasets": sorted({row["dataset"] for row in rows}),
        "results": str(args.output_dir / "ablation_results.csv"),
    })
    print(f"Wrote report and figures under {args.output_dir}")


if __name__ == "__main__":
    main()
