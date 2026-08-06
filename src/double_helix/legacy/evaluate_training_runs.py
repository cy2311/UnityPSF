from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


@dataclass(frozen=True)
class RunSpec:
    name: str
    density_um2: float
    run_root: Path
    prefix_run_root: Path | None = None


@dataclass(frozen=True)
class RunEvaluation:
    spec: RunSpec
    training_rows: tuple[dict[str, Any], ...]
    eval_rows: tuple[dict[str, Any], ...]
    gamma_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class EvaluationArtifacts:
    summary_json: Path
    summary_csv: Path
    report_markdown: Path
    localization_png: Path
    localization_pdf: Path
    gamma_png: Path
    gamma_pdf: Path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON objects in {path}")
            rows.append(payload)
    return rows


def _merge_rows(paths: list[Path], *, key) -> tuple[dict[str, Any], ...]:
    merged: dict[Any, dict[str, Any]] = {}
    for path in paths:
        for row in _load_jsonl(path):
            merged[key(row)] = row
    return tuple(merged[item] for item in sorted(merged))


def _metric_paths(spec: RunSpec, name: str) -> list[Path]:
    paths = []
    if spec.prefix_run_root is not None:
        prefix = spec.prefix_run_root / "metrics" / name
        if prefix.is_file():
            paths.append(prefix)
    paths.append(spec.run_root / "metrics" / name)
    return paths


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "path": str(path.resolve()),
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "eval_loss": None if checkpoint.get("eval_loss") is None else float(checkpoint["eval_loss"]),
    }


def _eval_summary(row: dict[str, Any], *, steps_per_epoch: int) -> dict[str, Any]:
    keys = ("global_step", "eval_loss", "jaccard", "precision", "recall", "rmse_lat", "rmse_ax")
    output = {key: row.get(key) for key in keys}
    output["epoch"] = int(row["global_step"]) // int(steps_per_epoch)
    return output


def evaluate_run(
    spec: RunSpec,
    *,
    expected_epochs: int = 300,
    steps_per_epoch: int = 417,
    expected_gamma_updates: int = 28,
    require_complete: bool = True,
) -> RunEvaluation:
    training_rows = _merge_rows(
        _metric_paths(spec, "training_metrics.jsonl"),
        key=lambda row: int(row["global_step"]),
    )
    eval_rows = _merge_rows(
        _metric_paths(spec, "eval_metrics.jsonl"),
        key=lambda row: int(row["global_step"]),
    )
    gamma_rows = _merge_rows(
        _metric_paths(spec, "gamma_update_metrics.jsonl"),
        key=lambda row: (int(row["epoch"]), str(row.get("domain_name", ""))),
    )
    expected_steps = int(expected_epochs) * int(steps_per_epoch)
    completed_epochs = int(eval_rows[-1]["global_step"]) // int(steps_per_epoch)
    if require_complete:
        if int(training_rows[-1]["global_step"]) != expected_steps:
            raise ValueError(f"{spec.name} training is incomplete")
        if completed_epochs != int(expected_epochs) or len(eval_rows) != int(expected_epochs):
            raise ValueError(f"{spec.name} evaluation is incomplete")
        if len(gamma_rows) != int(expected_gamma_updates):
            raise ValueError(f"{spec.name} gamma updates are incomplete")

    best_jaccard = max(eval_rows, key=lambda row: float(row["jaccard"]))
    best_eval_loss = min(eval_rows, key=lambda row: float(row["eval_loss"]))
    checkpoints_dir = spec.run_root / "checkpoints"
    summary = {
        "name": spec.name,
        "density_um2": float(spec.density_um2),
        "completed_epochs": completed_epochs,
        "training_steps": int(training_rows[-1]["global_step"]),
        "eval_rows": len(eval_rows),
        "gamma_updates": len(gamma_rows),
        "final": _eval_summary(eval_rows[-1], steps_per_epoch=steps_per_epoch),
        "best_jaccard": _eval_summary(best_jaccard, steps_per_epoch=steps_per_epoch),
        "best_eval_loss": _eval_summary(best_eval_loss, steps_per_epoch=steps_per_epoch),
        "checkpoints": {
            "best": _checkpoint_metadata(checkpoints_dir / "checkpoint_best.pt"),
            "latest": _checkpoint_metadata(checkpoints_dir / "checkpoint_latest.pt"),
        },
    }
    return RunEvaluation(spec, training_rows, eval_rows, gamma_rows, summary)


_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
_LINESTYLES = ("-", "--", "-.", ":")


def _style_axis(ax, *, xlabel: str = "Epoch") -> None:
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, png_path: Path, pdf_path: Path) -> None:
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_localization_metrics(
    evaluations: list[RunEvaluation],
    *,
    steps_per_epoch: int,
    png_path: Path,
    pdf_path: Path,
) -> None:
    panels = (
        ("jaccard", "Jaccard"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("rmse_lat", "RMSE xy (nm)"),
        ("rmse_ax", "RMSE z (nm)"),
        ("eval_loss", "Evaluation loss"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), constrained_layout=True)
    for index, evaluation in enumerate(evaluations):
        epochs = [float(row["global_step"]) / float(steps_per_epoch) for row in evaluation.eval_rows]
        label = f"{evaluation.spec.density_um2:g} emitter/um2"
        for ax, (key, ylabel) in zip(axes.flat, panels):
            values = [float(row[key]) for row in evaluation.eval_rows]
            ax.plot(
                epochs,
                values,
                color=_COLORS[index % len(_COLORS)],
                linestyle=_LINESTYLES[index % len(_LINESTYLES)],
                linewidth=1.5,
                label=label,
            )
            ax.set_ylabel(ylabel)
            _style_axis(ax)
        best = evaluation.summary["best_jaccard"]
        axes.flat[0].scatter(
            [best["epoch"]],
            [best["jaccard"]],
            marker="*",
            s=70,
            color=_COLORS[index % len(_COLORS)],
            edgecolor="black",
            linewidth=0.4,
            zorder=4,
        )
    axes.flat[0].legend(frameon=False, fontsize=8)
    for label, ax in zip("ABCDEF", axes.flat):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
    _save_figure(fig, png_path, pdf_path)


def _plot_gamma_metrics(
    evaluations: list[RunEvaluation],
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:
    panels = (
        ("best_loss", "Gamma objective"),
        ("gamma_after_norm", "Gamma norm after update"),
        ("gamma_delta_norm", "Gamma delta norm"),
        ("selected_sampled_emitter_count", "Sampled emitters"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), constrained_layout=True)
    for index, evaluation in enumerate(evaluations):
        epochs = [int(row["epoch"]) for row in evaluation.gamma_rows]
        label = f"{evaluation.spec.density_um2:g} emitter/um2"
        for ax, (key, ylabel) in zip(axes.flat, panels):
            values = [float(row.get(key, float("nan"))) for row in evaluation.gamma_rows]
            ax.plot(
                epochs,
                values,
                color=_COLORS[index % len(_COLORS)],
                linestyle=_LINESTYLES[index % len(_LINESTYLES)],
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                label=label,
            )
            ax.set_ylabel(ylabel)
            _style_axis(ax)
    axes.flat[0].legend(frameon=False, fontsize=8)
    for label, ax in zip("ABCD", axes.flat):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
    _save_figure(fig, png_path, pdf_path)


def _summary_csv_rows(evaluations: list[RunEvaluation]) -> list[dict[str, Any]]:
    rows = []
    for evaluation in evaluations:
        summary = evaluation.summary
        final = summary["final"]
        best = summary["best_jaccard"]
        rows.append(
            {
                "name": summary["name"],
                "density_um2": summary["density_um2"],
                "completed_epochs": summary["completed_epochs"],
                "gamma_updates": summary["gamma_updates"],
                "final_jaccard": final["jaccard"],
                "final_precision": final["precision"],
                "final_recall": final["recall"],
                "final_rmse_xy_nm": final["rmse_lat"],
                "final_rmse_z_nm": final["rmse_ax"],
                "best_jaccard_epoch": best["epoch"],
                "best_jaccard": best["jaccard"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "best_rmse_xy_nm": best["rmse_lat"],
                "best_rmse_z_nm": best["rmse_ax"],
                "checkpoint_best": summary["checkpoints"]["best"]["path"],
                "checkpoint_latest": summary["checkpoints"]["latest"]["path"],
            }
        )
    return rows


def _report_markdown(evaluations: list[RunEvaluation]) -> str:
    lines = [
        "# Double-helix training evaluation",
        "",
        "## Final metrics",
        "",
        "| Density (emitter/um2) | Epoch | Jaccard | Precision | Recall | RMSE xy (nm) | RMSE z (nm) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for evaluation in evaluations:
        row = evaluation.summary["final"]
        lines.append(
            f"| {evaluation.spec.density_um2:g} | {row['epoch']} | {row['jaccard']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['rmse_lat']:.2f} | {row['rmse_ax']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Best Jaccard by epoch",
            "",
            "| Density (emitter/um2) | Epoch | Jaccard | Precision | Recall | RMSE xy (nm) | RMSE z (nm) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for evaluation in evaluations:
        row = evaluation.summary["best_jaccard"]
        lines.append(
            f"| {evaluation.spec.density_um2:g} | {row['epoch']} | {row['jaccard']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['rmse_lat']:.2f} | {row['rmse_ax']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The best-Jaccard epoch is a historical metric unless a checkpoint exists at that exact epoch. "
            "Use the checkpoint paths recorded in summary.json for reproducible inference.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_package(
    evaluations: list[RunEvaluation],
    output_dir: str | Path,
    *,
    steps_per_epoch: int = 417,
) -> EvaluationArtifacts:
    output = Path(output_dir).resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    artifacts = EvaluationArtifacts(
        summary_json=output / "summary.json",
        summary_csv=output / "summary.csv",
        report_markdown=output / "report.md",
        localization_png=figures / "localization_metrics.png",
        localization_pdf=figures / "localization_metrics.pdf",
        gamma_png=figures / "gamma_diagnostics.png",
        gamma_pdf=figures / "gamma_diagnostics.pdf",
    )
    payload = {
        "schema_version": "double_helix_training_evaluation.v1",
        "evaluation_status": "complete",
        "runs": [evaluation.summary for evaluation in evaluations],
    }
    artifacts.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_rows = _summary_csv_rows(evaluations)
    with artifacts.summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    artifacts.report_markdown.write_text(_report_markdown(evaluations), encoding="utf-8")
    _plot_localization_metrics(
        evaluations,
        steps_per_epoch=steps_per_epoch,
        png_path=artifacts.localization_png,
        pdf_path=artifacts.localization_pdf,
    )
    _plot_gamma_metrics(evaluations, png_path=artifacts.gamma_png, pdf_path=artifacts.gamma_pdf)
    return artifacts


def parse_run_specs(
    run_values: list[list[str]],
    prefix_values: list[list[str]],
) -> list[RunSpec]:
    prefixes = {name: Path(path) for name, path in prefix_values}
    names = [values[0] for values in run_values]
    unknown_prefixes = sorted(set(prefixes) - set(names))
    if unknown_prefixes:
        raise ValueError(f"prefix runs do not match a run name: {unknown_prefixes}")
    return [
        RunSpec(
            name=name,
            density_um2=float(density),
            run_root=Path(run_root),
            prefix_run_root=prefixes.get(name),
        )
        for name, density, run_root in run_values
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate completed double-helix localization training runs.")
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "DENSITY_UM2", "RUN_ROOT"),
    )
    parser.add_argument(
        "--prefix-run",
        action="append",
        nargs=2,
        default=[],
        metavar=("NAME", "RUN_ROOT"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=417)
    parser.add_argument("--expected-gamma-updates", type=int, default=28)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = parse_run_specs(args.run, args.prefix_run)
    evaluations = [
        evaluate_run(
            spec,
            expected_epochs=args.expected_epochs,
            steps_per_epoch=args.steps_per_epoch,
            expected_gamma_updates=args.expected_gamma_updates,
            require_complete=True,
        )
        for spec in specs
    ]
    artifacts = write_evaluation_package(
        evaluations,
        args.output_dir,
        steps_per_epoch=args.steps_per_epoch,
    )
    print(
        json.dumps(
            {name: str(getattr(artifacts, name)) for name in artifacts.__dataclass_fields__},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
