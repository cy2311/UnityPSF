from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unity_psf.localization.legacy_decode import (
    LegacyEmitterSet,
    match_legacy_localizations,
)


@dataclass(frozen=True)
class ZBinMetrics:
    label: str
    z_min_nm: float
    z_max_nm: float
    upper_inclusive: bool
    true_positive: int
    false_positive: int
    false_negative: int
    predicted_emitters: int
    target_emitters: int
    precision: float
    recall: float
    jaccard: float
    rmse_xy_nm: float
    rmse_z_nm: float
    target_photons_mean: float
    target_photons_std: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ZBinEvaluation:
    bins: tuple[ZBinMetrics, ...]
    central_1000nm: ZBinMetrics
    global_metrics: ZBinMetrics
    out_of_range_predicted_emitters: int
    out_of_range_target_emitters: int

    def to_dict(self) -> dict[str, object]:
        return {
            "bins": [row.to_dict() for row in self.bins],
            "central_1000nm": self.central_1000nm.to_dict(),
            "global": self.global_metrics.to_dict(),
            "out_of_range_predicted_emitters": self.out_of_range_predicted_emitters,
            "out_of_range_target_emitters": self.out_of_range_target_emitters,
        }


@dataclass(frozen=True)
class RunZBinResult:
    name: str
    density_um2: float
    checkpoint_path: str
    checkpoint_epoch: int
    checkpoint_global_step: int
    physical_state_path: str
    evaluation: ZBinEvaluation
    evaluation_density_um2: float | None = None
    config_path: str | None = None
    device: str | None = None
    gpu_name: str | None = None
    provenance: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "density_um2": self.density_um2,
            "training_density_um2": self.density_um2,
            "evaluation_density_um2": self.eval_density_um2,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_global_step": self.checkpoint_global_step,
            "physical_state_path": self.physical_state_path,
            "config_path": self.config_path,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "provenance": self.provenance,
            "evaluation": self.evaluation.to_dict(),
        }

    @property
    def eval_density_um2(self) -> float:
        if self.evaluation_density_um2 is None:
            return self.density_um2
        return self.evaluation_density_um2


@dataclass(frozen=True)
class ZBinArtifacts:
    summary_json: Path
    summary_csv: Path
    report_markdown: Path
    metrics_png: Path
    heatmap_png: Path


def evaluate_z_bins(
    pred: LegacyEmitterSet,
    target: LegacyEmitterSet,
    *,
    z_edges_nm: Sequence[float],
    dist_tol_xy_px: float | None = 1.0,
    dist_tol_xy_nm: float | None = None,
    dist_tol_z_nm: float | None = None,
    pixel_size_nm_x: float = 1.0,
    pixel_size_nm_y: float = 1.0,
    match_dims: int = 2,
) -> ZBinEvaluation:
    edges = np.asarray(z_edges_nm, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("z_edges_nm must be a strictly increasing one-dimensional sequence")
    matches = match_legacy_localizations(
        pred,
        target,
        dist_tol_xy_px=dist_tol_xy_px,
        dist_tol_xy_nm=dist_tol_xy_nm,
        dist_tol_z_nm=dist_tol_z_nm,
        pixel_size_nm_x=pixel_size_nm_x,
        pixel_size_nm_y=pixel_size_nm_y,
        match_dims=match_dims,
    )
    target_z = target.xyz_px_nm[:, 2].numpy()
    pred_z = pred.xyz_px_nm[:, 2].numpy()
    target_photons = target.photons.numpy()
    matched_target_indices = np.asarray(matches.matched_target_indices, dtype=np.int64)
    unmatched_target_indices = np.asarray(matches.unmatched_target_indices, dtype=np.int64)
    unmatched_prediction_indices = np.asarray(matches.unmatched_prediction_indices, dtype=np.int64)
    lateral_sq = np.asarray(matches.lateral_sq_errors_nm2, dtype=np.float64)
    axial_sq = np.asarray(matches.axial_sq_errors_nm2, dtype=np.float64)

    bins = tuple(
        _summarize_interval(
            label=_interval_label(float(low), float(high), upper_inclusive=index == len(edges) - 2),
            low=float(low),
            high=float(high),
            upper_inclusive=index == len(edges) - 2,
            target_z=target_z,
            pred_z=pred_z,
            target_photons=target_photons,
            matched_target_indices=matched_target_indices,
            unmatched_target_indices=unmatched_target_indices,
            unmatched_prediction_indices=unmatched_prediction_indices,
            lateral_sq=lateral_sq,
            axial_sq=axial_sq,
        )
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]))
    )
    central = _summarize_interval(
        label="|z| <= 1000 nm",
        low=-1000.0,
        high=1000.0,
        upper_inclusive=True,
        target_z=target_z,
        pred_z=pred_z,
        target_photons=target_photons,
        matched_target_indices=matched_target_indices,
        unmatched_target_indices=unmatched_target_indices,
        unmatched_prediction_indices=unmatched_prediction_indices,
        lateral_sq=lateral_sq,
        axial_sq=axial_sq,
    )
    global_metrics = _summarize_interval(
        label=f"{_format_nm(edges[0])} <= z <= {_format_nm(edges[-1])} nm",
        low=float(edges[0]),
        high=float(edges[-1]),
        upper_inclusive=True,
        target_z=target_z,
        pred_z=pred_z,
        target_photons=target_photons,
        matched_target_indices=matched_target_indices,
        unmatched_target_indices=unmatched_target_indices,
        unmatched_prediction_indices=unmatched_prediction_indices,
        lateral_sq=lateral_sq,
        axial_sq=axial_sq,
    )
    return ZBinEvaluation(
        bins=bins,
        central_1000nm=central,
        global_metrics=global_metrics,
        out_of_range_predicted_emitters=int(np.count_nonzero((pred_z < edges[0]) | (pred_z > edges[-1]))),
        out_of_range_target_emitters=int(np.count_nonzero((target_z < edges[0]) | (target_z > edges[-1]))),
    )


def _summarize_interval(
    *,
    label: str,
    low: float,
    high: float,
    upper_inclusive: bool,
    target_z: np.ndarray,
    pred_z: np.ndarray,
    target_photons: np.ndarray,
    matched_target_indices: np.ndarray,
    unmatched_target_indices: np.ndarray,
    unmatched_prediction_indices: np.ndarray,
    lateral_sq: np.ndarray,
    axial_sq: np.ndarray,
) -> ZBinMetrics:
    target_mask = _interval_mask(target_z, low, high, upper_inclusive)
    matched_mask = _interval_mask(target_z[matched_target_indices], low, high, upper_inclusive)
    fn_mask = _interval_mask(target_z[unmatched_target_indices], low, high, upper_inclusive)
    fp_mask = _interval_mask(pred_z[unmatched_prediction_indices], low, high, upper_inclusive)
    tp = int(np.count_nonzero(matched_mask))
    fp = int(np.count_nonzero(fp_mask))
    fn = int(np.count_nonzero(fn_mask))
    photons = target_photons[target_mask]
    return ZBinMetrics(
        label=label,
        z_min_nm=float(low),
        z_max_nm=float(high),
        upper_inclusive=bool(upper_inclusive),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        predicted_emitters=tp + fp,
        target_emitters=tp + fn,
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        jaccard=tp / max(tp + fp + fn, 1),
        rmse_xy_nm=float(np.sqrt(np.mean(lateral_sq[matched_mask]))) if tp else 0.0,
        rmse_z_nm=float(np.sqrt(np.mean(axial_sq[matched_mask]))) if tp else 0.0,
        target_photons_mean=float(np.mean(photons)) if len(photons) else 0.0,
        target_photons_std=float(np.std(photons)) if len(photons) else 0.0,
    )


def _interval_mask(values: np.ndarray, low: float, high: float, upper_inclusive: bool) -> np.ndarray:
    if upper_inclusive:
        return (values >= low) & (values <= high)
    return (values >= low) & (values < high)


def _interval_label(low: float, high: float, *, upper_inclusive: bool) -> str:
    closing = "]" if upper_inclusive else ")"
    return f"[{_format_nm(low)}, {_format_nm(high)}{closing} nm"


def _format_nm(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def write_z_bin_package(runs: Sequence[RunZBinResult], output_dir: Path | str) -> ZBinArtifacts:
    if not runs:
        raise ValueError("write_z_bin_package requires at least one run")
    output = Path(output_dir)
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    artifacts = ZBinArtifacts(
        summary_json=output / "summary.json",
        summary_csv=output / "summary.csv",
        report_markdown=output / "report.md",
        metrics_png=figures / "z_bin_metrics.png",
        heatmap_png=figures / "z_bin_heatmaps.png",
    )
    payload = {
        "schema_version": "double_helix_z_bin_evaluation.v1",
        "bin_assignment": {
            "matching": "one_global_greedy_3d_match_per_frame_before_binning",
            "true_positive_and_false_negative": "ground_truth_z_nm",
            "false_positive": "predicted_z_nm",
            "intervals": "lower_inclusive_upper_exclusive_except_final_upper_inclusive",
        },
        "runs": [run.to_dict() for run in runs],
    }
    artifacts.summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(runs, artifacts.summary_csv, tables)
    artifacts.report_markdown.write_text(_markdown_report(runs), encoding="utf-8")
    _plot_metrics(runs, artifacts.metrics_png)
    _plot_heatmaps(runs, artifacts.heatmap_png)
    return artifacts


def _write_csv(runs: Sequence[RunZBinResult], summary_path: Path, tables_dir: Path) -> None:
    fieldnames = [
        "name", "density_um2", "training_density_um2", "evaluation_density_um2", "scope",
        "z_min_nm", "z_max_nm", "tp", "fp", "fn",
        "predicted_emitters", "target_emitters", "jaccard", "precision", "recall",
        "rmse_xy_nm", "rmse_z_nm", "target_photons_mean", "target_photons_std",
    ]
    all_rows: list[dict[str, object]] = []
    for run in runs:
        rows = [_csv_row(run, row, scope="z_bin") for row in run.evaluation.bins]
        rows.extend(
            (
                _csv_row(run, run.evaluation.central_1000nm, scope="central_1000nm"),
                _csv_row(run, run.evaluation.global_metrics, scope="global"),
            )
        )
        all_rows.extend(rows)
        _write_csv_rows(tables_dir / f"{run.name}_z_bins.csv", fieldnames, rows[: len(run.evaluation.bins)])
    _write_csv_rows(summary_path, fieldnames, all_rows)


def _csv_row(run: RunZBinResult, row: ZBinMetrics, *, scope: str) -> dict[str, object]:
    return {
        "name": run.name,
        "density_um2": run.density_um2,
        "training_density_um2": run.density_um2,
        "evaluation_density_um2": run.eval_density_um2,
        "scope": scope,
        "z_min_nm": row.z_min_nm,
        "z_max_nm": row.z_max_nm,
        "tp": row.true_positive,
        "fp": row.false_positive,
        "fn": row.false_negative,
        "predicted_emitters": row.predicted_emitters,
        "target_emitters": row.target_emitters,
        "jaccard": row.jaccard,
        "precision": row.precision,
        "recall": row.recall,
        "rmse_xy_nm": row.rmse_xy_nm,
        "rmse_z_nm": row.rmse_z_nm,
        "target_photons_mean": row.target_photons_mean,
        "target_photons_std": row.target_photons_std,
    }


def _write_csv_rows(path: Path, fieldnames: list[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(runs: Sequence[RunZBinResult]) -> str:
    lines = [
        "# Double-helix z-bin evaluation",
        "",
        "All bins use one global per-frame 3D matching pass. TP/FN are assigned by ground-truth z; unmatched FP are assigned by predicted z.",
        "",
    ]
    for run in runs:
        lines.extend(
            [
                f"## Train density {run.density_um2:g} -> eval density {run.eval_density_um2:g} emitters/um^2",
                "",
                "| GT z bin (nm) | Jaccard | Precision | Recall | RMSE xy (nm) | RMSE z (nm) | TP | FP | FN | GT photons mean |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in run.evaluation.bins:
            lines.append(
                f"| {row.label} | {row.jaccard:.4f} | {row.precision:.4f} | {row.recall:.4f} | "
                f"{row.rmse_xy_nm:.2f} | {row.rmse_z_nm:.2f} | {row.true_positive} | "
                f"{row.false_positive} | {row.false_negative} | {row.target_photons_mean:.1f} |"
            )
        for row in (run.evaluation.central_1000nm, run.evaluation.global_metrics):
            lines.append(
                f"| **{row.label}** | **{row.jaccard:.4f}** | **{row.precision:.4f}** | **{row.recall:.4f}** | "
                f"**{row.rmse_xy_nm:.2f}** | **{row.rmse_z_nm:.2f}** | **{row.true_positive}** | "
                f"**{row.false_positive}** | **{row.false_negative}** | **{row.target_photons_mean:.1f}** |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _plot_metrics(runs: Sequence[RunZBinResult], path: Path) -> None:
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    panels = (
        ("jaccard", "Jaccard", (0.0, 1.0)),
        ("recall", "Recall", (0.0, 1.0)),
        ("rmse_z_nm", "RMSE z (nm)", None),
        ("target_photons_mean", "Ground-truth photons (mean)", None),
    )
    for run_index, run in enumerate(runs):
        centers = [(row.z_min_nm + row.z_max_nm) / 2.0 for row in run.evaluation.bins]
        for axis, (field, ylabel, ylim) in zip(axes.flat, panels):
            values = [float(getattr(row, field)) for row in run.evaluation.bins]
            axis.plot(
                centers,
                values,
                color=colors[run_index % len(colors)],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=f"train density {run.density_um2:g}",
            )
            axis.set_ylabel(ylabel)
            if ylim is not None:
                axis.set_ylim(*ylim)
    for axis in axes.flat:
        axis.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        axis.axvspan(-1000.0, 1000.0, color="#999999", alpha=0.08)
        axis.set_xlabel("Ground-truth z-bin center (nm)")
        axis.grid(True, color="#DDDDDD", linewidth=0.7)
    axes[0, 0].legend(frameon=False, ncol=len(runs), loc="lower center")
    eval_densities = sorted({run.eval_density_um2 for run in runs})
    eval_label = f" | eval density {eval_densities[0]:g}" if len(eval_densities) == 1 else ""
    fig.suptitle(f"Double-helix localization performance by ground-truth z{eval_label}")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def _plot_heatmaps(runs: Sequence[RunZBinResult], path: Path) -> None:
    labels = [row.label.replace(" nm", "") for row in runs[0].evaluation.bins]
    densities = [f"train {run.density_um2:g}" for run in runs]
    jaccard = np.asarray([[row.jaccard for row in run.evaluation.bins] for run in runs])
    rmse_z = np.asarray([[row.rmse_z_nm for row in run.evaluation.bins] for run in runs])
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 5.8), constrained_layout=True)
    for axis, matrix, title, cmap, limits in (
        (axes[0], jaccard, "Jaccard", "viridis", (0.0, 1.0)),
        (axes[1], rmse_z, "RMSE z (nm)", "magma", (0.0, None)),
    ):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_xticks(range(len(labels)), labels=labels, rotation=25, ha="right")
        axis.set_yticks(range(len(densities)), labels=densities)
        axis.set_ylabel("Training density (emitters/um^2)")
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                text_color = "white" if value > (0.55 if title == "Jaccard" else 0.55 * np.max(matrix)) else "black"
                axis.text(column_index, row_index, f"{value:.3f}" if title == "Jaccard" else f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)
        fig.colorbar(image, ax=axis, pad=0.01)
    axes[-1].set_xlabel("Ground-truth z bin (nm)")
    eval_densities = sorted({run.eval_density_um2 for run in runs})
    eval_label = f" | eval density {eval_densities[0]:g}" if len(eval_densities) == 1 else ""
    fig.suptitle(f"Double-helix z-bin evaluation heatmaps{eval_label}")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


__all__ = [
    "RunZBinResult",
    "ZBinArtifacts",
    "ZBinEvaluation",
    "ZBinMetrics",
    "evaluate_z_bins",
    "write_z_bin_package",
]
