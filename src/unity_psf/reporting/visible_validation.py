"""Publication-style fixed figure pack for multimodal training validation."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unity_psf.contracts.joint_checkpoint import JointExpertKey


_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
_LINESTYLES = ("-", "--", "-.", ":")
_SCIENTIFIC_METRICS = (
    "precision",
    "recall",
    "Jaccard",
    "RMSE_XY_nm",
    "RMSE_Z_nm",
    "photon_relative_error",
)


@dataclass(frozen=True)
class InstanceVisualRecord:
    instance_key: str
    input_image: np.ndarray
    patches: tuple[np.ndarray, ...]
    loss_history: tuple[float, ...]
    route_count: int
    step_count: int
    sample_count: int
    prediction_xy: np.ndarray
    reconstruction: np.ndarray
    target_xy: np.ndarray | None = None
    z_values: np.ndarray | None = None
    z_errors: np.ndarray | None = None
    physical_initial: np.ndarray | None = None
    physical_current: np.ndarray | None = None
    status: str = "not-evaluated"
    checkpoint_hash: str | None = None
    heldout_metrics: Mapping[str, float | int | None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instance_key", JointExpertKey.parse(self.instance_key).storage_key)
        for field_name in ("input_image", "reconstruction"):
            value = np.asarray(getattr(self, field_name))
            if value.ndim != 2 or value.size == 0:
                raise ValueError(f"{field_name} must be a non-empty 2D array")
            object.__setattr__(self, field_name, value)
        patches = tuple(np.asarray(item) for item in self.patches)
        if not patches or any(item.ndim != 2 or item.size == 0 for item in patches):
            raise ValueError("patches must contain non-empty 2D arrays")
        object.__setattr__(self, "patches", patches)
        losses = tuple(float(item) for item in self.loss_history)
        if not losses or not np.isfinite(losses).all():
            raise ValueError("loss_history must contain finite values")
        object.__setattr__(self, "loss_history", losses)
        predictions = np.asarray(self.prediction_xy, dtype=np.float32)
        if predictions.ndim != 2 or predictions.shape[1] != 2:
            raise ValueError("prediction_xy must have shape (N,2)")
        object.__setattr__(self, "prediction_xy", predictions)
        if self.target_xy is not None:
            targets = np.asarray(self.target_xy, dtype=np.float32)
            if targets.ndim != 2 or targets.shape[1] != 2:
                raise ValueError("target_xy must have shape (N,2)")
            object.__setattr__(self, "target_xy", targets)
        for field_name in ("route_count", "step_count", "sample_count"):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        if (self.z_values is None) != (self.z_errors is None):
            raise ValueError("z_values and z_errors must be supplied together")
        if self.z_values is not None:
            z_values = np.asarray(self.z_values, dtype=np.float32).reshape(-1)
            z_errors = np.asarray(self.z_errors, dtype=np.float32).reshape(-1)
            if z_values.shape != z_errors.shape or not z_values.size:
                raise ValueError("z_values and z_errors must have equal non-empty shapes")
            object.__setattr__(self, "z_values", z_values)
            object.__setattr__(self, "z_errors", z_errors)
        if (self.physical_initial is None) != (self.physical_current is None):
            raise ValueError("physical_initial and physical_current must be supplied together")
        if self.physical_initial is not None:
            initial = np.asarray(self.physical_initial, dtype=np.float32)
            current = np.asarray(self.physical_current, dtype=np.float32)
            if initial.ndim != 2 or current.shape != initial.shape:
                raise ValueError("physical states must be equally shaped 2D arrays")
            object.__setattr__(self, "physical_initial", initial)
            object.__setattr__(self, "physical_current", current)
        if self.heldout_metrics is not None:
            metrics = {
                key: None if self.heldout_metrics.get(key) is None else float(self.heldout_metrics[key])
                for key in _SCIENTIFIC_METRICS
            }
            object.__setattr__(self, "heldout_metrics", metrics)


@dataclass(frozen=True)
class VisibleValidationResult:
    report_path: Path
    figure_paths: tuple[Path, ...]
    summary_path: Path


def _style() -> Mapping[str, object]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=_COLORS),
        "savefig.facecolor": "white",
    }


def _save(fig: plt.Figure, figures_dir: Path, name: str, *, vector: bool = False) -> Path:
    png = figures_dir / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    if vector:
        fig.savefig(figures_dir / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def _title(key: str) -> str:
    parsed = JointExpertKey.parse(key)
    return f"{parsed.modality.value} | {parsed.channel_id}"


def _input_audit(records: Sequence[InstanceVisualRecord], figures_dir: Path) -> Path:
    fig, axes = plt.subplots(len(records), 2, figsize=(7.2, 2.2 * len(records)), squeeze=False, constrained_layout=True)
    for row, record in enumerate(records):
        image = record.input_image
        axes[row, 0].imshow(image, cmap="gray", interpolation="nearest")
        axes[row, 0].set_title(_title(record.instance_key))
        axes[row, 0].set_axis_off()
        axes[row, 1].hist(image.ravel(), bins=40, color=_COLORS[row % len(_COLORS)], alpha=0.85)
        axes[row, 1].set_xlabel("Intensity (ADU)")
        axes[row, 1].set_ylabel("Pixels")
        axes[row, 1].text(0.98, 0.96, f"shape={image.shape}\nn={image.size}", transform=axes[row, 1].transAxes, ha="right", va="top")
    return _save(fig, figures_dir, "00_input_audit")


def _patch_montage(records: Sequence[InstanceVisualRecord], figures_dir: Path) -> Path:
    columns = max(len(record.patches) for record in records)
    fig, axes = plt.subplots(len(records), columns, figsize=(1.9 * columns, 1.9 * len(records)), squeeze=False, constrained_layout=True)
    for row, record in enumerate(records):
        shared_min = min(float(np.min(item)) for item in record.patches)
        shared_max = max(float(np.max(item)) for item in record.patches)
        for column in range(columns):
            ax = axes[row, column]
            if column < len(record.patches):
                ax.imshow(record.patches[column], cmap="viridis", vmin=shared_min, vmax=shared_max, interpolation="nearest")
                ax.set_title(f"{_title(record.instance_key)}\npatch {column + 1}")
            ax.set_axis_off()
    return _save(fig, figures_dir, "01_psf_patch_montage")


def _balance(records: Sequence[InstanceVisualRecord], figures_dir: Path) -> Path:
    labels = [record.instance_key.replace(":", "\n") for record in records]
    x = np.arange(len(records), dtype=np.float32)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for offset, (label, values, color, hatch) in enumerate((
        ("Routes", [item.route_count for item in records], _COLORS[0], ""),
        ("Optimizer steps", [item.step_count for item in records], _COLORS[1], "//"),
        ("Samples", [item.sample_count for item in records], _COLORS[2], ".."),
    )):
        ax.bar(x + (offset - 1) * width, values, width, label=label, color=color, hatch=hatch, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Count")
    ax.legend(frameon=False, ncols=3)
    return _save(fig, figures_dir, "02_route_and_step_balance", vector=True)


def _training_curves(records: Sequence[InstanceVisualRecord], figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    for index, record in enumerate(records):
        updates = np.arange(1, len(record.loss_history) + 1)
        ax.plot(updates, record.loss_history, label=record.instance_key, color=_COLORS[index % len(_COLORS)], linestyle=_LINESTYLES[index % len(_LINESTYLES)], marker=("o", "s", "^")[index % 3], markersize=3)
    ax.set_xlabel("Channel update")
    ax.set_ylabel("Instance loss (AU)")
    ax.legend(frameon=False)
    return _save(fig, figures_dir, "03_training_curves", vector=True)


def _per_instance_figures(record: InstanceVisualRecord, figures_dir: Path) -> list[Path]:
    slug = record.instance_key.replace(":", "_")
    paths = []
    fig, ax = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)
    ax.imshow(record.input_image, cmap="gray", interpolation="nearest")
    if record.target_xy is not None and len(record.target_xy):
        ax.scatter(
            record.target_xy[:, 0],
            record.target_xy[:, 1],
            s=24,
            marker="+",
            color=_COLORS[2],
            linewidths=1.0,
            label=f"GT (n={len(record.target_xy)})",
        )
    if len(record.prediction_xy):
        ax.scatter(record.prediction_xy[:, 0], record.prediction_xy[:, 1], s=30, facecolors="none", edgecolors=_COLORS[1], linewidths=0.9, label=f"Predictions (n={len(record.prediction_xy)})")
    if (record.target_xy is not None and len(record.target_xy)) or len(record.prediction_xy):
        ax.legend(frameon=False, loc="lower right")
    ax.set_title(_title(record.instance_key))
    ax.set_axis_off()
    paths.append(_save(fig, figures_dir, f"04_prediction_overlay_{slug}"))

    fig, ax = plt.subplots(figsize=(3.8, 2.8), constrained_layout=True)
    if record.z_values is None:
        ax.axhline(0.0, color="black", linewidth=0.8)
        modality = JointExpertKey.parse(record.instance_key).modality.value
        message = "z not applicable" if modality == "emitter_2d" else "z evaluation unavailable"
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
        ax.set_xlim(-1.0, 1.0)
    else:
        ax.scatter(record.z_values, record.z_errors, color=_COLORS[0], marker="o", s=24)
        ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Ground-truth z (nm)")
    ax.set_ylabel("z error (nm)")
    ax.set_title(_title(record.instance_key))
    paths.append(_save(fig, figures_dir, f"05_error_by_z_{slug}", vector=True))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), constrained_layout=True)
    axes[0].imshow(record.input_image, cmap="gray", interpolation="nearest")
    axes[0].set_title("Input")
    axes[1].imshow(record.reconstruction, cmap="magma", interpolation="nearest")
    axes[1].set_title("Localization reconstruction")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(_title(record.instance_key))
    paths.append(_save(fig, figures_dir, f"06_reconstruction_{slug}"))

    if record.physical_initial is not None:
        difference = record.physical_current - record.physical_initial
        limit = max(abs(float(np.min(difference))), abs(float(np.max(difference))), 1e-8)
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
        axes[0].imshow(record.physical_initial, cmap="viridis")
        axes[0].set_title("Initial state")
        axes[1].imshow(record.physical_current, cmap="viridis")
        axes[1].set_title("Current state")
        image = axes[2].imshow(difference, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[2].set_title("Current - initial")
        fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(_title(record.instance_key))
    else:
        fig, ax = plt.subplots(figsize=(3.8, 2.8), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.5, "physical state unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(_title(record.instance_key))
    paths.append(_save(fig, figures_dir, f"07_physical_state_{slug}"))
    return paths


def _scorecard(records: Sequence[InstanceVisualRecord], figures_dir: Path) -> Path:
    labels = [record.instance_key.replace(":", "\n") for record in records]
    final_losses = [record.loss_history[-1] for record in records]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 3.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (0.9, 1.6)},
    )
    axes[0].bar(np.arange(len(records)), final_losses, color=[_COLORS[index % len(_COLORS)] for index in range(len(records))], edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(np.arange(len(records)), labels)
    axes[0].set_ylabel("Final instance loss (AU)")
    axes[0].set_title("No cross-instance averaging")
    axes[1].axis("off")
    rows = [
        [
            label,
            record.status,
            str(record.sample_count),
            str(record.step_count),
            (record.checkpoint_hash or "missing")[:8],
        ]
        for label, record in zip(labels, records)
    ]
    table = axes[1].table(
        cellText=rows,
        colLabels=["Instance", "Status", "Samples", "Steps", "Hash"],
        colWidths=[0.25, 0.31, 0.14, 0.14, 0.16],
        bbox=(0.0, 0.02, 1.0, 0.82),
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    axes[1].set_title("Instance-level release evidence")
    return _save(fig, figures_dir, "08_cross_modality_scorecard", vector=True)


def _format_metric(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _heldout_metrics_scorecard(
    records: Sequence[InstanceVisualRecord],
    modality_metrics: Mapping[str, Mapping[str, float | int | None]],
    figures_dir: Path,
) -> Path:
    rows = []
    for record in records:
        metrics = record.heldout_metrics or {}
        rows.append(
            [record.instance_key.replace(":", "\n")]
            + [_format_metric(metrics.get(key)) for key in _SCIENTIFIC_METRICS]
        )
    for modality, metrics in modality_metrics.items():
        rows.append(
            [f"{modality}\naggregate"]
            + [_format_metric(metrics.get(key)) for key in _SCIENTIFIC_METRICS]
        )
    height = max(2.8, 0.55 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(11.2, height), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Instance", "Precision", "Recall", "Jaccard", "RMSE XY (nm)", "RMSE Z (nm)", "Photon error"],
        colWidths=[0.20, 0.12, 0.12, 0.12, 0.15, 0.15, 0.14],
        bbox=(0.0, 0.02, 1.0, 0.84),
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    ax.set_title("Held-out localization metrics by channel and modality")
    return _save(fig, figures_dir, "09_heldout_metrics_scorecard", vector=True)


def generate_visible_validation_report(
    output_root: str | Path,
    records: Sequence[InstanceVisualRecord],
    *,
    run_id: str,
    provenance: Mapping[str, object] | None = None,
    modality_metrics: Mapping[str, Mapping[str, float | int | None]] | None = None,
) -> VisibleValidationResult:
    """Generate the fixed dual-modality figure pack and an offline HTML index."""

    normalized = tuple(records)
    if not normalized or len({item.instance_key for item in normalized}) != len(normalized):
        raise ValueError("records must be non-empty and have unique instance keys")
    root = Path(output_root)
    figures_dir = root / "figures"
    metrics_dir = root / "metrics"
    report_dir = root / "report"
    for directory in (figures_dir, metrics_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    with plt.rc_context(_style()):
        figure_paths.extend((
            _input_audit(normalized, figures_dir),
            _patch_montage(normalized, figures_dir),
            _balance(normalized, figures_dir),
            _training_curves(normalized, figures_dir),
        ))
        for record in normalized:
            figure_paths.extend(_per_instance_figures(record, figures_dir))
        figure_paths.append(_scorecard(normalized, figures_dir))
        figure_paths.append(
            _heldout_metrics_scorecard(normalized, modality_metrics or {}, figures_dir)
        )

    summary = {
        "schema_version": "unitypsf.visible_validation.v1",
        "run_id": str(run_id),
        "instances": {
            record.instance_key: {
                "status": record.status,
                "route_count": record.route_count,
                "step_count": record.step_count,
                "sample_count": record.sample_count,
                "final_loss": record.loss_history[-1],
                "checkpoint_hash": record.checkpoint_hash,
                "heldout_metrics": None
                if record.heldout_metrics is None
                else dict(record.heldout_metrics),
            }
            for record in normalized
        },
        "modality_metrics": {
            str(modality): {
                key: None if metrics.get(key) is None else float(metrics[key])
                for key in _SCIENTIFIC_METRICS
            }
            for modality, metrics in (modality_metrics or {}).items()
        },
        "provenance": dict(provenance or {}),
    }
    summary_path = metrics_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_index = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in figure_paths
    }
    (report_dir / "figure_index.json").write_text(json.dumps(figure_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sections = "\n".join(
        f'<figure><img src="../figures/{escape(path.name)}" alt="{escape(path.stem)}"><figcaption>{escape(path.stem.replace("_", " "))}</figcaption></figure>'
        for path in figure_paths
    )
    instance_list = "".join(f"<li>{escape(record.instance_key)}: {escape(record.status)}</li>" for record in normalized)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>UnityPSF {escape(str(run_id))}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:28px auto;color:#111}}figure{{margin:28px 0;border-bottom:1px solid #ccc;padding-bottom:20px}}img{{max-width:100%;height:auto}}figcaption{{font-size:13px;margin-top:6px;color:#333}}</style></head>
<body><h1>UnityPSF validation: {escape(str(run_id))}</h1><ul>{instance_list}</ul>{sections}</body></html>"""
    report_path = report_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return VisibleValidationResult(report_path, tuple(figure_paths), summary_path)


__all__ = ["InstanceVisualRecord", "VisibleValidationResult", "generate_visible_validation_report"]
