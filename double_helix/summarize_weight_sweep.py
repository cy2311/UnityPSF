from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CENTRAL_DISPLAY_Z_NM = (-520.0, -240.0, 0.0, 240.0, 480.0)


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    weights: tuple[float, float, float, float]
    calibration_dir: Path


@dataclass(frozen=True)
class CandidateData:
    summary: dict[str, Any]
    stage_z_nm: np.ndarray
    raw_signal: np.ndarray
    model_signal: np.ndarray


def lobe_valley_ratio(image: np.ndarray) -> float:
    values = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    peak = float(values.max())
    if peak <= 0.0:
        return float("nan")
    values /= peak
    smoothed = gaussian_filter(values, sigma=0.65)
    maxima = smoothed == maximum_filter(smoothed, size=3, mode="nearest")
    peak_y, peak_x = np.nonzero(maxima)
    keep = (
        (peak_y > 0)
        & (peak_y < values.shape[0] - 1)
        & (peak_x > 0)
        & (peak_x < values.shape[1] - 1)
        & (smoothed[peak_y, peak_x] > 0.18)
    )
    peak_y, peak_x = peak_y[keep], peak_x[keep]
    intensities = smoothed[peak_y, peak_x]
    order = np.argsort(intensities)[::-1][:12]
    peak_y, peak_x, intensities = peak_y[order], peak_x[order], intensities[order]

    pairs: list[tuple[float, int, int]] = []
    for first in range(len(intensities)):
        for second in range(first + 1, len(intensities)):
            distance = float(
                np.hypot(
                    peak_x[second] - peak_x[first],
                    peak_y[second] - peak_y[first],
                )
            )
            if 2.2 <= distance <= 8.0:
                balance = min(intensities[first], intensities[second]) / max(
                    intensities[first], intensities[second]
                )
                score = (
                    np.sqrt(intensities[first] * intensities[second]) * balance
                    - 0.015 * abs(distance - 4.5)
                )
                pairs.append((float(score), first, second))
    if not pairs:
        return float("nan")

    _, first, second = max(pairs)
    y0, x0 = float(peak_y[first]), float(peak_x[first])
    y1, x1 = float(peak_y[second]), float(peak_x[second])
    dy, dx = y1 - y0, x1 - x0
    separation = float(np.hypot(dx, dy))
    t = np.linspace(0.0, 1.0, 241)
    profiles = []
    for offset in np.linspace(-1.0, 1.0, 5):
        sample_y = y0 + t * dy - offset * dx / separation
        sample_x = x0 + t * dx + offset * dy / separation
        profiles.append(
            map_coordinates(values, [sample_y, sample_x], order=1, mode="nearest")
        )
    profile = np.mean(profiles, axis=0)
    lobe_intensity = 0.5 * (
        float(profile[t <= 0.2].max()) + float(profile[t >= 0.8].max())
    )
    valley_intensity = float(profile[(t >= 0.3) & (t <= 0.7)].min())
    return valley_intensity / lobe_intensity


def choose_candidate(
    candidates: Sequence[dict[str, Any]],
    *,
    minimum_heldout_ncc: float,
    maximum_edge_flux: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one candidate is required.")
    eligible = [
        candidate
        for candidate in candidates
        if candidate["heldout_median_ncc"] >= minimum_heldout_ncc
        and candidate["edge_flux_fraction"] <= maximum_edge_flux
    ]
    pool = eligible or list(candidates)
    return min(
        pool,
        key=lambda candidate: (
            candidate["central_valley_mae"],
            -candidate["central_median_ncc"],
            -candidate["heldout_median_ncc"],
            candidate["edge_flux_fraction"],
        ),
    )


def load_candidate(spec: CandidateSpec, *, central_z_limit_nm: float) -> CandidateData:
    fit_path = spec.calibration_dir / "arrays" / "calibration_fit.npz"
    diagnostics_path = (
        spec.calibration_dir
        / "arrays"
        / "extended_z2000_photometry_matched_diagnostics.npz"
    )
    metrics_path = spec.calibration_dir / "metadata" / "metrics.json"
    with np.load(fit_path, allow_pickle=False) as fit:
        stage_z_nm = np.asarray(fit["stage_z_nm"], dtype=np.float64)
        photons = np.asarray(fit["photons_adu"], dtype=np.float64)
        background = np.asarray(fit["background_adu"], dtype=np.float64)
        observed = np.asarray(fit["observed_adu"], dtype=np.float64)
        model_signal = np.asarray(fit["reconstruction_unit_flux"], dtype=np.float64)
    raw_signal = np.maximum(
        (observed - background[:, None, None]) / photons[:, None, None],
        0.0,
    )
    with np.load(diagnostics_path, allow_pickle=False) as diagnostics:
        diagnostics_z_nm = np.asarray(diagnostics["stage_z_nm"], dtype=np.float64)
        shape_ncc = np.asarray(diagnostics["cropped_shape_ncc"], dtype=np.float64)
    if not np.array_equal(stage_z_nm, diagnostics_z_nm):
        raise ValueError(f"Stage Z mismatch in {spec.calibration_dir}.")

    central_mask = np.abs(stage_z_nm) <= central_z_limit_nm
    raw_valley = np.asarray(
        [lobe_valley_ratio(image) for image in raw_signal[central_mask]],
        dtype=np.float64,
    )
    model_valley = np.asarray(
        [lobe_valley_ratio(image) for image in model_signal[central_mask]],
        dtype=np.float64,
    )
    finite = np.isfinite(raw_valley) & np.isfinite(model_valley)
    if not np.any(finite):
        raise ValueError(f"No central-Z lobe pairs detected in {spec.calibration_dir}.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = {
        "label": spec.label,
        "z_bin_weights": list(spec.weights),
        "calibration_dir": str(spec.calibration_dir.resolve()),
        "central_z_limit_nm": float(central_z_limit_nm),
        "central_plane_count": int(np.count_nonzero(finite)),
        "central_raw_valley_median": float(np.median(raw_valley[finite])),
        "central_model_valley_median": float(np.median(model_valley[finite])),
        "central_valley_mae": float(np.median(np.abs(model_valley[finite] - raw_valley[finite]))),
        "central_valley_excess": float(np.median(model_valley[finite] - raw_valley[finite])),
        "central_median_ncc": float(np.median(shape_ncc[central_mask])),
        "heldout_median_ncc": float(metrics["heldout_median_ncc"]),
        "heldout_p10_ncc": float(metrics["heldout_p10_ncc"]),
        "edge_flux_fraction": float(metrics["edge_flux_fraction"]),
        "fitted_z_offset_nm": float(metrics["fitted_z_offset_nm"]),
        "fitted_z_scale": float(metrics["fitted_z_scale"]),
    }
    return CandidateData(
        summary=summary,
        stage_z_nm=stage_z_nm,
        raw_signal=raw_signal,
        model_signal=model_signal,
    )


def _peak_normalize(image: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    return values / max(float(values.max()), 1e-12)


def render_central_psfs(
    candidates: Sequence[CandidateData],
    recommended_label: str,
    output_path: Path,
) -> None:
    target_z_nm = np.asarray(CENTRAL_DISPLAY_Z_NM, dtype=np.float64)
    reference = candidates[0]
    plane_indices = np.asarray(
        [int(np.argmin(np.abs(reference.stage_z_nm - target))) for target in target_z_nm]
    )
    fig, axes = plt.subplots(
        len(candidates) + 1,
        len(plane_indices),
        figsize=(12.5, 2.35 * (len(candidates) + 1)),
        constrained_layout=True,
    )
    rows = [("Raw", reference.raw_signal)] + [
        (candidate.summary["label"], candidate.model_signal) for candidate in candidates
    ]
    for row_index, (label, volume) in enumerate(rows):
        is_recommended = label == recommended_label
        for column_index, plane_index in enumerate(plane_indices):
            axis = axes[row_index, column_index]
            axis.imshow(_peak_normalize(volume[plane_index]), cmap="magma", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"stage {reference.stage_z_nm[plane_index]:+.0f} nm", fontsize=10)
            if column_index == 0:
                suffix = "  RECOMMENDED" if is_recommended else ""
                axis.set_ylabel(f"{label}{suffix}", fontsize=9)
            if is_recommended:
                for spine in axis.spines.values():
                    spine.set_color("#00a676")
                    spine.set_linewidth(2.5)
    fig.suptitle("64-mode Z-weight sweep | central double-helix lobe separation", fontsize=15)
    fig.savefig(output_path, dpi=300, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def render_metric_summary(
    summaries: Sequence[dict[str, Any]],
    recommended_label: str,
    output_path: Path,
) -> None:
    labels = [summary["label"] for summary in summaries]
    colors = ["#00a676" if label == recommended_label else "#557a95" for label in labels]
    metrics = (
        ("central_valley_mae", "Central valley MAE", True),
        ("central_median_ncc", "Central median NCC", False),
        ("heldout_median_ncc", "Held-out median NCC", False),
        ("edge_flux_fraction", "Edge flux fraction", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (key, title, lower_is_better) in zip(axes.ravel(), metrics, strict=True):
        values = [summary[key] for summary in summaries]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(f"{title} ({'lower' if lower_is_better else 'higher'} is better)")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(f"64-mode Z-weight sweep | recommended: {recommended_label}", fontsize=15)
    fig.savefig(output_path, dpi=300, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a 64-mode double-helix Z-weight sweep.")
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=6,
        required=True,
        metavar=("LABEL", "W0", "W1", "W2", "W3", "CALIBRATION_DIR"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--central-z-limit-nm", type=float, default=600.0)
    parser.add_argument("--minimum-heldout-ncc", type=float, default=0.90)
    parser.add_argument("--maximum-edge-flux", type=float, default=0.12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = [
        CandidateSpec(
            label=raw[0],
            weights=tuple(float(value) for value in raw[1:5]),
            calibration_dir=Path(raw[5]),
        )
        for raw in args.candidate
    ]
    candidates = [
        load_candidate(spec, central_z_limit_nm=args.central_z_limit_nm) for spec in specs
    ]
    summaries = [candidate.summary for candidate in candidates]
    recommended = choose_candidate(
        summaries,
        minimum_heldout_ncc=args.minimum_heldout_ncc,
        maximum_edge_flux=args.maximum_edge_flux,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_central_psfs(
        candidates,
        recommended["label"],
        output_dir / "weight_sweep_central_psfs.png",
    )
    render_metric_summary(
        summaries,
        recommended["label"],
        output_dir / "weight_sweep_metrics.png",
    )
    report = {
        "selection_rule": {
            "primary": "minimum median central-Z lobe-valley absolute error",
            "tie_breakers": ["central median NCC", "held-out median NCC", "edge flux"],
            "minimum_heldout_ncc": float(args.minimum_heldout_ncc),
            "maximum_edge_flux": float(args.maximum_edge_flux),
        },
        "recommended_label": recommended["label"],
        "recommended_z_bin_weights": recommended["z_bin_weights"],
        "candidates": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateData",
    "CandidateSpec",
    "choose_candidate",
    "load_candidate",
    "lobe_valley_ratio",
    "render_central_psfs",
    "render_metric_summary",
]
