from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..optics.psf.double_helix.dataset import Microscope1Dataset
from ..optics.psf.double_helix.gamma_field import DirectGammaZernikeField
from ..optics.psf.double_helix.localization import (
    AngleZCalibration,
    LobeDetectionConfig,
    LocalizationConfig,
    MatchResult,
    localize_frames,
    match_localizations,
)
from ..optics.psf.double_helix.vector_model import DoubleHelixVectorPSF


PROJECT_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "datasets/training_sets/double_helix/Simulated_datasets_Microscope1"
DEFAULT_GAMMA_PATH = PROJECT_ROOT / "output/double_helix/microscope1/field_gamma/arrays/gamma_coefficients.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/double_helix/microscope1/evaluation"


def localization_metrics(match: MatchResult) -> dict[str, float | int]:
    tp = int(match.true_positives)
    fp = int(match.false_positives)
    fn = int(match.false_negatives)
    lateral_squared = match.dx_nm**2 + match.dy_nm**2
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "lateral_rmse_nm": float(np.sqrt(np.mean(lateral_squared))) if tp else float("nan"),
        "axial_rmse_nm": float(np.sqrt(np.mean(match.dz_nm**2))) if tp else float("nan"),
        "rmse_3d_nm": float(np.sqrt(np.mean(lateral_squared + match.dz_nm**2))) if tp else float("nan"),
        "x_bias_nm": float(np.mean(match.dx_nm)) if tp else float("nan"),
        "y_bias_nm": float(np.mean(match.dy_nm)) if tp else float("nan"),
        "z_bias_nm": float(np.mean(match.dz_nm)) if tp else float("nan"),
    }


def run_evaluation(
    *,
    dataset_root: Path,
    gamma_path: Path,
    output_dir: Path,
    config: LocalizationConfig,
    max_frames: int,
    lateral_tolerance_nm: float,
    axial_tolerance_nm: float,
    carrier_path: Path | None = None,
) -> dict[str, Any]:
    dataset = Microscope1Dataset(dataset_root)
    dataset.validate()
    with np.load(gamma_path, allow_pickle=False) as payload:
        gamma_nm = np.asarray(payload["gamma_nm"], dtype=np.float32)
        mode_order = tuple(tuple(int(value) for value in row) for row in payload["mode_order"])
    carrier_complex = None
    if carrier_path is not None:
        with np.load(carrier_path, allow_pickle=False) as payload:
            key = "carrier_complex" if "carrier_complex" in payload else "complex_pupil"
            carrier_complex = np.asarray(payload[key], dtype=np.complex64)
    calibration_stack = dataset.read_calibration()
    calibration_z_nm = dataset.z_sign * (
        np.arange(calibration_stack.shape[0], dtype=np.float64) + dataset.z_index_origin
    ) * dataset.z_step_nm
    angle_calibration = AngleZCalibration.from_stack(calibration_stack, calibration_z_nm)
    frame_count = min(int(max_frames), dataset.config.frame_count)
    frames = np.asarray(dataset.open_frames()[:frame_count])
    predictions = localize_frames(
        frames,
        gamma_nm=gamma_nm,
        mode_order=mode_order,
        angle_calibration=angle_calibration,
        config=config,
        carrier_complex=carrier_complex,
    )
    prediction_xyz_nm = np.column_stack(
        (
            (predictions.x_px - config.image_origin_xy_px[0]) * config.pixel_size_nm,
            (predictions.y_px - config.image_origin_xy_px[1]) * config.pixel_size_nm,
            predictions.z_nm,
        )
    )
    gt = dataset.load_ground_truth()
    gt_keep = gt.frame_index < frame_count
    gt_indices = np.flatnonzero(gt_keep)
    gt_xyz_nm = np.column_stack((gt.x_nm[gt_keep], gt.y_nm[gt_keep], gt.z_nm[gt_keep]))
    match = match_localizations(
        predictions.frame_index,
        prediction_xyz_nm,
        gt.frame_index[gt_keep],
        gt_xyz_nm,
        lateral_tolerance_nm=lateral_tolerance_nm,
        axial_tolerance_nm=axial_tolerance_nm,
    )
    metrics = localization_metrics(match)
    metrics.update(
        {
            "frames_evaluated": frame_count,
            "predictions": int(predictions.frame_index.size),
            "ground_truth_emitters": int(gt_xyz_nm.shape[0]),
            "median_fit_ncc": float(np.median(predictions.ncc)) if predictions.ncc.size else float("nan"),
            "lateral_match_tolerance_nm": float(lateral_tolerance_nm),
            "axial_match_tolerance_nm": float(axial_tolerance_nm),
        }
    )
    _write_outputs(
        output_dir,
        frames=frames,
        calibration_stack=calibration_stack,
        calibration_z_nm=calibration_z_nm,
        predictions=predictions,
        prediction_xyz_nm=prediction_xyz_nm,
        gt_frame=gt.frame_index[gt_keep],
        gt_xyz_nm=gt_xyz_nm,
        gt_indices=gt_indices,
        match=match,
        metrics=metrics,
        gamma_nm=gamma_nm,
        mode_order=mode_order,
        config=config,
        gamma_path=gamma_path,
        carrier_complex=carrier_complex,
        carrier_path=carrier_path,
    )
    return metrics


def _write_outputs(
    output_dir: Path,
    *,
    frames: np.ndarray,
    calibration_stack: np.ndarray,
    calibration_z_nm: np.ndarray,
    predictions: Any,
    prediction_xyz_nm: np.ndarray,
    gt_frame: np.ndarray,
    gt_xyz_nm: np.ndarray,
    gt_indices: np.ndarray,
    match: MatchResult,
    metrics: dict[str, Any],
    gamma_nm: np.ndarray,
    mode_order: tuple[tuple[int, int], ...],
    config: LocalizationConfig,
    gamma_path: Path,
    carrier_complex: np.ndarray | None,
    carrier_path: Path | None,
) -> None:
    arrays_dir = output_dir / "arrays"
    figures_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    tables_dir = output_dir / "tables"
    for directory in (arrays_dir, figures_dir, metadata_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_dir / "independent_localizations.npz",
        frame_index=predictions.frame_index,
        x_nm=prediction_xyz_nm[:, 0],
        y_nm=prediction_xyz_nm[:, 1],
        z_nm=prediction_xyz_nm[:, 2],
        x_px=predictions.x_px,
        y_px=predictions.y_px,
        photons_adu=predictions.photons_adu,
        background_adu=predictions.background_adu,
        ncc=predictions.ncc,
        lobe_angle_rad=predictions.lobe_angle_rad,
        lobe_separation_px=predictions.lobe_separation_px,
        prediction_to_gt=match.prediction_to_gt,
    )
    _write_localization_table(
        tables_dir / "emitter_localizations.csv",
        predictions=predictions,
        prediction_xyz_nm=prediction_xyz_nm,
        match=match,
        gt_indices=gt_indices,
        gt_xyz_nm=gt_xyz_nm,
    )
    (metadata_dir / "localization_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    residual_model = carrier_path is not None
    manifest = {
        "localization_contract": (
            "independent full-frame lobe-pair detection followed by shared-carrier plus residual-gamma PSF fitting"
            if residual_model
            else "independent full-frame lobe-pair detection followed by direct-gamma PSF fitting"
        ),
        "ground_truth_usage": "post-hoc matching and metrics only; not used for candidate detection or fitting",
        "gamma_semantics": (
            "field-dependent residual OPD above fixed shared DH carrier"
            if residual_model
            else "direct total pupil coefficient field; no residual decomposition"
        ),
        "gamma_input": str(gamma_path.resolve()),
        "gamma_sha256": _sha256(gamma_path),
        "shared_carrier_input": str(carrier_path.resolve()) if carrier_path is not None else None,
        "shared_carrier_sha256": _sha256(carrier_path) if carrier_path is not None else None,
        "config": asdict(config),
    }
    (metadata_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rendered = _render_psf_artifacts(
        calibration_stack,
        calibration_z_nm,
        gamma_nm=gamma_nm,
        mode_order=mode_order,
        config=config,
        figures_dir=figures_dir,
        carrier_complex=carrier_complex,
    )
    np.savez_compressed(arrays_dir / "selected_xy_psf_grid.npz", **rendered)
    _render_localization_overlay(
        frames,
        predictions=predictions,
        gt_frame=gt_frame,
        gt_xyz_nm=gt_xyz_nm,
        config=config,
        path=figures_dir / "emitter_localization_gt_overlay.png",
    )
    _render_error_figure(match, gt_xyz_nm, path=figures_dir / "localization_errors.png")


def _write_localization_table(
    path: Path,
    *,
    predictions: Any,
    prediction_xyz_nm: np.ndarray,
    match: MatchResult,
    gt_indices: np.ndarray,
    gt_xyz_nm: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "frame_0based",
                "x_nm",
                "y_nm",
                "z_nm",
                "photons_adu",
                "background_adu",
                "ncc",
                "matched_gt_row_0based",
                "dx_nm",
                "dy_nm",
                "dz_nm",
            )
        )
        for index in range(predictions.frame_index.size):
            gt_local = int(match.prediction_to_gt[index])
            if gt_local >= 0:
                error = prediction_xyz_nm[index] - gt_xyz_nm[gt_local]
                gt_row = int(gt_indices[gt_local])
            else:
                error = (np.nan, np.nan, np.nan)
                gt_row = -1
            writer.writerow(
                (
                    int(predictions.frame_index[index]),
                    *prediction_xyz_nm[index],
                    float(predictions.photons_adu[index]),
                    float(predictions.background_adu[index]),
                    float(predictions.ncc[index]),
                    gt_row,
                    *error,
                )
            )


def _render_psf_artifacts(
    calibration_stack: np.ndarray,
    calibration_z_nm: np.ndarray,
    *,
    gamma_nm: np.ndarray,
    mode_order: tuple[tuple[int, int], ...],
    config: LocalizationConfig,
    figures_dir: Path,
    carrier_complex: np.ndarray | None,
) -> dict[str, np.ndarray]:
    model = DoubleHelixVectorPSF(
        mode_order=mode_order,
        na=config.na,
        wavelength_nm=config.wavelength_nm,
        pixel_size_nm=config.pixel_size_nm,
        refractive_index=config.refractive_index,
        npupil=config.npupil,
        psf_size=config.psf_size,
        device=config.device,
    )
    field = DirectGammaZernikeField(
        gamma_nm=torch.as_tensor(gamma_nm, dtype=torch.float32, device=config.device),
        mode_order=mode_order,
    )
    sample_planes = np.linspace(0, calibration_stack.shape[0] - 1, 5, dtype=int)
    center_x = torch.full((5,), 75.0, device=config.device)
    center_y = torch.full((5,), 75.0, device=config.device)
    center_coefficients = field.evaluate(-1.0 + 2.0 * center_x / 150.0, -1.0 + 2.0 * center_y / 150.0)
    with torch.no_grad():
        center_psf = model.render(
            coefficients_nm=center_coefficients,
            z_nm=torch.as_tensor(calibration_z_nm[sample_planes], dtype=torch.float32, device=config.device),
            carrier_complex=carrier_complex,
        ).cpu().numpy()
    observed = np.asarray(calibration_stack[sample_planes], dtype=np.float32)
    border = np.median(
        np.concatenate((observed[:, :3].reshape(5, -1), observed[:, -3:].reshape(5, -1)), axis=1),
        axis=1,
    )
    observed_unit = np.maximum(observed - border[:, None, None], 0.0)
    observed_unit /= observed_unit.sum(axis=(1, 2), keepdims=True)
    _plot_spot_comparison(
        observed_unit,
        center_psf,
        calibration_z_nm[sample_planes],
        figures_dir / "original_vs_zmap_reconstructed_spots.png",
    )

    positions = np.asarray(((20, 20), (75, 20), (130, 20), (20, 75), (75, 75), (130, 75), (20, 130), (75, 130), (130, 130)), dtype=np.float32)
    position_t = torch.as_tensor(positions, dtype=torch.float32, device=config.device)
    coefficients = field.evaluate(-1.0 + 2.0 * position_t[:, 0] / 150.0, -1.0 + 2.0 * position_t[:, 1] / 150.0)
    z_grid = np.full(positions.shape[0], 2000.0, dtype=np.float32)
    with torch.no_grad():
        psf_grid = model.render(
            coefficients_nm=coefficients,
            z_nm=torch.as_tensor(z_grid, device=config.device),
            carrier_complex=carrier_complex,
        ).cpu().numpy()
    _plot_xy_psf_grid(psf_grid, positions, figures_dir / "selected_xy_psf_grid.png")
    return {
        "xy_px": positions,
        "z_nm": z_grid,
        "psf_unit_flux": psf_grid.astype(np.float32),
        "comparison_plane_index": sample_planes,
        "comparison_z_nm": calibration_z_nm[sample_planes],
        "comparison_observed_unit_flux": observed_unit.astype(np.float32),
        "comparison_reconstructed_unit_flux": center_psf.astype(np.float32),
    }


def _plotting() -> Any:
    cache_dir = PROJECT_ROOT / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


def _plot_spot_comparison(observed: np.ndarray, reconstructed: np.ndarray, z_nm: np.ndarray, path: Path) -> None:
    plt = _plotting()
    fig, axes = plt.subplots(2, observed.shape[0], figsize=(7.2, 3.2), constrained_layout=True)
    for column in range(observed.shape[0]):
        limit = max(float(observed[column].max()), float(reconstructed[column].max()))
        axes[0, column].imshow(observed[column], cmap="viridis", vmin=0.0, vmax=limit)
        axes[1, column].imshow(reconstructed[column], cmap="viridis", vmin=0.0, vmax=limit)
        axes[0, column].set_title(f"z = {z_nm[column]:.1f} nm", fontsize=8)
        axes[1, column].set_xlabel("2.00 um / 10 px")
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Observed")
    axes[1, 0].set_ylabel("Zmap PSF")
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _plot_xy_psf_grid(psfs: np.ndarray, positions: np.ndarray, path: Path) -> None:
    plt = _plotting()
    fig, axes = plt.subplots(3, 3, figsize=(5.6, 5.6), constrained_layout=True)
    limit = float(psfs.max())
    for axis, psf, (x, y) in zip(axes.reshape(-1), psfs, positions, strict=True):
        axis.imshow(psf, cmap="viridis", vmin=0.0, vmax=limit)
        axis.set_title(f"({x * 0.2:.1f}, {y * 0.2:.1f}) um", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Recovered PSF across field at z = 2000 nm", fontsize=10)
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _render_localization_overlay(
    frames: np.ndarray,
    *,
    predictions: Any,
    gt_frame: np.ndarray,
    gt_xyz_nm: np.ndarray,
    config: LocalizationConfig,
    path: Path,
) -> None:
    plt = _plotting()
    selected_frames = np.linspace(0, frames.shape[0] - 1, min(4, frames.shape[0]), dtype=int)
    fig, axes = plt.subplots(1, selected_frames.size, figsize=(7.2, 2.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, frame_index in zip(axes, selected_frames, strict=True):
        image = frames[frame_index]
        axis.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99.8))
        gt_rows = gt_frame == frame_index
        gt_x = gt_xyz_nm[gt_rows, 0] / config.pixel_size_nm + config.image_origin_xy_px[0]
        gt_y = gt_xyz_nm[gt_rows, 1] / config.pixel_size_nm + config.image_origin_xy_px[1]
        pred_rows = predictions.frame_index == frame_index
        axis.scatter(gt_x, gt_y, s=36, facecolors="none", edgecolors="#56B4E9", linewidths=1.0, label="GT")
        axis.scatter(predictions.x_px[pred_rows], predictions.y_px[pred_rows], s=28, marker="x", color="#E69F00", linewidths=1.0, label="Detected")
        axis.set_title(f"Frame {frame_index + 1}")
        axis.set_xticks([])
        axis.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="outside upper center", ncols=2)
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _render_error_figure(match: MatchResult, gt_xyz_nm: np.ndarray, *, path: Path) -> None:
    plt = _plotting()
    matched_prediction = np.flatnonzero(match.prediction_to_gt >= 0)
    matched_gt = match.prediction_to_gt[matched_prediction]
    lateral = np.hypot(match.dx_nm, match.dy_nm)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    axes[0].hist(lateral, bins=40, color="#0072B2")
    axes[0].set_xlabel("Lateral error (nm)")
    axes[0].set_ylabel("Matched emitters")
    axes[1].hist(match.dz_nm, bins=40, color="#D55E00")
    axes[1].set_xlabel("Axial error (nm)")
    axes[2].scatter(gt_xyz_nm[matched_gt, 2], match.dz_nm, s=3, alpha=0.25, color="#009E73", rasterized=True)
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].set_xlabel("GT Z (nm)")
    axes[2].set_ylabel("Z error (nm)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent double-helix localization using direct field gamma.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--gamma-path", type=Path, default=DEFAULT_GAMMA_PATH)
    parser.add_argument("--carrier-complex", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--npupil", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--refinement-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold-sigma", type=float, default=15.0)
    parser.add_argument("--minimum-ncc", type=float, default=0.35)
    parser.add_argument("--lateral-tolerance-nm", type=float, default=500.0)
    parser.add_argument("--axial-tolerance-nm", type=float, default=500.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = LocalizationConfig(
        detection=LobeDetectionConfig(threshold_sigma=args.threshold_sigma),
        device=args.device,
        npupil=args.npupil,
        refinement_steps=args.refinement_steps,
        batch_size=args.batch_size,
        minimum_ncc=args.minimum_ncc,
    )
    provenance = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "gamma_sha256": _sha256(args.gamma_path),
        "shared_carrier_sha256": _sha256(args.carrier_complex) if args.carrier_complex is not None else None,
        "config": asdict(config),
    }
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    metrics = run_evaluation(
        dataset_root=args.dataset_root,
        gamma_path=args.gamma_path,
        output_dir=args.output_dir,
        config=config,
        max_frames=args.max_frames,
        lateral_tolerance_nm=args.lateral_tolerance_nm,
        axial_tolerance_nm=args.axial_tolerance_nm,
        carrier_path=args.carrier_complex,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": metrics}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["localization_metrics", "run_evaluation"]
