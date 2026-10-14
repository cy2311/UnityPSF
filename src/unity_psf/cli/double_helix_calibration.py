from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from ..optics.psf.double_helix.calibration import (
    SUPPORTED_CALIBRATION_MODE_COUNTS,
    CalibrationFitConfig,
    CalibrationFitResult,
    calibration_fit_plane_indices,
    calibration_fit_z_nm,
    calibration_config_dict,
    calibration_mode_order,
    fit_calibration_stack,
)
from ..optics.psf.double_helix.dataset import Microscope1Dataset
from ..optics.psf.double_helix.vector_model import evaluate_normalized_zernike


PROJECT_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "datasets/training_sets/double_helix/Simulated_datasets_Microscope1"


def default_calibration_output_dir(mode_count: int) -> Path:
    if mode_count not in SUPPORTED_CALIBRATION_MODE_COUNTS:
        raise ValueError(
            f"Supported mode counts are {SUPPORTED_CALIBRATION_MODE_COUNTS}; got {mode_count}."
        )
    return PROJECT_ROOT / f"output/double_helix/microscope1_mode{mode_count}/calibration"


DEFAULT_OUTPUT_DIR = default_calibration_output_dir(13)


@dataclass(frozen=True)
class CalibrationOutputs:
    output_dir: Path
    gamma_path: Path
    zmap_path: Path
    fit_path: Path
    metrics_path: Path
    manifest_path: Path
    closure_figure_path: Path
    pupil_figure_path: Path
    coefficients_figure_path: Path


def write_calibration_outputs(
    output_dir: str | Path,
    *,
    observed_adu: np.ndarray,
    result: CalibrationFitResult,
    config: CalibrationFitConfig,
) -> CalibrationOutputs:
    root = Path(output_dir)
    arrays_dir = root / "arrays"
    stacks_dir = root / "stacks"
    figures_dir = root / "figures"
    metadata_dir = root / "metadata"
    config_dir = root / "config"
    for directory in (arrays_dir, stacks_dir, figures_dir, metadata_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    selected_observed_adu = np.asarray(observed_adu)[result.source_plane_indices]

    mode_order = np.asarray(result.mode_order, dtype=np.int64)
    gamma_path = arrays_dir / "gamma_coefficients.npz"
    np.savez_compressed(
        gamma_path,
        gamma_nm=np.asarray(result.gamma_nm, dtype=np.float32),
        mode_order=mode_order,
        spatial_order=np.asarray(0, dtype=np.int64),
    )
    zmap_path = arrays_dir / "alternating_full_roi_zernike_maps_nm.npz"
    coefficient_maps = np.broadcast_to(
        np.asarray(result.gamma_nm, dtype=np.float32),
        (len(result.mode_order), 150, 150),
    ).copy()
    np.savez_compressed(zmap_path, zernike_maps_nm=coefficient_maps, mode_order=mode_order)

    fit_path = arrays_dir / "calibration_fit.npz"
    np.savez_compressed(
        fit_path,
        observed_adu=np.asarray(selected_observed_adu, dtype=np.float32),
        reconstruction_adu=np.asarray(result.reconstruction_adu, dtype=np.float32),
        reconstruction_unit_flux=np.asarray(result.reconstruction_unit_flux, dtype=np.float32),
        z_nm=np.asarray(result.z_nm, dtype=np.float64),
        stage_z_nm=np.asarray(
            result.z_nm if result.stage_z_nm is None else result.stage_z_nm,
            dtype=np.float64,
        ),
        photons_adu=np.asarray(result.photons_adu, dtype=np.float32),
        background_adu=np.asarray(result.background_adu, dtype=np.float32),
        dx_affine_px=np.asarray(result.dx_affine_px, dtype=np.float32),
        dy_affine_px=np.asarray(result.dy_affine_px, dtype=np.float32),
        source_plane_indices=np.asarray(result.source_plane_indices, dtype=np.int64),
        train_indices=np.asarray(result.train_indices, dtype=np.int64),
        heldout_indices=np.asarray(result.heldout_indices, dtype=np.int64),
    )
    tifffile.imwrite(stacks_dir / "reconstruction_adu.tif", result.reconstruction_adu)
    tifffile.imwrite(stacks_dir / "reconstruction_unit_flux.tif", result.reconstruction_unit_flux)

    metrics_path = metadata_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "accepted": bool(result.metrics.get("accepted", False)),
        "mode_contract": f"fixed {len(result.mode_order)} modes after excluded (2,0) defocus gauge",
        "gamma_semantics": "direct total pupil coefficient field; no residual decomposition",
        "gauge": "piston, tip, tilt, and defocus excluded",
        "wavelength_nm": config.wavelength_nm,
        "wavelength_source": config.wavelength_source,
        "na": config.na,
        "refractive_index": config.refractive_index,
        "pixel_size_nm": config.pixel_size_nm,
        "z_step_nm": config.z_step_nm,
        "z_sign": config.z_sign,
        "z_index_origin": config.z_index_origin,
        "mode_count": len(result.mode_order),
        "fit_z_range_nm": config.fit_z_range_nm,
        "z_coordinate_semantics": (
            "relative to calibration-stack midpoint"
            if config.fit_z_range_nm is not None
            else "absolute calibration-stack coordinate"
        ),
        "source_plane_count": int(len(result.source_plane_indices)),
        "learning_rate_schedule": config.learning_rate_schedule,
        "minimum_learning_rate_ratio": config.minimum_learning_rate_ratio,
        "warm_start_mode_count": config.warm_start_mode_count,
        "deep_z_loss": config.deep_z_loss,
        "optimize_z_calibration": config.optimize_z_calibration,
        "fitted_z_offset_nm": float(result.metrics.get("fitted_z_offset_nm", 0.0)),
        "fitted_z_scale": float(result.metrics.get("fitted_z_scale", 1.0)),
        "z_bin_edges_nm": list(config.z_bin_edges_nm),
        "z_bin_weights": list(config.z_bin_weights),
        "paired_helix_loss_weight": config.paired_helix_loss_weight,
        "paired_helix_component_weights": {
            "angle": config.paired_angle_weight,
            "separation": config.paired_separation_weight,
            "center": config.paired_center_weight,
        },
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (config_dir / "resolved_config.json").write_text(
        json.dumps(calibration_config_dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    closure_path = figures_dir / "calibration_closure.png"
    pupil_path = figures_dir / "pupil_phase.png"
    coefficients_path = figures_dir / "zernike_coefficients.png"
    _render_calibration_closure(selected_observed_adu, result, config, closure_path)
    _render_pupil_phase(result, config, pupil_path)
    _render_coefficients(result, coefficients_path)
    return CalibrationOutputs(
        output_dir=root,
        gamma_path=gamma_path,
        zmap_path=zmap_path,
        fit_path=fit_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        closure_figure_path=closure_path,
        pupil_figure_path=pupil_path,
        coefficients_figure_path=coefficients_path,
    )


def _plotting() -> Any:
    cache_dir = PROJECT_ROOT / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


def _render_calibration_closure(
    observed_adu: np.ndarray,
    result: CalibrationFitResult,
    config: CalibrationFitConfig,
    path: Path,
) -> None:
    plt = _plotting()
    plane_indices = np.linspace(0, len(result.z_nm) - 1, min(5, len(result.z_nm)), dtype=int)
    observed = np.asarray(observed_adu)[plane_indices]
    reconstructed = np.asarray(result.reconstruction_adu)[plane_indices]
    difference = observed - reconstructed
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7}):
        fig, axes = plt.subplots(3, len(plane_indices), figsize=(7.2, 4.2), constrained_layout=True)
        axes = np.asarray(axes).reshape(3, len(plane_indices))
        intensity_limit = float(np.percentile(observed, 99.8))
        difference_limit = max(float(np.percentile(np.abs(difference), 99.0)), 1.0)
        for column, plane in enumerate(plane_indices):
            axes[0, column].imshow(observed[column], cmap="viridis", vmin=0.0, vmax=intensity_limit)
            axes[1, column].imshow(reconstructed[column], cmap="viridis", vmin=0.0, vmax=intensity_limit)
            axes[2, column].imshow(difference[column], cmap="RdBu_r", vmin=-difference_limit, vmax=difference_limit)
            axes[0, column].set_title(f"z = {result.z_nm[plane]:.1f} nm")
            axes[2, column].set_xlabel(f"X ({config.pixel_size_nm:g} nm/pixel)")
            for row in range(3):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        for row, label in enumerate(("Observed (ADU)", "Reconstructed (ADU)", "Difference (ADU)")):
            axes[row, 0].set_ylabel(label)
        fig.savefig(path, dpi=300, facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)


def _render_pupil_phase(
    result: CalibrationFitResult,
    config: CalibrationFitConfig,
    path: Path,
) -> None:
    import torch

    plt = _plotting()
    coordinates = torch.linspace(-1.0, 1.0, 257, dtype=torch.float32)
    y_pupil, x_pupil = torch.meshgrid(coordinates, coordinates, indexing="ij")
    basis = evaluate_normalized_zernike(result.mode_order, x_pupil, y_pupil).numpy()
    optical_path_nm = np.einsum("c,chw->hw", result.gamma_nm[:, 0, 0], basis)
    phase = np.angle(np.exp(2j * np.pi * optical_path_nm / config.wavelength_nm))
    phase[x_pupil.square().add(y_pupil.square()).numpy() >= 1.0] = np.nan
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    artist = ax.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=(-1, 1, 1, -1))
    ax.set_xlabel("Normalized pupil X")
    ax.set_ylabel("Normalized pupil Y")
    ax.set_title("Recovered double-helix pupil phase")
    colorbar = fig.colorbar(artist, ax=ax)
    colorbar.set_label("Phase (rad)")
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _render_coefficients(result: CalibrationFitResult, path: Path) -> None:
    plt = _plotting()
    values = result.gamma_nm[:, 0, 0]
    labels = [f"({n},{m:+d})" for n, m in result.mode_order]
    colors = np.where(values >= 0, "#0072B2", "#D55E00")
    width = max(7.2, 0.16 * len(values))
    fig, ax = plt.subplots(figsize=(width, 3.2), constrained_layout=True)
    ax.bar(np.arange(len(values)), values, color=colors, width=0.8)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(np.arange(len(values)), labels, rotation=90)
    ax.set_xlabel("Zernike mode (n,m)")
    ax.set_ylabel("Direct gamma coefficient (nm)")
    ax.set_title("Recovered total pupil coefficients")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(
    input_paths: dict[str, Path],
    config: CalibrationFitConfig,
) -> dict[str, Any]:
    try:
        git_sha = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
        "git_sha": git_sha,
        "torch_cuda_available": __import__("torch").cuda.is_available(),
        "config": calibration_config_dict(config),
        "inputs_sha256": {name: _sha256(path) for name, path in input_paths.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a direct-gamma Zernike double-helix pupil.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--calibration-tiff", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode-count", type=int, choices=SUPPORTED_CALIBRATION_MODE_COUNTS, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wavelength-nm", type=float, default=660.0)
    parser.add_argument("--na", type=float, default=1.27)
    parser.add_argument("--refractive-index", type=float, default=1.33)
    parser.add_argument("--pixel-size-nm", type=float, default=200.0)
    parser.add_argument("--z-step-nm", type=float, default=33.3)
    parser.add_argument("--z-index-origin", type=float, default=1.0)
    parser.add_argument("--z-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--npupil", type=int, default=128)
    parser.add_argument("--psf-size", type=int, default=31)
    parser.add_argument("--adam-steps", type=int, default=300)
    parser.add_argument("--fit-z-range-nm", type=float)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--restart-count", type=int, default=4)
    parser.add_argument("--warm-start-gamma", type=Path)
    parser.add_argument("--warm-start-noise-nm", type=float, default=10.0)
    parser.add_argument("--optimize-z-calibration", action="store_true")
    parser.add_argument("--z-calibration-learning-rate", type=float, default=0.01)
    parser.add_argument("--deep-z-loss", action="store_true")
    parser.add_argument("--poisson-loss-weight", type=float, default=1.0)
    parser.add_argument("--ncc-loss-weight", type=float, default=2.0)
    parser.add_argument("--lobe-geometry-loss-weight", type=float, default=3.0)
    parser.add_argument("--paired-helix-loss-weight", type=float, default=0.0)
    parser.add_argument("--paired-angle-weight", type=float, default=1.0)
    parser.add_argument("--paired-separation-weight", type=float, default=1.0)
    parser.add_argument("--paired-center-weight", type=float, default=1.0)
    parser.add_argument("--z-bin-edges-nm", type=float, nargs="+", default=(400.0, 800.0))
    parser.add_argument("--z-bin-weights", type=float, nargs="+", default=(1.0, 1.5, 2.5))
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    initial_gamma_nm = None
    warm_start_mode_count = 0
    if args.warm_start_gamma is not None:
        with np.load(args.warm_start_gamma, allow_pickle=False) as payload:
            initial_gamma_nm = np.asarray(payload["gamma_nm"], dtype=np.float32)
            initial_mode_order = tuple(tuple(map(int, mode)) for mode in payload["mode_order"])
        target_mode_order = calibration_mode_order(args.mode_count)
        if target_mode_order[: len(initial_mode_order)] != initial_mode_order:
            raise ValueError("Warm-start Zernike mode order is not a prefix of the target mode order.")
        warm_start_mode_count = len(initial_mode_order)
    config = CalibrationFitConfig(
        mode_count=args.mode_count,
        wavelength_nm=args.wavelength_nm,
        na=args.na,
        refractive_index=args.refractive_index,
        pixel_size_nm=args.pixel_size_nm,
        z_step_nm=args.z_step_nm,
        z_index_origin=args.z_index_origin,
        z_sign=args.z_sign,
        npupil=args.npupil,
        psf_size=args.psf_size,
        adam_steps=args.adam_steps,
        fit_z_range_nm=args.fit_z_range_nm,
        learning_rate_schedule=args.learning_rate_schedule,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        restart_count=args.restart_count,
        warm_start_mode_count=warm_start_mode_count,
        warm_start_noise_nm=args.warm_start_noise_nm,
        optimize_z_calibration=args.optimize_z_calibration,
        z_calibration_learning_rate=args.z_calibration_learning_rate,
        deep_z_loss=args.deep_z_loss,
        poisson_loss_weight=args.poisson_loss_weight,
        ncc_loss_weight=args.ncc_loss_weight,
        lobe_geometry_loss_weight=args.lobe_geometry_loss_weight,
        paired_helix_loss_weight=args.paired_helix_loss_weight,
        paired_angle_weight=args.paired_angle_weight,
        paired_separation_weight=args.paired_separation_weight,
        paired_center_weight=args.paired_center_weight,
        z_bin_edges_nm=tuple(args.z_bin_edges_nm),
        z_bin_weights=tuple(args.z_bin_weights),
        seed=args.seed,
        device=args.device,
    )
    if args.calibration_tiff is None:
        dataset = Microscope1Dataset(args.dataset_root)
        dataset.validate()
        observed = dataset.read_calibration()
        input_paths = {
            "Calib.tif": dataset.calibration_path,
            "Dens5_noisy5000.tif": dataset.frames_path,
            "Dens5_noisy5000_GT.txt": dataset.gt_path,
        }
    else:
        observed = np.asarray(tifffile.imread(args.calibration_tiff), dtype=np.float32)
        input_paths = {"calibration_tiff": args.calibration_tiff}
    if args.warm_start_gamma is not None:
        input_paths["warm_start_gamma"] = args.warm_start_gamma
    provenance = _provenance(input_paths, config)
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    source_plane_indices = calibration_fit_plane_indices(len(observed), config=config)
    fit_z_nm = calibration_fit_z_nm(len(observed), config=config)
    print(
        json.dumps(
            {
                "fit_selection": {
                    "source_plane_count": int(len(source_plane_indices)),
                    "source_plane_first": int(source_plane_indices[0]),
                    "source_plane_last": int(source_plane_indices[-1]),
                    "z_min_nm": float(fit_z_nm.min()),
                    "z_max_nm": float(fit_z_nm.max()),
                }
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    result = fit_calibration_stack(observed, config=config, initial_gamma_nm=initial_gamma_nm)
    output_dir = args.output_dir or default_calibration_output_dir(args.mode_count)
    outputs = write_calibration_outputs(output_dir, observed_adu=observed, result=result, config=config)
    print(json.dumps({"outputs": str(outputs.output_dir), "metrics": result.metrics}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationOutputs",
    "default_calibration_output_dir",
    "parse_args",
    "write_calibration_outputs",
]
