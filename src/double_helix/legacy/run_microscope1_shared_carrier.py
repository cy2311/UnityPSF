from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from .._paths import PROJECT_ROOT as V04_ROOT
from ..calibration import calibration_mode_order
from ..dataset import Microscope1Dataset
from ..pixel_pupil_calibration import (
    PixelPupilFitConfig,
    PixelPupilFitResult,
    fit_single_pixel_pupil,
    load_zernike_phase_initialization,
)


DEFAULT_DATASET_ROOT = (
    V04_ROOT.parent / "datasets/training_sets/double_helix/Simulated_datasets_Microscope1"
)
DEFAULT_INITIALIZATION = (
    V04_ROOT
    / "output/double_helix/microscope1_sample_mode21_steps3000_20260725"
    / "calibration/arrays/gamma_coefficients.npz"
)
DEFAULT_OUTPUT_DIR = (
    V04_ROOT
    / "output/double_helix/microscope1_shared_carrier_residual21_20260725/calibration"
)


@dataclass(frozen=True)
class Microscope1SharedCarrierOutputs:
    output_dir: Path
    carrier_path: Path
    residual_gamma_path: Path
    fit_path: Path
    metrics_path: Path
    manifest_path: Path
    required_paths: tuple[Path, ...]


def write_microscope1_shared_carrier_outputs(
    output_dir: str | Path,
    *,
    observed_adu: np.ndarray,
    result: PixelPupilFitResult,
    config: PixelPupilFitConfig,
    dataset_root: str | Path,
    source_tiff: str | Path,
    initialization_path: str | Path,
) -> Microscope1SharedCarrierOutputs:
    root = Path(output_dir)
    arrays_dir = root / "arrays"
    figures_dir = root / "figures"
    metadata_dir = root / "metadata"
    stacks_dir = root / "stacks"
    for directory in (arrays_dir, figures_dir, metadata_dir, stacks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    carrier_path = arrays_dir / "shared_double_helix_carrier.npz"
    np.savez_compressed(
        carrier_path,
        carrier_phase_rad=np.asarray(result.pupil_phase_rad, dtype=np.float32),
        carrier_complex=np.asarray(result.complex_pupil, dtype=np.complex64),
        pupil_shape_yx=np.asarray(result.complex_pupil.shape, dtype=np.int64),
        gauge=np.asarray("piston, tip, tilt, and Z(2,0) defocus fixed to zero"),
    )
    residual_mode_order = calibration_mode_order(21)
    residual_gamma_path = arrays_dir / "residual_gamma_initialization.npz"
    np.savez_compressed(
        residual_gamma_path,
        gamma_nm=np.zeros((len(residual_mode_order), 1, 1), dtype=np.float32),
        mode_order=np.asarray(residual_mode_order, dtype=np.int64),
        spatial_order=np.asarray(0, dtype=np.int64),
        semantics=np.asarray("zero residual OPD above independent shared DH carrier"),
    )
    fit_path = arrays_dir / "calibration_fit.npz"
    np.savez_compressed(
        fit_path,
        observed_adu=np.asarray(observed_adu, dtype=np.float32),
        reconstruction_adu=np.asarray(result.reconstruction_adu, dtype=np.float32),
        reconstruction_unit_flux=np.asarray(result.reconstruction_unit_flux, dtype=np.float32),
        photons_adu=np.asarray(result.photons_adu, dtype=np.float32),
        background_adu=np.asarray(result.background_adu, dtype=np.float32),
        z_nm=np.asarray(result.z_nm, dtype=np.float64),
        source_plane_indices=np.arange(len(result.z_nm), dtype=np.int64),
        train_indices=np.asarray(result.train_indices, dtype=np.int64),
        heldout_indices=np.asarray(result.heldout_indices, dtype=np.int64),
        dx_affine_px=np.asarray(result.dx_affine_px, dtype=np.float32),
        dy_affine_px=np.asarray(result.dy_affine_px, dtype=np.float32),
        per_plane_ncc=np.asarray(result.per_plane_ncc, dtype=np.float32),
        loss_history=np.asarray(result.loss_history, dtype=np.float32),
    )

    observed_stack_path = stacks_dir / "observed_adu.tif"
    reconstruction_stack_path = stacks_dir / "shared_carrier_reconstruction_adu.tif"
    tifffile.imwrite(observed_stack_path, np.asarray(observed_adu, dtype=np.float32))
    tifffile.imwrite(
        reconstruction_stack_path,
        np.asarray(result.reconstruction_adu, dtype=np.float32),
    )

    metrics = dict(result.metrics)
    metrics.update(
        {
            "calibration_plane_count": int(len(result.z_nm)),
            "carrier_pupil_shape_yx": list(map(int, result.complex_pupil.shape)),
            "residual_mode_count": len(residual_mode_order),
            "residual_initialization_nonzero_count": 0,
        }
    )
    metrics_path = metadata_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source = Path(source_tiff)
    initialization = Path(initialization_path)
    manifest = {
        "parameterization": (
            f"independent {config.npupil}x{config.npupil} shared DH pixel-pupil carrier "
            "plus 21 residual Zernike modes"
        ),
        "carrier_semantics": "shared phase-only free-form complex pupil recovered from Calib.tif",
        "carrier_counted_as_zernike_mode": False,
        "residual_semantics": "21 Zernike OPD maps above the fixed shared carrier",
        "residual_mode_count": len(residual_mode_order),
        "residual_mode_order": [list(mode) for mode in residual_mode_order],
        "residual_initialization": "all zeros; spatial field is selected by held-out validation later",
        "gauge": "piston, tip, tilt, and Z(2,0) defocus fixed to zero",
        "calibration_site_label": "calibration_site",
        "calibration_planes_used": int(len(result.z_nm)),
        "z_coordinate_semantics": "absolute calibration-stack coordinate",
        "z_min_nm": float(np.min(result.z_nm)),
        "z_max_nm": float(np.max(result.z_nm)),
        "dataset_root": str(Path(dataset_root).resolve()),
        "source_tiff": str(source.resolve()),
        "source_sha256": _sha256(source),
        "phase_initialization_npz": str(initialization.resolve()),
        "phase_initialization_sha256": _sha256(initialization),
        "phase_initialization_usage": "initial optimizer state only; not the final pupil model",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
        "config": asdict(config),
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    raw_unit = _photometry_normalized_raw(observed_adu, result)
    phase_figure_path = figures_dir / "shared_carrier_phase.png"
    xy_figure_path = figures_dir / "calibration_raw_vs_carrier_recon.png"
    sections_figure_path = figures_dir / "raw_vs_carrier_xz_yz.png"
    _render_phase(result.pupil_phase_rad, result.complex_pupil, phase_figure_path)
    _render_xy(raw_unit, result.reconstruction_unit_flux, result.z_nm, xy_figure_path)
    _render_sections(
        raw_unit,
        result.reconstruction_unit_flux,
        result.z_nm,
        config.pixel_size_nm,
        sections_figure_path,
    )
    required_paths = (
        carrier_path,
        residual_gamma_path,
        fit_path,
        metrics_path,
        manifest_path,
        observed_stack_path,
        reconstruction_stack_path,
        phase_figure_path,
        xy_figure_path,
        sections_figure_path,
    )
    return Microscope1SharedCarrierOutputs(
        output_dir=root,
        carrier_path=carrier_path,
        residual_gamma_path=residual_gamma_path,
        fit_path=fit_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        required_paths=required_paths,
    )


def recover_microscope1_shared_carrier(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    initialization_path: str | Path,
    config: PixelPupilFitConfig,
) -> Microscope1SharedCarrierOutputs:
    dataset = Microscope1Dataset(dataset_root)
    contract = dataset.validate()
    if config.npupil != 128 or config.roi_size != 31 or config.pixel_size_nm != 200.0:
        raise ValueError("Microscope1 shared-carrier recovery requires npupil=128, roi_size=31, pixel_size_nm=200.")
    observed = dataset.read_calibration()
    z_nm = dataset.z_sign * (
        np.arange(contract.calibration_shape[0], dtype=np.float64) + dataset.z_index_origin
    ) * dataset.z_step_nm
    initial_phase = load_zernike_phase_initialization(
        initialization_path,
        npupil=config.npupil,
        wavelength_nm=config.wavelength_nm,
    )
    result = fit_single_pixel_pupil(
        observed,
        z_nm=z_nm,
        initial_phase_rad=initial_phase,
        config=config,
    )
    return write_microscope1_shared_carrier_outputs(
        output_dir,
        observed_adu=observed,
        result=result,
        config=config,
        dataset_root=dataset_root,
        source_tiff=dataset.calibration_path,
        initialization_path=initialization_path,
    )


def _photometry_normalized_raw(
    observed_adu: np.ndarray,
    result: PixelPupilFitResult,
) -> np.ndarray:
    signal = np.maximum(
        np.asarray(observed_adu, dtype=np.float32) - result.background_adu[:, None, None],
        0.0,
    )
    unit = signal / result.photons_adu[:, None, None].clip(min=1e-6)
    return unit / unit.sum(axis=(-2, -1), keepdims=True).clip(min=1e-12)


def _plotting() -> Any:
    cache_dir = V04_ROOT.parent / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


def _render_xy(raw: np.ndarray, recon: np.ndarray, z_nm: np.ndarray, path: Path) -> None:
    plt = _plotting()
    selected = np.linspace(0, len(z_nm) - 1, min(11, len(z_nm)), dtype=np.int64)
    figure, axes = plt.subplots(2, len(selected), figsize=(14.0, 3.5), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(selected))
    for column, plane in enumerate(selected):
        vmax = max(float(raw[plane].max()), float(recon[plane].max()), 1e-8)
        axes[0, column].imshow(raw[plane], cmap="magma", vmin=0.0, vmax=vmax)
        axes[1, column].imshow(recon[plane], cmap="magma", vmin=0.0, vmax=vmax)
        axes[0, column].set_title(f"{z_nm[plane]:.1f} nm", fontsize=7)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Raw")
    axes[1, 0].set_ylabel("Shared carrier recon")
    figure.suptitle("Microscope1 calibration site: raw versus shared-carrier reconstruction")
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _render_sections(
    raw: np.ndarray,
    recon: np.ndarray,
    z_nm: np.ndarray,
    pixel_size_nm: float,
    path: Path,
) -> None:
    plt = _plotting()
    half_width_um = 0.5 * raw.shape[-1] * pixel_size_nm / 1000.0
    extent = (-half_width_um, half_width_um, float(z_nm[-1] / 1000.0), float(z_nm[0] / 1000.0))
    volumes = (raw.max(axis=1), recon.max(axis=1), raw.max(axis=2), recon.max(axis=2))
    vmax = max(float(volume.max()) for volume in volumes)
    figure, axes = plt.subplots(1, 4, figsize=(11.0, 4.0), constrained_layout=True)
    for axis, volume, title in zip(
        axes,
        volumes,
        ("Raw XZ", "Carrier recon XZ", "Raw YZ", "Carrier recon YZ"),
        strict=True,
    ):
        axis.imshow(volume, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto", extent=extent)
        axis.set_title(title)
        axis.set_xlabel("X (um)" if "XZ" in title else "Y (um)")
    axes[0].set_ylabel("Z (um)")
    figure.suptitle("Microscope1 calibration site axial PSF sections")
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _render_phase(phase_rad: np.ndarray, complex_pupil: np.ndarray, path: Path) -> None:
    plt = _plotting()
    wrapped = np.angle(np.exp(1j * np.asarray(phase_rad)))
    aperture = np.abs(complex_pupil) > 0.0
    figure, axis = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    artist = axis.imshow(
        np.ma.masked_where(~aperture, wrapped),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axis.set_title("Independent shared double-helix carrier phase")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(artist, ax=axis, label="Phase (rad)")
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover the independent Microscope1 shared DH carrier before residual-21 fitting."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--initialization", type=Path, default=DEFAULT_INITIALIZATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alternating-rounds", type=int, default=2)
    parser.add_argument("--local-steps", type=int, default=100)
    parser.add_argument("--phase-adam-steps", type=int, default=500)
    parser.add_argument("--lbfgs-steps", type=int, default=80)
    parser.add_argument("--phase-learning-rate", type=float, default=0.03)
    parser.add_argument("--phase-anchor-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for formal shared-carrier recovery but is unavailable.")
    config = PixelPupilFitConfig(
        pixel_size_nm=200.0,
        roi_size=31,
        npupil=128,
        alternating_rounds=args.alternating_rounds,
        local_steps=args.local_steps,
        phase_adam_steps=args.phase_adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        phase_learning_rate=args.phase_learning_rate,
        phase_anchor_weight=args.phase_anchor_weight,
        seed=args.seed,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
                "architecture": "independent shared carrier + 21 residual Zernike modes",
                "config": asdict(config),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    outputs = recover_microscope1_shared_carrier(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        initialization_path=args.initialization,
        config=config,
    )
    print(
        json.dumps(
            {"output_dir": str(outputs.output_dir), "required_paths": [str(path) for path in outputs.required_paths]},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Microscope1SharedCarrierOutputs",
    "recover_microscope1_shared_carrier",
    "write_microscope1_shared_carrier_outputs",
]
