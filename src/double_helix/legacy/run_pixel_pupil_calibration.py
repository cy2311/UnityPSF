from __future__ import annotations

import argparse
import csv
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
from ..lg_calibration import detect_stable_dh_centers, extract_centered_roi_stacks
from ..pixel_pupil_calibration import (
    PixelPupilFitConfig,
    PixelPupilFitResult,
    fit_single_pixel_pupil,
    load_zernike_phase_initialization,
)


DEFAULT_TIFF = V04_ROOT.parent / "datasets/training_sets/double_helix/Real_dataset_Tcell/Calib.tif"
DEFAULT_INITIALIZATION = (
    V04_ROOT
    / "output/double_helix/real_tcell_deepz_mode64_weight_sweep_z2000/w5434"
    / "calibration/arrays/gamma_coefficients.npz"
)
DEFAULT_OUTPUT_DIR = (
    V04_ROOT
    / "output/double_helix/real_tcell_independent_pixel_pupils_w5434_init_z2000/calibration"
)


@dataclass(frozen=True)
class PixelPupilOutputs:
    pupil_path: Path
    fit_path: Path
    metrics_path: Path
    manifest_path: Path
    per_plane_metrics_path: Path
    figure_paths: tuple[Path, ...]


def write_single_bead_outputs(
    output_dir: str | Path,
    *,
    bead_index: int,
    center_yx: np.ndarray,
    observed_adu: np.ndarray,
    result: PixelPupilFitResult,
    config: PixelPupilFitConfig,
    source_tiff: str | Path,
    source_plane_indices: np.ndarray,
    initialization_path: str | Path,
) -> PixelPupilOutputs:
    root = Path(output_dir)
    arrays_dir = root / "arrays"
    figures_dir = root / "figures"
    metadata_dir = root / "metadata"
    tables_dir = root / "tables"
    stacks_dir = root / "stacks"
    for directory in (arrays_dir, figures_dir, metadata_dir, tables_dir, stacks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bead_number = int(bead_index) + 1
    stem = f"bead_{bead_number:02d}"
    pupil_path = arrays_dir / f"{stem}_complex_pupil.npz"
    np.savez_compressed(
        pupil_path,
        pupil_phase_rad=result.pupil_phase_rad,
        complex_pupil=result.complex_pupil,
        center_yx=np.asarray(center_yx, dtype=np.float32),
        gauge=np.asarray("piston, tip, tilt, and Z(2,0) fixed to zero"),
    )
    fit_path = arrays_dir / f"{stem}_calibration_fit.npz"
    np.savez_compressed(
        fit_path,
        observed_adu=np.asarray(observed_adu, dtype=np.float32),
        reconstruction_adu=result.reconstruction_adu,
        reconstruction_unit_flux=result.reconstruction_unit_flux,
        photons_adu=result.photons_adu,
        background_adu=result.background_adu,
        z_nm=result.z_nm,
        source_plane_indices=np.asarray(source_plane_indices, dtype=np.int64),
        train_indices=result.train_indices,
        heldout_indices=result.heldout_indices,
        dx_affine_px=result.dx_affine_px,
        dy_affine_px=result.dy_affine_px,
        per_plane_ncc=result.per_plane_ncc,
        loss_history=result.loss_history,
    )
    _write_tiff(stacks_dir / f"{stem}_observed_adu.tif", observed_adu)
    _write_tiff(stacks_dir / f"{stem}_reconstruction_adu.tif", result.reconstruction_adu)

    metrics = dict(result.metrics)
    metrics.update(
        {
            "bead_index": int(bead_index),
            "bead_number": bead_number,
            "center_yx": np.asarray(center_yx, dtype=float).tolist(),
        }
    )
    metrics_path = metadata_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    source_path = Path(source_tiff)
    initialization = Path(initialization_path)
    manifest = {
        "source_tiff": str(source_path.resolve()),
        "source_sha256": _sha256(source_path) if source_path.is_file() else None,
        "initialization_npz": str(initialization.resolve()),
        "initialization_sha256": _sha256(initialization) if initialization.is_file() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "parameterization": "independent phase-only pixel pupil",
        "gauge": "piston, tip, tilt, and Z(2,0) defocus fixed to zero",
        "z_calibration": "fixed z0=0 nm and z_scale=1",
        "config": asdict(config),
        "bead_index": int(bead_index),
        "center_yx": np.asarray(center_yx, dtype=float).tolist(),
    }
    manifest_path = metadata_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    per_plane_metrics_path = tables_dir / f"{stem}_per_plane_metrics.csv"
    train_set = set(map(int, result.train_indices))
    with per_plane_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("plane_index", "source_plane_index", "z_nm", "split", "ncc"))
        for plane, (source_plane, z_value, ncc) in enumerate(
            zip(source_plane_indices, result.z_nm, result.per_plane_ncc, strict=True)
        ):
            writer.writerow(
                (
                    plane,
                    int(source_plane),
                    f"{float(z_value):.6f}",
                    "train" if plane in train_set else "heldout",
                    f"{float(ncc):.8f}",
                )
            )

    raw_unit = _photometry_normalized_raw(observed_adu, result)
    xy_path = figures_dir / f"{stem}_raw_vs_recon_xy.png"
    sections_path = figures_dir / f"{stem}_raw_vs_recon_xz_yz.png"
    phase_path = figures_dir / f"{stem}_pixel_pupil_phase.png"
    complex_path = figures_dir / f"{stem}_pixel_pupil_complex.png"
    _render_xy(raw_unit, result.reconstruction_unit_flux, result.z_nm, bead_number, center_yx, xy_path)
    _render_sections(
        raw_unit,
        result.reconstruction_unit_flux,
        result.z_nm,
        config.pixel_size_nm,
        bead_number,
        sections_path,
    )
    _render_phase(result.pupil_phase_rad, bead_number, phase_path)
    _render_complex_pupil(result.complex_pupil, bead_number, complex_path)
    return PixelPupilOutputs(
        pupil_path=pupil_path,
        fit_path=fit_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        per_plane_metrics_path=per_plane_metrics_path,
        figure_paths=(xy_path, sections_path, phase_path, complex_path),
    )


def _write_tiff(path: Path, values: np.ndarray) -> None:
    import tifffile

    tifffile.imwrite(path, np.asarray(values, dtype=np.float32))


def _plotting() -> Any:
    cache_dir = V04_ROOT.parent / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


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


def _render_xy(
    raw: np.ndarray,
    recon: np.ndarray,
    z_nm: np.ndarray,
    bead_number: int,
    center_yx: np.ndarray,
    path: Path,
) -> None:
    plt = _plotting()
    selected = np.linspace(0, len(z_nm) - 1, min(11, len(z_nm)), dtype=int)
    figure, axes = plt.subplots(2, len(selected), figsize=(14.0, 3.4), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(selected))
    for column, plane in enumerate(selected):
        vmax = max(float(raw[plane].max()), float(recon[plane].max()), 1e-8)
        axes[0, column].imshow(raw[plane], cmap="magma", vmin=0.0, vmax=vmax)
        axes[1, column].imshow(recon[plane], cmap="magma", vmin=0.0, vmax=vmax)
        axes[0, column].set_title(f"{z_nm[plane]:.0f} nm", fontsize=8)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Raw")
    axes[1, 0].set_ylabel("Recon")
    figure.suptitle(
        f"Bead {bead_number} raw vs pixel-pupil reconstruction | "
        f"center (y,x)=({center_yx[0]:.0f},{center_yx[1]:.0f})"
    )
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def _render_sections(
    raw: np.ndarray,
    recon: np.ndarray,
    z_nm: np.ndarray,
    pixel_size_nm: float,
    bead_number: int,
    path: Path,
) -> None:
    plt = _plotting()
    half_width_um = 0.5 * raw.shape[-1] * pixel_size_nm / 1000.0
    extent = (-half_width_um, half_width_um, float(z_nm[0]), float(z_nm[-1]))
    volumes = (raw.max(axis=1), recon.max(axis=1), raw.max(axis=2), recon.max(axis=2))
    vmax = max(float(volume.max()) for volume in volumes)
    figure, axes = plt.subplots(1, 4, figsize=(11.0, 4.0), constrained_layout=True)
    for axis, volume, title in zip(
        axes,
        volumes,
        ("Raw XZ", "Recon XZ", "Raw YZ", "Recon YZ"),
        strict=True,
    ):
        axis.imshow(
            volume,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            origin="lower",
            aspect="auto",
            extent=extent,
        )
        axis.set(xlabel="X (um)" if "XZ" in title else "Y (um)", title=title)
    axes[0].set_ylabel("Z (nm)")
    figure.suptitle(f"Bead {bead_number} axial PSF sections")
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def _render_phase(phase_rad: np.ndarray, bead_number: int, path: Path) -> None:
    plt = _plotting()
    figure, axis = plt.subplots(figsize=(4.5, 4.0), constrained_layout=True)
    yy, xx = np.indices(phase_rad.shape)
    center = 0.5 * (np.asarray(phase_rad.shape) - 1.0)
    aperture = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 < (phase_rad.shape[0] / 2.0) ** 2
    wrapped = np.angle(np.exp(1j * phase_rad))
    masked = np.ma.masked_where(~aperture, wrapped)
    artist = axis.imshow(masked, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axis.set_title(f"Bead {bead_number} recovered pixel-pupil phase")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(artist, ax=axis, label="Phase (rad)")
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def _render_complex_pupil(pupil: np.ndarray, bead_number: int, path: Path) -> None:
    plt = _plotting()
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    for axis, values, title, cmap in zip(
        axes,
        (np.abs(pupil), pupil.real, pupil.imag),
        ("Amplitude", "Real", "Imaginary"),
        ("gray", "RdBu_r", "RdBu_r"),
        strict=True,
    ):
        limit = 1.0
        artist = axis.imshow(values, cmap=cmap, vmin=0.0 if title == "Amplitude" else -limit, vmax=limit)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(artist, ax=axis, shrink=0.8)
    figure.suptitle(f"Bead {bead_number} recovered complex pupil")
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_real_tcell_bead(
    *,
    source_tiff: str | Path,
    output_dir: str | Path,
    initialization_path: str | Path,
    bead_number: int,
    fit_z_range_nm: float,
    z_step_nm: float,
    config: PixelPupilFitConfig,
) -> PixelPupilOutputs:
    if not 1 <= int(bead_number) <= 5:
        raise ValueError("bead_number must be between 1 and 5.")
    source_stack = tifffile.imread(source_tiff)
    centers = detect_stable_dh_centers(
        source_stack,
        roi_size=config.roi_size,
        maximum_count=5,
        minimum_distance_px=9,
    )
    roi_stacks = extract_centered_roi_stacks(source_stack, centers, roi_size=config.roi_size)
    full_z_nm = (
        np.arange(source_stack.shape[0], dtype=np.float64)
        - 0.5 * (source_stack.shape[0] - 1)
    ) * float(z_step_nm)
    source_plane_indices = np.flatnonzero(np.abs(full_z_nm) <= float(fit_z_range_nm) + 1e-6)
    initial_phase = load_zernike_phase_initialization(
        initialization_path,
        npupil=config.npupil,
        wavelength_nm=config.wavelength_nm,
    )
    bead_index = int(bead_number) - 1
    result = fit_single_pixel_pupil(
        roi_stacks[bead_index, source_plane_indices],
        z_nm=full_z_nm[source_plane_indices],
        initial_phase_rad=initial_phase,
        config=config,
    )
    return write_single_bead_outputs(
        output_dir,
        bead_index=bead_index,
        center_yx=centers[bead_index],
        observed_adu=roi_stacks[bead_index, source_plane_indices],
        result=result,
        config=config,
        source_tiff=source_tiff,
        source_plane_indices=source_plane_indices,
        initialization_path=initialization_path,
    )


def aggregate_pixel_pupil_outputs(
    output_dir: str | Path,
    *,
    bead_count: int = 5,
) -> dict[str, str]:
    root = Path(output_dir)
    figures_dir = root / "figures"
    metadata_dir = root / "metadata"
    tables_dir = root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    phases = []
    centers = []
    raw_stacks = []
    recon_stacks = []
    z_values = []
    for bead_number in range(1, int(bead_count) + 1):
        stem = f"bead_{bead_number:02d}"
        metrics_path = metadata_dir / f"{stem}_metrics.json"
        fit_path = root / "arrays" / f"{stem}_calibration_fit.npz"
        pupil_path = root / "arrays" / f"{stem}_complex_pupil.npz"
        metrics.append(json.loads(metrics_path.read_text()))
        with np.load(fit_path, allow_pickle=False) as fit:
            raw = np.asarray(fit["observed_adu"], dtype=np.float32)
            background = np.asarray(fit["background_adu"], dtype=np.float32)
            photons = np.asarray(fit["photons_adu"], dtype=np.float32)
            signal = np.maximum(raw - background[:, None, None], 0.0)
            raw_unit = signal / photons[:, None, None].clip(min=1e-6)
            raw_unit /= raw_unit.sum(axis=(-2, -1), keepdims=True).clip(min=1e-12)
            raw_stacks.append(raw_unit)
            recon_stacks.append(np.asarray(fit["reconstruction_unit_flux"], dtype=np.float32))
            z_values.append(np.asarray(fit["z_nm"], dtype=np.float64))
        with np.load(pupil_path, allow_pickle=False) as pupil:
            phases.append(np.asarray(pupil["pupil_phase_rad"], dtype=np.float32))
            centers.append(np.asarray(pupil["center_yx"], dtype=np.float32))

    per_bead_csv = tables_dir / "per_bead_metrics.csv"
    columns = (
        "bead_number",
        "center_y",
        "center_x",
        "initial_train_objective",
        "final_train_objective",
        "train_median_ncc",
        "heldout_median_ncc",
        "heldout_p10_ncc",
        "negative_edge_ncc",
        "positive_edge_ncc",
    )
    with per_bead_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for item in metrics:
            writer.writerow(
                (
                    item["bead_number"],
                    item["center_yx"][0],
                    item["center_yx"][1],
                    *[item[key] for key in columns[3:]],
                )
            )

    combined_metrics = {
        "bead_count": int(bead_count),
        "heldout_median_ncc_across_beads": float(
            np.median([item["heldout_median_ncc"] for item in metrics])
        ),
        "minimum_bead_heldout_median_ncc": float(
            np.min([item["heldout_median_ncc"] for item in metrics])
        ),
        "per_bead": metrics,
    }
    metrics_path = metadata_dir / "metrics.json"
    metrics_path.write_text(json.dumps(combined_metrics, indent=2, sort_keys=True) + "\n")
    manifest = {
        "parameterization": "five independently recovered phase-only pixel pupils",
        "field_decomposition_status": "not started; requires per-bead recovery acceptance first",
        "gauge": "piston, tip, tilt, and Z(2,0) defocus fixed to zero",
        "bead_centers_yx": np.asarray(centers).tolist(),
        "individual_manifests": [
            str((metadata_dir / f"bead_{index:02d}_manifest.json").resolve())
            for index in range(1, int(bead_count) + 1)
        ],
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    sections_path = figures_dir / "all_beads_raw_vs_recon_xz_yz.png"
    pupils_path = figures_dir / "all_beads_pupil_phase_comparison.png"
    _render_all_bead_sections(
        np.stack(raw_stacks),
        np.stack(recon_stacks),
        np.stack(z_values),
        np.asarray(centers),
        pixel_size_nm=207.0,
        path=sections_path,
    )
    _render_all_pupils(np.stack(phases), np.asarray(centers), pupils_path)
    return {
        "metrics": str(metrics_path),
        "manifest": str(manifest_path),
        "per_bead_metrics": str(per_bead_csv),
        "sections_figure": str(sections_path),
        "pupils_figure": str(pupils_path),
    }


def _render_all_bead_sections(
    raw: np.ndarray,
    recon: np.ndarray,
    z_nm: np.ndarray,
    centers_yx: np.ndarray,
    *,
    pixel_size_nm: float,
    path: Path,
) -> None:
    plt = _plotting()
    bead_count, _, size, _ = raw.shape
    half_width_um = 0.5 * size * pixel_size_nm / 1000.0
    figure, axes = plt.subplots(
        bead_count,
        4,
        figsize=(10.5, 2.3 * bead_count),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(bead_count, 4)
    for bead in range(bead_count):
        extent = (-half_width_um, half_width_um, float(z_nm[bead, 0]), float(z_nm[bead, -1]))
        volumes = (
            raw[bead].max(axis=1),
            recon[bead].max(axis=1),
            raw[bead].max(axis=2),
            recon[bead].max(axis=2),
        )
        vmax = max(float(volume.max()) for volume in volumes)
        for column, volume in enumerate(volumes):
            axes[bead, column].imshow(
                volume,
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                origin="lower",
                aspect="auto",
                extent=extent,
            )
            axes[bead, column].set_xlabel("X (um)" if column < 2 else "Y (um)")
        axes[bead, 0].set_ylabel(
            f"Bead {bead + 1}\nZ (nm)\n({centers_yx[bead,0]:.0f},{centers_yx[bead,1]:.0f})"
        )
    for axis, title in zip(
        axes[0],
        ("Raw XZ", "Recon XZ", "Raw YZ", "Recon YZ"),
        strict=True,
    ):
        axis.set_title(title)
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def _render_all_pupils(phases: np.ndarray, centers_yx: np.ndarray, path: Path) -> None:
    plt = _plotting()
    figure, axes = plt.subplots(1, len(phases), figsize=(14.0, 3.0), constrained_layout=True)
    for bead, (axis, phase) in enumerate(zip(np.asarray(axes).reshape(-1), phases, strict=True)):
        yy, xx = np.indices(phase.shape)
        center = 0.5 * (np.asarray(phase.shape) - 1.0)
        aperture = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 < (phase.shape[0] / 2.0) ** 2
        shown = np.ma.masked_where(~aperture, np.angle(np.exp(1j * phase)))
        artist = axis.imshow(shown, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axis.set_title(
            f"Bead {bead + 1}\n(y,x)=({centers_yx[bead,0]:.0f},{centers_yx[bead,1]:.0f})"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(artist, ax=np.asarray(axes).reshape(-1).tolist(), label="Phase (rad)", shrink=0.75)
    figure.suptitle("Independent recovered pixel-pupil phases | Z(2,0) gauge fixed")
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover independent pixel pupils for Real T-cell DH beads.")
    parser.add_argument("--source-tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--initialization", type=Path, default=DEFAULT_INITIALIZATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bead-number", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-z-range-nm", type=float, default=2000.0)
    parser.add_argument("--z-step-nm", type=float, default=40.0)
    parser.add_argument("--roi-size", type=int, default=19)
    parser.add_argument("--npupil", type=int, default=128)
    parser.add_argument("--alternating-rounds", type=int, default=2)
    parser.add_argument("--local-steps", type=int, default=100)
    parser.add_argument("--phase-adam-steps", type=int, default=500)
    parser.add_argument("--lbfgs-steps", type=int, default=80)
    parser.add_argument("--phase-learning-rate", type=float, default=0.03)
    parser.add_argument("--phase-anchor-weight", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.aggregate_only:
        print(json.dumps(aggregate_pixel_pupil_outputs(args.output_dir), indent=2), flush=True)
        return 0
    if args.bead_number is None:
        raise ValueError("--bead-number is required unless --aggregate-only is used.")
    config = PixelPupilFitConfig(
        roi_size=args.roi_size,
        npupil=args.npupil,
        alternating_rounds=args.alternating_rounds,
        local_steps=args.local_steps,
        phase_adam_steps=args.phase_adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        phase_learning_rate=args.phase_learning_rate,
        phase_anchor_weight=args.phase_anchor_weight,
        seed=20260725 + int(args.bead_number),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "bead_number": args.bead_number,
                "config": asdict(config),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    outputs = fit_real_tcell_bead(
        source_tiff=args.source_tiff,
        output_dir=args.output_dir,
        initialization_path=args.initialization,
        bead_number=args.bead_number,
        fit_z_range_nm=args.fit_z_range_nm,
        z_step_nm=args.z_step_nm,
        config=config,
    )
    print(json.dumps({"metrics": str(outputs.metrics_path), "figures": [str(p) for p in outputs.figure_paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PixelPupilOutputs",
    "aggregate_pixel_pupil_outputs",
    "fit_real_tcell_bead",
    "write_single_bead_outputs",
]
