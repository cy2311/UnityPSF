from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from .._paths import PROJECT_ROOT as V04_ROOT
from ..shared_carrier_field import (
    SharedCarrierFieldConfig,
    SharedCarrierFieldResult,
    SharedFieldCalibrationResult,
    fit_shared_carrier_field,
    render_shared_field_calibration,
)


DEFAULT_PIXEL_PUPIL_DIR = (
    V04_ROOT
    / "output/double_helix/real_tcell_independent_pixel_pupils_w5434_init_z2000/calibration"
)
DEFAULT_OUTPUT_DIR = (
    V04_ROOT
    / "output/double_helix/real_tcell_shared_carrier_fd_zernike_mode21_complexcircle_z2000/calibration"
)


@dataclass(frozen=True)
class SharedCarrierFieldOutputs:
    carrier_path: Path
    gamma_path: Path
    zmap_path: Path
    fit_path: Path
    metrics_path: Path
    manifest_path: Path
    gamma_table_path: Path
    raw_vs_recon_xy_paths: tuple[Path, ...]
    summary_figure_paths: tuple[Path, ...]


def load_independent_pixel_pupil_calibration(
    source_dir: str | Path,
    *,
    bead_count: int = 5,
) -> dict[str, np.ndarray]:
    root = Path(source_dir)
    phases = []
    centers = []
    observed = []
    z_values = []
    dx_affine = []
    dy_affine = []
    source_plane_indices = []
    for bead_number in range(1, int(bead_count) + 1):
        stem = f"bead_{bead_number:02d}"
        with np.load(root / "arrays" / f"{stem}_complex_pupil.npz", allow_pickle=False) as pupil:
            phases.append(np.asarray(pupil["pupil_phase_rad"], dtype=np.float32))
            centers.append(np.asarray(pupil["center_yx"], dtype=np.float32))
        with np.load(root / "arrays" / f"{stem}_calibration_fit.npz", allow_pickle=False) as fit:
            observed.append(np.asarray(fit["observed_adu"], dtype=np.float32))
            z_values.append(np.asarray(fit["z_nm"], dtype=np.float64))
            dx_affine.append(np.asarray(fit["dx_affine_px"], dtype=np.float32))
            dy_affine.append(np.asarray(fit["dy_affine_px"], dtype=np.float32))
            source_plane_indices.append(np.asarray(fit["source_plane_indices"], dtype=np.int64))
    if not all(np.array_equal(z_values[0], values) for values in z_values[1:]):
        raise ValueError("All independent bead fits must use identical z coordinates.")
    if not all(
        np.array_equal(source_plane_indices[0], values) for values in source_plane_indices[1:]
    ):
        raise ValueError("All independent bead fits must use identical source planes.")
    return {
        "pupil_phases_rad": np.stack(phases),
        "centers_yx": np.stack(centers),
        "observed_adu": np.stack(observed),
        "z_nm": z_values[0],
        "dx_affine_px": np.stack(dx_affine),
        "dy_affine_px": np.stack(dy_affine),
        "source_plane_indices": source_plane_indices[0],
    }


def write_shared_carrier_field_outputs(
    output_dir: str | Path,
    *,
    observed_stacks_adu: np.ndarray,
    decomposition: SharedCarrierFieldResult,
    calibration: SharedFieldCalibrationResult,
    config: SharedCarrierFieldConfig,
    source_pixel_pupil_dir: str | Path,
    source_plane_indices: np.ndarray | None = None,
) -> SharedCarrierFieldOutputs:
    root = Path(output_dir)
    arrays_dir = root / "arrays"
    figures_dir = root / "figures"
    metadata_dir = root / "metadata"
    tables_dir = root / "tables"
    stacks_dir = root / "stacks"
    for directory in (arrays_dir, figures_dir, metadata_dir, tables_dir, stacks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mode_order = np.asarray(decomposition.mode_order, dtype=np.int64)
    carrier_path = arrays_dir / "shared_double_helix_carrier.npz"
    np.savez_compressed(
        carrier_path,
        carrier_phase_rad=decomposition.shared_carrier_phase_rad,
        carrier_complex=decomposition.shared_carrier_complex,
        gauge=np.asarray("shared carrier from gauge-fixed independent total pupils"),
    )
    gamma_path = arrays_dir / "residual_gamma_observations.npz"
    np.savez_compressed(
        gamma_path,
        centers_yx=decomposition.centers_yx,
        residual_gamma_nm=decomposition.residual_gamma_nm,
        field_gamma_at_beads_nm=decomposition.field_gamma_at_beads_nm,
        field_coefficients_nm=decomposition.field_coefficients_nm,
        mode_order=mode_order,
        gamma_semantics=np.asarray("field-dependent residual OPD above shared DH carrier"),
    )
    zmap_path = arrays_dir / "field_dependent_gamma_zernike_maps_nm.npz"
    np.savez_compressed(
        zmap_path,
        zernike_maps_nm=decomposition.zernike_maps_nm,
        field_coefficients_nm=decomposition.field_coefficients_nm,
        mode_order=mode_order,
        centers_yx=decomposition.centers_yx,
        field_shape_yx=np.asarray(decomposition.zernike_maps_nm.shape[1:], dtype=np.int64),
        spatial_model=np.asarray("affine in normalized field x and y"),
    )
    fit_path = arrays_dir / "shared_carrier_field_model_calibration_fit.npz"
    np.savez_compressed(
        fit_path,
        observed_adu=np.asarray(observed_stacks_adu, dtype=np.float32),
        reconstruction_adu=calibration.reconstruction_adu,
        reconstruction_unit_flux=calibration.reconstruction_unit_flux,
        photons_adu=calibration.photons_adu,
        background_adu=calibration.background_adu,
        per_plane_ncc=calibration.per_plane_ncc,
        z_nm=calibration.z_nm,
        train_indices=calibration.train_indices,
        heldout_indices=calibration.heldout_indices,
        source_plane_indices=(
            np.asarray(source_plane_indices, dtype=np.int64)
            if source_plane_indices is not None
            else np.arange(len(calibration.z_nm), dtype=np.int64)
        ),
    )
    tifffile.imwrite(
        stacks_dir / "observed_multi_bead_adu.tif",
        np.asarray(observed_stacks_adu, dtype=np.float32),
    )
    tifffile.imwrite(
        stacks_dir / "shared_carrier_field_reconstruction_adu.tif",
        calibration.reconstruction_adu,
    )
    tifffile.imwrite(
        stacks_dir / "shared_carrier_field_reconstruction_unit_flux.tif",
        calibration.reconstruction_unit_flux,
    )

    metrics = {
        "decomposition": decomposition.metrics,
        "field_model_calibration": calibration.metrics,
    }
    metrics_path = metadata_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest = {
        "source_pixel_pupil_dir": str(Path(source_pixel_pupil_dir).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "model": "shared phase-only DH carrier plus affine field-dependent residual Zernike gamma",
        "mode_count": len(decomposition.mode_order),
        "residual_gauge": "piston, tip, tilt, Z(2,0), and cross-bead mean gamma fixed to zero",
        "carrier_semantics": "common high-frequency commercial DH phase-mask structure",
        "gamma_semantics": "smooth field-dependent residual OPD in nm",
        "field_shape_yx": list(decomposition.zernike_maps_nm.shape[1:]),
        "field_sampling_caveat": "five bead positions support only an affine field model",
        "config": asdict(config),
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    gamma_table_path = tables_dir / "per_bead_residual_gamma_coefficients.csv"
    with gamma_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "bead_number",
                "center_y_px",
                "center_x_px",
                "n",
                "m",
                "decomposed_gamma_nm",
                "field_map_gamma_nm",
            )
        )
        for bead, center in enumerate(decomposition.centers_yx):
            for (n, m), gamma, field_gamma in zip(
                decomposition.mode_order,
                decomposition.residual_gamma_nm[bead],
                decomposition.field_gamma_at_beads_nm[bead],
                strict=True,
            ):
                writer.writerow(
                    (
                        bead + 1,
                        float(center[0]),
                        float(center[1]),
                        n,
                        m,
                        f"{float(gamma):.8f}",
                        f"{float(field_gamma):.8f}",
                    )
                )

    raw_unit = _photometry_normalized_raw(observed_stacks_adu, calibration)
    xy_paths = []
    for bead in range(len(decomposition.centers_yx)):
        path = figures_dir / f"bead_{bead + 1:02d}_raw_vs_shared_fd_recon_xy.png"
        _render_bead_xy(
            bead,
            raw_unit[bead],
            calibration.reconstruction_unit_flux[bead],
            calibration.z_nm,
            decomposition.centers_yx[bead],
            path,
        )
        xy_paths.append(path)
    carrier_figure = figures_dir / "shared_double_helix_carrier.png"
    zmap_figure = figures_dir / "field_dependent_gamma_zernike_maps.png"
    gamma_figure = figures_dir / "per_bead_residual_gamma_coefficients.png"
    sections_figure = figures_dir / "all_beads_raw_vs_shared_fd_recon_xz_yz.png"
    _render_carrier(decomposition, carrier_figure)
    _render_zmaps(decomposition, zmap_figure)
    _render_gamma_sets(decomposition, gamma_figure)
    _render_all_sections(
        raw_unit,
        calibration.reconstruction_unit_flux,
        calibration.z_nm,
        decomposition.centers_yx,
        config.pixel_size_nm,
        sections_figure,
    )
    return SharedCarrierFieldOutputs(
        carrier_path=carrier_path,
        gamma_path=gamma_path,
        zmap_path=zmap_path,
        fit_path=fit_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        gamma_table_path=gamma_table_path,
        raw_vs_recon_xy_paths=tuple(xy_paths),
        summary_figure_paths=(carrier_figure, zmap_figure, gamma_figure, sections_figure),
    )


def _plotting() -> Any:
    cache_dir = V04_ROOT.parent / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


def _photometry_normalized_raw(
    observed: np.ndarray,
    calibration: SharedFieldCalibrationResult,
) -> np.ndarray:
    signal = np.maximum(
        np.asarray(observed, dtype=np.float32) - calibration.background_adu[:, :, None, None],
        0.0,
    )
    unit = signal / calibration.photons_adu[:, :, None, None].clip(min=1e-6)
    return unit / unit.sum(axis=(-2, -1), keepdims=True).clip(min=1e-12)


def _render_bead_xy(
    bead: int,
    raw: np.ndarray,
    recon: np.ndarray,
    z_nm: np.ndarray,
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
    axes[1, 0].set_ylabel("Shared+FD recon")
    figure.suptitle(
        f"Bead {bead + 1} | field position (y,x)=({center_yx[0]:.0f},{center_yx[1]:.0f})"
    )
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def _render_carrier(result: SharedCarrierFieldResult, path: Path) -> None:
    plt = _plotting()
    phase = np.angle(result.shared_carrier_complex)
    amplitude = np.abs(result.shared_carrier_complex)
    mask = amplitude < 0.5
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.1), constrained_layout=True)
    values = (
        np.ma.masked_where(mask, phase),
        amplitude,
        result.shared_carrier_complex.real,
        result.shared_carrier_complex.imag,
    )
    titles = ("Wrapped phase", "Amplitude", "Real", "Imaginary")
    cmaps = ("twilight", "gray", "RdBu_r", "RdBu_r")
    for axis, image, title, cmap in zip(axes, values, titles, cmaps, strict=True):
        vmin, vmax = ((-np.pi, np.pi) if title == "Wrapped phase" else ((0.0, 1.0) if title == "Amplitude" else (-1.0, 1.0)))
        artist = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(artist, ax=axis, shrink=0.78)
    figure.suptitle("Recovered shared double-helix carrier")
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _render_zmaps(result: SharedCarrierFieldResult, path: Path) -> None:
    plt = _plotting()
    columns = 4
    rows = int(np.ceil(len(result.mode_order) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10.5, 2.35 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, values, (n, m) in zip(
        axes,
        result.zernike_maps_nm,
        result.mode_order,
        strict=False,
    ):
        midpoint = float(np.mean(values))
        spread = max(float(np.max(np.abs(values - midpoint))), 1e-3)
        artist = axis.imshow(
            values,
            cmap="RdBu_r",
            vmin=midpoint - spread,
            vmax=midpoint + spread,
            origin="upper",
        )
        axis.scatter(result.centers_yx[:, 1], result.centers_yx[:, 0], c="black", marker="+", s=18)
        axis.set_title(f"Z({n},{m:+d}) | range {np.ptp(values):.2f} nm", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(artist, ax=axis, shrink=0.72, label="Gamma (nm)")
    for axis in axes[len(result.mode_order) :]:
        axis.axis("off")
    figure.suptitle("75 x 75 field-dependent residual Zernike gamma maps")
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def _render_gamma_sets(result: SharedCarrierFieldResult, path: Path) -> None:
    plt = _plotting()
    labels = [f"({n},{m:+d})" for n, m in result.mode_order]
    x = np.arange(len(labels))
    width = 0.15
    figure, axis = plt.subplots(figsize=(12.0, 4.2), constrained_layout=True)
    for bead in range(len(result.centers_yx)):
        axis.bar(
            x + (bead - 2) * width,
            result.field_gamma_at_beads_nm[bead],
            width=width,
            label=f"Bead {bead + 1}",
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xticks(x, labels, rotation=90)
    axis.set(xlabel="Residual Zernike mode (n,m)", ylabel="Gamma from field map (nm)")
    axis.legend(ncol=5, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def _render_all_sections(
    raw: np.ndarray,
    recon: np.ndarray,
    z_nm: np.ndarray,
    centers_yx: np.ndarray,
    pixel_size_nm: float,
    path: Path,
) -> None:
    plt = _plotting()
    bead_count, _, size, _ = raw.shape
    half_width_um = 0.5 * size * pixel_size_nm / 1000.0
    extent = (-half_width_um, half_width_um, float(z_nm[0]), float(z_nm[-1]))
    figure, axes = plt.subplots(bead_count, 4, figsize=(10.5, 2.3 * bead_count), constrained_layout=True)
    axes = np.asarray(axes).reshape(bead_count, 4)
    for bead in range(bead_count):
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
        ("Raw XZ", "Shared+FD XZ", "Raw YZ", "Shared+FD YZ"),
        strict=True,
    ):
        axis.set_title(title)
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose Real T-cell DH pupils into a shared carrier and FD residual gamma maps."
    )
    parser.add_argument("--pixel-pupil-dir", type=Path, default=DEFAULT_PIXEL_PUPIL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode-count", type=int, default=21)
    parser.add_argument("--alternating-rounds", type=int, default=6)
    parser.add_argument("--gamma-steps", type=int, default=300)
    parser.add_argument("--gamma-learning-rate-nm", type=float, default=1.0)
    parser.add_argument("--field-ridge", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SharedCarrierFieldConfig(
        mode_count=args.mode_count,
        alternating_rounds=args.alternating_rounds,
        gamma_steps=args.gamma_steps,
        gamma_learning_rate_nm=args.gamma_learning_rate_nm,
        field_ridge=args.field_ridge,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
                "config": asdict(config),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    source = load_independent_pixel_pupil_calibration(args.pixel_pupil_dir)
    decomposition = fit_shared_carrier_field(
        source["pupil_phases_rad"],
        centers_yx=source["centers_yx"],
        field_shape_yx=(75, 75),
        config=config,
    )
    calibration = render_shared_field_calibration(
        source["observed_adu"],
        z_nm=source["z_nm"],
        dx_affine_px=source["dx_affine_px"],
        dy_affine_px=source["dy_affine_px"],
        decomposition=decomposition,
        config=config,
    )
    outputs = write_shared_carrier_field_outputs(
        args.output_dir,
        observed_stacks_adu=source["observed_adu"],
        decomposition=decomposition,
        calibration=calibration,
        config=config,
        source_pixel_pupil_dir=args.pixel_pupil_dir,
        source_plane_indices=source["source_plane_indices"],
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "accepted": calibration.metrics["accepted"],
                "metrics": str(outputs.metrics_path),
                "figures": [str(path) for path in outputs.summary_figure_paths],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SharedCarrierFieldOutputs",
    "load_independent_pixel_pupil_calibration",
    "write_shared_carrier_field_outputs",
]
