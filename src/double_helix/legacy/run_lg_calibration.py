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
from ..lg_calibration import (
    LGResidualFitConfig,
    LGResidualFitResult,
    detect_stable_dh_centers,
    extract_centered_roi_stacks,
    fit_lg_residual_calibration,
)


DEFAULT_TIFF = (
    V04_ROOT.parent / "datasets/training_sets/double_helix/Real_dataset_Tcell/Calib.tif"
)
DEFAULT_OUTPUT_DIR = (
    V04_ROOT / "output/double_helix/real_tcell_lgcarrier_residual_zernike/calibration"
)


@dataclass(frozen=True)
class LGCalibrationOutputs:
    output_dir: Path
    metrics_path: Path
    manifest_path: Path
    center_coefficients_csv: Path
    bead_coefficients_csv: Path
    zmap_figure_path: Path
    center_coefficients_figure_path: Path
    carrier_figure_path: Path
    raw_vs_recon_xy_paths: tuple[Path, ...]
    raw_vs_recon_xz_yz_path: Path


def write_lg_calibration_outputs(
    output_dir: str | Path,
    *,
    roi_stacks_adu: np.ndarray,
    result: LGResidualFitResult,
    config: LGResidualFitConfig,
    source_tiff: str | Path,
) -> LGCalibrationOutputs:
    root = Path(output_dir)
    arrays_dir = root / "arrays"
    stacks_dir = root / "stacks"
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    metadata_dir = root / "metadata"
    for directory in (arrays_dir, stacks_dir, figures_dir, tables_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mode_order = np.asarray(result.mode_order, dtype=np.int64)
    lg_mode_order = np.asarray(result.lg_mode_order, dtype=np.int64)
    np.savez_compressed(
        arrays_dir / "lg_dh_carrier.npz",
        lg_mode_order=lg_mode_order,
        lg_weights=result.lg_weights,
        lg_weight_logits=result.lg_weight_logits,
        lg_phase_offsets_rad=result.lg_phase_offsets_rad,
        lg_rotation_rad=np.asarray(result.lg_rotation_rad, dtype=np.float32),
        lg_waist=np.asarray(config.lg_waist, dtype=np.float32),
        carrier_phase_rad=result.carrier_phase_rad,
    )
    np.savez_compressed(
        arrays_dir / "residual_gamma_observations.npz",
        centers_yx=result.centers_yx,
        gamma_nm=result.residual_gamma_nm,
        mode_order=mode_order,
    )
    np.savez_compressed(
        arrays_dir / "full_roi_residual_zernike_maps_nm.npz",
        zernike_maps_nm=result.zernike_maps_nm,
        field_coefficients_nm=result.residual_field_coefficients_nm,
        mode_order=mode_order,
        centers_yx=result.centers_yx,
        gamma_semantics=np.asarray(
            "field-dependent residual Zernike OPD above global LG-DH carrier"
        ),
    )
    selected_raw = np.asarray(roi_stacks_adu, dtype=np.float32)[:, result.source_plane_indices]
    np.savez_compressed(
        arrays_dir / "calibration_fit.npz",
        observed_adu=selected_raw,
        reconstruction_adu=result.reconstruction_adu,
        reconstruction_unit_flux=result.reconstruction_unit_flux,
        photons_adu=result.photons_adu,
        background_adu=result.background_adu,
        z_nm=result.z_nm,
        source_plane_indices=result.source_plane_indices,
        train_indices=result.train_indices,
        heldout_indices=result.heldout_indices,
        dx_affine_px=result.dx_affine_px,
        dy_affine_px=result.dy_affine_px,
    )
    tifffile.imwrite(stacks_dir / "observed_multi_bead_adu.tif", selected_raw)
    tifffile.imwrite(stacks_dir / "reconstruction_multi_bead_adu.tif", result.reconstruction_adu)
    tifffile.imwrite(
        stacks_dir / "reconstruction_multi_bead_unit_flux.tif",
        result.reconstruction_unit_flux,
    )

    metrics_path = metadata_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n")
    source_path = Path(source_tiff)
    manifest = {
        "source_tiff": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "source_shape_zyx": list(tifffile.TiffFile(source_path).series[0].shape),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device": str(config.device),
        "parameterization": "global phase-only LG-DH carrier plus field residual Zernike",
        "gamma_semantics": "field-dependent residual OPD in nm; carrier stored separately",
        "gauge": "piston, tip, tilt, and Zernike (2,0) defocus excluded",
        "field_shape_yx": list(result.zernike_maps_nm.shape[1:]),
        "bead_centers_yx": result.centers_yx.tolist(),
        "config": asdict(config),
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    center_y = result.zernike_maps_nm.shape[1] // 2
    center_x = result.zernike_maps_nm.shape[2] // 2
    center_gamma = result.zernike_maps_nm[:, center_y, center_x]
    center_coefficients_csv = tables_dir / "center_residual_zernike_coefficients.csv"
    with center_coefficients_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode_index", "radial_order_n", "azimuthal_order_m", "gamma_nm"))
        for index, ((n, m), value) in enumerate(
            zip(result.mode_order, center_gamma, strict=True), start=1
        ):
            writer.writerow((index, n, m, f"{float(value):.8f}"))

    bead_coefficients_csv = tables_dir / "bead_residual_zernike_coefficients.csv"
    with bead_coefficients_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("bead_index", "center_y_px", "center_x_px", "n", "m", "gamma_nm"))
        for bead, (center, gamma) in enumerate(
            zip(result.centers_yx, result.residual_gamma_nm, strict=True), start=1
        ):
            for (n, m), value in zip(result.mode_order, gamma, strict=True):
                writer.writerow((bead, center[0], center[1], n, m, f"{float(value):.8f}"))

    normalized_raw = _photometry_normalized_raw(selected_raw, result)
    xy_paths = tuple(
        _render_bead_xy(
            bead,
            normalized_raw[bead],
            result.reconstruction_unit_flux[bead],
            result.z_nm,
            result.centers_yx[bead],
            figures_dir / f"bead_{bead + 1:02d}_raw_vs_recon_xy.png",
        )
        for bead in range(len(result.centers_yx))
    )
    xz_yz_path = figures_dir / "all_beads_raw_vs_recon_xz_yz.png"
    _render_all_bead_sections(
        normalized_raw,
        result.reconstruction_unit_flux,
        result.z_nm,
        result.centers_yx,
        config.pixel_size_nm,
        xz_yz_path,
    )
    zmap_path = figures_dir / "full_roi_residual_zernike_zmap.png"
    _render_zmaps(result, zmap_path)
    coefficient_figure = figures_dir / "center_residual_zernike_coefficients.png"
    _render_center_coefficients(result.mode_order, center_gamma, coefficient_figure)
    carrier_figure = figures_dir / "lg_dh_carrier_phase_and_weights.png"
    _render_carrier(result, carrier_figure)
    return LGCalibrationOutputs(
        output_dir=root,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        center_coefficients_csv=center_coefficients_csv,
        bead_coefficients_csv=bead_coefficients_csv,
        zmap_figure_path=zmap_path,
        center_coefficients_figure_path=coefficient_figure,
        carrier_figure_path=carrier_figure,
        raw_vs_recon_xy_paths=xy_paths,
        raw_vs_recon_xz_yz_path=xz_yz_path,
    )


def _plotting() -> Any:
    cache_dir = V04_ROOT / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


def _photometry_normalized_raw(
    raw_adu: np.ndarray,
    result: LGResidualFitResult,
) -> np.ndarray:
    signal = np.maximum(
        raw_adu - result.background_adu[:, :, None, None],
        0.0,
    )
    normalized = signal / result.photons_adu[:, :, None, None].clip(min=1e-6)
    return normalized / normalized.sum(axis=(-2, -1), keepdims=True).clip(min=1e-12)


def _render_bead_xy(
    bead: int,
    raw: np.ndarray,
    recon: np.ndarray,
    z_nm: np.ndarray,
    center_yx: np.ndarray,
    path: Path,
) -> Path:
    plt = _plotting()
    selected = np.linspace(0, len(z_nm) - 1, min(9, len(z_nm)), dtype=int)
    fig, axes = plt.subplots(2, len(selected), figsize=(12.0, 3.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(selected))
    for column, plane in enumerate(selected):
        vmax = max(float(raw[plane].max()), float(recon[plane].max()), 1e-8)
        axes[0, column].imshow(raw[plane], cmap="magma", vmin=0.0, vmax=vmax)
        axes[1, column].imshow(recon[plane], cmap="magma", vmin=0.0, vmax=vmax)
        axes[0, column].set_title(f"z={z_nm[plane]:.0f} nm", fontsize=8)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Raw\nphotometry matched")
    axes[1, 0].set_ylabel("LG+residual\nreconstruction")
    fig.suptitle(
        f"Bead {bead + 1} at full-frame (y,x)=({center_yx[0]:.0f},{center_yx[1]:.0f})",
        fontsize=10,
    )
    fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)
    return path


def _render_all_bead_sections(
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
    fig, axes = plt.subplots(bead_count, 4, figsize=(10.0, 2.25 * bead_count), constrained_layout=True)
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
        ("Raw XZ", "Recon XZ", "Raw YZ", "Recon YZ"),
        strict=True,
    ):
        axis.set_title(title)
    fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)


def _render_zmaps(result: LGResidualFitResult, path: Path) -> None:
    plt = _plotting()
    columns = 4
    rows = int(np.ceil(len(result.mode_order) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10.0, 2.35 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, values, (n, m) in zip(
        axes,
        result.zernike_maps_nm,
        result.mode_order,
        strict=False,
    ):
        limit = max(float(np.max(np.abs(values))), 1.0)
        artist = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper")
        axis.scatter(result.centers_yx[:, 1], result.centers_yx[:, 0], s=10, c="black", marker="+")
        axis.set_title(f"Z({n},{m:+d}) residual")
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(artist, ax=axis, shrink=0.72, label="nm")
    for axis in axes[len(result.mode_order) :]:
        axis.axis("off")
    fig.suptitle("Full 75 x 75 residual Zernike zmap | crosses = measured bead positions")
    fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)


def _render_center_coefficients(
    mode_order: tuple[tuple[int, int], ...],
    values: np.ndarray,
    path: Path,
) -> None:
    plt = _plotting()
    labels = [f"({n},{m:+d})" for n, m in mode_order]
    colors = np.where(values >= 0.0, "#0072B2", "#D55E00")
    fig, axis = plt.subplots(figsize=(max(7.0, 0.42 * len(values)), 3.2), constrained_layout=True)
    axis.bar(np.arange(len(values)), values, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xticks(np.arange(len(values)), labels, rotation=90)
    axis.set(xlabel="Residual Zernike mode (n,m)", ylabel="Gamma (nm)")
    axis.set_title("Residual gamma coefficient set at full-field center (37,37)")
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)


def _render_carrier(result: LGResidualFitResult, path: Path) -> None:
    plt = _plotting()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True)
    pupil = result.carrier_phase_rad.copy()
    artist = axes[0].imshow(pupil, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("Global LG-DH carrier phase")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(artist, ax=axes[0], label="rad")
    labels = [f"LG({p},{l})" for p, l in result.lg_mode_order]
    axes[1].bar(np.arange(len(labels)), result.lg_weights, color="#009E73")
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axes[1].set_ylabel("Normalized positive weight")
    axes[1].set_title(f"LG weights | rotation={result.lg_rotation_rad:.3f} rad")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[2].bar(np.arange(len(labels)), result.lg_phase_offsets_rad, color="#CC79A7")
    axes[2].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axes[2].set_ylabel("Relative phase (rad)")
    axes[2].set_title("Complex LG coefficient phases")
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a global LG-DH carrier plus field residual Zernike model."
    )
    parser.add_argument("--source-tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode-count", type=int, default=13)
    parser.add_argument("--maximum-beads", type=int, default=5)
    parser.add_argument("--roi-size", type=int, default=17)
    parser.add_argument("--npupil", type=int, default=128)
    parser.add_argument("--fit-z-range-nm", type=float, default=2000.0)
    parser.add_argument("--alternating-rounds", type=int, default=4)
    parser.add_argument("--local-steps", type=int, default=100)
    parser.add_argument("--global-steps", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_stack = tifffile.imread(args.source_tiff)
    centers = detect_stable_dh_centers(
        source_stack,
        roi_size=args.roi_size,
        maximum_count=args.maximum_beads,
        minimum_distance_px=9,
    )
    roi_stacks = extract_centered_roi_stacks(source_stack, centers, roi_size=args.roi_size)
    config = LGResidualFitConfig(
        mode_count=args.mode_count,
        roi_size=args.roi_size,
        npupil=args.npupil,
        fit_z_range_nm=args.fit_z_range_nm,
        alternating_rounds=args.alternating_rounds,
        local_steps=args.local_steps,
        global_steps=args.global_steps,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "centers_yx": centers.tolist(),
                "config": asdict(config),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    result = fit_lg_residual_calibration(
        roi_stacks,
        centers_yx=centers,
        field_shape_yx=source_stack.shape[1:],
        config=config,
    )
    outputs = write_lg_calibration_outputs(
        args.output_dir,
        roi_stacks_adu=roi_stacks,
        result=result,
        config=config,
        source_tiff=args.source_tiff,
    )
    print(json.dumps({"output_dir": str(outputs.output_dir), "metrics": result.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LGCalibrationOutputs", "write_lg_calibration_outputs"]
