from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .._paths import PROJECT_ROOT as V04_ROOT
from .render_wavefront_diagnostics import orthogonal_max_projections
from ..vector_model import evaluate_normalized_zernike


DEFAULT_CALIBRATION_DIR = (
    V04_ROOT
    / "output/double_helix/real_tcell_deepz_mode64_warm42_roi17_zopt/calibration"
)


def per_plane_ncc(observed: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    first = np.asarray(observed, dtype=np.float64)
    second = np.asarray(reconstructed, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("observed and reconstructed must have matching (Z,Y,X) shapes.")
    first = first.reshape(len(first), -1)
    second = second.reshape(len(second), -1)
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )


def nearest_plane_indices(z_nm: np.ndarray, targets_nm: np.ndarray) -> np.ndarray:
    z_values = np.asarray(z_nm, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets_nm, dtype=np.float64).reshape(-1)
    selected: list[int] = []
    for target in targets:
        index = int(np.argmin(np.abs(z_values - target)))
        if index not in selected:
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def summarize_global_zmap(
    zernike_maps_nm: np.ndarray,
    mode_order: np.ndarray,
) -> dict[str, Any]:
    maps = np.asarray(zernike_maps_nm, dtype=np.float64)
    modes = np.asarray(mode_order, dtype=np.int64)
    if maps.ndim != 3 or modes.shape != (maps.shape[0], 2):
        raise ValueError("zernike_maps_nm and mode_order must have shapes (C,Y,X) and (C,2).")
    coefficients = maps.mean(axis=(1, 2))
    spatial_deviation = float(np.max(np.std(maps, axis=(1, 2))))
    return {
        "shape_cyx": [int(value) for value in maps.shape],
        "mode_count": int(maps.shape[0]),
        "field_shape_yx": [int(value) for value in maps.shape[1:]],
        "spatially_constant": bool(np.allclose(maps, coefficients[:, None, None])),
        "defocus_2_0_present": bool(np.any(np.all(modes == (2, 0), axis=1))),
        "spatial_standard_deviation_nm": spatial_deviation,
        "coefficient_rms_nm": float(np.sqrt(np.mean(coefficients**2))),
    }


def axial_projection_extents(
    stage_z_nm: np.ndarray,
    fitted_z_nm: np.ndarray,
    *,
    psf_size: int,
    pixel_size_nm: float,
) -> dict[str, tuple[float, float, float, float]]:
    radius_um = psf_size * pixel_size_nm / 2000.0
    return {
        "raw": (-radius_um, radius_um, float(stage_z_nm[0] / 1000.0), float(stage_z_nm[-1] / 1000.0)),
        "recovered": (
            -radius_um,
            radius_um,
            float(fitted_z_nm[0] / 1000.0),
            float(fitted_z_nm[-1] / 1000.0),
        ),
    }


def render_calibration_diagnostics(
    calibration_dir: str | Path,
    *,
    pixel_size_nm: float = 207.0,
) -> dict[str, Any]:
    root = Path(calibration_dir)
    fit_path = root / "arrays/calibration_fit.npz"
    zmap_path = root / "arrays/alternating_full_roi_zernike_maps_nm.npz"
    resolved_config = json.loads((root / "config/resolved_config.json").read_text(encoding="utf-8"))
    wavelength_nm = float(resolved_config["wavelength_nm"])
    with np.load(fit_path, allow_pickle=False) as payload:
        fit = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(zmap_path, allow_pickle=False) as payload:
        zernike_maps_nm = np.asarray(payload["zernike_maps_nm"], dtype=np.float32)
        mode_order = np.asarray(payload["mode_order"], dtype=np.int64)

    observed = np.asarray(fit["observed_adu"], dtype=np.float32)
    reconstructed = np.asarray(fit["reconstruction_adu"], dtype=np.float32)
    background = np.asarray(fit["background_adu"], dtype=np.float32)
    stage_z_nm = np.asarray(fit["stage_z_nm"], dtype=np.float64)
    fitted_z_nm = np.asarray(fit["z_nm"], dtype=np.float64)
    ncc = per_plane_ncc(observed, reconstructed)
    observed_signal, reconstructed_signal = _normalized_signal_volumes(
        observed,
        reconstructed,
        background,
    )
    signal_nrmse = _per_plane_signal_nrmse(observed, reconstructed, background)
    targets_nm = np.linspace(float(stage_z_nm[0]), float(stage_z_nm[-1]), 9)
    selected = nearest_plane_indices(stage_z_nm, targets_nm)
    zmap_summary = summarize_global_zmap(zernike_maps_nm, mode_order)

    figures_dir = root / "figures"
    arrays_dir = root / "arrays"
    metadata_dir = root / "metadata"
    for directory in (figures_dir, arrays_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics_path = arrays_dir / "per_plane_closure_metrics.npz"
    np.savez_compressed(
        metrics_path,
        stage_z_nm=stage_z_nm,
        fitted_z_nm=fitted_z_nm,
        ncc=ncc.astype(np.float32),
        signal_nrmse=signal_nrmse.astype(np.float32),
        selected_plane_indices=selected,
    )
    zmap_figure = figures_dir / "global_zmap_overview.png"
    xy_figure = figures_dir / "raw_vs_recovered_psf_xy.png"
    axial_figure = figures_dir / "recovered_psf_xy_xz_yz.png"
    _render_global_zmap(
        zernike_maps_nm,
        mode_order,
        stage_z_nm=stage_z_nm,
        ncc=ncc,
        heldout_indices=np.asarray(fit["heldout_indices"], dtype=np.int64),
        pixel_size_nm=pixel_size_nm,
        wavelength_nm=wavelength_nm,
        path=zmap_figure,
    )
    _render_xy_closure(
        observed_signal,
        reconstructed_signal,
        stage_z_nm=stage_z_nm,
        fitted_z_nm=fitted_z_nm,
        ncc=ncc,
        selected=selected,
        pixel_size_nm=pixel_size_nm,
        path=xy_figure,
    )
    _render_axial_psf(
        observed_signal,
        reconstructed_signal,
        stage_z_nm=stage_z_nm,
        fitted_z_nm=fitted_z_nm,
        ncc=ncc,
        selected=selected,
        pixel_size_nm=pixel_size_nm,
        path=axial_figure,
    )

    summary = {
        **zmap_summary,
        "gauge": (
            "Zernike (2,0) defocus present"
            if zmap_summary["defocus_2_0_present"]
            else "Zernike (2,0) defocus excluded and fixed at 0"
        ),
        "wavelength_nm": wavelength_nm,
        "zmap_semantics": (
            f"{zmap_summary['mode_count']} global direct-gamma coefficients broadcast over a canonical "
            f"{zmap_summary['field_shape_yx'][0]}x{zmap_summary['field_shape_yx'][1]} field"
        ),
        "stage_z_range_nm": [float(stage_z_nm[0]), float(stage_z_nm[-1])],
        "fitted_optical_z_range_nm": [float(fitted_z_nm[0]), float(fitted_z_nm[-1])],
        "median_ncc": float(np.median(ncc)),
        "minimum_ncc": float(np.min(ncc)),
        "endpoint_ncc": [float(ncc[0]), float(ncc[-1])],
        "selected_stage_z_nm": stage_z_nm[selected].tolist(),
        "selected_fitted_z_nm": fitted_z_nm[selected].tolist(),
        "figures": [str(zmap_figure), str(xy_figure), str(axial_figure)],
        "per_plane_metrics": str(metrics_path),
    }
    metadata_path = metadata_dir / "calibration_diagnostics.json"
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["metadata"] = str(metadata_path)
    return summary


def _normalized_signal_volumes(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_signal = np.maximum(observed - background[:, None, None], 0.0)
    reconstructed_signal = np.maximum(reconstructed - background[:, None, None], 0.0)
    observed_signal /= observed_signal.max(axis=(1, 2), keepdims=True).clip(min=1e-12)
    reconstructed_signal /= reconstructed_signal.max(axis=(1, 2), keepdims=True).clip(min=1e-12)
    return observed_signal, reconstructed_signal


def _per_plane_signal_nrmse(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    observed_signal = np.maximum(observed - background[:, None, None], 0.0).reshape(len(observed), -1)
    reconstructed_signal = np.maximum(reconstructed - background[:, None, None], 0.0).reshape(
        len(reconstructed), -1
    )
    denominator = np.linalg.norm(observed_signal, axis=1)
    return np.divide(
        np.linalg.norm(observed_signal - reconstructed_signal, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )


def _render_global_zmap(
    zernike_maps_nm: np.ndarray,
    mode_order: np.ndarray,
    *,
    stage_z_nm: np.ndarray,
    ncc: np.ndarray,
    heldout_indices: np.ndarray,
    pixel_size_nm: float,
    wavelength_nm: float,
    path: Path,
) -> None:
    import torch
    from matplotlib.colors import TwoSlopeNorm

    plt = _plotting()
    coefficients = zernike_maps_nm.mean(axis=(1, 2))
    coordinates = torch.linspace(-1.0, 1.0, 257, dtype=torch.float32)
    y_pupil, x_pupil = torch.meshgrid(coordinates, coordinates, indexing="ij")
    basis = evaluate_normalized_zernike(tuple(map(tuple, mode_order.tolist())), x_pupil, y_pupil).numpy()
    optical_path_nm = np.einsum("c,chw->hw", coefficients, basis)
    phase = np.angle(np.exp(2j * np.pi * optical_path_nm / wavelength_nm))
    phase[(x_pupil.square() + y_pupil.square()).numpy() >= 1.0] = np.nan
    rms_map = np.sqrt(np.mean(np.square(zernike_maps_nm), axis=0))
    tile_columns = 8
    tile_rows = int(np.ceil(len(coefficients) / tile_columns))
    coefficient_tiles = np.full(tile_rows * tile_columns, np.nan, dtype=np.float32)
    coefficient_tiles[: len(coefficients)] = coefficients
    coefficient_tiles = coefficient_tiles.reshape(tile_rows, tile_columns)
    coefficient_limit = max(float(np.max(np.abs(coefficients))), 1.0)
    field_size_um = zernike_maps_nm.shape[-1] * pixel_size_nm / 1000.0

    with plt.rc_context(_figure_style()):
        fig = plt.figure(figsize=(11.0, 7.0), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.45, 1.15))
        phase_axis = fig.add_subplot(grid[0, 0])
        phase_artist = phase_axis.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=(-1, 1, 1, -1))
        phase_axis.set(title="A  Recovered pupil phase", xlabel="Normalized pupil X", ylabel="Normalized pupil Y")
        fig.colorbar(phase_artist, ax=phase_axis, shrink=0.78, label="Wrapped phase (rad)")

        tile_axis = fig.add_subplot(grid[:, 1])
        tile_artist = tile_axis.imshow(
            coefficient_tiles,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-coefficient_limit, vcenter=0.0, vmax=coefficient_limit),
        )
        for index, ((n, m), value) in enumerate(zip(mode_order, coefficients, strict=True)):
            row, column = divmod(index, tile_columns)
            tile_axis.text(column, row, f"({n},{m:+d})\n{value:.0f}", ha="center", va="center", fontsize=5.5)
        tile_axis.set(
            title=f"B  All {len(coefficients)} global gamma coefficients",
            xlabel="Mode tile",
            ylabel="Mode tile",
        )
        tile_axis.set_xticks([])
        tile_axis.set_yticks([])
        fig.colorbar(tile_artist, ax=tile_axis, shrink=0.72, label="Coefficient (nm)")

        rms_axis = fig.add_subplot(grid[0, 2])
        rms_artist = rms_axis.imshow(
            rms_map,
            cmap="viridis",
            extent=(0.0, field_size_um, field_size_um, 0.0),
            vmin=0.0,
            vmax=max(float(rms_map.max()), 1.0),
        )
        rms_axis.set(title="C  Coefficient RMS zmap", xlabel="Field X (um)", ylabel="Field Y (um)")
        rms_axis.text(
            0.5,
            0.06,
            f"constant = {float(rms_map[0, 0]):.1f} nm",
            transform=rms_axis.transAxes,
            ha="center",
            color="white",
            fontsize=8,
        )
        fig.colorbar(rms_artist, ax=rms_axis, shrink=0.78, label="RMS coefficient (nm)")

        ncc_axis = fig.add_subplot(grid[1, 2])
        train_mask = np.ones(len(ncc), dtype=bool)
        train_mask[heldout_indices] = False
        ncc_axis.plot(stage_z_nm, ncc, color="#777777", linewidth=0.8)
        ncc_axis.scatter(stage_z_nm[train_mask], ncc[train_mask], s=15, color="#0072B2", label="Train")
        ncc_axis.scatter(stage_z_nm[heldout_indices], ncc[heldout_indices], s=22, marker="s", color="#D55E00", label="Held out")
        ncc_axis.set(title="D  Per-plane closure", xlabel="Stage Z (nm)", ylabel="NCC", ylim=(0.0, 1.02))
        ncc_axis.legend(frameon=False, fontsize=7)
        ncc_axis.spines[["top", "right"]].set_visible(False)
        defocus_state = "present" if np.any(np.all(mode_order == (2, 0), axis=1)) else "excluded and fixed at 0"
        fig.suptitle(f"Real T-cell global double-helix zmap | Zernike (2,0) {defocus_state}", fontsize=11)
        _save_png_pdf(fig, path)
        plt.close(fig)


def _render_xy_closure(
    observed_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
    *,
    stage_z_nm: np.ndarray,
    fitted_z_nm: np.ndarray,
    ncc: np.ndarray,
    selected: np.ndarray,
    pixel_size_nm: float,
    path: Path,
) -> None:
    from matplotlib.colors import PowerNorm

    plt = _plotting()
    radius_um = observed_signal.shape[-1] * pixel_size_nm / 2000.0
    extent = (-radius_um, radius_um, radius_um, -radius_um)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    with plt.rc_context(_figure_style()):
        fig, axes = plt.subplots(2, len(selected), figsize=(12.5, 3.7), constrained_layout=True)
        for column, index in enumerate(selected):
            axes[0, column].imshow(observed_signal[index], cmap="magma", norm=norm, extent=extent)
            axes[1, column].imshow(reconstructed_signal[index], cmap="magma", norm=norm, extent=extent)
            axes[0, column].set_title(
                f"stage {stage_z_nm[index]:+.0f} nm\nmodel {fitted_z_nm[index]:+.0f} nm",
                fontsize=7,
            )
            axes[1, column].set_xlabel(f"NCC {ncc[index]:.3f}", fontsize=7)
            for row in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        axes[0, 0].set_ylabel("Raw\nper-plane normalized")
        axes[1, 0].set_ylabel("Recovered\nper-plane normalized")
        fig.suptitle("Real T-cell raw versus recovered double-helix PSF across stage Z", fontsize=10)
        _save_png_pdf(fig, path)
        plt.close(fig)


def _render_axial_psf(
    observed_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
    *,
    stage_z_nm: np.ndarray,
    fitted_z_nm: np.ndarray,
    ncc: np.ndarray,
    selected: np.ndarray,
    pixel_size_nm: float,
    path: Path,
) -> None:
    from matplotlib.colors import PowerNorm

    plt = _plotting()
    raw_xz, raw_yz = orthogonal_max_projections(observed_signal)
    recon_xz, recon_yz = orthogonal_max_projections(reconstructed_signal)
    radius_um = observed_signal.shape[-1] * pixel_size_nm / 2000.0
    lateral_extent = (-radius_um, radius_um, radius_um, -radius_um)
    axial_extents = axial_projection_extents(
        stage_z_nm,
        fitted_z_nm,
        psf_size=observed_signal.shape[-1],
        pixel_size_nm=pixel_size_nm,
    )
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    with plt.rc_context(_figure_style()):
        fig = plt.figure(figsize=(12.5, 7.8))
        grid = fig.add_gridspec(
            3,
            18,
            height_ratios=(1.0, 1.15, 1.15),
            left=0.07,
            right=0.96,
            bottom=0.08,
            top=0.88,
            hspace=0.65,
            wspace=0.7,
        )
        for position, index in enumerate(selected):
            axis = fig.add_subplot(grid[0, 2 * position : 2 * position + 2])
            axis.imshow(reconstructed_signal[index], cmap="magma", norm=norm, extent=lateral_extent)
            axis.set_title(
                f"stage {stage_z_nm[index]:+.0f} nm\n"
                f"model {fitted_z_nm[index]:+.0f} nm\n"
                f"NCC {ncc[index]:.3f}",
                fontsize=6.5,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        side_specs = (
            (raw_xz, "Raw XZ (max over Y)", "X (um)", "Stage Z (um)", axial_extents["raw"]),
            (
                recon_xz,
                "Recovered XZ (max over Y)",
                "X (um)",
                "Fitted optical Z (um)",
                axial_extents["recovered"],
            ),
            (raw_yz, "Raw YZ (max over X)", "Y (um)", "Stage Z (um)", axial_extents["raw"]),
            (
                recon_yz,
                "Recovered YZ (max over X)",
                "Y (um)",
                "Fitted optical Z (um)",
                axial_extents["recovered"],
            ),
        )
        side_axes = []
        for position, (projection, title, xlabel, ylabel, extent) in enumerate(side_specs):
            row = 1 + position // 2
            column = position % 2
            axis = fig.add_subplot(grid[row, column * 9 : column * 9 + 8])
            axis.imshow(projection, cmap="magma", norm=norm, origin="lower", aspect="auto", extent=extent)
            axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
            side_axes.append(axis)
        colorbar_axis = fig.add_subplot(grid[1:, 17])
        fig.colorbar(side_axes[-1].images[0], cax=colorbar_axis, label="Per-plane normalized intensity")
        fig.suptitle(
            "Real T-cell PSF through Z | raw uses stage Z; recovered uses fitted optical Z",
            fontsize=10,
        )
        _save_png_pdf(fig, path)
        plt.close(fig)


def _figure_style() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
    }


def _plotting() -> Any:
    cache_dir = V04_ROOT / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


def _save_png_pdf(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render global-zmap and axial diagnostics for a calibration fit.")
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--pixel-size-nm", type=float, default=207.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = render_calibration_diagnostics(args.calibration_dir, pixel_size_nm=args.pixel_size_nm)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "axial_projection_extents",
    "nearest_plane_indices",
    "per_plane_ncc",
    "render_calibration_diagnostics",
    "summarize_global_zmap",
]
