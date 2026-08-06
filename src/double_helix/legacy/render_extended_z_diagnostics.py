from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from .._paths import PROJECT_ROOT as V04_ROOT
from .render_calibration_diagnostics import nearest_plane_indices, per_plane_ncc
from .render_wavefront_diagnostics import orthogonal_max_projections
from ..vector_model import DoubleHelixVectorPSF


DEFAULT_CALIBRATION_DIR = (
    V04_ROOT
    / "output/double_helix/real_tcell_deepz_mode64_warm42_roi17_zopt/calibration"
)
DEFAULT_SOURCE_TIFF = (
    V04_ROOT.parent / "datasets/training_sets/double_helix/Real_dataset_Tcell/Calib.tif"
)
REAL_TCELL_ROI_ORIGIN_YX = (5, 40)
REAL_TCELL_ROI_SIZE = 17


def centered_stage_z_nm(plane_count: int, *, z_step_nm: float) -> np.ndarray:
    indices = np.arange(int(plane_count), dtype=np.float64)
    return (indices - 0.5 * (plane_count - 1)) * float(z_step_nm)


def select_symmetric_z_range(stage_z_nm: np.ndarray, *, z_limit_nm: float) -> np.ndarray:
    values = np.asarray(stage_z_nm, dtype=np.float64).reshape(-1)
    return np.flatnonzero(np.abs(values) <= float(z_limit_nm) + 1e-9)


def select_dense_z_planes(
    stage_z_nm: np.ndarray,
    *,
    z_limit_nm: float,
    plane_count: int,
) -> np.ndarray:
    if plane_count < 2:
        raise ValueError("plane_count must be at least two.")
    targets_nm = np.linspace(-float(z_limit_nm), float(z_limit_nm), int(plane_count))
    selected = nearest_plane_indices(stage_z_nm, targets_nm)
    if selected.size != int(plane_count):
        raise ValueError("plane_count exceeds the number of unique source Z planes.")
    return selected


def fitted_optical_z_nm(
    stage_z_nm: np.ndarray,
    *,
    z_scale: float,
    z_offset_nm: float,
) -> np.ndarray:
    return float(z_scale) * np.asarray(stage_z_nm, dtype=np.float64) + float(z_offset_nm)


def fit_extrapolation_coordinates(
    stage_z_nm: np.ndarray,
    *,
    fit_range_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(stage_z_nm, dtype=np.float64)
    coordinates = values / float(fit_range_nm)
    optimized_mask = np.abs(values) <= float(fit_range_nm) + 1e-9
    return coordinates, optimized_mask


def edge_flux_fraction(psf_volume: np.ndarray) -> np.ndarray:
    psf = np.asarray(psf_volume, dtype=np.float64)
    if psf.ndim != 3:
        raise ValueError("psf_volume must have shape (Z,Y,X).")
    border = np.concatenate(
        (psf[:, 0], psf[:, -1], psf[:, 1:-1, 0], psf[:, 1:-1, -1]),
        axis=1,
    )
    return border.sum(axis=1) / psf.sum(axis=(1, 2)).clip(min=1e-12)


def outside_fit_statistics(
    ncc: np.ndarray,
    edge_flux: np.ndarray,
    optimized_mask: np.ndarray,
) -> dict[str, float | None]:
    outside = ~np.asarray(optimized_mask, dtype=bool)
    outside_ncc = np.asarray(ncc, dtype=np.float64)[outside]
    outside_edge_flux = np.asarray(edge_flux, dtype=np.float64)[outside]
    if outside_ncc.size == 0:
        return {
            "outside_fit_median_cropped_shape_ncc": None,
            "outside_fit_median_edge_flux_fraction": None,
            "outside_fit_max_edge_flux_fraction": None,
        }
    return {
        "outside_fit_median_cropped_shape_ncc": float(np.median(outside_ncc)),
        "outside_fit_median_edge_flux_fraction": float(np.median(outside_edge_flux)),
        "outside_fit_max_edge_flux_fraction": float(np.max(outside_edge_flux)),
    }


def load_saved_model_volume(
    calibration_dir: str | Path,
    *,
    expected_shape: tuple[int, int, int],
) -> np.ndarray | None:
    path = Path(calibration_dir) / "stacks/reconstruction_unit_flux.tif"
    if not path.is_file():
        return None
    volume = np.asarray(tifffile.imread(path), dtype=np.float32)
    if volume.shape != expected_shape:
        return None
    return volume


def photometry_matched_signals(
    observed_adu: np.ndarray,
    model_psf: np.ndarray,
    *,
    photons_adu: np.ndarray,
    background_adu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed_adu, dtype=np.float32)
    model = np.asarray(model_psf, dtype=np.float32)
    photons = np.asarray(photons_adu, dtype=np.float32).reshape(-1)
    background = np.asarray(background_adu, dtype=np.float32).reshape(-1)
    if observed.shape != model.shape or photons.shape != (len(observed),):
        raise ValueError("Photometry arrays must match the observed PSF volume.")
    if background.shape != photons.shape:
        raise ValueError("background_adu must contain one value per plane.")

    raw_unit = np.maximum(
        (observed - background[:, None, None]) / photons[:, None, None],
        0.0,
    )
    model_unit = np.maximum(model, 0.0)
    raw_unit /= raw_unit.sum(axis=(1, 2), keepdims=True).clip(min=1e-12)
    model_unit /= model_unit.sum(axis=(1, 2), keepdims=True).clip(min=1e-12)
    raw_signal = raw_unit / raw_unit.max(axis=(1, 2), keepdims=True).clip(min=1e-12)
    model_signal = model_unit / model_unit.max(axis=(1, 2), keepdims=True).clip(min=1e-12)
    return raw_signal, model_signal


def render_extended_z_diagnostics(
    calibration_dir: str | Path,
    source_tiff: str | Path,
    *,
    z_limit_nm: float = 2000.0,
    selected_plane_count: int = 9,
    device: str = "cpu",
) -> dict[str, Any]:
    root = Path(calibration_dir)
    source_path = Path(source_tiff)
    config = json.loads((root / "config/resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "metadata/manifest.json").read_text(encoding="utf-8"))
    with np.load(root / "arrays/gamma_coefficients.npz", allow_pickle=False) as payload:
        gamma_nm = np.asarray(payload["gamma_nm"], dtype=np.float32).reshape(-1)
        mode_order = tuple(tuple(int(value) for value in row) for row in payload["mode_order"])
    with np.load(root / "arrays/calibration_fit.npz", allow_pickle=False) as payload:
        dx_affine_px = np.asarray(payload["dx_affine_px"], dtype=np.float32)
        dy_affine_px = np.asarray(payload["dy_affine_px"], dtype=np.float32)
        fit_observed_adu = np.asarray(payload["observed_adu"], dtype=np.float32)
        fit_photons_adu = np.asarray(payload["photons_adu"], dtype=np.float32)
        fit_background_adu = np.asarray(payload["background_adu"], dtype=np.float32)
        fit_source_indices = np.asarray(payload["source_plane_indices"], dtype=np.int64)

    raw_stack = np.asarray(tifffile.imread(source_path), dtype=np.float32)
    full_stage_z_nm = centered_stage_z_nm(len(raw_stack), z_step_nm=float(config["z_step_nm"]))
    source_indices = select_symmetric_z_range(full_stage_z_nm, z_limit_nm=z_limit_nm)
    stage_z_nm = full_stage_z_nm[source_indices]
    z_scale = float(manifest["fitted_z_scale"])
    z_offset_nm = float(manifest["fitted_z_offset_nm"])
    optical_z_nm = fitted_optical_z_nm(stage_z_nm, z_scale=z_scale, z_offset_nm=z_offset_nm)
    fit_range_nm = float(config["fit_z_range_nm"])
    fit_coordinate, optimized_mask = fit_extrapolation_coordinates(
        stage_z_nm,
        fit_range_nm=fit_range_nm,
    )
    dx_px = dx_affine_px[0] + dx_affine_px[1] * fit_coordinate
    dy_px = dy_affine_px[0] + dy_affine_px[1] * fit_coordinate

    y0, x0 = REAL_TCELL_ROI_ORIGIN_YX
    raw_full = raw_stack[source_indices]
    raw_roi = raw_full[:, y0 : y0 + REAL_TCELL_ROI_SIZE, x0 : x0 + REAL_TCELL_ROI_SIZE]
    model_psf = load_saved_model_volume(
        root,
        expected_shape=tuple(int(value) for value in raw_roi.shape),
    )
    if model_psf is None:
        model_psf = _render_model_volume(
            gamma_nm,
            mode_order,
            optical_z_nm=optical_z_nm,
            dx_px=dx_px,
            dy_px=dy_px,
            config=config,
            device=device,
        )
        model_volume_source = f"fresh vector PSF render on {device}"
    else:
        model_volume_source = "saved formal calibration reconstruction_unit_flux.tif"
    photometry_matched = (
        fit_observed_adu.shape == raw_roi.shape
        and fit_photons_adu.shape == (len(raw_roi),)
        and fit_background_adu.shape == (len(raw_roi),)
        and np.array_equal(fit_source_indices, source_indices)
    )
    if photometry_matched:
        raw_signal, model_signal = photometry_matched_signals(
            raw_roi,
            model_psf,
            photons_adu=fit_photons_adu,
            background_adu=fit_background_adu,
        )
        normalization_label = "photometry matched: fitted background and photons"
        output_suffix = "_photometry_matched"
    else:
        raw_signal = _plane_normalized_signal(raw_roi)
        model_signal = model_psf / model_psf.max(axis=(1, 2), keepdims=True).clip(min=1e-12)
        normalization_label = "raw border-subtracted; model unit flux"
        output_suffix = ""
    ncc = per_plane_ncc(raw_signal, model_signal)
    edge_flux = edge_flux_fraction(model_psf)
    selected = select_dense_z_planes(
        stage_z_nm,
        z_limit_nm=z_limit_nm,
        plane_count=selected_plane_count,
    )

    figures_dir = root / "figures"
    arrays_dir = root / "arrays"
    metadata_dir = root / "metadata"
    for directory in (figures_dir, arrays_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    range_tag = f"z{float(z_limit_nm):g}".replace(".", "p")
    full_raw_figure = figures_dir / f"raw_tiff_full_75px_{range_tag}.png"
    comparison_figure = figures_dir / (
        f"raw_vs_model_psf_xy_{range_tag}_n{selected_plane_count}{output_suffix}.png"
    )
    axial_figure = figures_dir / f"raw_vs_model_psf_xz_yz_{range_tag}{output_suffix}.png"
    arrays_path = arrays_dir / f"extended_{range_tag}{output_suffix}_diagnostics.npz"
    _render_full_raw_tiff(
        raw_full,
        stage_z_nm=stage_z_nm,
        selected=selected,
        roi_origin_yx=REAL_TCELL_ROI_ORIGIN_YX,
        roi_size=REAL_TCELL_ROI_SIZE,
        path=full_raw_figure,
    )
    _render_extended_comparison(
        raw_signal,
        model_signal,
        stage_z_nm=stage_z_nm,
        optical_z_nm=optical_z_nm,
        ncc=ncc,
        optimized_mask=optimized_mask,
        selected=selected,
        pixel_size_nm=float(config["pixel_size_nm"]),
        z_limit_nm=float(z_limit_nm),
        fit_range_nm=fit_range_nm,
        normalization_label=normalization_label,
        path=comparison_figure,
    )
    _render_axial_comparison(
        raw_signal,
        model_signal,
        stage_z_nm=stage_z_nm,
        optical_z_nm=optical_z_nm,
        pixel_size_nm=float(config["pixel_size_nm"]),
        fit_range_nm=fit_range_nm,
        normalization_label=normalization_label,
        path=axial_figure,
    )
    np.savez_compressed(
        arrays_path,
        source_plane_indices=source_indices,
        stage_z_nm=stage_z_nm,
        fitted_optical_z_nm=optical_z_nm,
        raw_roi_adu=raw_roi,
        model_psf_unit_flux=model_psf,
        cropped_shape_ncc=ncc.astype(np.float32),
        edge_flux_fraction=edge_flux.astype(np.float32),
        optimized_mask=optimized_mask,
        selected_plane_indices=selected,
    )

    result = {
        "source_tiff": str(source_path),
        "source_shape_zyx": [int(value) for value in raw_stack.shape],
        "full_frame_size_px": [int(raw_stack.shape[1]), int(raw_stack.shape[2])],
        "roi_origin_yx": list(REAL_TCELL_ROI_ORIGIN_YX),
        "roi_size_px": REAL_TCELL_ROI_SIZE,
        "stage_z_range_nm": [float(stage_z_nm[0]), float(stage_z_nm[-1])],
        "fitted_optical_z_range_nm": [float(optical_z_nm[0]), float(optical_z_nm[-1])],
        "optimized_stage_z_range_nm": [-fit_range_nm, fit_range_nm],
        "outside_fit_semantics": (
            f"forward-model extrapolation only; no planes outside +/-{fit_range_nm:g} nm participated in optimization"
        ),
        "ncc_semantics": (
            "17x17 cropped photometry-matched, per-plane peak-normalized shape NCC; "
            "not full-PSF energy closure"
            if photometry_matched
            else "17x17 cropped, border-subtracted, per-plane-normalized shape NCC; not full-PSF energy closure"
        ),
        "edge_flux_semantics": "fraction of the unit-flux 17x17 model PSF on the one-pixel border; high values indicate truncation risk",
        "inside_fit_median_cropped_shape_ncc": float(np.median(ncc[optimized_mask])),
        **outside_fit_statistics(ncc, edge_flux, optimized_mask),
        "selected_stage_z_nm": stage_z_nm[selected].tolist(),
        "selected_fitted_optical_z_nm": optical_z_nm[selected].tolist(),
        "selected_ncc": ncc[selected].tolist(),
        "selected_edge_flux_fraction": edge_flux[selected].tolist(),
        "selected_plane_count": int(selected_plane_count),
        "device": device,
        "model_volume_source": model_volume_source,
        "photometry_matched": bool(photometry_matched),
        "signal_normalization": normalization_label,
        "figures": [str(full_raw_figure), str(comparison_figure), str(axial_figure)],
        "arrays": str(arrays_path),
    }
    metadata_path = metadata_dir / f"extended_{range_tag}{output_suffix}_diagnostics.json"
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["metadata"] = str(metadata_path)
    return result


def _render_model_volume(
    gamma_nm: np.ndarray,
    mode_order: tuple[tuple[int, int], ...],
    *,
    optical_z_nm: np.ndarray,
    dx_px: np.ndarray,
    dy_px: np.ndarray,
    config: dict[str, Any],
    device: str,
) -> np.ndarray:
    model = DoubleHelixVectorPSF(
        mode_order=mode_order,
        na=float(config["na"]),
        wavelength_nm=float(config["wavelength_nm"]),
        pixel_size_nm=float(config["pixel_size_nm"]),
        refractive_index=float(config["refractive_index"]),
        npupil=int(config["npupil"]),
        psf_size=int(config["psf_size"]),
        device=device,
    )
    rendered = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(optical_z_nm), batch_size):
            stop = min(start + batch_size, len(optical_z_nm))
            coefficients = np.broadcast_to(gamma_nm, (stop - start, len(gamma_nm))).copy()
            batch = model.render(
                coefficients_nm=coefficients,
                z_nm=optical_z_nm[start:stop],
                dx_px=dx_px[start:stop],
                dy_px=dy_px[start:stop],
            )
            rendered.append(batch.cpu().numpy().astype(np.float32))
    return np.concatenate(rendered, axis=0)


def _plane_normalized_signal(stack: np.ndarray) -> np.ndarray:
    values = np.asarray(stack, dtype=np.float32)
    border = np.concatenate(
        (
            values[:, :2].reshape(len(values), -1),
            values[:, -2:].reshape(len(values), -1),
            values[:, 2:-2, :2].reshape(len(values), -1),
            values[:, 2:-2, -2:].reshape(len(values), -1),
        ),
        axis=1,
    )
    background = np.median(border, axis=1)
    signal = np.maximum(values - background[:, None, None], 0.0)
    return signal / signal.max(axis=(1, 2), keepdims=True).clip(min=1e-12)


def _render_full_raw_tiff(
    raw_full: np.ndarray,
    *,
    stage_z_nm: np.ndarray,
    selected: np.ndarray,
    roi_origin_yx: tuple[int, int],
    roi_size: int,
    path: Path,
) -> None:
    from matplotlib.patches import Rectangle

    plt = _plotting()
    with plt.rc_context(_figure_style()):
        width = max(14.0, 1.45 * len(selected))
        fig, axes = plt.subplots(1, len(selected), figsize=(width, 2.35), constrained_layout=True)
        y0, x0 = roi_origin_yx
        for axis, index in zip(axes, selected, strict=True):
            image = raw_full[index]
            vmin, vmax = np.percentile(image, (1.0, 99.8))
            axis.imshow(image, cmap="gray", vmin=float(vmin), vmax=float(vmax), origin="upper")
            axis.add_patch(
                Rectangle(
                    (x0 - 0.5, y0 - 0.5),
                    roi_size,
                    roi_size,
                    fill=False,
                    edgecolor="#00BFC4",
                    linewidth=1.0,
                )
            )
            axis.set_title(f"stage {stage_z_nm[index]:+.0f} nm", fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[0].set_ylabel("Raw 75 x 75 pixels\nper-plane contrast")
        fig.suptitle("Real T-cell Calib.tif full-size raw frames | cyan box = fitted 17 x 17 ROI", fontsize=10)
        _save_png_pdf(fig, path)
        plt.close(fig)


def _render_extended_comparison(
    raw_signal: np.ndarray,
    model_signal: np.ndarray,
    *,
    stage_z_nm: np.ndarray,
    optical_z_nm: np.ndarray,
    ncc: np.ndarray,
    optimized_mask: np.ndarray,
    selected: np.ndarray,
    pixel_size_nm: float,
    z_limit_nm: float,
    fit_range_nm: float,
    normalization_label: str,
    path: Path,
) -> None:
    from matplotlib.colors import PowerNorm

    plt = _plotting()
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    radius_um = raw_signal.shape[-1] * pixel_size_nm / 2000.0
    extent = (-radius_um, radius_um, radius_um, -radius_um)
    with plt.rc_context(_figure_style()):
        width = max(13.5, 1.45 * len(selected))
        fig, axes = plt.subplots(2, len(selected), figsize=(width, 4.2), constrained_layout=True)
        for column, index in enumerate(selected):
            is_optimized = bool(optimized_mask[index])
            status = "FIT" if is_optimized else "EXTRAP"
            border_color = "black" if is_optimized else "#E69F00"
            axes[0, column].imshow(raw_signal[index], cmap="magma", norm=norm, extent=extent)
            axes[1, column].imshow(model_signal[index], cmap="magma", norm=norm, extent=extent)
            axes[0, column].set_title(
                f"stage {stage_z_nm[index]:+.0f} nm\n"
                f"model {optical_z_nm[index]:+.0f} nm\n{status}",
                fontsize=7,
                color=border_color,
            )
            axes[1, column].set_xlabel(f"shape NCC {ncc[index]:.3f}", fontsize=7)
            for row in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                for spine in axes[row, column].spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(1.4 if not is_optimized else 0.8)
        axes[0, 0].set_ylabel("Raw 17 x 17 ROI\nper-plane normalized")
        axes[1, 0].set_ylabel("Global-pupil model\nper-plane normalized")
        fig.suptitle(
            f"Real T-cell PSF to stage +/-{z_limit_nm:g} nm | "
            f"{normalization_label} | "
            f"orange EXTRAP = outside optimized +/-{fit_range_nm:g} nm",
            fontsize=10,
        )
        _save_png_pdf(fig, path)
        plt.close(fig)


def _render_axial_comparison(
    raw_signal: np.ndarray,
    model_signal: np.ndarray,
    *,
    stage_z_nm: np.ndarray,
    optical_z_nm: np.ndarray,
    pixel_size_nm: float,
    fit_range_nm: float,
    normalization_label: str,
    path: Path,
) -> None:
    from matplotlib.colors import PowerNorm

    raw_xz, raw_yz = orthogonal_max_projections(raw_signal)
    model_xz, model_yz = orthogonal_max_projections(model_signal)
    radius_um = raw_signal.shape[-1] * pixel_size_nm / 2000.0
    z_min_um = float(stage_z_nm[0]) / 1000.0
    z_max_um = float(stage_z_nm[-1]) / 1000.0
    xz_extent = (-radius_um, radius_um, z_min_um, z_max_um)
    yz_extent = (-radius_um, radius_um, z_min_um, z_max_um)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    panels = (
        (raw_xz, "Raw XZ", "X (um)", xz_extent),
        (raw_yz, "Raw YZ", "Y (um)", yz_extent),
        (model_xz, "Recovered model XZ", "X (um)", xz_extent),
        (model_yz, "Recovered model YZ", "Y (um)", yz_extent),
    )

    plt = _plotting()
    with plt.rc_context(_figure_style()):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.8), sharey=True, constrained_layout=True)
        images = []
        for axis, (image, title, lateral_label, extent) in zip(
            axes.ravel(), panels, strict=True
        ):
            images.append(
                axis.imshow(
                    image,
                    cmap="magma",
                    norm=norm,
                    extent=extent,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                )
            )
            axis.set_title(title)
            axis.set_xlabel(lateral_label)
            axis.set_ylabel("Stage Z (um)")
            if fit_range_nm < max(abs(float(stage_z_nm[0])), abs(float(stage_z_nm[-1]))):
                for boundary_nm in (-fit_range_nm, fit_range_nm):
                    axis.axhline(
                        boundary_nm / 1000.0,
                        color="#E69F00",
                        linewidth=1.0,
                        linestyle="--",
                    )
        colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label("Per-plane normalized max intensity")
        fig.suptitle(
            "Real T-cell PSF axial shape\n"
            f"Stage Z {stage_z_nm[0]:+.0f} to {stage_z_nm[-1]:+.0f} nm | "
            f"model optical Z {optical_z_nm[0]:+.0f} to {optical_z_nm[-1]:+.0f} nm\n"
            f"{normalization_label}",
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
    parser = argparse.ArgumentParser(description="Render full-size raw TIFF and extended-Z model diagnostics.")
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--source-tiff", type=Path, default=DEFAULT_SOURCE_TIFF)
    parser.add_argument("--z-limit-nm", type=float, default=2000.0)
    parser.add_argument("--selected-plane-count", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = render_extended_z_diagnostics(
        args.calibration_dir,
        args.source_tiff,
        z_limit_nm=args.z_limit_nm,
        selected_plane_count=args.selected_plane_count,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "centered_stage_z_nm",
    "edge_flux_fraction",
    "fit_extrapolation_coordinates",
    "fitted_optical_z_nm",
    "load_saved_model_volume",
    "outside_fit_statistics",
    "photometry_matched_signals",
    "render_extended_z_diagnostics",
    "select_dense_z_planes",
    "select_symmetric_z_range",
]
