from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .._paths import PROJECT_ROOT as V04_ROOT

DEFAULT_INPUT_TIFF = (
    V04_ROOT.parent
    / "datasets/training_sets/double_helix/Real_dataset_Tcell/Calib.tif"
)
DEFAULT_OUTPUT_DIR = V04_ROOT / "output/double_helix/real_tcell/raw_calibration"
REAL_TCELL_ROI_ORIGIN_YX = (4, 39)
REAL_TCELL_ROI_SIZE = 19
REAL_TCELL_PIXEL_SIZE_NM = 207.0
REAL_TCELL_Z_STEP_NM = 40.0


def extract_real_tcell_calibration_roi(stack: np.ndarray) -> np.ndarray:
    values = np.asarray(stack)
    if values.ndim != 3:
        raise ValueError("Real T-cell calibration stack must have shape (Z,Y,X).")
    y0, x0 = REAL_TCELL_ROI_ORIGIN_YX
    y1 = y0 + REAL_TCELL_ROI_SIZE
    x1 = x0 + REAL_TCELL_ROI_SIZE
    if values.shape[1] < y1 or values.shape[2] < x1:
        raise ValueError("Real T-cell calibration stack is smaller than the verified ROI.")
    return values[:, y0:y1, x0:x1]


def _plotting() -> Any:
    cache_dir = V04_ROOT / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


def _plane_normalized(stack: np.ndarray) -> np.ndarray:
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
    return signal / signal.max(axis=(1, 2), keepdims=True).clip(min=1.0)


def _render_full_fov(
    stack: np.ndarray,
    *,
    z_nm: np.ndarray,
    output_path: Path,
) -> None:
    from matplotlib.patches import Rectangle

    plt = _plotting()
    selected = np.linspace(3, len(stack) - 4, 7, dtype=int)
    shown = np.asarray(stack)[selected]
    vmin, vmax = np.percentile(shown, (1.0, 99.8))
    fig, axes = plt.subplots(1, len(selected), figsize=(10.5, 1.9), constrained_layout=True)
    y0, x0 = REAL_TCELL_ROI_ORIGIN_YX
    for ax, plane, image in zip(axes, selected, shown, strict=True):
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        ax.add_patch(
            Rectangle(
                (x0 - 0.5, y0 - 0.5),
                REAL_TCELL_ROI_SIZE,
                REAL_TCELL_ROI_SIZE,
                fill=False,
                edgecolor="#00BFC4",
                linewidth=1.0,
            )
        )
        ax.set_title(f"z = {z_nm[plane]:.0f} nm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("Raw 75 x 75 (ADU)", fontsize=8)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def _render_roi_psf(
    roi_stack: np.ndarray,
    *,
    z_nm: np.ndarray,
    output_path: Path,
) -> None:
    plt = _plotting()
    normalized = _plane_normalized(roi_stack)
    selected = np.linspace(3, len(roi_stack) - 4, 7, dtype=int)
    half_width_um = 0.5 * REAL_TCELL_ROI_SIZE * REAL_TCELL_PIXEL_SIZE_NM / 1000.0
    figure = plt.figure(figsize=(10.5, 4.8))
    grid = figure.add_gridspec(
        2,
        14,
        height_ratios=(1.0, 1.45),
        left=0.06,
        right=0.88,
        bottom=0.12,
        top=0.92,
        hspace=0.42,
        wspace=0.55,
    )
    for column, plane in enumerate(selected):
        ax = figure.add_subplot(grid[0, 2 * column : 2 * column + 2])
        ax.imshow(normalized[plane], cmap="magma", vmin=0.0, vmax=1.0, origin="upper")
        ax.set_title(f"z = {z_nm[plane]:.0f} nm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if column == 0:
            ax.set_ylabel("Per-plane\nnormalized", fontsize=8)

    xz = normalized.max(axis=1)
    yz = normalized.max(axis=2)
    extent = (-half_width_um, half_width_um, float(z_nm[0]), float(z_nm[-1]))
    xz_ax = figure.add_subplot(grid[1, :7])
    yz_ax = figure.add_subplot(grid[1, 7:])
    xz_artist = xz_ax.imshow(
        xz,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    yz_ax.imshow(
        yz,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    xz_ax.set(xlabel="X (um)", ylabel="Z (nm)", title="Raw PSF XZ maximum projection")
    yz_ax.set(xlabel="Y (um)", title="Raw PSF YZ maximum projection")
    yz_ax.tick_params(axis="y", labelleft=False)
    colorbar_ax = figure.add_axes((0.91, 0.16, 0.015, 0.30))
    figure.colorbar(xz_artist, cax=colorbar_ax, label="Normalized intensity")
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def prepare_real_tcell_calibration(
    input_tiff: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    source_path = Path(input_tiff)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stack = tifffile.imread(source_path)
    roi_stack = extract_real_tcell_calibration_roi(stack)
    z_nm = (np.arange(len(stack), dtype=np.float64) - 0.5 * (len(stack) - 1)) * REAL_TCELL_Z_STEP_NM

    roi_tiff = output_root / "real_tcell_calib_cut1_19px.tif"
    full_fov_png = output_root / "raw_calibration_full_fov.png"
    roi_psf_png = output_root / "raw_calib_cut1_psf_xy_xz_yz.png"
    tifffile.imwrite(roi_tiff, roi_stack, imagej=True, metadata={"axes": "ZYX"})
    _render_full_fov(stack, z_nm=z_nm, output_path=full_fov_png)
    _render_roi_psf(roi_stack, z_nm=z_nm, output_path=roi_psf_png)

    manifest = {
        "source_tiff": str(source_path.resolve()),
        "source_shape_zyx": list(stack.shape),
        "roi_origin_yx": list(REAL_TCELL_ROI_ORIGIN_YX),
        "roi_size_px": REAL_TCELL_ROI_SIZE,
        "roi_shape_zyx": list(roi_stack.shape),
        "pixel_size_nm": REAL_TCELL_PIXEL_SIZE_NM,
        "z_step_nm": REAL_TCELL_Z_STEP_NM,
        "z_coordinate_semantics": "relative to stack midpoint",
        "roi_verification": "Calib.xls peak pixels match raw_ADU = 2 * origValue + 1 at x=39, y=4",
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "roi_tiff": str(roi_tiff),
        "full_fov_png": str(full_fov_png),
        "roi_psf_png": str(roi_psf_png),
        "manifest": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the verified real T-cell DH calibration ROI.")
    parser.add_argument("--input-tiff", type=Path, default=DEFAULT_INPUT_TIFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(prepare_real_tcell_calibration(args.input_tiff, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REAL_TCELL_PIXEL_SIZE_NM",
    "REAL_TCELL_ROI_ORIGIN_YX",
    "REAL_TCELL_ROI_SIZE",
    "REAL_TCELL_Z_STEP_NM",
    "extract_real_tcell_calibration_roi",
    "prepare_real_tcell_calibration",
]
