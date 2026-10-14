#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, shift as image_shift
from scipy.optimize import least_squares
from skimage.registration import phase_cross_correlation


def solve_redundant_shifts(
    block_count: int,
    pairs: list[tuple[int, int, float, float]],
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve d[j] - d[i] = pair displacement with block zero fixed at zero."""
    if block_count < 2:
        return np.zeros(block_count, dtype=np.float64), np.zeros(block_count, dtype=np.float64)
    if not pairs:
        raise ValueError("RCC requires at least one valid block pair")
    matrix = np.zeros((len(pairs), block_count - 1), dtype=np.float64)
    measured_y = np.empty(len(pairs), dtype=np.float64)
    measured_x = np.empty(len(pairs), dtype=np.float64)
    for row, (i, j, dy, dx) in enumerate(pairs):
        if i > 0:
            matrix[row, i - 1] = -1.0
        if j > 0:
            matrix[row, j - 1] = 1.0
        measured_y[row] = float(dy)
        measured_x[row] = float(dx)
    sqrt_weight = np.sqrt(np.ones(len(pairs)) if weights is None else np.asarray(weights, dtype=np.float64))

    def solve(values: np.ndarray) -> np.ndarray:
        initial, *_ = np.linalg.lstsq(matrix, values, rcond=None)
        result = least_squares(
            lambda candidate: sqrt_weight * (matrix @ candidate - values),
            initial,
            loss="soft_l1",
            f_scale=0.5,
        )
        return np.concatenate(([0.0], result.x))

    return solve(measured_y), solve(measured_x)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RCC drift diagnostic for Neptune localization H5 files.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-block-size", type=int, default=500)
    parser.add_argument("--camera-pixel-nm-x", type=float, default=101.11)
    parser.add_argument("--camera-pixel-nm-y", type=float, default=98.83)
    parser.add_argument("--width-px", type=int, default=400)
    parser.add_argument("--height-px", type=int, default=400)
    parser.add_argument("--rcc-pixel-nm", type=float, default=50.0)
    parser.add_argument("--rcc-sigma-px", type=float, default=1.0)
    parser.add_argument("--upsample-factor", type=int, default=10)
    parser.add_argument("--max-pair-gap", type=int, default=8)
    parser.add_argument("--max-shift-nm", type=float, default=1000.0)
    parser.add_argument("--min-pair-correlation", type=float, default=0.02)
    return parser.parse_args()


def _render_blocks(
    frame: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    block_edges: np.ndarray,
    width_px: int,
    height_px: int,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
    rcc_pixel_nm: float,
    sigma_px: float,
) -> tuple[list[np.ndarray], list[int]]:
    width = int(np.ceil(width_px * camera_pixel_nm_x / rcc_pixel_nm))
    height = int(np.ceil(height_px * camera_pixel_nm_y / rcc_pixel_nm))
    x_bin = np.floor(x_px * camera_pixel_nm_x / rcc_pixel_nm).astype(np.int64)
    y_bin = np.floor(y_px * camera_pixel_nm_y / rcc_pixel_nm).astype(np.int64)
    valid_xy = (x_bin >= 0) & (x_bin < width) & (y_bin >= 0) & (y_bin < height)
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    images: list[np.ndarray] = []
    counts: list[int] = []
    for start, stop in zip(block_edges[:-1], block_edges[1:]):
        keep = valid_xy & (frame >= start) & (frame < stop)
        density = np.zeros((height, width), dtype=np.float32)
        np.add.at(density, (y_bin[keep], x_bin[keep]), 1.0)
        density = gaussian_filter(density, sigma=float(sigma_px), mode="constant")
        density = np.sqrt(density, dtype=np.float32) * window
        images.append(density)
        counts.append(int(keep.sum()))
    return images, counts


def _normalized_correlation(reference: np.ndarray, moving: np.ndarray, shift_yx: np.ndarray) -> float:
    aligned = image_shift(moving, shift=shift_yx, order=1, mode="constant", prefilter=False)
    ref = reference.ravel().astype(np.float64, copy=False)
    mov = aligned.ravel().astype(np.float64, copy=False)
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(mov))
    return float(np.dot(ref, mov) / denominator) if denominator > 0 else 0.0


def _write_corrected_h5(
    source: Path,
    destination: Path,
    *,
    frame: np.ndarray,
    drift_x_px: np.ndarray,
    drift_y_px: np.ndarray,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
) -> None:
    shutil.copy2(source, destination)
    frame_int = frame.astype(np.int64, copy=False)
    with h5py.File(destination, "r+") as handle:
        group = handle["locs"]
        for key in ("x_px", "x_px_full"):
            if key in group:
                group[key][:] = group[key][:] - drift_x_px[frame_int]
        for key in ("y_px", "y_px_full"):
            if key in group:
                group[key][:] = group[key][:] - drift_y_px[frame_int]
        for key in ("x_nm", "x_nm_full"):
            if key in group:
                group[key][:] = group[key][:] - drift_x_px[frame_int] * float(camera_pixel_nm_x)
        for key in ("y_nm", "y_nm_full"):
            if key in group:
                group[key][:] = group[key][:] - drift_y_px[frame_int] * float(camera_pixel_nm_y)
        for key, values in (
            ("drift_x_px", drift_x_px[frame_int]),
            ("drift_y_px", drift_y_px[frame_int]),
            ("drift_x_nm", drift_x_px[frame_int] * float(camera_pixel_nm_x)),
            ("drift_y_nm", drift_y_px[frame_int] * float(camera_pixel_nm_y)),
        ):
            if key in group:
                del group[key]
            group.create_dataset(key, data=values.astype(np.float32), compression="gzip", shuffle=True)
        handle.attrs["derived_kind"] = "rcc_drift_corrected"
        handle.attrs["rcc_source_predictions"] = str(source)


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.predictions, "r") as handle:
        group = handle["locs"]
        frame = group["frame"][:].astype(np.int64)
        x_px = group["x_px"][:].astype(np.float32)
        y_px = group["y_px"][:].astype(np.float32)
    frame_min = int(frame.min())
    frame_max = int(frame.max())
    first_edge = (frame_min // int(args.frame_block_size)) * int(args.frame_block_size)
    block_edges = np.arange(first_edge, frame_max + int(args.frame_block_size) + 1, int(args.frame_block_size), dtype=np.int64)
    images, counts = _render_blocks(
        frame,
        x_px,
        y_px,
        block_edges=block_edges,
        width_px=int(args.width_px),
        height_px=int(args.height_px),
        camera_pixel_nm_x=float(args.camera_pixel_nm_x),
        camera_pixel_nm_y=float(args.camera_pixel_nm_y),
        rcc_pixel_nm=float(args.rcc_pixel_nm),
        sigma_px=float(args.rcc_sigma_px),
    )

    pair_rows: list[dict[str, float | int]] = []
    pairs: list[tuple[int, int, float, float]] = []
    weights: list[float] = []
    max_shift_px = float(args.max_shift_nm) / float(args.rcc_pixel_nm)
    for i in range(len(images)):
        for j in range(i + 1, min(len(images), i + int(args.max_pair_gap) + 1)):
            shift_yx, error, _ = phase_cross_correlation(
                images[i],
                images[j],
                upsample_factor=int(args.upsample_factor),
                normalization="phase",
            )
            correlation = _normalized_correlation(images[i], images[j], shift_yx)
            displacement_y = -float(shift_yx[0])
            displacement_x = -float(shift_yx[1])
            accepted = (
                abs(displacement_y) <= max_shift_px
                and abs(displacement_x) <= max_shift_px
                and correlation >= float(args.min_pair_correlation)
            )
            pair_rows.append(
                {
                    "block_i": i,
                    "block_j": j,
                    "dy_rcc_px": displacement_y,
                    "dx_rcc_px": displacement_x,
                    "dy_nm": displacement_y * float(args.rcc_pixel_nm),
                    "dx_nm": displacement_x * float(args.rcc_pixel_nm),
                    "correlation": correlation,
                    "phase_error": float(error),
                    "accepted": int(accepted),
                }
            )
            if accepted:
                pairs.append((i, j, displacement_y, displacement_x))
                weights.append(max(correlation, 1e-3))

    drift_y_rcc, drift_x_rcc = solve_redundant_shifts(len(images), pairs, np.asarray(weights))
    block_centers = (block_edges[:-1] + block_edges[1:] - 1) / 2.0
    all_frames = np.arange(frame_max + 1, dtype=np.float64)
    drift_x_nm = np.interp(all_frames, block_centers, drift_x_rcc * float(args.rcc_pixel_nm))
    drift_y_nm = np.interp(all_frames, block_centers, drift_y_rcc * float(args.rcc_pixel_nm))
    drift_x_px = drift_x_nm / float(args.camera_pixel_nm_x)
    drift_y_px = drift_y_nm / float(args.camera_pixel_nm_y)

    pair_csv = args.output_dir / "rcc_pairwise_shifts.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    trajectory_csv = args.output_dir / "rcc_drift_trajectory.csv"
    with trajectory_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "drift_x_nm", "drift_y_nm", "drift_x_px", "drift_y_px"])
        writer.writerows(zip(all_frames.astype(np.int64), drift_x_nm, drift_y_nm, drift_x_px, drift_y_px))

    trajectory_png = args.output_dir / "rcc_drift_trajectory.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(all_frames, drift_x_nm, color="#0072B2", linewidth=2)
    axes[0].scatter(block_centers, drift_x_rcc * float(args.rcc_pixel_nm), color="#0072B2", s=24)
    axes[0].set_ylabel("drift x (nm)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(all_frames, drift_y_nm, color="#D55E00", linewidth=2)
    axes[1].scatter(block_centers, drift_y_rcc * float(args.rcc_pixel_nm), color="#D55E00", s=24)
    axes[1].set_ylabel("drift y (nm)")
    axes[1].set_xlabel("frame")
    axes[1].grid(alpha=0.25)
    fig.savefig(trajectory_png, dpi=180)
    plt.close(fig)

    corrected_h5 = args.output_dir / "predictions_degrid_rcc_corrected.h5"
    _write_corrected_h5(
        args.predictions,
        corrected_h5,
        frame=frame,
        drift_x_px=drift_x_px,
        drift_y_px=drift_y_px,
        camera_pixel_nm_x=float(args.camera_pixel_nm_x),
        camera_pixel_nm_y=float(args.camera_pixel_nm_y),
    )
    summary = {
        "source_predictions": str(args.predictions),
        "corrected_predictions": str(corrected_h5),
        "frame_range": [frame_min, frame_max],
        "frame_block_size": int(args.frame_block_size),
        "block_count": len(images),
        "block_localization_counts": counts,
        "pair_count_total": len(pair_rows),
        "pair_count_accepted": len(pairs),
        "drift_x_range_nm": [float(drift_x_nm.min()), float(drift_x_nm.max())],
        "drift_y_range_nm": [float(drift_y_nm.min()), float(drift_y_nm.max())],
        "trajectory_csv": str(trajectory_csv),
        "trajectory_png": str(trajectory_png),
        "pairwise_csv": str(pair_csv),
    }
    summary_path = args.output_dir / "rcc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
