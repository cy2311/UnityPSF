from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def render_bin_count(
    rows: Iterable[dict[str, object]],
    *,
    pixel_size_nm: float,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
    width_px: int,
    height_px: int,
) -> np.ndarray:
    width_bins = max(1, int(np.ceil(width_px * camera_pixel_nm_x / pixel_size_nm)))
    height_bins = max(1, int(np.ceil(height_px * camera_pixel_nm_y / pixel_size_nm)))
    counts = np.zeros((height_bins, width_bins), dtype=np.uint16)
    x_scale = camera_pixel_nm_x / pixel_size_nm
    y_scale = camera_pixel_nm_y / pixel_size_nm
    for row in rows:
        x_px = float(row["x_px"])
        y_px = float(row["y_px"])
        bx = int(np.floor(x_px * x_scale))
        by = int(np.floor(y_px * y_scale))
        if 0 <= bx < width_bins and 0 <= by < height_bins:
            if counts[by, bx] < np.iinfo(np.uint16).max:
                counts[by, bx] += 1
    return counts


def save_count_png(counts: np.ndarray, out_path: Path) -> None:
    arr = counts.astype(np.float32)
    non_zero = arr[arr > 0]
    vmax = float(np.percentile(non_zero, 99.5)) if non_zero.size else 1.0
    vmax = max(vmax, 1.0)
    scaled = np.clip(arr / vmax, 0.0, 1.0)
    image = (scaled * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(out_path)


def save_rgb_overlay(red_counts: np.ndarray, green_counts: np.ndarray, out_path: Path) -> None:
    red = red_counts.astype(np.float32)
    green = green_counts.astype(np.float32)
    red_scale = float(np.percentile(red[red > 0], 99.5)) if np.any(red > 0) else 1.0
    green_scale = float(np.percentile(green[green > 0], 99.5)) if np.any(green > 0) else 1.0
    red_img = np.clip(red / max(red_scale, 1.0), 0.0, 1.0)
    green_img = np.clip(green / max(green_scale, 1.0), 0.0, 1.0)
    rgb = np.zeros(red.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (red_img * 255.0).astype(np.uint8)
    rgb[..., 1] = (green_img * 255.0).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def write_rows_csv(rows: Iterable[dict[str, object]], out_path: Path) -> None:
    rows = list(rows)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
