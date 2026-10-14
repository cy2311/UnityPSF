#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unity_psf.infer_recon.recon.render_subpixel import (
    NEPTUNE_DEFAULT_IMAX_MIN,
    _colorize_density_display,
    _normalize_display,
    _parse_normalization_fov,
    _smap_quantile_from_imax_min,
)


RAW_TIFF = Path("/home/guest/Others/main/race/neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Union left/right emitter sets, then render raw-TIFF ratiometric bicolor.")
    parser.add_argument("--left-predictions", type=Path, required=True)
    parser.add_argument("--right-predictions", type=Path, required=True)
    parser.add_argument("--sample-tiff", type=Path, default=RAW_TIFF)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratio-threshold", type=float, default=0.4)
    parser.add_argument("--color-mode", choices=("binary", "channel_z_gradient"), default="binary")
    parser.add_argument("--union-dist-px", type=float, default=2.0)
    parser.add_argument("--union-policy", choices=("right_priority_union", "matched_only"), default="right_priority_union")
    parser.add_argument("--signal-radius", type=int, default=2)
    parser.add_argument("--bg-inner-radius", type=int, default=6)
    parser.add_argument("--bg-outer-radius", type=int, default=10)
    parser.add_argument("--min-total-intensity", type=float, default=0.0)
    parser.add_argument("--right-crop-left", type=int, default=600)
    parser.add_argument("--alignment-mode", choices=("none", "auto_translation"), default="auto_translation")
    parser.add_argument("--left-to-right-dx-px", type=float, default=None)
    parser.add_argument("--left-to-right-dy-px", type=float, default=None)
    parser.add_argument("--alignment-bin-px", type=float, default=4.0)
    parser.add_argument("--alignment-max-shift-px", type=float, default=80.0)
    parser.add_argument("--alignment-sample-max", type=int, default=1000000)
    parser.add_argument("--qc-match-radius-px", type=float, default=2.0)
    parser.add_argument("--qc-sample-max", type=int, default=200000)
    parser.add_argument("--width-px", type=int, default=600)
    parser.add_argument("--height-px", type=int, default=1200)
    parser.add_argument("--camera-pixel-nm-x", type=float, default=101.11)
    parser.add_argument("--camera-pixel-nm-y", type=float, default=98.83)
    parser.add_argument("--render-pixel-nm", type=float, default=20.0)
    parser.add_argument("--spot-radius-nm", type=float, default=28.0)
    parser.add_argument("--radius-mode", choices=("fixed", "xy_uncertainty_mean"), default="fixed")
    parser.add_argument("--render-weight", choices=("count", "probability"), default="count")
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--max-frame", type=int, default=None)
    parser.add_argument("--emitter-z-min", "--emitter-z-min-nm", dest="emitter_z_min_nm", type=float, default=None)
    parser.add_argument("--emitter-z-max", "--emitter-z-max-nm", dest="emitter_z_max_nm", type=float, default=None)
    parser.add_argument("--log-every-frames", type=int, default=500)
    parser.add_argument("--uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--uncertainty-min-sigma-px", type=float, default=0.75)
    parser.add_argument("--uncertainty-max-sigma-px", type=float, default=6.0)
    parser.add_argument("--uncertainty-bin-size-px", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--display-mode", choices=("quantile", "fixed_imax"), default="quantile")
    parser.add_argument("--display-imax", type=float, default=None)
    parser.add_argument("--display-imax-min", type=float, default=NEPTUNE_DEFAULT_IMAX_MIN)
    parser.add_argument("--normalization-fov", type=str, default=None, metavar="X0,Y0,WIDTH,HEIGHT")
    parser.add_argument("--save-ratio-map", action="store_true")
    parser.add_argument("--suffix", default="union_raw_ratio_bicolor_thr040_right_priority")
    return parser.parse_args()


def _sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= int(max_n):
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, int(max_n), dtype=np.int64)


def estimate_left_to_right_translation(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    width_px: int,
    height_px: int,
    bin_px: float,
    max_shift_px: float,
    sample_max: int,
) -> dict[str, float]:
    left_idx = _sample_indices(int(left["x_px"].shape[0]), int(sample_max))
    right_idx = _sample_indices(int(right["x_px"].shape[0]), int(sample_max))
    lx = left["x_px"][left_idx].astype(np.float32, copy=False)
    ly = left["y_px"][left_idx].astype(np.float32, copy=False)
    rx = right["x_px"][right_idx].astype(np.float32, copy=False)
    ry = right["y_px"][right_idx].astype(np.float32, copy=False)
    lg = np.isfinite(lx) & np.isfinite(ly) & (lx >= 0) & (lx < float(width_px)) & (ly >= 0) & (ly < float(height_px))
    rg = np.isfinite(rx) & np.isfinite(ry) & (rx >= 0) & (rx < float(width_px)) & (ry >= 0) & (ry < float(height_px))
    lx, ly = lx[lg], ly[lg]
    rx, ry = rx[rg], ry[rg]
    bw = max(float(bin_px), 1e-6)
    hist_w = int(math.ceil(float(width_px) / bw))
    hist_h = int(math.ceil(float(height_px) / bw))
    left_hist = np.zeros((hist_h, hist_w), dtype=np.float32)
    right_hist = np.zeros((hist_h, hist_w), dtype=np.float32)
    li = np.clip((lx / bw).astype(np.int32), 0, hist_w - 1)
    lj = np.clip((ly / bw).astype(np.int32), 0, hist_h - 1)
    ri = np.clip((rx / bw).astype(np.int32), 0, hist_w - 1)
    rj = np.clip((ry / bw).astype(np.int32), 0, hist_h - 1)
    np.add.at(left_hist, (lj, li), 1.0)
    np.add.at(right_hist, (rj, ri), 1.0)
    left_hist = (left_hist - float(left_hist.mean())) / (float(left_hist.std()) + 1e-6)
    right_hist = (right_hist - float(right_hist.mean())) / (float(right_hist.std()) + 1e-6)
    shape = (hist_h * 2, hist_w * 2)
    corr = np.fft.irfft2(np.fft.rfft2(right_hist, shape) * np.conj(np.fft.rfft2(left_hist, shape)), shape)
    corr = np.fft.fftshift(corr)
    cy, cx = np.array(corr.shape) // 2
    max_bins = max(1, int(round(float(max_shift_px) / bw)))
    win = corr[cy - max_bins : cy + max_bins + 1, cx - max_bins : cx + max_bins + 1]
    yy, xx = np.unravel_index(int(np.argmax(win)), win.shape)
    dx = float((xx - max_bins) * bw)
    dy = float((yy - max_bins) * bw)
    return {
        "left_to_right_dx_px": dx,
        "left_to_right_dy_px": dy,
        "alignment_peak": float(win[yy, xx]),
        "alignment_left_sample_count": int(lx.size),
        "alignment_right_sample_count": int(rx.size),
        "alignment_bin_px": float(bw),
        "alignment_max_shift_px": float(max_shift_px),
    }


def estimate_match_fraction(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    left_to_right_dx_px: float,
    left_to_right_dy_px: float,
    radius_px: float,
    sample_max: int,
) -> dict[str, float]:
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return {}
    left_idx = _sample_indices(int(left["x_px"].shape[0]), int(sample_max))
    right_idx = _sample_indices(int(right["x_px"].shape[0]), int(sample_max))
    lxy = np.column_stack(
        (
            left["x_px"][left_idx].astype(np.float32, copy=False) + float(left_to_right_dx_px),
            left["y_px"][left_idx].astype(np.float32, copy=False) + float(left_to_right_dy_px),
        )
    )
    rxy = np.column_stack(
        (
            right["x_px"][right_idx].astype(np.float32, copy=False),
            right["y_px"][right_idx].astype(np.float32, copy=False),
        )
    )
    lg = np.isfinite(lxy).all(axis=1)
    rg = np.isfinite(rxy).all(axis=1)
    lxy = lxy[lg]
    rxy = rxy[rg]
    if lxy.size == 0 or rxy.size == 0:
        return {"match_fraction_left_to_right": float("nan"), "match_fraction_right_to_left": float("nan")}
    radius = float(radius_px)
    right_tree = cKDTree(rxy)
    left_to_right, _ = right_tree.query(lxy, k=1, distance_upper_bound=radius)
    left_tree = cKDTree(lxy)
    right_to_left, _ = left_tree.query(rxy, k=1, distance_upper_bound=radius)
    return {
        "match_radius_px": radius,
        "match_fraction_left_to_right": float(np.isfinite(left_to_right).mean()),
        "match_fraction_right_to_left": float(np.isfinite(right_to_left).mean()),
        "median_nn_left_to_right_px": float(np.nanmedian(np.where(np.isfinite(left_to_right), left_to_right, np.nan))),
        "median_nn_right_to_left_px": float(np.nanmedian(np.where(np.isfinite(right_to_left), right_to_left, np.nan))),
        "qc_left_sample_count": int(lxy.shape[0]),
        "qc_right_sample_count": int(rxy.shape[0]),
    }


def disk_offsets(radius: int) -> np.ndarray:
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                out.append((dy, dx))
    return np.asarray(out, dtype=np.int16)


def annulus_offsets(inner: int, outer: int) -> np.ndarray:
    out = []
    inner2 = inner * inner
    outer2 = outer * outer
    for dy in range(-outer, outer + 1):
        for dx in range(-outer, outer + 1):
            d2 = dx * dx + dy * dy
            if inner2 <= d2 <= outer2:
                out.append((dy, dx))
    return np.asarray(out, dtype=np.int16)


class FrameReader:
    def __init__(self, path: Path) -> None:
        self.tif = tifffile.TiffFile(path)
        self.series = self.tif.series[0]
        self.n_frames = int(self.series.shape[0])
        self.shape_hw = (int(self.series.shape[1]), int(self.series.shape[2]))
        self.cache: dict[int, np.ndarray] = {}

    def get(self, frame: int) -> np.ndarray:
        frame = int(frame)
        if frame not in self.cache:
            self.cache[frame] = np.asarray(self.series.asarray(key=frame), dtype=np.float32)
            if len(self.cache) > 12:
                for key in sorted(self.cache)[:-12]:
                    self.cache.pop(key, None)
        return self.cache[frame]

    def close(self) -> None:
        self.tif.close()


def read_h5(path: Path) -> dict[str, np.ndarray]:
    columns = ("frame", "x_px", "y_px", "z", "photon", "prob", "x_sig", "y_sig")
    with h5py.File(path, "r") as handle:
        group = handle["locs"]
        missing = [key for key in columns if key not in group]
        if missing:
            raise KeyError(f"{path} missing columns: {missing}")
        return {key: np.asarray(group[key][:]) for key in columns}


def filter_emitters_by_z(
    predictions: dict[str, np.ndarray],
    *,
    z_min_nm: float | None,
    z_max_nm: float | None,
) -> dict[str, np.ndarray]:
    if z_min_nm is None and z_max_nm is None:
        return predictions
    z_nm = predictions["z"].astype(np.float32, copy=False)
    keep = np.isfinite(z_nm)
    if z_min_nm is not None:
        keep &= z_nm > float(z_min_nm)
    if z_max_nm is not None:
        keep &= z_nm < float(z_max_nm)
    return {key: values[keep] for key, values in predictions.items()}


def channel_z_colors(
    z_nm: np.ndarray,
    right_class: np.ndarray,
    *,
    z_min_nm: float,
    z_max_nm: float,
) -> np.ndarray:
    scale = max(float(z_max_nm) - float(z_min_nm), 1e-6)
    depth = np.clip((z_nm.astype(np.float32, copy=False) - float(z_min_nm)) / scale, 0.0, 1.0)[:, None]
    left_low = np.asarray([0.02, 0.05, 0.35], dtype=np.float32)
    left_high = np.asarray([0.10, 1.00, 1.00], dtype=np.float32)
    right_low = np.asarray([0.35, 0.02, 0.08], dtype=np.float32)
    right_high = np.asarray([1.00, 0.82, 0.06], dtype=np.float32)
    left_colors = left_low + depth * (left_high - left_low)
    right_colors = right_low + depth * (right_high - right_low)
    return np.where(right_class[:, None], right_colors, left_colors).astype(np.float32, copy=False)


def frame_index(frame: np.ndarray, max_frame: int | None) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    order = np.argsort(frame.astype(np.int64), kind="stable")
    sorted_frame = frame[order].astype(np.int64, copy=False)
    if max_frame is not None:
        keep_n = int(np.searchsorted(sorted_frame, int(max_frame) + 1, side="left"))
        order = order[:keep_n]
        sorted_frame = sorted_frame[:keep_n]
    unique, starts, counts = np.unique(sorted_frame, return_index=True, return_counts=True)
    return order, {int(f): (int(s), int(s + c)) for f, s, c in zip(unique, starts, counts)}


def union_frame(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    left_ix: np.ndarray,
    right_ix: np.ndarray,
    *,
    max_dist_px: float,
    left_to_right_dx_px: float = 0.0,
    left_to_right_dy_px: float = 0.0,
) -> dict[str, np.ndarray]:
    # Right detections are sorted first so duplicate suppression keeps right coordinates/precision.
    source = np.concatenate(
        [
            np.ones(right_ix.size, dtype=np.int8),
            np.zeros(left_ix.size, dtype=np.int8),
        ]
    )
    ix = np.concatenate([right_ix, left_ix]).astype(np.int64, copy=False)
    x = np.concatenate(
        [
            right["x_px"][right_ix],
            left["x_px"][left_ix] + float(left_to_right_dx_px),
        ]
    ).astype(np.float32, copy=False)
    y = np.concatenate(
        [
            right["y_px"][right_ix],
            left["y_px"][left_ix] + float(left_to_right_dy_px),
        ]
    ).astype(np.float32, copy=False)
    prob = np.concatenate([right["prob"][right_ix], left["prob"][left_ix]]).astype(np.float32, copy=False)
    z = np.concatenate([right["z"][right_ix], left["z"][left_ix]]).astype(np.float32, copy=False)
    photon = np.concatenate([right["photon"][right_ix], left["photon"][left_ix]]).astype(np.float32, copy=False)
    x_sig = np.concatenate([right["x_sig"][right_ix], left["x_sig"][left_ix]]).astype(np.float32, copy=False)
    y_sig = np.concatenate([right["y_sig"][right_ix], left["y_sig"][left_ix]]).astype(np.float32, copy=False)

    priority = np.lexsort((-prob, -source))
    cell = max(float(max_dist_px), 1e-6)
    max_d2 = float(max_dist_px) * float(max_dist_px)
    grid: dict[tuple[int, int], list[int]] = {}
    keep: list[int] = []
    kept_x: list[float] = []
    kept_y: list[float] = []
    for local in priority:
        xx = float(x[local])
        yy = float(y[local])
        cx = int(xx // cell)
        cy = int(yy // cell)
        duplicate = False
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                for prev in grid.get((gx, gy), []):
                    dx = xx - kept_x[prev]
                    dy = yy - kept_y[prev]
                    if dx * dx + dy * dy <= max_d2:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if duplicate:
            continue
        grid.setdefault((cx, cy), []).append(len(keep))
        keep.append(int(local))
        kept_x.append(xx)
        kept_y.append(yy)
    keep_arr = np.asarray(keep, dtype=np.int64)
    sort_out = np.lexsort((x[keep_arr], y[keep_arr]))
    keep_arr = keep_arr[sort_out]
    return {
        "source": source[keep_arr],
        "source_index": ix[keep_arr],
        "matched_left_index": np.full(keep_arr.shape, -1, dtype=np.int64),
        "x_px": x[keep_arr],
        "y_px": y[keep_arr],
        "z": z[keep_arr],
        "photon": photon[keep_arr],
        "prob": prob[keep_arr],
        "x_sig": x_sig[keep_arr],
        "y_sig": y_sig[keep_arr],
    }


def matched_only_frame(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    left_ix: np.ndarray,
    right_ix: np.ndarray,
    *,
    max_dist_px: float,
    left_to_right_dx_px: float = 0.0,
    left_to_right_dy_px: float = 0.0,
) -> dict[str, np.ndarray]:
    if left_ix.size == 0 or right_ix.size == 0:
        empty_f = np.asarray([], dtype=np.float32)
        empty_i = np.asarray([], dtype=np.int64)
        return {
            "source": np.asarray([], dtype=np.int8),
            "source_index": empty_i,
            "matched_left_index": empty_i,
            "x_px": empty_f,
            "y_px": empty_f,
            "z": empty_f,
            "photon": empty_f,
            "prob": empty_f,
            "x_sig": empty_f,
            "y_sig": empty_f,
        }
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("matched_only union requires scipy.spatial.cKDTree") from exc

    left_xy = np.column_stack(
        (
            left["x_px"][left_ix].astype(np.float32, copy=False) + float(left_to_right_dx_px),
            left["y_px"][left_ix].astype(np.float32, copy=False) + float(left_to_right_dy_px),
        )
    )
    right_xy = np.column_stack(
        (
            right["x_px"][right_ix].astype(np.float32, copy=False),
            right["y_px"][right_ix].astype(np.float32, copy=False),
        )
    )
    good_left = np.isfinite(left_xy).all(axis=1)
    good_right = np.isfinite(right_xy).all(axis=1)
    if not np.any(good_left) or not np.any(good_right):
        return matched_only_frame(left, right, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), max_dist_px=max_dist_px)
    left_valid_pos = np.flatnonzero(good_left)
    right_valid_pos = np.flatnonzero(good_right)
    tree = cKDTree(right_xy[good_right])
    dist, nn = tree.query(left_xy[good_left], k=1, distance_upper_bound=float(max_dist_px))
    keep = np.isfinite(dist)
    if not np.any(keep):
        return matched_only_frame(left, right, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), max_dist_px=max_dist_px)
    left_src = left_ix[left_valid_pos[keep]].astype(np.int64, copy=False)
    right_src = right_ix[right_valid_pos[nn[keep]]].astype(np.int64, copy=False)
    # Multiple left detections can choose the same right detection. Keep the highest-probability left partner
    # per right while retaining the right coordinates/precision for rendering.
    left_prob = left["prob"][left_src].astype(np.float32, copy=False)
    order = np.lexsort((-left_prob, right_src))
    right_sorted = right_src[order]
    first = np.r_[True, right_sorted[1:] != right_sorted[:-1]]
    chosen = order[first]
    left_src = left_src[chosen]
    right_src = right_src[chosen]
    sort_out = np.lexsort((right["x_px"][right_src], right["y_px"][right_src]))
    right_src = right_src[sort_out]
    left_src = left_src[sort_out]
    return {
        "source": np.ones(right_src.size, dtype=np.int8),
        "source_index": right_src,
        "matched_left_index": left_src,
        "x_px": right["x_px"][right_src].astype(np.float32, copy=False),
        "y_px": right["y_px"][right_src].astype(np.float32, copy=False),
        "z": right["z"][right_src].astype(np.float32, copy=False),
        "photon": right["photon"][right_src].astype(np.float32, copy=False),
        "prob": right["prob"][right_src].astype(np.float32, copy=False),
        "x_sig": right["x_sig"][right_src].astype(np.float32, copy=False),
        "y_sig": right["y_sig"][right_src].astype(np.float32, copy=False),
    }


def intensity_cuda(
    reader: FrameReader,
    frame: int,
    xy: np.ndarray,
    signal: np.ndarray,
    bg: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    if xy.size == 0:
        return np.empty((0,), dtype=np.float32)
    height, width = reader.shape_hw
    sig_y = torch.as_tensor(signal[:, 0], device=device, dtype=torch.long)
    sig_x = torch.as_tensor(signal[:, 1], device=device, dtype=torch.long)
    bg_y = torch.as_tensor(bg[:, 0], device=device, dtype=torch.long)
    bg_x = torch.as_tensor(bg[:, 1], device=device, dtype=torch.long)
    coords = torch.as_tensor(xy, device=device, dtype=torch.float32)
    xi = torch.round(coords[:, 0]).to(torch.long)
    yi = torch.round(coords[:, 1]).to(torch.long)
    total = torch.zeros((coords.shape[0],), device=device, dtype=torch.float32)
    area = torch.zeros_like(total)
    bg_chunks = []
    for ff in (int(frame) - 1, int(frame), int(frame) + 1):
        if ff < 0 or ff >= reader.n_frames:
            continue
        img = torch.as_tensor(reader.get(ff), device=device)
        flat = img.reshape(-1)
        ys = yi[:, None] + sig_y[None, :]
        xs = xi[:, None] + sig_x[None, :]
        mask = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        idx = torch.clamp(ys, 0, height - 1) * width + torch.clamp(xs, 0, width - 1)
        vals = flat[idx] * mask.to(torch.float32)
        total += vals.sum(dim=1)
        area += mask.sum(dim=1).to(torch.float32)
        yb = yi[:, None] + bg_y[None, :]
        xb = xi[:, None] + bg_x[None, :]
        bmask = (yb >= 0) & (yb < height) & (xb >= 0) & (xb < width)
        bidx = torch.clamp(yb, 0, height - 1) * width + torch.clamp(xb, 0, width - 1)
        bvals = flat[bidx].to(torch.float32)
        bvals = torch.where(bmask, bvals, torch.full_like(bvals, float("nan")))
        bg_chunks.append(bvals)
    bg_vals = torch.cat(bg_chunks, dim=1)
    base = torch.nanmedian(bg_vals, dim=1).values
    out = torch.clamp(total - base * area, min=0.0)
    out = torch.where((area > 0) & torch.isfinite(base), out, torch.full_like(out, float("nan")))
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def write_union_h5(path: Path, rows: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {key: np.concatenate(value) if value else np.asarray([], dtype=np.float32) for key, value in rows.items()}
    int_cols = {"frame", "source", "source_index", "matched_left_index", "color_class"}
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "neptune_v03_union_raw_ratio_v0.1"
        handle.attrs["columns_json"] = json.dumps(list(merged))
        group = handle.create_group("locs")
        for key, value in merged.items():
            dtype = np.dtype("int32") if key in int_cols else np.dtype("float32")
            group.create_dataset(key, data=np.asarray(value).astype(dtype, copy=False), compression="lzf", shuffle=True)
        handle.attrs["count"] = int(next(iter(merged.values())).shape[0]) if merged else 0
    return merged


def add_bilinear(canvas: np.ndarray, x: np.ndarray, y: np.ndarray, colors: np.ndarray) -> None:
    height, width, _ = canvas.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32, copy=False)
    fy = (y - y0).astype(np.float32, copy=False)
    for dx, wx in ((0, 1.0 - fx), (1, fx)):
        ix = x0 + dx
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            iy = y0 + dy
            keep = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            if not np.any(keep):
                continue
            weight = (wx[keep] * wy[keep]).astype(np.float32, copy=False)
            for channel in range(canvas.shape[2]):
                np.add.at(canvas[:, :, channel], (iy[keep], ix[keep]), colors[keep, channel] * weight)


def render(rows: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(1, int(np.ceil(args.width_px * args.camera_pixel_nm_x / args.render_pixel_nm)))
    height = max(1, int(np.ceil(args.height_px * args.camera_pixel_nm_y / args.render_pixel_nm)))
    x = rows["x_px"].astype(np.float32) * float(args.camera_pixel_nm_x) / float(args.render_pixel_nm)
    y = rows["y_px"].astype(np.float32) * float(args.camera_pixel_nm_y) / float(args.render_pixel_nm)
    ratio = rows["ratio_right"].astype(np.float32)
    prob = rows["prob"].astype(np.float32)
    colors = np.zeros((x.size, 3), dtype=np.float32)
    high = ratio >= float(args.ratio_threshold)
    if str(args.color_mode) == "channel_z_gradient":
        if args.emitter_z_min_nm is None or args.emitter_z_max_nm is None:
            raise ValueError("channel_z_gradient requires --emitter-z-min and --emitter-z-max")
        colors = channel_z_colors(
            rows["z"],
            high,
            z_min_nm=float(args.emitter_z_min_nm),
            z_max_nm=float(args.emitter_z_max_nm),
        )
    else:
        colors[high] = np.asarray([1.0, 0.12, 0.08], dtype=np.float32)
        colors[~high] = np.asarray([0.05, 0.55, 1.0], dtype=np.float32)
    weights = np.ones_like(prob) if str(args.render_weight) == "count" else prob
    colors *= weights[:, None]
    ratio_colors = plt.get_cmap("turbo")(np.clip(ratio, 0.0, 1.0))[:, :3].astype(np.float32) * weights[:, None]

    if str(args.radius_mode) == "fixed":
        fixed_sigma = max((float(args.spot_radius_nm) / float(args.render_pixel_nm)) / 2.0, 0.7)
        sigma = np.full(x.size, fixed_sigma, dtype=np.float32)
    else:
        sx = np.abs(rows["x_sig"].astype(np.float32)) * float(args.camera_pixel_nm_x) / float(args.render_pixel_nm)
        sy = np.abs(rows["y_sig"].astype(np.float32)) * float(args.camera_pixel_nm_y) / float(args.render_pixel_nm)
        sigma = np.sqrt((sx * sx + sy * sy) / 2.0).astype(np.float32)
        sigma *= float(args.uncertainty_scale)
        sigma = np.nan_to_num(sigma, nan=float(args.uncertainty_min_sigma_px), posinf=float(args.uncertainty_max_sigma_px))
        sigma = np.clip(sigma, float(args.uncertainty_min_sigma_px), float(args.uncertainty_max_sigma_px))
    bins = np.round(sigma / float(args.uncertainty_bin_size_px)) * float(args.uncertainty_bin_size_px)
    bins = np.clip(bins, float(sigma.min()) if sigma.size else 0.75, float(sigma.max()) if sigma.size else 0.75)

    def draw(color_values: np.ndarray) -> np.ndarray:
        out = np.zeros((height, width, color_values.shape[1]), dtype=np.float32)
        for sig in np.unique(bins):
            idx = np.flatnonzero(np.isclose(bins, sig))
            if idx.size == 0:
                continue
            tmp = np.zeros_like(out)
            for start in range(0, idx.size, int(args.chunk_size)):
                sel = idx[start : start + int(args.chunk_size)]
                add_bilinear(tmp, x[sel], y[sel], color_values[sel])
            for channel in range(out.shape[2]):
                out[:, :, channel] += gaussian_filter(tmp[:, :, channel], sigma=float(sig), mode="constant")
        return out

    return draw(colors), draw(ratio_colors), draw(weights[:, None])[:, :, 0]


def save_rgb(canvas: np.ndarray, density: np.ndarray, path: Path, *, args: argparse.Namespace) -> dict[str, object]:
    normalization_fov = _parse_normalization_fov(
        str(args.normalization_fov) if args.normalization_fov else None,
        width=int(canvas.shape[1]),
        height=int(canvas.shape[0]),
    )
    _, imax, source = _normalize_display(
        density,
        mode=str(args.display_mode),
        fixed_imax=args.display_imax,
        imax_min=float(args.display_imax_min),
        gamma=float(args.gamma),
        brightness=float(args.brightness),
        normalize_roi=normalization_fov,
    )
    image = _colorize_density_display(
        density,
        canvas,
        imax=imax,
        gamma=float(args.gamma),
        brightness=float(args.brightness),
    )
    rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)
    tifffile.imwrite(path.with_suffix(".tiff"), rgb, photometric="rgb")
    return {
        "display_mode": str(args.display_mode),
        "display_imax": float(imax),
        "display_imax_source": source,
        "display_imax_min": float(args.display_imax_min),
        "display_quantile": _smap_quantile_from_imax_min(float(args.display_imax_min)) if str(args.display_mode) == "quantile" else None,
        "normalization_fov_rendered_px": list(normalization_fov) if normalization_fov is not None else None,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("union raw-ratio run requires CUDA for raw TIFF intensity extraction")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    left = read_h5(args.left_predictions)
    right = read_h5(args.right_predictions)
    left_unfiltered_count = int(left["z"].size)
    right_unfiltered_count = int(right["z"].size)
    left = filter_emitters_by_z(left, z_min_nm=args.emitter_z_min_nm, z_max_nm=args.emitter_z_max_nm)
    right = filter_emitters_by_z(right, z_min_nm=args.emitter_z_min_nm, z_max_nm=args.emitter_z_max_nm)
    if args.left_to_right_dx_px is not None or args.left_to_right_dy_px is not None:
        left_to_right_dx_px = float(args.left_to_right_dx_px or 0.0)
        left_to_right_dy_px = float(args.left_to_right_dy_px or 0.0)
        alignment = {
            "alignment_mode": "manual_translation",
            "left_to_right_dx_px": left_to_right_dx_px,
            "left_to_right_dy_px": left_to_right_dy_px,
        }
    elif str(args.alignment_mode) == "auto_translation":
        alignment = estimate_left_to_right_translation(
            left,
            right,
            width_px=int(args.width_px),
            height_px=int(args.height_px),
            bin_px=float(args.alignment_bin_px),
            max_shift_px=float(args.alignment_max_shift_px),
            sample_max=int(args.alignment_sample_max),
        )
        alignment["alignment_mode"] = "auto_translation"
        left_to_right_dx_px = float(alignment["left_to_right_dx_px"])
        left_to_right_dy_px = float(alignment["left_to_right_dy_px"])
    else:
        left_to_right_dx_px = 0.0
        left_to_right_dy_px = 0.0
        alignment = {
            "alignment_mode": "none",
            "left_to_right_dx_px": 0.0,
            "left_to_right_dy_px": 0.0,
        }
    alignment_before = estimate_match_fraction(
        left,
        right,
        left_to_right_dx_px=0.0,
        left_to_right_dy_px=0.0,
        radius_px=float(args.qc_match_radius_px),
        sample_max=int(args.qc_sample_max),
    )
    alignment_after = estimate_match_fraction(
        left,
        right,
        left_to_right_dx_px=left_to_right_dx_px,
        left_to_right_dy_px=left_to_right_dy_px,
        radius_px=float(args.qc_match_radius_px),
        sample_max=int(args.qc_sample_max),
    )
    print(
        json.dumps(
            {
                "alignment": alignment,
                "qc_before": alignment_before,
                "qc_after": alignment_after,
            },
            indent=2,
        ),
        flush=True,
    )
    left_order, left_frames = frame_index(left["frame"], args.max_frame)
    right_order, right_frames = frame_index(right["frame"], args.max_frame)
    reader = FrameReader(args.sample_tiff)
    signal = disk_offsets(int(args.signal_radius))
    bg = annulus_offsets(int(args.bg_inner_radius), int(args.bg_outer_radius))
    keys = (
        "frame",
        "x_px",
        "y_px",
        "z",
        "source",
        "source_index",
        "matched_left_index",
        "photon",
        "prob",
        "x_sig",
        "y_sig",
        "I_left_raw3",
        "I_right_raw3",
        "ratio_right",
        "color_class",
    )
    buffers: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    stats = {
        "left_unfiltered": left_unfiltered_count,
        "right_unfiltered": right_unfiltered_count,
        "emitter_z_min_nm": args.emitter_z_min_nm,
        "emitter_z_max_nm": args.emitter_z_max_nm,
        "left_raw": 0,
        "right_raw": 0,
        "union_total": 0,
        "kept_total": 0,
        "right_source": 0,
        "left_source": 0,
    }
    frames = sorted(set(left_frames).union(right_frames))
    for frame in frames:
        ls, le = left_frames.get(frame, (0, 0))
        rs, re = right_frames.get(frame, (0, 0))
        li = left_order[ls:le]
        ri = right_order[rs:re]
        stats["left_raw"] += int(li.size)
        stats["right_raw"] += int(ri.size)
        if str(args.union_policy) == "matched_only":
            union = matched_only_frame(
                left,
                right,
                li,
                ri,
                max_dist_px=float(args.union_dist_px),
                left_to_right_dx_px=left_to_right_dx_px,
                left_to_right_dy_px=left_to_right_dy_px,
            )
        else:
            union = union_frame(
                left,
                right,
                li,
                ri,
                max_dist_px=float(args.union_dist_px),
                left_to_right_dx_px=left_to_right_dx_px,
                left_to_right_dy_px=left_to_right_dy_px,
            )
        if union["x_px"].size == 0:
            continue
        stats["union_total"] += int(union["x_px"].size)
        for start in range(0, union["x_px"].size, int(args.batch_size)):
            stop = min(union["x_px"].size, start + int(args.batch_size))
            output_xy = np.column_stack((union["x_px"][start:stop], union["y_px"][start:stop])).astype(np.float32, copy=False)
            source = union["source"][start:stop].astype(np.int8, copy=False)
            source_index = union["source_index"][start:stop].astype(np.int64, copy=False)
            matched_left_index = union["matched_left_index"][start:stop].astype(np.int64, copy=False)
            left_xy = np.empty_like(output_xy)
            right_xy = np.empty_like(output_xy)
            right_mask = source == 1
            left_mask = ~right_mask
            if np.any(right_mask):
                ri_src = source_index[right_mask]
                right_xy[right_mask, 0] = right["x_px"][ri_src]
                right_xy[right_mask, 1] = right["y_px"][ri_src]
                matched_left = matched_left_index[right_mask]
                has_matched_left = matched_left >= 0
                if np.any(has_matched_left):
                    left_src = matched_left[has_matched_left]
                    right_local = np.flatnonzero(right_mask)
                    target = right_local[has_matched_left]
                    left_xy[target, 0] = left["x_px"][left_src]
                    left_xy[target, 1] = left["y_px"][left_src]
                if np.any(~has_matched_left):
                    right_local = np.flatnonzero(right_mask)
                    target = right_local[~has_matched_left]
                    left_xy[target, 0] = right_xy[target, 0] - float(left_to_right_dx_px)
                    left_xy[target, 1] = right_xy[target, 1] - float(left_to_right_dy_px)
            if np.any(left_mask):
                li_src = source_index[left_mask]
                left_xy[left_mask, 0] = left["x_px"][li_src]
                left_xy[left_mask, 1] = left["y_px"][li_src]
                right_xy[left_mask, 0] = left_xy[left_mask, 0] + float(left_to_right_dx_px)
                right_xy[left_mask, 1] = left_xy[left_mask, 1] + float(left_to_right_dy_px)
            left_i = intensity_cuda(reader, frame, left_xy, signal, bg, device=str(args.device))
            right_xy[:, 0] += float(args.right_crop_left)
            right_i = intensity_cuda(reader, frame, right_xy, signal, bg, device=str(args.device))
            total_i = left_i + right_i
            keep = np.isfinite(left_i) & np.isfinite(right_i) & (total_i >= float(args.min_total_intensity))
            if not np.any(keep):
                continue
            ratio = right_i[keep] / (total_i[keep] + 1e-12)
            color = (ratio >= float(args.ratio_threshold)).astype(np.int32)
            local_idx = np.flatnonzero(keep)
            idx = local_idx + start
            buffers["frame"].append(np.full(idx.size, int(frame), dtype=np.int32))
            for key in ("x_px", "y_px", "z", "source", "source_index", "matched_left_index", "photon", "prob", "x_sig", "y_sig"):
                buffers[key].append(union[key][idx])
            buffers["I_left_raw3"].append(left_i[keep])
            buffers["I_right_raw3"].append(right_i[keep])
            buffers["ratio_right"].append(ratio.astype(np.float32, copy=False))
            buffers["color_class"].append(color)
            stats["kept_total"] += int(idx.size)
            stats["right_source"] += int((union["source"][idx] == 1).sum())
            stats["left_source"] += int((union["source"][idx] == 0).sum())
        if args.log_every_frames and (int(frame) + 1) % int(args.log_every_frames) == 0:
            print({"frame": int(frame), **stats}, flush=True)
    reader.close()
    out_h5 = args.output_dir / f"{args.suffix}_union_points.h5"
    rows = write_union_h5(out_h5, buffers)
    dual_canvas, ratio_canvas, density = render(rows, args)
    dual_png = args.output_dir / f"{args.suffix}.png"
    ratio_png = args.output_dir / f"{args.suffix}_ratio_map.png"
    dual_display = save_rgb(dual_canvas, density, dual_png, args=args)
    ratio_display = save_rgb(ratio_canvas, density, ratio_png, args=args) if args.save_ratio_map else None
    ratio = rows["ratio_right"].astype(np.float32)
    source = rows["source"].astype(np.int32)
    color = rows["color_class"].astype(np.int32)
    summary = {
        "method": "union_left_right_emitters_then_raw_tiff_ratio",
        "left_predictions": str(args.left_predictions),
        "right_predictions": str(args.right_predictions),
        "sample_tiff": str(args.sample_tiff),
        "output_h5": str(out_h5),
        "dual_png": str(dual_png),
        "dual_tiff": str(dual_png.with_suffix(".tiff")),
        "ratio_png": str(ratio_png) if args.save_ratio_map else None,
        "ratio_tiff": str(ratio_png.with_suffix(".tiff")) if args.save_ratio_map else None,
        "ratio_threshold": float(args.ratio_threshold),
        "color_mode": str(args.color_mode),
        "render_pixel_nm": float(args.render_pixel_nm),
        "spot_radius_nm": float(args.spot_radius_nm),
        "radius_mode": str(args.radius_mode),
        "render_weight": str(args.render_weight),
        "union_dist_px": float(args.union_dist_px),
        "union_policy": str(args.union_policy),
        "min_total_intensity": float(args.min_total_intensity),
        "right_priority_for_duplicates": True,
        "dual_display": dual_display,
        "ratio_display": ratio_display,
        "alignment": alignment,
        "alignment_qc_before": alignment_before,
        "alignment_qc_after": alignment_after,
        "left_unfiltered": int(stats["left_unfiltered"]),
        "right_unfiltered": int(stats["right_unfiltered"]),
        "emitter_z_min_nm": stats["emitter_z_min_nm"],
        "emitter_z_max_nm": stats["emitter_z_max_nm"],
        "left_raw": int(stats["left_raw"]),
        "right_raw": int(stats["right_raw"]),
        "union_total": int(stats["union_total"]),
        "kept_total": int(stats["kept_total"]),
        "right_source": int((source == 1).sum()),
        "left_source": int((source == 0).sum()),
        "right_color_count": int(color.sum()),
        "left_color_count": int(color.shape[0] - color.sum()),
        "ratio_percentiles": {str(q): float(np.percentile(ratio, q)) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)} if ratio.size else {},
    }
    summary_path = args.output_dir / f"{args.suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
