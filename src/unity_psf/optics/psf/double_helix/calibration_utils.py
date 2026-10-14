from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def detect_stable_dh_centers(
    stack: np.ndarray,
    *,
    roi_size: int,
    maximum_count: int,
    minimum_distance_px: int,
) -> np.ndarray:
    values = np.asarray(stack)
    if values.ndim != 3 or roi_size <= 0 or roi_size % 2 == 0:
        raise ValueError("stack must be (z,y,x) and roi_size must be positive and odd")
    signal = np.maximum(values.astype(np.float32) - np.median(values, axis=(1, 2))[:, None, None], 0.0)
    aggregate = gaussian_filter(signal.mean(axis=0), sigma=2.0)
    peaks = maximum_filter(aggregate, size=int(minimum_distance_px), mode="nearest")
    y, x = np.nonzero((aggregate == peaks) & (aggregate >= np.quantile(aggregate, 0.85)))
    half = roi_size // 2
    keep = (y >= half) & (y < values.shape[1] - half) & (x >= half) & (x < values.shape[2] - half)
    candidates = np.stack((y[keep], x[keep]), axis=1)
    order = np.argsort(aggregate[candidates[:, 0], candidates[:, 1]])[::-1]
    selected: list[np.ndarray] = []
    for index in order:
        candidate = candidates[index]
        if all(np.linalg.norm(candidate - previous) >= minimum_distance_px for previous in selected):
            selected.append(candidate)
        if len(selected) == maximum_count:
            break
    if len(selected) != maximum_count:
        raise ValueError(f"Detected only {len(selected)} complete centers; requested {maximum_count}.")
    return np.asarray(selected, dtype=np.int64)


def extract_centered_roi_stacks(stack: np.ndarray, centers_yx: np.ndarray, *, roi_size: int) -> np.ndarray:
    values = np.asarray(stack)
    centers = np.asarray(centers_yx)
    if values.ndim != 3 or centers.ndim != 2 or centers.shape[1] != 2 or roi_size <= 0 or roi_size % 2 == 0:
        raise ValueError("stack must be (z,y,x), centers_yx must be (bead,2), and roi_size must be odd")
    half = roi_size // 2
    rois = []
    for center_y, center_x in np.rint(centers).astype(np.int64):
        y0, y1 = center_y - half, center_y + half + 1
        x0, x1 = center_x - half, center_x + half + 1
        if y0 < 0 or x0 < 0 or y1 > values.shape[1] or x1 > values.shape[2]:
            raise ValueError("Every center must define a complete ROI")
        rois.append(values[:, y0:y1, x0:x1])
    return np.stack(rois)


def fit_affine_residual_gamma_maps(
    centers_yx: np.ndarray,
    gamma_nm: np.ndarray,
    *,
    field_shape_yx: tuple[int, int],
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers_yx, dtype=np.float64)
    gamma = np.asarray(gamma_nm, dtype=np.float64)
    height, width = map(int, field_shape_yx)
    if centers.ndim != 2 or centers.shape[1] != 2 or gamma.ndim != 2 or gamma.shape[0] != centers.shape[0] or height <= 1 or width <= 1 or ridge < 0:
        raise ValueError("invalid centers, gamma, field shape, or ridge")
    y = -1.0 + 2.0 * centers[:, 0] / (height - 1)
    x = -1.0 + 2.0 * centers[:, 1] / (width - 1)
    design = np.stack((np.ones_like(x), x, y), axis=1)
    penalty = np.diag((0.0, float(ridge), float(ridge)))
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ gamma).T
    fy = np.linspace(-1.0, 1.0, height)
    fx = np.linspace(-1.0, 1.0, width)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    maps = np.einsum("ct,tyx->cyx", coefficients, np.stack((np.ones_like(xx), xx, yy), axis=0))
    return maps.astype(np.float32), coefficients.astype(np.float32)
