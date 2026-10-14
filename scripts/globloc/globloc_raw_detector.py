from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def detect_channel(
    image: np.ndarray,
    *,
    sigma_signal: float = 1.0,
    sigma_background: float = 3.0,
    threshold_sigma: float = 6.0,
    min_distance: int = 3,
    exclude_border: int = 10,
) -> dict[str, Any]:
    """Detect raw-TIFF candidates without using a neural prediction file.

    The detector is deliberately a small, deterministic raw-image front end:
    difference-of-Gaussians response, robust MAD threshold, and local maxima.
    GlobLoc subsequently refits the corresponding raw ROIs with its official
    multichannel likelihood model.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got {image.shape}")
    if sigma_signal <= 0 or sigma_background <= sigma_signal:
        raise ValueError("expected 0 < sigma_signal < sigma_background")
    if threshold_sigma <= 0 or min_distance < 1 or exclude_border < 0:
        raise ValueError("invalid detector parameters")

    response = gaussian_filter(image, sigma_signal) - gaussian_filter(image, sigma_background)
    center = float(np.median(response))
    mad = float(np.median(np.abs(response - center)))
    threshold = max(
        center + float(threshold_sigma) * 1.4826 * mad,
        float(np.finfo(np.float32).eps),
    )

    neighborhood = 2 * int(min_distance) + 1
    local_maximum = response == maximum_filter(response, size=neighborhood, mode="reflect")
    accepted = local_maximum & (response >= threshold)
    if exclude_border:
        accepted[:exclude_border, :] = False
        accepted[-exclude_border:, :] = False
        accepted[:, :exclude_border] = False
        accepted[:, -exclude_border:] = False

    y_px, x_px = np.nonzero(accepted)
    score = response[y_px, x_px]
    if len(score):
        order = np.lexsort((x_px, y_px, -score))
        x_px = x_px[order]
        y_px = y_px[order]
        score = score[order]

    return {
        "x_px": x_px.astype(np.float32, copy=False),
        "y_px": y_px.astype(np.float32, copy=False),
        "score": score.astype(np.float32, copy=False),
        "threshold": float(threshold),
        "response_median": center,
        "response_mad": mad,
    }
