"""LUNAR-compatible photon-range estimation for experimental TIFF stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage, stats

from ppg.lunar_background import _read_frame_chunk


def estimate_lunar_photon_range(
    source: str | Path,
    *,
    psf_size: int = 21,
    max_frames: int = 1000,
    max_rois: int = 20_000,
    min_range_max: float = 500.0,
    max_range_max: float = 300_000.0,
) -> tuple[float, float, dict[str, float | int | str]]:
    """Estimate LUNAR's training range from sparse, background-subtracted ROIs.

    This mirrors LUNAR's sparse-ROI sum and best-fit (exponential/Gaussian/Gamma)
    selection, while keeping the implementation independent of napari.
    """
    if psf_size < 5 or psf_size % 2 == 0:
        raise ValueError("psf_size must be odd and >= 5")
    with tifffile.TiffFile(Path(source)) as handle:
        shape = tuple(int(v) for v in handle.series[0].shape)
        if len(shape) != 3:
            raise ValueError(f"sample TIFF must be (T,H,W), got {shape}")
        nframes, height, width = shape
        stop = min(nframes, int(max_frames))
        frames = _read_frame_chunk(handle, start=0, stop=stop).astype(np.float32, copy=False)

    radius = psf_size // 2
    # LUNAR uses a +/-50-frame local temporal background.
    sums: list[float] = []
    for index, image in enumerate(frames):
        lo = max(0, index - 50)
        hi = min(stop, index + 51)
        local_bg = frames[lo:hi].mean(axis=0)
        signal = np.clip(image - local_bg, 0.0, None)
        dog = ndimage.gaussian_filter(signal, 1.5) - ndimage.gaussian_filter(signal, 2.0)
        maxima = ndimage.maximum_filter(dog, size=3) == dog
        values = dog[maxima]
        if values.size == 0:
            continue
        threshold = float(np.quantile(values, 1.0 - 1e-4) * 0.3)
        coords = np.argwhere(maxima & (dog > threshold))
        for y, x in coords:
            if y < radius or x < radius or y >= height - radius or x >= width - radius:
                continue
            patch = signal[y - radius:y + radius + 1, x - radius:x + radius + 1]
            sums.append(float(patch.sum()))
            if len(sums) >= max_rois:
                break
        if len(sums) >= max_rois:
            break
    values = np.asarray(sums, dtype=np.float64)
    if values.size < 32:
        raise ValueError(f"only {values.size} sparse ROIs found; cannot estimate photon range")

    limit = (float(min_range_max), float(max_range_max))
    fits = [
        ("exponential", stats.expon.fit(values), lambda p: p[0] + 2.0 * p[1], stats.expon),
        ("gaussian", stats.norm.fit(values), lambda p: p[0] + 2.0 * p[1], stats.norm),
        ("gamma", stats.gamma.fit(values), lambda p: stats.gamma.ppf(0.95, *p), stats.gamma),
    ]
    ks = [float(stats.kstest(values, dist.cdf, args=params).statistic) for _, params, _, dist in fits]
    best = int(np.argmin(ks))
    name, params, estimate, _ = fits[best]
    range_max = float(np.clip(estimate(params), *limit))
    range_min = max(range_max / 20.0, 100.0)
    metadata: dict[str, float | int | str] = {
        "method": "lunar_sparse_roi_best_ks",
        "fit": name,
        "roi_count": int(values.size),
        "sampled_frames": int(stop),
        "signal_sum_median": float(np.median(values)),
        "signal_sum_p95": float(np.percentile(values, 95.0)),
        "range_min": float(range_min),
        "range_max": float(range_max),
        "ks_statistic": float(ks[best]),
    }
    return (float(range_min), float(range_max)), metadata
