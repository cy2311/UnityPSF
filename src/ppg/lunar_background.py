from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True)
class LunarBackgroundEstimate:
    background_range: tuple[float, float]
    frame_range: tuple[int, int]
    percentile: float
    camera_backward: str = "identity"

    @property
    def background_mean(self) -> float:
        return (self.background_range[0] + self.background_range[1]) / 2.0


def estimate_lunar_background_range(
    images: np.ndarray,
    *,
    percentile: float = 10.0,
) -> tuple[float, float]:
    frames = np.asarray(images)
    if frames.ndim != 3:
        raise ValueError(f"LUNAR background input must be (T,H,W), got {frames.shape}")
    temporal_mean = frames.mean(axis=0)
    dark_mask = temporal_mean < np.percentile(temporal_mean, float(percentile))
    values = frames[:, dark_mask].reshape(-1)
    mean = float(values.mean())
    sigma = float(values.std(ddof=0))
    return max(mean - 2.0 * sigma, 0.0), max(mean + 2.0 * sigma, 0.0)


def _read_frame_chunk(
    handle: tifffile.TiffFile,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    if len(handle.pages) == 1:
        return np.asarray(handle.series[0].asarray()[start:stop])
    return np.asarray(handle.asarray(key=range(start, stop), series=0))


def estimate_lunar_tiff_background(
    source: str | Path,
    *,
    percentile: float = 10.0,
    max_size_gb: float = 2.0,
    chunk_frames: int = 64,
) -> LunarBackgroundEstimate:
    path = Path(source)
    with tifffile.TiffFile(path) as handle:
        shape = tuple(int(value) for value in handle.series[0].shape)
        dtype = np.dtype(handle.series[0].dtype)
        if len(shape) != 3:
            raise ValueError(f"sample TIFF must be (T,H,W), got {shape}")
        frame_count, height, width = shape
        bytes_per_frame = height * width * dtype.itemsize
        selected_frames = min(
            frame_count,
            max(1, int(float(max_size_gb) * 1024**3 / bytes_per_frame)),
        )
        temporal_sum = np.zeros((height, width), dtype=np.float64)
        for start in range(0, selected_frames, int(chunk_frames)):
            stop = min(selected_frames, start + int(chunk_frames))
            chunk = _read_frame_chunk(handle, start=start, stop=stop)
            temporal_sum += chunk.sum(axis=0, dtype=np.float64)
        temporal_mean = temporal_sum / selected_frames
        dark_mask = temporal_mean < np.percentile(temporal_mean, float(percentile))

        value_count = 0
        value_sum = 0.0
        value_square_sum = 0.0
        for start in range(0, selected_frames, int(chunk_frames)):
            stop = min(selected_frames, start + int(chunk_frames))
            values = _read_frame_chunk(handle, start=start, stop=stop)[:, dark_mask]
            values = values.astype(np.float64, copy=False)
            value_count += int(values.size)
            value_sum += float(values.sum(dtype=np.float64))
            value_square_sum += float(np.square(values).sum(dtype=np.float64))

    mean = value_sum / value_count
    variance = max(value_square_sum / value_count - mean * mean, 0.0)
    sigma = float(np.sqrt(variance))
    background_range = (
        max(mean - 2.0 * sigma, 0.0),
        max(mean + 2.0 * sigma, 0.0),
    )
    return LunarBackgroundEstimate(
        background_range=background_range,
        frame_range=(0, selected_frames),
        percentile=float(percentile),
    )
