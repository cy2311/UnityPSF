from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .contract import PeakBootstrapConfig


@dataclass(frozen=True)
class PeakCandidate:
    frame_index: int
    x_px: float
    y_px: float
    score: float


@dataclass(frozen=True)
class PeakHarvestResult:
    config: PeakBootstrapConfig
    candidate_count: int
    kept_count: int
    harvest_path: Path
    summary_path: Path

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_json_dict(),
            "candidate_count": int(self.candidate_count),
            "kept_count": int(self.kept_count),
            "harvest_pt": str(self.harvest_path),
            "summary_path": str(self.summary_path),
        }


def run_peak_harvest(*, config: PeakBootstrapConfig, output_dir: Path) -> dict[str, Any]:
    if config.tiff_path is None:
        raise ValueError("Peak harvest requires config.tiff_path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    start, stop = int(config.frame_range[0]), int(config.frame_range[1])
    frames = _read_frame_range(Path(config.tiff_path), start, stop)
    full_h, full_w = int(frames.shape[1]), int(frames.shape[2])
    crop_x0 = max(int(config.crop_x0), 0)
    crop_y0 = max(int(config.crop_y0), 0)
    crop_x1 = full_w if config.crop_x1 is None else min(int(config.crop_x1), full_w)
    crop_y1 = full_h if config.crop_y1 is None else min(int(config.crop_y1), full_h)
    half = int(config.patch_size_px) // 2

    candidates: list[PeakCandidate] = []
    frame_lookup: dict[int, np.ndarray] = {}
    for local_index, raw_frame in enumerate(frames):
        frame_index = start + int(local_index)
        frame = np.asarray(raw_frame, dtype=np.float32)
        frame_lookup[frame_index] = frame
        crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
        for candidate in _detect_peaks_in_frame(crop, frame_index=frame_index, x0=crop_x0, y0=crop_y0, config=config):
            cx = int(round(candidate.x_px - 0.5))
            cy = int(round(candidate.y_px - 0.5))
            if cx - half >= 0 and cy - half >= 0 and cx + half < full_w and cy + half < full_h:
                candidates.append(candidate)

    kept = greedy_distance_filter(candidates, min_distance_px=float(config.min_distance_px))
    max_candidates = int(config.max_candidates if config.max_candidates is not None else config.max_emitters)
    kept = kept[:max_candidates]
    max_score = max((candidate.score for candidate in kept), default=1.0)
    emitters = _build_harvest_payload(
        [(candidate, frame_lookup[int(candidate.frame_index)]) for candidate in kept],
        patch_size_px=int(config.patch_size_px),
        max_score=float(max_score),
    )

    harvest_path = output_dir / "harvest.pt"
    torch.save(
        {
            "candidate_count": int(len(candidates)),
            "accepted_count": int(emitters["frame_index"].shape[0]),
            "metrics": {
                "method": "numpy_gaussian_peak",
                "frame_range": [start, stop],
                "crop_xy": [int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)],
                "candidate_count": int(len(candidates)),
                "kept_after_distance": int(len(kept)),
                "accepted_count": int(emitters["frame_index"].shape[0]),
            },
            "payload": emitters,
        },
        harvest_path,
    )
    summary_path = output_dir / "peak_harvest_summary.json"
    result = PeakHarvestResult(
        config=config,
        candidate_count=int(len(candidates)),
        kept_count=int(emitters["frame_index"].shape[0]),
        harvest_path=harvest_path,
        summary_path=summary_path,
    )
    summary_path.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "candidate_count": result.candidate_count,
        "kept_count": result.kept_count,
        "harvest_path": result.harvest_path,
        "summary_path": result.summary_path,
    }


def greedy_distance_filter(candidates: Iterable[PeakCandidate], *, min_distance_px: float) -> list[PeakCandidate]:
    ordered = sorted(candidates, key=lambda item: float(item.score), reverse=True)
    min_dist_sq = float(min_distance_px) ** 2
    kept: list[PeakCandidate] = []
    kept_by_frame: dict[int, list[PeakCandidate]] = {}
    for peak in ordered:
        frame_peaks = kept_by_frame.setdefault(int(peak.frame_index), [])
        if all((peak.x_px - prev.x_px) ** 2 + (peak.y_px - prev.y_px) ** 2 >= min_dist_sq for prev in frame_peaks):
            frame_peaks.append(peak)
            kept.append(peak)
    return kept


def _read_frame_range(path: Path, start: int, stop: int) -> np.ndarray:
    import tifffile

    if int(stop) <= int(start):
        raise ValueError("frame_range stop must be greater than start")
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        shape = tuple(int(v) for v in series.shape)
        if len(shape) == 2:
            if int(start) > 0 or int(stop) > 1:
                raise ValueError(f"frame_range [{start}, {stop}] exceeds single-frame TIFF")
            frames = np.asarray(series.asarray(), dtype=np.float32)[None, ...]
        elif len(shape) == 3:
            if int(stop) > int(shape[0]):
                raise ValueError(f"frame_range stop {stop} exceeds TIFF frame count {shape[0]}")
            frames = np.asarray(series.asarray(key=range(int(start), int(stop))), dtype=np.float32)
        else:
            raise ValueError(f"Expected TIFF stack of shape (T,H,W), got {shape}")
    while frames.ndim > 3 and 1 in frames.shape:
        axes = tuple(index for index, size in enumerate(frames.shape) if size == 1)
        frames = np.squeeze(frames, axis=axes[:1])
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim != 3:
        raise ValueError(f"Expected TIFF stack of shape (T,H,W), got {frames.shape}")
    return np.ascontiguousarray(frames[int(start) : int(stop)], dtype=np.float32)


def _detect_peaks_in_frame(
    frame: np.ndarray,
    *,
    frame_index: int,
    x0: int,
    y0: int,
    config: PeakBootstrapConfig,
) -> list[PeakCandidate]:
    image = np.asarray(frame, dtype=np.float32)
    smoothed = _smooth3(image, sigma=float(config.gaussian_sigma_px))
    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median)))
    robust_sigma = max(1.4826 * mad, 1e-6)
    threshold = median + float(config.threshold_sigma) * robust_sigma
    local_max = _local_max_3x3(smoothed)
    yy, xx = np.nonzero(local_max & (smoothed > threshold))
    return [
        PeakCandidate(
            frame_index=int(frame_index),
            x_px=float(x + x0) + 0.5,
            y_px=float(y + y0) + 0.5,
            score=float(smoothed[y, x] - threshold),
        )
        for y, x in zip(yy.tolist(), xx.tolist())
    ]


def _smooth3(image: np.ndarray, *, sigma: float) -> np.ndarray:
    if float(sigma) <= 0.0:
        return np.asarray(image, dtype=np.float32)
    padded = np.pad(np.asarray(image, dtype=np.float32), 1, mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    kernel = np.asarray([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    for dy in range(3):
        for dx in range(3):
            out += kernel[dy, dx] * padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return out


def _local_max_3x3(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    result = np.ones_like(image, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            result &= image >= padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return result


def _build_harvest_payload(
    candidates_with_frames: Iterable[tuple[PeakCandidate, np.ndarray]],
    *,
    patch_size_px: int,
    max_score: float,
) -> dict[str, torch.Tensor]:
    frame_index: list[int] = []
    probability: list[float] = []
    x_px: list[float] = []
    y_px: list[float] = []
    photons: list[float] = []
    background: list[float] = []
    local_x_nm: list[float] = []
    local_y_nm: list[float] = []
    half = int(patch_size_px) // 2
    center = (float(patch_size_px) - 1.0) / 2.0
    score_norm = max(float(max_score), 1e-6)

    for candidate, frame in candidates_with_frames:
        cx = int(round(float(candidate.x_px) - 0.5))
        cy = int(round(float(candidate.y_px) - 0.5))
        patch = np.asarray(frame[cy - half : cy + half + 1, cx - half : cx + half + 1], dtype=np.float32)
        if patch.shape != (int(patch_size_px), int(patch_size_px)):
            continue
        border = np.concatenate([patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]])
        bg = float(np.median(border))
        signal = np.clip(patch - bg, 0.0, None)
        total_signal = float(signal.sum())
        if total_signal <= 0.0:
            continue
        yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]].astype(np.float32)
        dx_px = float((signal * (xx - center)).sum() / max(total_signal, 1e-6))
        dy_px = float((signal * (yy - center)).sum() / max(total_signal, 1e-6))
        frame_index.append(int(candidate.frame_index))
        probability.append(float(np.clip(candidate.score / score_norm, 0.0, 1.0)))
        x_px.append(float(cx) + 0.5 + dx_px)
        y_px.append(float(cy) + 0.5 + dy_px)
        photons.append(total_signal)
        background.append(bg)
        local_x_nm.append(dx_px * 100.0)
        local_y_nm.append(dy_px * 100.0)

    count = len(frame_index)
    ones = torch.ones((count,), dtype=torch.float32)
    return {
        "frame_index": torch.as_tensor(frame_index, dtype=torch.int64),
        "probability": torch.as_tensor(probability, dtype=torch.float32),
        "x_px": torch.as_tensor(x_px, dtype=torch.float32),
        "y_px": torch.as_tensor(y_px, dtype=torch.float32),
        "z_um": torch.zeros((count,), dtype=torch.float32),
        "photons": torch.as_tensor(photons, dtype=torch.float32),
        "x_sigma_px": ones.clone(),
        "y_sigma_px": ones.clone(),
        "z_sigma_um": ones.clone(),
        "photon_sigma": ones.clone(),
        "background_adu": torch.as_tensor(background, dtype=torch.float32),
        "local_x_nm": torch.as_tensor(local_x_nm, dtype=torch.float32),
        "local_y_nm": torch.as_tensor(local_y_nm, dtype=torch.float32),
    }


__all__ = [
    "PeakCandidate",
    "PeakHarvestResult",
    "greedy_distance_filter",
    "run_peak_harvest",
]
