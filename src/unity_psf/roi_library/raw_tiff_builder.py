from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unity_psf.field_origin import build_sliding_window_origin_bank

from .geometry import build_emitter_centered_candidates, build_sliding_window_guided_candidates, select_fov_balanced_candidates
from .hdf5 import save_roi_bank
from .types import EmitterPosterior, ROIBank, ROICandidate, ROIRecord


@dataclass(frozen=True)
class ROIBankDomain:
    name: str
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int


@dataclass(frozen=True)
class InferredEmitter:
    probability: float
    mu_xy_px: tuple[float, float]
    sigma_xy_px: tuple[float, float]
    mu_z_nm: float
    sigma_z_nm: float
    mu_photons: float
    sigma_photons: float
    cell_xy_px: tuple[float, float] | None = None
    frame_index: int | None = None


@dataclass(frozen=True)
class RawInferenceResult:
    emitters: tuple[InferredEmitter, ...]
    background_mu: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ROIBankBuildConfig:
    roi_size_px: int = 256
    window_size: int = 3
    frame_range: tuple[int, int] | None = None
    grid_shape: tuple[int, int] = (4, 4)
    max_rois: int = 20
    target_emitters: int = 5000
    candidate_probability_threshold: float = 0.3
    probability_threshold: float = 0.5
    max_overlap_fraction: float = 0.95
    seed: int = 0
    background_smoothing_kernel: int = 9
    camera_backward: dict[str, Any] | None = None
    over_cut_px: int = 0
    origin_mode: str = "emitter_centered"
    origin_stride_px: int | None = None
    valid_core_size_px: int | None = None


InferFn = Callable[..., RawInferenceResult]


def build_roi_bank_from_inference(
    *,
    raw_frames_photon: np.ndarray | str | Path,
    domains: Sequence[ROIBankDomain],
    infer_fn: InferFn,
    config: ROIBankBuildConfig | None = None,
    output_h5_path: str | Path | None = None,
) -> ROIBank:
    cfg = config or ROIBankBuildConfig()
    use_lazy_domain_tiff = isinstance(raw_frames_photon, (str, Path)) and cfg.frame_range is not None
    raw_frames = None if use_lazy_domain_tiff else _load_raw_frames(raw_frames_photon)
    frames = None if raw_frames is None else raw_frames
    window_config = cfg
    if use_lazy_domain_tiff:
        frame_count = int(cfg.frame_range[1]) - int(cfg.frame_range[0])
        window_config = ROIBankBuildConfig(
            roi_size_px=cfg.roi_size_px,
            window_size=cfg.window_size,
            frame_range=None,
            grid_shape=cfg.grid_shape,
            max_rois=cfg.max_rois,
            target_emitters=cfg.target_emitters,
            candidate_probability_threshold=cfg.candidate_probability_threshold,
            probability_threshold=cfg.probability_threshold,
            max_overlap_fraction=cfg.max_overlap_fraction,
            seed=cfg.seed,
            background_smoothing_kernel=cfg.background_smoothing_kernel,
            camera_backward=None,
            over_cut_px=cfg.over_cut_px,
            origin_mode=cfg.origin_mode,
            origin_stride_px=cfg.origin_stride_px,
            valid_core_size_px=cfg.valid_core_size_px,
        )
    else:
        frame_count = int(frames.shape[0])
    windows = _iter_frame_windows(frame_count, config=window_config)
    records: list[ROIRecord] = []
    empty_cells: set[int] = set()
    camera_backward_by_domain: dict[str, dict[str, Any]] = {}

    for domain in domains:
        if use_lazy_domain_tiff:
            domain_frames_adu = _load_raw_domain_frames(raw_frames_photon, domain=domain, frame_range=cfg.frame_range)
        else:
            domain_frames_adu = _crop_domain(frames, domain)
        domain_camera_backward = _camera_backward_params_for_domain(cfg.camera_backward, domain=domain, frames_adu=domain_frames_adu)
        if domain_camera_backward is not None:
            camera_backward_by_domain[str(domain.name)] = domain_camera_backward
        domain_frames = _camera_backward_photons(domain_frames_adu, domain_camera_backward)
        domain_candidates = []
        candidate_context: dict[int, tuple[ROIBankDomain, tuple[int, int], RawInferenceResult]] = {}
        next_candidate_id = 0
        for window in windows:
            result = infer_fn(
                domain=domain,
                frame_window=window,
                raw_domain_frames_photon=domain_frames[window[0] : window[1]],
            )
            emitters = tuple(
                e for e in result.emitters if float(e.probability) >= float(cfg.candidate_probability_threshold)
            )
            if not emitters:
                continue
            xy = np.asarray([e.mu_xy_px for e in emitters], dtype=np.float32)
            probs = np.asarray([e.probability for e in emitters], dtype=np.float32)
            candidates = _build_candidates(
                xy_px=xy,
                probabilities=probs,
                domain=domain,
                config=cfg,
            )
            for candidate in candidates:
                reindexed = _replace_candidate_id(candidate, candidate_id=next_candidate_id)
                next_candidate_id += 1
                domain_candidates.append(reindexed)
                candidate_context[reindexed.candidate_id] = (
                    domain,
                    window,
                    RawInferenceResult(emitters=emitters, background_mu=result.background_mu),
                )

        selection = select_fov_balanced_candidates(
            domain_candidates,
            max_rois=cfg.max_rois,
            target_emitters=cfg.target_emitters,
            grid_cell_count=int(cfg.grid_shape[0]) * int(cfg.grid_shape[1]),
            roi_size_px=cfg.roi_size_px,
            max_overlap_fraction=cfg.max_overlap_fraction,
        )
        empty_cells.update(range(int(cfg.grid_shape[0]) * int(cfg.grid_shape[1])))
        empty_cells.difference_update(int(candidate.grid_cell_id) for candidate in domain_candidates)

        for candidate in selection.candidates:
            domain_for_record, window, result = candidate_context[candidate.candidate_id]
            roi_id = len(records)
            records.append(
                _build_record(
                    roi_id=roi_id,
                    domain=domain_for_record,
                    frame_window=window,
                    candidate=candidate,
                    result=result,
                    domain_frames=domain_frames,
                    config=cfg,
                )
            )

    bank = ROIBank(
        records=tuple(records),
        config={
            "roi_size_px": int(cfg.roi_size_px),
            "window_size": int(cfg.window_size),
            "frame_range": cfg.frame_range,
            "grid_shape": cfg.grid_shape,
            "max_rois": int(cfg.max_rois),
            "target_emitters": int(cfg.target_emitters),
            "candidate_probability_threshold": float(cfg.candidate_probability_threshold),
            "probability_threshold": float(cfg.probability_threshold),
            "origin_mode": str(cfg.origin_mode),
            "origin_stride_px": None if cfg.origin_stride_px is None else int(cfg.origin_stride_px),
            "valid_core_size_px": None if cfg.valid_core_size_px is None else int(cfg.valid_core_size_px),
            "origin_bank_size": _origin_bank_size(cfg, domains=domains),
        },
        metadata={
            "seed": int(cfg.seed),
            **({"camera_backward": _camera_backward_metadata(cfg.camera_backward, camera_backward_by_domain)} if cfg.camera_backward else {}),
        },
        empty_grid_cell_ids=tuple(sorted(empty_cells)),
    )
    if output_h5_path is not None:
        save_roi_bank(bank, output_h5_path)
    return bank


def _replace_candidate_id(candidate: ROICandidate, *, candidate_id: int) -> ROICandidate:
    return ROICandidate(
        candidate_id=int(candidate_id),
        origin_xy_px=candidate.origin_xy_px,
        center_xy_px=candidate.center_xy_px,
        grid_cell_id=int(candidate.grid_cell_id),
        emitter_indices=candidate.emitter_indices,
        emitter_count=int(candidate.emitter_count),
        quality_score=float(candidate.quality_score),
        origin_bank_index=candidate.origin_bank_index,
        valid_core_origin_xy_px=candidate.valid_core_origin_xy_px,
        valid_core_offset_xy_px=candidate.valid_core_offset_xy_px,
        valid_core_size_px=candidate.valid_core_size_px,
    )


def _build_candidates(
    *,
    xy_px: np.ndarray,
    probabilities: np.ndarray,
    domain: ROIBankDomain,
    config: ROIBankBuildConfig,
) -> tuple[ROICandidate, ...]:
    mode = str(config.origin_mode or "emitter_centered").strip().lower()
    if mode in {"emitter_centered", "legacy", "clamp"}:
        return build_emitter_centered_candidates(
            xy_px=xy_px,
            probabilities=probabilities,
            domain_width_px=domain.crop_width,
            domain_height_px=domain.crop_height,
            roi_size_px=config.roi_size_px,
            grid_shape=config.grid_shape,
        )
    if mode in {"sliding_window_guided", "sliding_window", "field_origin_bank"}:
        stride = int(config.origin_stride_px or config.roi_size_px)
        return build_sliding_window_guided_candidates(
            xy_px=xy_px,
            probabilities=probabilities,
            domain_width_px=domain.crop_width,
            domain_height_px=domain.crop_height,
            roi_size_px=config.roi_size_px,
            stride_px=stride,
            grid_shape=config.grid_shape,
            valid_core_size_px=config.valid_core_size_px,
        )
    raise ValueError("origin_mode must be 'emitter_centered' or 'sliding_window_guided'")


def _origin_bank_size(config: ROIBankBuildConfig, *, domains: Sequence[ROIBankDomain]) -> int | None:
    mode = str(config.origin_mode or "emitter_centered").strip().lower()
    if mode not in {"sliding_window_guided", "sliding_window", "field_origin_bank"}:
        return None
    if not domains:
        return 0
    domain = domains[0]
    origins = build_sliding_window_origin_bank(
        field_width_px=int(domain.crop_width),
        field_height_px=int(domain.crop_height),
        roi_width_px=int(config.valid_core_size_px or config.roi_size_px),
        roi_height_px=int(config.valid_core_size_px or config.roi_size_px),
        stride_px=int(config.origin_stride_px or config.roi_size_px),
    )
    return int(len(origins))


def _build_record(
    *,
    roi_id: int,
    domain: ROIBankDomain,
    frame_window: tuple[int, int],
    candidate: Any,
    result: RawInferenceResult,
    domain_frames: np.ndarray,
    config: ROIBankBuildConfig,
) -> ROIRecord:
    x0, y0 = int(candidate.origin_xy_px[0]), int(candidate.origin_xy_px[1])
    roi = int(config.roi_size_px)
    raw = np.asarray(domain_frames[frame_window[0] : frame_window[1], y0 : y0 + roi, x0 : x0 + roi], dtype=np.float32)
    background = np.asarray(result.background_mu, dtype=np.float32)[y0 : y0 + roi, x0 : x0 + roi]
    emitters = tuple(
        _to_posterior(
            emitter,
            domain=domain,
            origin_xy_px=(x0, y0),
            frame_index=frame_window[0],
        )
        for emitter in result.emitters
        if _inside_roi_inner(emitter, origin_xy_px=(x0, y0), roi_size_px=roi, over_cut_px=config.over_cut_px)
        and float(emitter.probability) >= float(config.probability_threshold)
    )
    full_origin_xy = (float(domain.crop_left + x0), float(domain.crop_top + y0))
    valid_core_origin_xy = None
    valid_core_offset_xy = None
    if candidate.valid_core_origin_xy_px is not None:
        valid_core_origin_xy = (
            float(domain.crop_left + int(candidate.valid_core_origin_xy_px[0])),
            float(domain.crop_top + int(candidate.valid_core_origin_xy_px[1])),
        )
        valid_core_offset_xy = (
            float(int(candidate.valid_core_origin_xy_px[0]) - x0),
            float(int(candidate.valid_core_origin_xy_px[1]) - y0),
        )
    return ROIRecord(
        roi_id=int(roi_id),
        domain_name=domain.name,
        frame_window=frame_window,
        roi_origin_xy_px=full_origin_xy,
        raw_frames_photon=raw,
        background_mu=background,
        background_smoothed=_smooth_background(background, kernel_size=config.background_smoothing_kernel),
        grid_cell_id=int(candidate.grid_cell_id),
        emitters=emitters,
        summary={
            "candidate_id": int(candidate.candidate_id),
            "emitter_count": len(emitters),
            "domain_local_roi_origin_xy_px": (float(x0), float(y0)),
            "full_fov_roi_origin_xy_px": full_origin_xy,
            **({} if candidate.origin_bank_index is None else {"roi_origin_index": int(candidate.origin_bank_index)}),
            **({} if config.origin_stride_px is None else {"roi_origin_stride_px": int(config.origin_stride_px)}),
            **({} if candidate.valid_core_size_px is None else {"valid_core_size_px": int(candidate.valid_core_size_px)}),
            **({} if valid_core_origin_xy is None else {"valid_core_origin_xy_px": valid_core_origin_xy}),
            **({} if valid_core_offset_xy is None else {"valid_core_offset_xy_px": valid_core_offset_xy}),
            **({"context_roi_origin_xy_px": full_origin_xy} if candidate.valid_core_size_px is not None else {}),
            **({"context_roi_size_px": int(config.roi_size_px)} if candidate.valid_core_size_px is not None else {}),
        },
    )


def _to_posterior(
    emitter: InferredEmitter,
    *,
    domain: ROIBankDomain,
    origin_xy_px: tuple[int, int],
    frame_index: int,
) -> EmitterPosterior:
    x, y = float(emitter.mu_xy_px[0]), float(emitter.mu_xy_px[1])
    x0, y0 = origin_xy_px
    local_xy = (float(x - x0), float(y - y0))
    return EmitterPosterior(
        probability=float(emitter.probability),
        cell_xy_px=tuple(float(v) for v in (emitter.cell_xy_px or emitter.mu_xy_px)),
        mu_xy_px=(x, y),
        sigma_xy_px=(float(emitter.sigma_xy_px[0]), float(emitter.sigma_xy_px[1])),
        mu_z_nm=float(emitter.mu_z_nm),
        sigma_z_nm=float(emitter.sigma_z_nm),
        mu_photons=float(emitter.mu_photons),
        sigma_photons=float(emitter.sigma_photons),
        local_xy_px=local_xy,
        full_xy_px=(float(domain.crop_left + x), float(domain.crop_top + y)),
        frame_index=int(frame_index if emitter.frame_index is None else emitter.frame_index),
    )


def _load_raw_frames(
    raw_frames_photon: np.ndarray | str | Path,
    *,
    frame_range: tuple[int, int] | None = None,
) -> np.ndarray:
    if isinstance(raw_frames_photon, (str, Path)):
        import tifffile

        if frame_range is None:
            frames = np.asarray(tifffile.imread(str(raw_frames_photon)), dtype=np.float32)
        else:
            start, stop = int(frame_range[0]), int(frame_range[1])
            if stop <= start:
                raise ValueError("frame_range stop must be greater than start")
            with tifffile.TiffFile(str(raw_frames_photon)) as tif:
                frames = np.asarray(tif.asarray(key=range(start, stop)), dtype=np.float32)
    else:
        frames = np.asarray(raw_frames_photon, dtype=np.float32)
    frames = np.squeeze(frames)
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim != 3:
        raise ValueError(f"raw_frames_photon must have shape (T,H,W), got {frames.shape}")
    return np.ascontiguousarray(frames, dtype=np.float32)


def _load_raw_domain_frames(
    raw_frames_photon: str | Path,
    *,
    domain: ROIBankDomain,
    frame_range: tuple[int, int],
) -> np.ndarray:
    import tifffile

    start, stop = int(frame_range[0]), int(frame_range[1])
    if stop <= start:
        raise ValueError("frame_range stop must be greater than start")
    left = int(domain.crop_left)
    top = int(domain.crop_top)
    width = int(domain.crop_width)
    height = int(domain.crop_height)
    if left < 0 or top < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid ROI bank domain: {domain}")
    with tifffile.TiffFile(str(raw_frames_photon)) as tif:
        series_shape = tuple(int(v) for v in tif.series[0].shape)
        if len(series_shape) == 2:
            frame_count, frame_height, frame_width = 1, series_shape[0], series_shape[1]
        elif len(series_shape) == 3:
            frame_count, frame_height, frame_width = series_shape
        else:
            raise ValueError(f"raw_frames_photon must have shape (T,H,W), got {series_shape}")
        if stop > frame_count:
            raise ValueError(f"frame_range stop {stop} exceeds TIFF frame count {frame_count}")
        if top + height > frame_height or left + width > frame_width:
            raise ValueError(f"Domain {domain.name!r} crop exceeds raw frame bounds {(frame_height, frame_width)}")
        crops = []
        chunk = 16
        for chunk_start in range(start, stop, chunk):
            chunk_stop = min(stop, chunk_start + chunk)
            frames = np.asarray(tif.series[0].asarray(key=range(chunk_start, chunk_stop)), dtype=np.float32)
            frames = np.squeeze(frames)
            if frames.ndim == 2:
                frames = frames[None, ...]
            if frames.ndim != 3:
                raise ValueError(f"expected TIFF chunk shape (T,H,W), got {frames.shape}")
            crops.append(frames[:, top : top + height, left : left + width])
    return np.ascontiguousarray(np.concatenate(crops, axis=0), dtype=np.float32)


def _camera_backward_params_for_domain(
    params: dict[str, Any] | None,
    *,
    domain: ROIBankDomain,
    frames_adu: np.ndarray,
) -> dict[str, Any] | None:
    if not params:
        return None
    resolved = dict(params)
    baseline_by_domain = resolved.get("baseline_by_domain")
    if isinstance(baseline_by_domain, dict) and str(domain.name) in baseline_by_domain:
        resolved["baseline"] = float(baseline_by_domain[str(domain.name)])
    elif str(resolved.get("baseline_mode", "")).strip().lower() in {"per_domain_percentile", "domain_percentile"}:
        percentile = float(resolved.get("baseline_percentile", 1.0))
        resolved["baseline"] = _estimate_domain_baseline(frames_adu, percentile=percentile)
    resolved.pop("baseline_mode", None)
    resolved.pop("baseline_percentile", None)
    resolved.pop("baseline_frame_range", None)
    return resolved


def _estimate_domain_baseline(frames_adu: np.ndarray, *, percentile: float) -> float:
    frames = np.asarray(frames_adu, dtype=np.float32)
    if frames.size == 0:
        return 0.0
    pct = float(np.clip(percentile, 0.0, 100.0))
    per_frame = np.percentile(frames.reshape((frames.shape[0], -1)), pct, axis=1)
    return float(np.median(per_frame))


def _camera_backward_metadata(
    params: dict[str, Any] | None,
    resolved_by_domain: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not params:
        return {}
    metadata = {key: value for key, value in dict(params).items() if key != "baseline_by_domain"}
    has_domain_specific_baseline = (
        isinstance(params.get("baseline_by_domain"), dict)
        or str(params.get("baseline_mode", "")).strip().lower() in {"per_domain_percentile", "domain_percentile"}
    )
    if resolved_by_domain and has_domain_specific_baseline:
        metadata["baseline_by_domain"] = {
            domain: float(resolved.get("baseline", 0.0)) for domain, resolved in sorted(resolved_by_domain.items())
        }
        metadata["resolved_by_domain"] = {
            domain: {
                key: float(value)
                for key, value in resolved.items()
                if key in {"baseline", "baseline_adu", "e_per_adu", "em_gain", "qe", "spurious_charge"}
            }
            for domain, resolved in sorted(resolved_by_domain.items())
        }
    return metadata


def _camera_backward_photons(frames: np.ndarray, params: dict[str, Any] | None) -> np.ndarray:
    if not params:
        return np.asarray(frames, dtype=np.float32)
    qe = float(params.get("qe", 1.0))
    e_per_adu = float(params.get("e_per_adu", 1.0))
    baseline = float(params.get("baseline", params.get("baseline_adu", 0.0)))
    spurious = float(params.get("spurious_charge", 0.0))
    em_gain = float(params.get("em_gain", 1.0))
    electrons = (np.asarray(frames, dtype=np.float32) - baseline) * e_per_adu
    photons = ((electrons / max(em_gain, 1e-12)) - spurious) / max(qe, 1e-12)
    return np.maximum(photons, np.float32(1e-10)).astype(np.float32, copy=False)


def _iter_frame_windows(frame_count: int, *, config: ROIBankBuildConfig) -> list[tuple[int, int]]:
    start, stop = (0, int(frame_count)) if config.frame_range is None else (int(config.frame_range[0]), int(config.frame_range[1]))
    start = max(0, start)
    stop = min(int(frame_count), stop)
    window = int(config.window_size)
    if window <= 0:
        raise ValueError("window_size must be positive")
    if stop <= start:
        return []
    windows: list[tuple[int, int]] = []
    cursor = start
    while cursor + window <= stop:
        windows.append((cursor, cursor + window))
        cursor += 1
    if not windows:
        windows.append((start, stop))
    return windows


def _crop_domain(frames: np.ndarray, domain: ROIBankDomain) -> np.ndarray:
    left = int(domain.crop_left)
    top = int(domain.crop_top)
    width = int(domain.crop_width)
    height = int(domain.crop_height)
    if left < 0 or top < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid ROI bank domain: {domain}")
    if top + height > frames.shape[-2] or left + width > frames.shape[-1]:
        raise ValueError(f"Domain {domain.name!r} crop exceeds raw frame bounds {tuple(frames.shape[-2:])}")
    return np.ascontiguousarray(frames[:, top : top + height, left : left + width])


def _smooth_background(background: np.ndarray, *, kernel_size: int) -> np.ndarray:
    bg = np.asarray(background, dtype=np.float32)
    kernel = int(kernel_size)
    if kernel <= 1:
        return bg.copy()
    if kernel % 2 == 0:
        raise ValueError("background_smoothing_kernel must be odd")
    pad = kernel // 2
    padded = np.pad(bg, ((pad, pad), (pad, pad)), mode="edge")
    out = np.empty_like(bg, dtype=np.float32)
    for y in range(bg.shape[0]):
        for x in range(bg.shape[1]):
            out[y, x] = float(padded[y : y + kernel, x : x + kernel].mean())
    return out


def _inside_roi_inner(
    emitter: InferredEmitter,
    *,
    origin_xy_px: tuple[int, int],
    roi_size_px: int,
    over_cut_px: int,
) -> bool:
    x, y = float(emitter.mu_xy_px[0]), float(emitter.mu_xy_px[1])
    x0, y0 = origin_xy_px
    roi = int(roi_size_px)
    over = max(0, int(over_cut_px))
    if roi <= 2 * over:
        raise ValueError("roi_size_px must be larger than 2 * over_cut_px")
    return (x0 + over) <= x < (x0 + roi - over) and (y0 + over) <= y < (y0 + roi - over)
