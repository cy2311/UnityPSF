from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
import json

import numpy as np
import torch
import torch.nn.functional as F
import tifffile


@dataclass(frozen=True)
class StarterHarvesterConfig:
    tiff_path: Path
    input_offset: float
    input_scale: float
    phot_max: float
    z_max: float
    bkg_max: float
    pixel_size_x_nm: float
    pixel_size_y_nm: float
    raw_th: float = 0.5
    split_th: float = 0.6
    accept_prob_threshold: float = 0.7
    phot_min: float = 100.0
    sigma_max_px: float = 2.5
    frame_window: int = 3
    frame_start: int = 0
    frame_stop: int | None = None
    temporal_batch_size: int = 16
    spatial_tile_size: int = 64
    spatial_overlap_px: int = 8
    spatial_tile_batch_size: int = 64
    device: str = "cpu"
    crop_x0: int = 0
    crop_x1: int | None = None
    crop_y0: int = 0
    crop_y1: int | None = None


@dataclass(frozen=True)
class HarvestedEmitters:
    frame_index: torch.Tensor
    probability: torch.Tensor
    x_px: torch.Tensor
    y_px: torch.Tensor
    z_um: torch.Tensor
    photons: torch.Tensor
    x_sigma_px: torch.Tensor
    y_sigma_px: torch.Tensor
    z_sigma_um: torch.Tensor
    photon_sigma: torch.Tensor
    background_adu: torch.Tensor
    local_x_nm: torch.Tensor
    local_y_nm: torch.Tensor

    def __len__(self) -> int:
        return int(self.frame_index.numel())


@dataclass(frozen=True)
class StarterHarvesterOutput:
    candidate_count: int
    accepted_count: int
    metrics: dict[str, Any]
    emitters: HarvestedEmitters


@dataclass(frozen=True)
class _TileSpec:
    x0: int
    x1: int
    y0: int
    y1: int
    keep_x0: int
    keep_x1: int
    keep_y0: int
    keep_y1: int


def _load_pipeline_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_starter_harvester_config(
    *,
    pipeline_config_path: str | Path,
    tiff_path: str | Path | None = None,
    device: str = "cpu",
    spatial_tile_size: int = 64,
    temporal_batch_size: int = 16,
    frame_start: int = 0,
    frame_stop: int | None = None,
    spatial_overlap_px: int = 8,
    spatial_tile_batch_size: int = 64,
    raw_th: float | None = None,
    split_th: float | None = None,
    input_offset: float | None = None,
    input_scale: float | None = None,
    accept_prob_threshold: float = 0.7,
    phot_min: float = 100.0,
    sigma_max_px: float = 2.5,
    crop_x0: int = 0,
    crop_x1: int | None = None,
    crop_y0: int = 0,
    crop_y1: int | None = None,
) -> StarterHarvesterConfig:
    pipeline_path = Path(pipeline_config_path)
    cfg = _load_pipeline_json(pipeline_path)
    optical = cfg.get("optical") or {}
    scaling = ((cfg.get("train") or {}).get("scaling") or {})
    source_tiff = tiff_path if tiff_path is not None else ((cfg.get("phase_retrieval") or {}).get("input") or {}).get("tiff_path")
    if source_tiff is None:
        raise ValueError("No TIFF path provided and pipeline config has no phase_retrieval.input.tiff_path.")
    pp_raw = 0.5 if raw_th is None else float(raw_th)
    pp_split = 0.6 if split_th is None else float(split_th)
    tile_size = int(spatial_tile_size)
    overlap_px = max(int(spatial_overlap_px), 0)
    if tile_size > 1:
        overlap_px = min(overlap_px, tile_size - 1)
    else:
        overlap_px = 0

    return StarterHarvesterConfig(
        tiff_path=Path(source_tiff),
        input_offset=float(scaling.get("input_offset", 0.0)) if input_offset is None else float(input_offset),
        input_scale=max(float(scaling.get("input_scale", 1.0)) if input_scale is None else float(input_scale), 1e-6),
        phot_max=float(scaling.get("photon_max", 1.0)),
        z_max=float(scaling.get("z_max", 1.0)),
        bkg_max=float(scaling.get("bg_max", 1.0)),
        pixel_size_x_nm=float(optical.get("pixel_size_nm_x", 100.0)),
        pixel_size_y_nm=float(optical.get("pixel_size_nm_y", 100.0)),
        raw_th=pp_raw,
        split_th=pp_split,
        accept_prob_threshold=float(accept_prob_threshold),
        phot_min=float(phot_min),
        sigma_max_px=float(sigma_max_px),
        frame_start=max(int(frame_start), 0),
        frame_stop=None if frame_stop is None else max(int(frame_stop), 0),
        temporal_batch_size=int(temporal_batch_size),
        spatial_tile_size=tile_size,
        spatial_overlap_px=overlap_px,
        spatial_tile_batch_size=max(int(spatial_tile_batch_size), 1),
        device=str(device),
        crop_x0=max(int(crop_x0), 0),
        crop_x1=None if crop_x1 is None else max(int(crop_x1), 0),
        crop_y0=max(int(crop_y0), 0),
        crop_y1=None if crop_y1 is None else max(int(crop_y1), 0),
    )


def _canonicalize_stack_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    non_singleton = tuple(int(dim) for dim in shape if int(dim) != 1)
    if len(non_singleton) == 2:
        return 1, int(non_singleton[0]), int(non_singleton[1])
    if len(non_singleton) == 3:
        return int(non_singleton[0]), int(non_singleton[1]), int(non_singleton[2])
    raise ValueError(f"Unsupported TIFF stack shape: {shape}")


def _read_tiff_shape(path: Path) -> tuple[int, int, int]:
    try:
        mem = tifffile.memmap(path)
        return _canonicalize_stack_shape(tuple(int(dim) for dim in mem.shape))
    except ValueError:
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError(f"TIFF has no image series: {path}")
            return _canonicalize_stack_shape(tuple(int(dim) for dim in tif.series[0].shape))


def _canonicalize_stack_array(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames)
    while array.ndim > 3 and 1 in array.shape:
        squeeze_axes = tuple(ix for ix, dim in enumerate(array.shape) if dim == 1)
        array = np.squeeze(array, axis=squeeze_axes[:1])
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected frame stack of shape (N,H,W), got {tuple(array.shape)}")
    return array


@lru_cache(maxsize=64)
def _build_normalization_lut(input_offset: float, input_scale: float, max_value: int = 65535) -> np.ndarray:
    values = np.arange(int(max_value) + 1, dtype=np.float32)
    return (values - float(input_offset)) / float(input_scale)


def _normalize_frame_stack(frames: torch.Tensor, *, input_offset: float, input_scale: float) -> torch.Tensor:
    if torch.is_floating_point(frames):
        return (frames - float(input_offset)) / float(input_scale)
    if frames.device.type != "cpu":
        frames = frames.cpu()
    frames_np = np.asarray(frames.numpy())
    if np.issubdtype(frames_np.dtype, np.integer) and frames_np.size:
        max_value = int(np.iinfo(frames_np.dtype).max)
        lut = _build_normalization_lut(float(input_offset), float(input_scale), max_value=max_value)
        normalized = lut[frames_np]
        return torch.as_tensor(normalized, dtype=torch.float32)
    return (frames.to(dtype=torch.float32) - float(input_offset)) / float(input_scale)


def _read_tiff_frame_range(path: Path, start: int, stop: int) -> torch.Tensor:
    try:
        mem = tifffile.memmap(path)
        frames = _canonicalize_stack_array(np.asarray(mem))[int(start) : int(stop)]
    except ValueError:
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError(f"TIFF has no image series: {path}")
            frames = _canonicalize_stack_array(tif.series[0].asarray(key=range(int(start), int(stop))))
    tensor = torch.as_tensor(frames)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected frame stack of shape (N,H,W), got {tuple(tensor.shape)}")
    return tensor


@lru_cache(maxsize=32)
def _load_tiff_frame_stack_cached(path_str: str, start: int, stop: int) -> torch.Tensor:
    return _read_tiff_frame_range(Path(path_str), start, stop)


def load_tiff_frame_stack(path: str | Path, start: int, stop: int) -> torch.Tensor:
    return _load_tiff_frame_stack_cached(str(Path(path)), int(start), int(stop))


def _sliding_windows(frames: torch.Tensor, window: int) -> torch.Tensor:
    if frames.dim() != 3:
        raise ValueError(f"Expected frames with shape (N,H,W), got {tuple(frames.shape)}")
    if window <= 0 or window % 2 != 1:
        raise ValueError(f"frame_window must be a positive odd integer, got {window}")
    if frames.size(0) < window:
        return torch.empty((0, window, frames.size(1), frames.size(2)), dtype=frames.dtype)
    windows = frames.unfold(0, window, 1)
    return windows.permute(0, 3, 1, 2).contiguous()


def _tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if overlap >= tile:
        raise ValueError(f"spatial_overlap_px must be smaller than tile size, got overlap={overlap}, tile={tile}")
    if size <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, max(size - tile, 0) + 1, stride))
    end_start = size - tile
    if starts[-1] != end_start:
        starts.append(end_start)
    return starts


def _iter_tiles(height: int, width: int, tile: int, overlap: int) -> Iterator[_TileSpec]:
    x_starts = _tile_starts(width, tile, overlap)
    y_starts = _tile_starts(height, tile, overlap)
    left_margin = overlap // 2
    right_margin = overlap - left_margin
    for iy, y0 in enumerate(y_starts):
        y1 = min(y0 + tile, height)
        keep_y0 = y0 if iy == 0 else y0 + left_margin
        keep_y1 = y1 if iy == len(y_starts) - 1 else y1 - right_margin
        for ix, x0 in enumerate(x_starts):
            x1 = min(x0 + tile, width)
            keep_x0 = x0 if ix == 0 else x0 + left_margin
            keep_x1 = x1 if ix == len(x_starts) - 1 else x1 - right_margin
            yield _TileSpec(
                x0=int(x0),
                x1=int(x1),
                y0=int(y0),
                y1=int(y1),
                keep_x0=int(keep_x0),
                keep_x1=int(keep_x1),
                keep_y0=int(keep_y0),
                keep_y1=int(keep_y1),
            )


def _flush_cuda_oom_state(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _spatial_integration(p: torch.Tensor, raw_th: float, split_th: float) -> torch.Tensor:
    if p.dim() != 3:
        raise ValueError(f"Expected detection map with shape (N,H,W), got {tuple(p.shape)}")
    diag = 0.0
    filt = torch.tensor(
        [[diag, 1.0, diag], [1.0, 1.0, 1.0], [diag, 1.0, diag]],
        dtype=p.dtype,
        device=p.device,
    ).view(1, 1, 3, 3)
    conv = F.conv2d(p.unsqueeze(1), filt, padding=1)
    p_clip = torch.where(p > raw_th, p, torch.zeros_like(p))
    pool = F.max_pool2d(p_clip.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    max_mask1 = torch.eq(p.unsqueeze(1), pool)
    p_ps1 = max_mask1.to(dtype=p.dtype) * conv
    p_copy = p.unsqueeze(1) * (1.0 - max_mask1.to(dtype=p.dtype))
    max_mask2 = torch.where(p_copy > split_th, torch.ones_like(p_copy), torch.zeros_like(p_copy))
    p_ps2 = max_mask2 * conv
    return torch.clamp(p_ps1 + p_ps2, min=0.0, max=1.0).squeeze(1)


def _extract_from_tile(
    y_out: torch.Tensor,
    *,
    tile: _TileSpec,
    frame_indices: torch.Tensor,
    cfg: StarterHarvesterConfig,
    width: int,
    height: int,
) -> dict[str, torch.Tensor]:
    if y_out.numel() == 0:
        return {}

    output = y_out.detach().to(dtype=torch.float32)
    nms_prob = _spatial_integration(output[:, 0], raw_th=cfg.raw_th, split_th=cfg.split_th)

    output = output.clone()
    output[:, 0] = nms_prob
    output[:, [1, 5]] *= cfg.phot_max
    output[:, [4, 8]] *= cfg.z_max
    output[:, 9] *= cfg.bkg_max

    device = output.device
    y_coords = torch.arange(tile.y0, tile.y1, device=device, dtype=output.dtype) + 0.5
    x_coords = torch.arange(tile.x0, tile.x1, device=device, dtype=output.dtype) + 0.5
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    x_abs = output[:, 2] + xx.unsqueeze(0)
    y_abs = output[:, 3] + yy.unsqueeze(0)
    prob = output[:, 0]
    photons = output[:, 1]
    x_sigma = output[:, 6]
    y_sigma = output[:, 7]

    keep_region = (
        (xx.unsqueeze(0) >= float(tile.keep_x0) + 0.5)
        & (xx.unsqueeze(0) < float(tile.keep_x1) + 0.5)
        & (yy.unsqueeze(0) >= float(tile.keep_y0) + 0.5)
        & (yy.unsqueeze(0) < float(tile.keep_y1) + 0.5)
    )
    candidate_mask = keep_region & (prob >= cfg.raw_th)
    accepted_mask = (
        candidate_mask
        & torch.isfinite(prob)
        & torch.isfinite(photons)
        & torch.isfinite(x_abs)
        & torch.isfinite(y_abs)
        & torch.isfinite(x_sigma)
        & torch.isfinite(y_sigma)
        & (prob >= cfg.accept_prob_threshold)
        & (photons >= cfg.phot_min)
        & (x_sigma <= cfg.sigma_max_px)
        & (y_sigma <= cfg.sigma_max_px)
        & (x_abs >= 0.0)
        & (x_abs < float(width))
        & (y_abs >= 0.0)
        & (y_abs < float(height))
    )

    candidate_idx = candidate_mask.nonzero(as_tuple=False)
    accepted_idx = accepted_mask.nonzero(as_tuple=False)
    if accepted_idx.numel() == 0:
        return {
            "candidate_count": torch.tensor(int(candidate_idx.size(0))),
            "accepted_count": torch.tensor(0),
        }

    batch_ix = accepted_idx[:, 0]
    row_ix = accepted_idx[:, 1]
    col_ix = accepted_idx[:, 2]
    frame_index = frame_indices.to(device=device)[batch_ix].to(dtype=torch.int64).cpu()
    local_x_px = y_out[batch_ix, 2, row_ix, col_ix].detach().cpu().to(dtype=torch.float32)
    local_y_px = y_out[batch_ix, 3, row_ix, col_ix].detach().cpu().to(dtype=torch.float32)

    return {
        "candidate_count": torch.tensor(int(candidate_idx.size(0))),
        "accepted_count": torch.tensor(int(accepted_idx.size(0))),
        "frame_index": frame_index,
        "probability": prob[batch_ix, row_ix, col_ix].detach().cpu(),
        "x_px": x_abs[batch_ix, row_ix, col_ix].detach().cpu(),
        "y_px": y_abs[batch_ix, row_ix, col_ix].detach().cpu(),
        "z_um": output[batch_ix, 4, row_ix, col_ix].detach().cpu(),
        "photons": photons[batch_ix, row_ix, col_ix].detach().cpu(),
        "x_sigma_px": x_sigma[batch_ix, row_ix, col_ix].detach().cpu(),
        "y_sigma_px": y_sigma[batch_ix, row_ix, col_ix].detach().cpu(),
        "z_sigma_um": output[batch_ix, 8, row_ix, col_ix].detach().cpu(),
        "photon_sigma": output[batch_ix, 5, row_ix, col_ix].detach().cpu(),
        "background_adu": output[batch_ix, 9, row_ix, col_ix].detach().cpu(),
        "local_x_nm": local_x_px * float(cfg.pixel_size_x_nm),
        "local_y_nm": local_y_px * float(cfg.pixel_size_y_nm),
    }


def _concat_emitters(chunks: list[dict[str, torch.Tensor]]) -> HarvestedEmitters:
    if not chunks:
        empty_i = torch.zeros((0,), dtype=torch.int64)
        empty_f = torch.zeros((0,), dtype=torch.float32)
        return HarvestedEmitters(
            frame_index=empty_i,
            probability=empty_f,
            x_px=empty_f,
            y_px=empty_f,
            z_um=empty_f,
            photons=empty_f,
            x_sigma_px=empty_f,
            y_sigma_px=empty_f,
            z_sigma_um=empty_f,
            photon_sigma=empty_f,
            background_adu=empty_f,
            local_x_nm=empty_f,
            local_y_nm=empty_f,
        )

    def cat(name: str) -> torch.Tensor:
        return torch.cat([chunk[name] for chunk in chunks if name in chunk], dim=0)

    return HarvestedEmitters(
        frame_index=cat("frame_index").to(dtype=torch.int64),
        probability=cat("probability").to(dtype=torch.float32),
        x_px=cat("x_px").to(dtype=torch.float32),
        y_px=cat("y_px").to(dtype=torch.float32),
        z_um=cat("z_um").to(dtype=torch.float32),
        photons=cat("photons").to(dtype=torch.float32),
        x_sigma_px=cat("x_sigma_px").to(dtype=torch.float32),
        y_sigma_px=cat("y_sigma_px").to(dtype=torch.float32),
        z_sigma_um=cat("z_sigma_um").to(dtype=torch.float32),
        photon_sigma=cat("photon_sigma").to(dtype=torch.float32),
        background_adu=cat("background_adu").to(dtype=torch.float32),
        local_x_nm=cat("local_x_nm").to(dtype=torch.float32),
        local_y_nm=cat("local_y_nm").to(dtype=torch.float32),
    )


def harvest_tiff_points(model: torch.nn.Module, cfg: StarterHarvesterConfig) -> StarterHarvesterOutput:
    path = Path(cfg.tiff_path)
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found: {path}")

    n_frames, height, width = _read_tiff_shape(path)
    frame_start = max(int(cfg.frame_start), 0)
    frame_stop = int(n_frames if cfg.frame_stop is None else min(int(cfg.frame_stop), n_frames))
    if frame_stop <= frame_start:
        emitters = _concat_emitters([])
        return StarterHarvesterOutput(
            candidate_count=0,
            accepted_count=0,
            metrics={
                "status": "empty_frame_range",
                "n_frames": int(n_frames),
                "frame_start": int(frame_start),
                "frame_stop": int(frame_stop),
                "frame_window": int(cfg.frame_window),
                "image_height_px": int(height),
                "image_width_px": int(width),
            },
            emitters=emitters,
        )

    effective_frames = int(frame_stop - frame_start)
    if effective_frames < cfg.frame_window:
        emitters = _concat_emitters([])
        return StarterHarvesterOutput(
            candidate_count=0,
            accepted_count=0,
            metrics={
                "status": "short_stack",
                "n_frames": int(n_frames),
                "frame_start": int(frame_start),
                "frame_stop": int(frame_stop),
                "effective_frames": int(effective_frames),
                "frame_window": int(cfg.frame_window),
                "image_height_px": int(height),
                "image_width_px": int(width),
            },
            emitters=emitters,
        )

    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    emitters_per_batch: list[dict[str, torch.Tensor]] = []
    total_candidates = 0
    total_windows = 0
    total_model_calls = 0
    temporal_batch = max(int(cfg.temporal_batch_size), 1)
    crop_x0 = min(max(int(cfg.crop_x0), 0), width)
    crop_x1 = width if cfg.crop_x1 is None else min(max(int(cfg.crop_x1), crop_x0), width)
    crop_y0 = min(max(int(cfg.crop_y0), 0), height)
    crop_y1 = height if cfg.crop_y1 is None else min(max(int(cfg.crop_y1), crop_y0), height)
    tile_specs = [
        tile
        for tile in _iter_tiles(height, width, cfg.spatial_tile_size, cfg.spatial_overlap_px)
        if tile.x1 > crop_x0 and tile.x0 < crop_x1 and tile.y1 > crop_y0 and tile.y0 < crop_y1
    ]
    tile_batch_size_requested = max(int(cfg.spatial_tile_batch_size), 1)
    tile_batch_size_effective = tile_batch_size_requested

    frame_stack = load_tiff_frame_stack(path, frame_start, frame_stop)
    frame_stack = _normalize_frame_stack(frame_stack, input_offset=cfg.input_offset, input_scale=cfg.input_scale)
    local_frame_stop = int(frame_stack.size(0))

    with torch.no_grad():
        for sample_start in range(0, local_frame_stop - (cfg.frame_window - 1), temporal_batch):
            sample_stop = min(sample_start + temporal_batch, local_frame_stop - (cfg.frame_window - 1))
            raw_start = sample_start
            raw_stop = sample_stop + (cfg.frame_window - 1)
            frames = frame_stack[raw_start:raw_stop]
            windows = _sliding_windows(frames, cfg.frame_window)
            if windows.numel() == 0:
                continue
            frame_indices = torch.arange(
                frame_start + sample_start + (cfg.frame_window - 1) // 2,
                frame_start + sample_start + (cfg.frame_window - 1) // 2 + windows.size(0),
                dtype=torch.int64,
            )
            tile_start = 0
            current_tile_batch_size = tile_batch_size_effective
            while tile_start < len(tile_specs):
                group_size = min(current_tile_batch_size, len(tile_specs) - tile_start)
                group = tile_specs[tile_start : tile_start + group_size]
                try:
                    packed = torch.cat(
                        [
                            windows[:, :, tile.y0 : tile.y1, tile.x0 : tile.x1]
                            for tile in group
                        ],
                        dim=0,
                    ).to(device=device, dtype=torch.float32)
                    y_out = model(packed)
                    total_model_calls += 1
                except torch.cuda.OutOfMemoryError:
                    if device.type != "cuda" or group_size <= 1:
                        raise
                    _flush_cuda_oom_state(device)
                    current_tile_batch_size = max(group_size // 2, 1)
                    tile_batch_size_effective = min(tile_batch_size_effective, current_tile_batch_size)
                    continue

                split_outputs = torch.split(y_out, int(windows.size(0)), dim=0)
                for tile, tile_out in zip(group, split_outputs):
                    extracted = _extract_from_tile(
                        tile_out,
                        tile=tile,
                        frame_indices=frame_indices,
                        cfg=cfg,
                        width=width,
                        height=height,
                    )
                    if not extracted:
                        continue
                    total_candidates += int(extracted["candidate_count"].item())
                    if int(extracted["accepted_count"].item()) > 0:
                        emitters_per_batch.append(extracted)

                tile_start += len(group)

            total_windows += int(windows.size(0))

    emitters = _concat_emitters(emitters_per_batch)
    accepted_count = len(emitters)
    mean_prob = float(emitters.probability.mean().item()) if accepted_count > 0 else 0.0
    mean_phot = float(emitters.photons.mean().item()) if accepted_count > 0 else 0.0
    metrics = {
        "status": "ok",
        "tiff_path": str(path),
        "n_frames": int(n_frames),
        "frame_start": int(frame_start),
        "frame_stop": int(frame_stop),
        "effective_frames": int(effective_frames),
        "processed_windows": int(total_windows),
        "image_height_px": int(height),
        "image_width_px": int(width),
        "frame_window": int(cfg.frame_window),
        "temporal_batch_size": int(cfg.temporal_batch_size),
        "spatial_tile_size": int(cfg.spatial_tile_size),
        "spatial_overlap_px": int(cfg.spatial_overlap_px),
        "spatial_tile_batch_size_requested": int(tile_batch_size_requested),
        "spatial_tile_batch_size_effective": int(tile_batch_size_effective),
        "model_calls": int(total_model_calls),
        "candidate_probability_threshold": float(cfg.raw_th),
        "accept_probability_threshold": float(cfg.accept_prob_threshold),
        "accepted_probability_mean": mean_prob,
        "accepted_photon_mean": mean_phot,
    }
    return StarterHarvesterOutput(
        candidate_count=int(total_candidates),
        accepted_count=int(accepted_count),
        metrics=metrics,
        emitters=emitters,
    )


__all__ = [
    "HarvestedEmitters",
    "StarterHarvesterConfig",
    "StarterHarvesterOutput",
    "build_starter_harvester_config",
    "load_tiff_frame_stack",
    "harvest_tiff_points",
]
