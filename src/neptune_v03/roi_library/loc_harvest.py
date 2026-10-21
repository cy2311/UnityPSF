from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from neptune_v03.localization.conditioning import FullResZernikeConditioning
from neptune_v03.localization.roi_posterior_sampling import (
    ROIPosteriorSamplingConfig,
    _dh_posterior_samples_from_loc_output,
)

from .raw_tiff_builder import InferredEmitter, ROIBankBuildConfig, ROIBankDomain, RawInferenceResult, build_roi_bank_from_inference
from .types import ROIBank


@dataclass(frozen=True)
class LocHarvestConfig:
    raw_path: str | Path
    domains: tuple[ROIBankDomain, ...]
    roi_bank_config: ROIBankBuildConfig
    input_offset: float = 0.0
    input_scale: float = 1.0
    photon_scale: float = 1.0
    z_scale: float = 1.0
    bg_scale: float = 1.0
    normalization_config: Mapping[str, Any] | None = None
    candidate_probability_threshold: float = 0.3
    probability_threshold: float = 0.5
    split_threshold: float = 0.6
    phot_min: float = 100.0
    sigma_max_px: float = 2.5
    tile_size_px: int = 128
    overlap_px: int = 16
    max_emitters_per_window: int | None = None
    dh_pair_shape_ncc_min: float | None = None
    dh_pair_shape_rmse_max: float | None = None
    dh_pair_probability_weight: float = 0.0
    dh_pair_shape_ncc_weight: float = 0.0
    dh_pair_shape_rmse_weight: float = 0.0
    dh_pair_maximum_cost: float | None = None
    dh_pair_fit_radius_px: int = 5
    dh_pair_quality: Mapping[str, Any] | None = None


def build_roi_bank_from_loc_harvest(
    *,
    model: torch.nn.Module,
    config: LocHarvestConfig,
    condition_context: Mapping[str, Any] | None = None,
) -> ROIBank:
    infer_fn = loc_harvest_infer_fn(
        model=model,
        config=config,
        condition_context=condition_context,
    )
    return build_roi_bank_from_inference(
        raw_frames_photon=str(config.raw_path),
        domains=config.domains,
        infer_fn=infer_fn,
        config=config.roi_bank_config,
    )


def _prepare_condition_vector(
    condition: torch.Tensor,
    *,
    feature_dim: int,
    append_domain_onehot: bool,
    domain_name: str,
    domain_names: tuple[str, ...],
    expected_dim: int | None,
) -> torch.Tensor:
    if int(condition.shape[0]) >= int(feature_dim):
        matched = condition[: int(feature_dim)].contiguous()
    else:
        matched = torch.zeros((int(feature_dim),), dtype=condition.dtype, device=condition.device)
        matched[: int(condition.shape[0])] = condition
    if append_domain_onehot:
        onehot = torch.zeros((len(domain_names),), dtype=condition.dtype, device=condition.device)
        onehot[list(domain_names).index(str(domain_name))] = 1.0
        matched = torch.cat((matched, onehot), dim=0)
    if expected_dim is not None and int(matched.shape[0]) != int(expected_dim):
        raise ValueError(f"Expected condition_dim={int(expected_dim)}, got {int(matched.shape[0])}")
    return matched


def loc_harvest_infer_fn(
    *,
    model: torch.nn.Module,
    config: LocHarvestConfig,
    condition_context: Mapping[str, Any] | None = None,
):
    context = dict(condition_context or {})
    frame_proc = _build_inference_frame_proc(config.normalization_config)

    def infer_fn(
        *,
        domain: ROIBankDomain,
        frame_window: tuple[int, int],
        raw_domain_frames_photon: np.ndarray,
    ) -> RawInferenceResult:
        raw = np.asarray(raw_domain_frames_photon, dtype=np.float32)
        if raw.ndim != 3:
            raise ValueError(f"raw TIFF inference window must have shape (T,H,W), got {raw.shape}")
        _, height, width = raw.shape
        device = _model_device(model)
        was_training = model.training
        model.eval()
        raw_tensor = torch.as_tensor(raw, dtype=torch.float32)
        providers = context.get("providers")
        provider = None if providers is None else providers.get(str(domain.name))
        append_domain_onehot = bool(context.get("append_domain_onehot", False))
        domain_names = tuple(str(name) for name in context.get("domain_names", ()))
        condition_feature_dim = context.get("condition_feature_dim")
        condition_dim = context.get("condition_dim")
        emitters: list[InferredEmitter] = []
        background_accum = np.zeros((height, width), dtype=np.float32)
        background_count = np.zeros((height, width), dtype=np.float32)
        center_frame = int(frame_window[0]) + int(raw.shape[0] // 2)
        with torch.no_grad():
            for tile in _iter_tiles(height, width, tile_size=int(config.tile_size_px), overlap_px=int(config.overlap_px)):
                tile_raw = raw_tensor[:, tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]]
                tile_image = _normalize_tile_input(
                    tile_raw,
                    frame_proc=frame_proc,
                    offset=config.input_offset,
                    scale=config.input_scale,
                ).unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                )
                model_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = tile_image
                if isinstance(provider, FullResZernikeConditioning) or provider is not None:
                    condition = provider.condition_vector_from_xy(
                        x0=int(tile["x0"]),
                        y0=int(tile["y0"]),
                        height=int(tile["y1"] - tile["y0"]),
                        width=int(tile["x1"] - tile["x0"]),
                        device=device,
                        dtype=tile_image.dtype,
                    )
                    condition = _prepare_condition_vector(
                        condition,
                        feature_dim=int(condition.shape[0]) if condition_feature_dim is None else int(condition_feature_dim),
                        append_domain_onehot=append_domain_onehot,
                        domain_name=str(domain.name),
                        domain_names=domain_names,
                        expected_dim=None if condition_dim is None else int(condition_dim),
                    )
                    model_input = (tile_image, condition.unsqueeze(0))
                output = model(model_input).detach().to(dtype=torch.float32)
                if _is_dh_lobe_output(output, model=model):
                    quality = dict(config.dh_pair_quality or {})
                    quality.setdefault("shape_ncc_min", config.dh_pair_shape_ncc_min)
                    quality.setdefault("shape_rmse_max", config.dh_pair_shape_rmse_max)
                    quality.setdefault("probability_weight", config.dh_pair_probability_weight)
                    quality.setdefault("shape_ncc_weight", config.dh_pair_shape_ncc_weight)
                    quality.setdefault("shape_rmse_weight", config.dh_pair_shape_rmse_weight)
                    quality.setdefault("maximum_cost", config.dh_pair_maximum_cost)
                    quality.setdefault("fit_radius_px", config.dh_pair_fit_radius_px)
                    tile_emitters = extract_dh_emitters_from_tile(
                        output,
                        raw_frame=np.asarray(tile_raw[int(raw.shape[0]) // 2].detach().cpu(), dtype=np.float32),
                        tile=tile,
                        candidate_threshold=float(config.candidate_probability_threshold),
                        accept_threshold=float(config.probability_threshold),
                        photon_scale=float(config.photon_scale),
                        frame_window=frame_window,
                        full_width=width,
                        full_height=height,
                        max_emitters=config.max_emitters_per_window,
                        dh_calibration=_dh_calibration(model),
                        pair_quality=quality,
                    )
                else:
                    tile_emitters = extract_old_smlm_emitters_from_tile(
                        output,
                        tile=tile,
                        candidate_threshold=float(config.candidate_probability_threshold),
                        accept_threshold=float(config.probability_threshold),
                        split_threshold=float(config.split_threshold),
                        photon_scale=float(config.photon_scale),
                        z_scale=float(config.z_scale),
                        phot_min=float(config.phot_min),
                        sigma_max_px=float(config.sigma_max_px),
                        frame_index=center_frame,
                        full_width=width,
                        full_height=height,
                        max_emitters=config.max_emitters_per_window,
                    )
                emitters.extend(tile_emitters)
                bg = _background_from_output_or_raw(output, raw, tile=tile, bg_scale=float(config.bg_scale))
                background_accum[tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]] += bg
                background_count[tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]] += 1.0
        if was_training:
            model.train()
        emitters.sort(key=lambda item: float(item.probability), reverse=True)
        return RawInferenceResult(
            emitters=tuple(emitters),
            background_mu=background_accum / np.maximum(background_count, 1.0),
            metadata={
                "domain": str(domain.name),
                "frame_window": frame_window,
                "source": "loc_harvest_raw_tiff",
                "decoder": "dh_lobe_pair_matching" if _is_dh_lobe_model(model) else "legacy_smlm_10ch",
            },
        )

    return infer_fn


def _is_dh_lobe_model(model: torch.nn.Module) -> bool:
    module = model.module if hasattr(model, "module") and isinstance(model.module, torch.nn.Module) else model
    return bool(getattr(module, "is_dh_lobe_model", False)) and getattr(module, "dh_calibration", None) is not None


def _dh_calibration(model: torch.nn.Module) -> Any:
    module = model.module if hasattr(model, "module") and isinstance(model.module, torch.nn.Module) else model
    calibration = getattr(module, "dh_calibration", None)
    if calibration is None:
        raise ValueError("DH lobe harvest requires model.dh_calibration")
    return calibration


def _is_dh_lobe_output(output: torch.Tensor, *, model: torch.nn.Module) -> bool:
    return _is_dh_lobe_model(model) and output.ndim == 4 and int(output.shape[1]) == 8


def extract_dh_emitters_from_tile(
    output: torch.Tensor,
    *,
    raw_frame: np.ndarray,
    tile: Mapping[str, int],
    candidate_threshold: float,
    accept_threshold: float,
    photon_scale: float,
    frame_window: tuple[int, int],
    full_width: int,
    full_height: int,
    max_emitters: int | None,
    dh_calibration: Any,
    pair_quality: Mapping[str, Any] | None = None,
) -> list[InferredEmitter]:
    if not (output.ndim == 4 and int(output.shape[1]) == 8):
        return []
    height = int(output.shape[-2])
    width = int(output.shape[-1])
    record = _dh_harvest_record(
        raw_frames=np.asarray(raw_frame, dtype=np.float32)[None, ...],
        frame_window=frame_window,
        roi_size_px=max(height, width),
    )
    samples = _dh_posterior_samples_from_loc_output(
        output,
        records=(record,),
        sample_index_base=0,
        posterior_group_id_base=0,
        config=ROIPosteriorSamplingConfig(
            candidate_probability_threshold=float(candidate_threshold),
            probability_threshold=float(accept_threshold),
            max_emitters_per_roi=max_emitters,
            roi_size_px=max(height, width),
            photon_scale=float(photon_scale),
            background_smoothing_kernel=1,
            dh_pair_shape_ncc_min=None if pair_quality is None else pair_quality.get("shape_ncc_min"),
            dh_pair_shape_rmse_max=None if pair_quality is None else pair_quality.get("shape_rmse_max"),
            dh_pair_probability_weight=0.0 if pair_quality is None else float(pair_quality.get("probability_weight", 0.0)),
            dh_pair_shape_ncc_weight=0.0 if pair_quality is None else float(pair_quality.get("shape_ncc_weight", 0.0)),
            dh_pair_shape_rmse_weight=0.0 if pair_quality is None else float(pair_quality.get("shape_rmse_weight", 0.0)),
            dh_pair_maximum_cost=None if pair_quality is None else pair_quality.get("maximum_cost"),
            dh_pair_fit_radius_px=5 if pair_quality is None else int(pair_quality.get("fit_radius_px", 5)),
        ),
        dh_calibration=dh_calibration,
    )
    if not samples:
        return []
    sample = samples[0]
    emitters: list[InferredEmitter] = []
    frame_index = int(frame_window[0]) + int((frame_window[1] - frame_window[0]) // 2)
    for index in range(int(sample.emitter_count)):
        xy = sample.xy_px[index]
        x = float(tile["x0"]) + float(xy[0])
        y = float(tile["y0"]) + float(xy[1])
        probability = float(sample.probability[index])
        photons = float(sample.photons[index])
        z_nm = float(sample.z_nm[index])
        if not (
            probability >= float(accept_threshold)
            and float(tile["keep_x0"]) <= x < float(tile["keep_x1"])
            and float(tile["keep_y0"]) <= y < float(tile["keep_y1"])
            and 0.0 <= x < float(full_width)
            and 0.0 <= y < float(full_height)
            and np.isfinite(x)
            and np.isfinite(y)
            and np.isfinite(photons)
            and np.isfinite(z_nm)
        ):
            continue
        emitters.append(
            InferredEmitter(
                probability=probability,
                mu_xy_px=(x, y),
                sigma_xy_px=(1.0, 1.0),
                mu_z_nm=z_nm,
                sigma_z_nm=0.0,
                mu_photons=max(photons, 1e-6),
                sigma_photons=max(0.5 * photons, 1e-6),
                cell_xy_px=(float(tile["x0"]) + float(sample.cell_xy_px[index, 0]), float(tile["y0"]) + float(sample.cell_xy_px[index, 1])),
                frame_index=frame_index,
            )
        )
    return emitters


def _dh_harvest_record(*, raw_frames: np.ndarray, frame_window: tuple[int, int], roi_size_px: int):
    from neptune_v03.roi_library import ROIRecord

    background = np.zeros((roi_size_px, roi_size_px), dtype=np.float32)
    return ROIRecord(
        roi_id=0,
        domain_name="double_helix",
        frame_window=frame_window,
        roi_origin_xy_px=(0.0, 0.0),
        raw_frames_photon=raw_frames,
        background_mu=background,
        background_smoothed=background,
        grid_cell_id=0,
    )


def extract_old_smlm_emitters_from_tile(
    output: torch.Tensor,
    *,
    tile: Mapping[str, int],
    candidate_threshold: float,
    accept_threshold: float,
    split_threshold: float,
    photon_scale: float,
    z_scale: float,
    phot_min: float,
    sigma_max_px: float,
    frame_index: int,
    full_width: int,
    full_height: int,
    max_emitters: int | None = None,
) -> list[InferredEmitter]:
    if not (output.ndim == 4 and int(output.shape[1]) == 10):
        return []
    out = output[0].detach().cpu().to(dtype=torch.float32).clone()
    p = _spatial_integration(out[0].unsqueeze(0), raw_th=float(candidate_threshold), split_th=float(split_threshold))[0]
    out[0] = p
    out[[1, 5]] *= float(photon_scale)
    out[[4, 8]] *= float(z_scale)
    scores = p.reshape(-1)
    if scores.numel() == 0:
        return []
    k = int(scores.numel()) if max_emitters is None else min(int(max_emitters), int(scores.numel()))
    values, indices = torch.topk(scores, k=k)
    tile_w = int(p.shape[-1])
    emitters = []
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        if float(value) < float(accept_threshold):
            continue
        row = int(flat_index) // tile_w
        col = int(flat_index) % tile_w
        x = float(tile["x0"]) + float(col) + float(out[2, row, col].item())
        y = float(tile["y0"]) + float(row) + float(out[3, row, col].item())
        photons = float(out[1, row, col].item())
        x_sigma = float(out[6, row, col].item())
        y_sigma = float(out[7, row, col].item())
        if not (
            float(tile["keep_x0"]) <= x < float(tile["keep_x1"])
            and float(tile["keep_y0"]) <= y < float(tile["keep_y1"])
            and 0.0 <= x < float(full_width)
            and 0.0 <= y < float(full_height)
            and photons >= float(phot_min)
            and np.isfinite(photons)
            and np.isfinite(x)
            and np.isfinite(y)
            and x_sigma <= float(sigma_max_px)
            and y_sigma <= float(sigma_max_px)
        ):
            continue
        emitter = InferredEmitter(
            probability=float(value),
            mu_xy_px=(x, y),
            sigma_xy_px=(x_sigma, y_sigma),
            mu_z_nm=float(out[4, row, col].item()),
            sigma_z_nm=abs(float(out[8, row, col].item())),
            mu_photons=max(photons, 1e-6),
            sigma_photons=max(float(out[5, row, col].item()), 1e-6),
            cell_xy_px=(float(tile["x0"] + col), float(tile["y0"] + row)),
            frame_index=int(frame_index),
        )
        emitters.append(emitter)
    return emitters


def _normalize_input(frames: torch.Tensor, *, offset: float, scale: float) -> torch.Tensor:
    return (frames.to(dtype=torch.float32) - float(offset)) / max(float(scale), 1e-6)


def _normalize_tile_input(
    frames: torch.Tensor,
    *,
    frame_proc: "_FDDeeplocTileNormalizer | None",
    offset: float,
    scale: float,
) -> torch.Tensor:
    if frame_proc is not None:
        return frame_proc.forward(frames.to(dtype=torch.float32)).to(dtype=torch.float32)
    return _normalize_input(frames, offset=offset, scale=scale)


@dataclass(frozen=True)
class _FDDeeplocTileNormalizer:
    train_background_adu: float = 495.58422534346505
    background_percentile: float = 50.0
    recenter_mode: str = "block"

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        arr = frames.detach().cpu().numpy()
        if self.recenter_mode == "frame":
            backgrounds = np.asarray(
                [self.estimate_background_numpy(image[None, ...]) for image in arr], dtype=np.float32
            )
            return frames.to(dtype=torch.float32) - torch.as_tensor(
                backgrounds, dtype=torch.float32, device=frames.device
            )[:, None, None] + float(self.train_background_adu)
        bg = self.estimate_background_numpy(arr)
        return frames.to(dtype=torch.float32) - float(bg) + float(self.train_background_adu)

    def estimate_background_numpy(self, images: np.ndarray) -> float:
        arr = np.asarray(images, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected FD-DeepLoc images with shape (frames,height,width), got {arr.shape}")
        safe = arr.copy()
        safe[safe <= 0] = 1.0
        mean_img = safe.mean(axis=0)
        mask = mean_img < np.percentile(mean_img, float(self.background_percentile))
        if not np.any(mask):
            return float(np.percentile(safe, float(self.background_percentile)))
        pixel_vals = safe[:, mask].reshape(-1)
        try:
            from scipy import stats

            fit_alpha, _fit_loc, fit_beta = stats.gamma.fit(pixel_vals, floc=0)
            return float(fit_alpha * fit_beta)
        except Exception:
            return float(np.percentile(pixel_vals, float(self.background_percentile)))

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "mode": "fd_deeploc_style",
            "model_input": "recentered_raw_adu",
            "recenter_mode": self.recenter_mode,
            "train_background_adu": float(self.train_background_adu),
            "background_percentile": float(self.background_percentile),
            "legacy_input_offset": None,
            "legacy_input_scale": None,
        }


def _build_inference_frame_proc(config: Mapping[str, Any] | None) -> _FDDeeplocTileNormalizer | None:
    if not isinstance(config, Mapping):
        return None
    norm_cfg = config.get("normalization") if isinstance(config.get("normalization"), Mapping) else config
    mode = str(norm_cfg.get("infer_mode", norm_cfg.get("mode", "legacy_input_scale"))).lower()
    if mode not in {"fd_deeploc_style", "fd_deeploc_exact_recenter", "fd-style", "fd_style", "fd_deeploc"}:
        return None
    train_background_adu = norm_cfg.get("train_background_adu", norm_cfg.get("backg", 495.58422534346505))
    recenter_mode = str(norm_cfg.get("recenter_mode", "block")).lower()
    if recenter_mode not in {"frame", "block"}:
        raise ValueError(f"Unsupported recenter_mode={recenter_mode!r}; expected 'frame' or 'block'.")
    return _FDDeeplocTileNormalizer(
        train_background_adu=float(train_background_adu),
        background_percentile=float(norm_cfg.get("background_percentile", 50.0)),
        recenter_mode=recenter_mode,
    )


def _spatial_integration(p: torch.Tensor, *, raw_th: float, split_th: float) -> torch.Tensor:
    diag = 0.0
    filt = torch.tensor(
        [[diag, 1.0, diag], [1.0, 1.0, 1.0], [diag, 1.0, diag]],
        dtype=p.dtype,
        device=p.device,
    ).view(1, 1, 3, 3)
    conv = F.conv2d(p.unsqueeze(1), filt, padding=1)
    p_clip = torch.where(p > float(raw_th), p, torch.zeros_like(p))
    pool = F.max_pool2d(p_clip.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    max_mask1 = torch.eq(p.unsqueeze(1), pool)
    p_ps1 = max_mask1.to(dtype=p.dtype) * conv
    p_copy = p.unsqueeze(1) * (1.0 - max_mask1.to(dtype=p.dtype))
    max_mask2 = torch.where(p_copy > float(split_th), torch.ones_like(p_copy), torch.zeros_like(p_copy))
    p_ps2 = max_mask2 * conv
    return torch.clamp(p_ps1 + p_ps2, min=0.0, max=1.0).squeeze(1)


def _tile_starts(size: int, tile_size: int, overlap_px: int) -> list[int]:
    if int(size) <= int(tile_size):
        return [0]
    stride = max(1, int(tile_size) - int(overlap_px))
    starts = list(range(0, int(size) - int(tile_size) + 1, stride))
    end = int(size) - int(tile_size)
    if starts[-1] != end:
        starts.append(end)
    return starts


def _iter_tiles(height: int, width: int, *, tile_size: int, overlap_px: int) -> list[dict[str, int]]:
    tile = min(int(tile_size), int(height), int(width))
    overlap = min(max(int(overlap_px), 0), max(tile - 1, 0))
    xs = _tile_starts(width, tile, overlap)
    ys = _tile_starts(height, tile, overlap)
    left_margin = overlap // 2
    right_margin = overlap - left_margin
    tiles = []
    for iy, y0 in enumerate(ys):
        y1 = min(y0 + tile, height)
        keep_y0 = y0 if iy == 0 else y0 + left_margin
        keep_y1 = y1 if iy == len(ys) - 1 else y1 - right_margin
        for ix, x0 in enumerate(xs):
            x1 = min(x0 + tile, width)
            keep_x0 = x0 if ix == 0 else x0 + left_margin
            keep_x1 = x1 if ix == len(xs) - 1 else x1 - right_margin
            tiles.append(
                {
                    "x0": int(x0),
                    "x1": int(x1),
                    "y0": int(y0),
                    "y1": int(y1),
                    "keep_x0": int(keep_x0),
                    "keep_x1": int(keep_x1),
                    "keep_y0": int(keep_y0),
                    "keep_y1": int(keep_y1),
                }
            )
    return tiles


def _background_from_output_or_raw(output: torch.Tensor, raw: np.ndarray, *, tile: Mapping[str, int], bg_scale: float) -> np.ndarray:
    if output.ndim == 4 and int(output.shape[1]) == 10:
        return (output[0, 9].detach().cpu().numpy() * float(bg_scale)).astype(np.float32, copy=False)
    projection = np.asarray(raw[:, tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]], dtype=np.float32).mean(axis=0)
    return np.full(projection.shape, float(np.median(projection)), dtype=np.float32)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


__all__ = [
    "LocHarvestConfig",
    "build_roi_bank_from_loc_harvest",
    "extract_old_smlm_emitters_from_tile",
    "loc_harvest_infer_fn",
]
