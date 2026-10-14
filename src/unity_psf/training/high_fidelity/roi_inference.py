"""Raw TIFF inference primitives used by high-fidelity ROI-bank building."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


from unity_psf.roi_library import InferredEmitter, ROIBankDomain, RawInferenceResult
from unity_psf.training.evaluation_mode import preserve_module_training_modes


def uses_model_infer(roi_source: Any) -> bool:
    return roi_source.alias == "loc_infer_raw_tiff" or roi_source.candidate_mode == "dense_tile_temporal"


def bright_pixel_infer_fn(
    *,
    domain: ROIBankDomain,
    frame_window: tuple[int, int],
    raw_domain_frames_photon: np.ndarray,
) -> RawInferenceResult:
    projection = np.asarray(raw_domain_frames_photon, dtype=np.float32).mean(axis=0)
    flat_indices = np.argsort(projection.reshape(-1))[::-1][:2]
    height, width = projection.shape
    emitters = []
    for flat_index in flat_indices:
        y, x = divmod(int(flat_index), int(width))
        emitters.append(
            InferredEmitter(
                probability=0.9,
                mu_xy_px=(float(x), float(y)),
                sigma_xy_px=(0.3, 0.3),
                mu_z_nm=0.0,
                sigma_z_nm=10.0,
                mu_photons=float(max(projection[y, x], 1.0)),
                sigma_photons=1.0,
                cell_xy_px=(float(x), float(y)),
            )
        )
    return RawInferenceResult(
        emitters=tuple(emitters),
        background_mu=np.full((height, width), float(np.median(projection)), dtype=np.float32),
        metadata={"domain": domain.name, "frame_window": frame_window},
    )


def model_raw_tiff_infer_fn(
    *,
    model: torch.nn.Module,
    threshold: float,
    max_emitters: int,
    seed: int,
    expected_channels: int,
    photon_scale: float | None,
    z_scale_nm: float | None,
    condition_context: dict[str, Any] | None = None,
):
    del seed
    condition_context = condition_context or {"providers": None, "append_domain_onehot": False, "domain_names": ()}

    def infer_fn(
        *,
        domain: ROIBankDomain,
        frame_window: tuple[int, int],
        raw_domain_frames_photon: np.ndarray,
    ) -> RawInferenceResult:
        raw = np.asarray(raw_domain_frames_photon, dtype=np.float32)
        if raw.ndim != 3:
            raise ValueError(f"raw TIFF inference window must have shape (T,H,W), got {raw.shape}")
        if int(raw.shape[0]) != int(expected_channels):
            raise ValueError(
                "raw TIFF inference window channel count must match model input channels: "
                f"got {int(raw.shape[0])}, expected {int(expected_channels)}"
            )
        _, height, width = raw.shape
        image = torch.as_tensor(raw, dtype=torch.float32)
        device = model_device(model)
        emitters = []
        background_accum = np.zeros((height, width), dtype=np.float32)
        background_count = np.zeros((height, width), dtype=np.float32)
        tile_size = min(128, height, width)
        overlap = 16 if tile_size > 32 else 0
        providers = condition_context.get("providers")
        provider = None if providers is None else providers.get(str(domain.name))
        append_domain_onehot = bool(condition_context.get("append_domain_onehot", False))
        domain_names = tuple(str(name) for name in condition_context.get("domain_names", ()))
        with preserve_module_training_modes(model), torch.no_grad():
            for tile in iter_inference_tiles(height, width, tile_size=tile_size, overlap_px=overlap):
                tile_image = image[:, tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]].unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                )
                model_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = tile_image
                if provider is not None:
                    condition = provider.condition_vector_from_xy(
                        x0=int(tile["x0"]),
                        y0=int(tile["y0"]),
                        height=int(tile["y1"] - tile["y0"]),
                        width=int(tile["x1"] - tile["x0"]),
                        device=device,
                        dtype=tile_image.dtype,
                    )
                    if append_domain_onehot:
                        onehot = torch.zeros((len(domain_names),), dtype=tile_image.dtype, device=device)
                        onehot[list(domain_names).index(str(domain.name))] = 1.0
                        condition = torch.cat((condition, onehot), dim=0)
                    model_input = (tile_image, condition.unsqueeze(0))
                output = model(model_input).detach().to(dtype=torch.float32)
                emitters.extend(
                    emitters_from_active_smlm_tile(
                        output,
                        tile=tile,
                        threshold=float(threshold),
                        max_emitters=int(max_emitters),
                        photon_scale=photon_scale,
                        z_scale_nm=z_scale_nm,
                        frame_window=frame_window,
                        full_width=width,
                        full_height=height,
                    )
                )
                bg = background_from_output_or_raw(output, tile_image)
                background_accum[tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]] += bg
                background_count[tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]] += 1.0
        background = background_accum / np.maximum(background_count, 1.0)
        emitters.sort(key=lambda item: float(item.probability), reverse=True)
        return RawInferenceResult(
            emitters=tuple(emitters[: int(max_emitters)]),
            background_mu=background,
            metadata={"domain": domain.name, "frame_window": frame_window, "source": "loc_infer_raw_tiff"},
        )

    return infer_fn


def tile_starts(size: int, tile_size: int, overlap_px: int) -> list[int]:
    if int(size) <= int(tile_size):
        return [0]
    stride = max(1, int(tile_size) - int(overlap_px))
    starts = list(range(0, int(size) - int(tile_size) + 1, stride))
    end = int(size) - int(tile_size)
    if starts[-1] != end:
        starts.append(end)
    return starts


def iter_inference_tiles(height: int, width: int, *, tile_size: int, overlap_px: int) -> list[dict[str, int]]:
    tiles = []
    xs = tile_starts(width, tile_size, overlap_px)
    ys = tile_starts(height, tile_size, overlap_px)
    left_margin = int(overlap_px) // 2
    right_margin = int(overlap_px) - left_margin
    for iy, y0 in enumerate(ys):
        y1 = min(y0 + int(tile_size), height)
        keep_y0 = y0 if iy == 0 else y0 + left_margin
        keep_y1 = y1 if iy == len(ys) - 1 else y1 - right_margin
        for ix, x0 in enumerate(xs):
            x1 = min(x0 + int(tile_size), width)
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


def emitters_from_active_smlm_tile(
    output: torch.Tensor,
    *,
    tile: dict[str, int],
    threshold: float,
    max_emitters: int,
    photon_scale: float | None,
    z_scale_nm: float | None,
    frame_window: tuple[int, int],
    full_width: int,
    full_height: int,
) -> list[InferredEmitter]:
    if not (output.ndim == 4 and int(output.shape[1]) == 10):
        return []
    out = output[0].detach().cpu().to(dtype=torch.float32).clone()
    p = spatial_integration(out[0].unsqueeze(0), raw_th=0.3, split_th=0.6)[0]
    out[0] = p
    scores = p.reshape(-1)
    if scores.numel() == 0:
        return []
    k = min(int(max_emitters), int(scores.numel()))
    values, indices = torch.topk(scores, k=k)
    tile_w = int(p.shape[-1])
    emitters = []
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        if float(value) < float(threshold):
            continue
        row = int(flat_index) // tile_w
        col = int(flat_index) % tile_w
        x = float(tile["x0"]) + float(col) + float(out[2, row, col].item())
        y = float(tile["y0"]) + float(row) + float(out[3, row, col].item())
        if not (
            float(tile["keep_x0"]) <= x < float(tile["keep_x1"])
            and float(tile["keep_y0"]) <= y < float(tile["keep_y1"])
            and 0.0 <= x < float(full_width)
            and 0.0 <= y < float(full_height)
        ):
            continue
        photons = physical_photons(float(out[1, row, col].item()), photon_scale=photon_scale)
        z_nm = physical_z_nm(float(out[4, row, col].item()), z_scale_nm=z_scale_nm)
        emitters.append(
            InferredEmitter(
                probability=float(value),
                mu_xy_px=(x, y),
                sigma_xy_px=(float(out[6, row, col].item()), float(out[7, row, col].item())),
                mu_z_nm=z_nm,
                sigma_z_nm=abs(physical_z_nm(float(out[8, row, col].item()), z_scale_nm=z_scale_nm)),
                mu_photons=max(float(photons), 1e-6),
                sigma_photons=max(physical_photons(float(out[5, row, col].item()), photon_scale=photon_scale), 1e-6),
                cell_xy_px=(float(tile["x0"] + col), float(tile["y0"] + row)),
            )
        )
    return emitters


def spatial_integration(p: torch.Tensor, *, raw_th: float, split_th: float) -> torch.Tensor:
    filt = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=p.dtype,
        device=p.device,
    ).view(1, 1, 3, 3)
    conv = torch.nn.functional.conv2d(p.unsqueeze(1), filt, padding=1)
    p_clip = torch.where(p > float(raw_th), p, torch.zeros_like(p))
    pool = torch.nn.functional.max_pool2d(p_clip.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    max_mask1 = torch.eq(p.unsqueeze(1), pool)
    p_ps1 = max_mask1.to(dtype=p.dtype) * conv
    p_copy = p.unsqueeze(1) * (1.0 - max_mask1.to(dtype=p.dtype))
    max_mask2 = torch.where(p_copy > float(split_th), torch.ones_like(p_copy), torch.zeros_like(p_copy))
    p_ps2 = max_mask2 * conv
    return torch.clamp(p_ps1 + p_ps2, min=0.0, max=1.0).squeeze(1)


def physical_photons(value: float, *, photon_scale: float | None) -> float:
    return float(value) if photon_scale is None else float(value) * float(photon_scale)


def physical_z_nm(value: float, *, z_scale_nm: float | None) -> float:
    if z_scale_nm is None:
        return float(value)
    return float(value) * float(z_scale_nm)


def background_from_output_or_raw(output: torch.Tensor, tile_image: torch.Tensor) -> np.ndarray:
    if output.ndim == 4 and int(output.shape[1]) == 10:
        return output[0, 9].detach().cpu().numpy().astype(np.float32, copy=False)
    projection = tile_image[0].detach().cpu().mean(dim=0).numpy().astype(np.float32, copy=False)
    return np.full(projection.shape, float(np.median(projection)), dtype=np.float32)


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


__all__ = [
    "background_from_output_or_raw",
    "bright_pixel_infer_fn",
    "emitters_from_active_smlm_tile",
    "iter_inference_tiles",
    "model_device",
    "model_raw_tiff_infer_fn",
    "physical_photons",
    "physical_z_nm",
    "spatial_integration",
    "tile_starts",
    "uses_model_infer",
]
