"""Raw-TIFF tile inference and active-SMLM emitter decoding."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from unity_psf.roi_library import InferredEmitter, RawInferenceResult, ROIBankDomain


def build_model_raw_tiff_infer_fn(
    *,
    model: torch.nn.Module,
    threshold: float,
    max_emitters: int,
    expected_channels: int,
    photon_scale: float | None,
    z_scale: float | None,
    condition_context: Mapping[str, Any] | None = None,
):
    """Build a model-backed inference callback for raw-TIFF ROI harvesting."""
    resolved_context = condition_context or {
        "providers": None,
        "append_domain_onehot": False,
        "domain_names": (),
    }

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
        was_training = model.training
        model.eval()
        emitters: list[InferredEmitter] = []
        background_accum = np.zeros((height, width), dtype=np.float32)
        background_count = np.zeros((height, width), dtype=np.float32)
        tile_size = min(128, height, width)
        overlap = 16 if tile_size > 32 else 0
        providers = resolved_context.get("providers")
        provider = None if providers is None else providers.get(str(domain.name))
        append_domain_onehot = bool(resolved_context.get("append_domain_onehot", False))
        domain_names = tuple(str(name) for name in resolved_context.get("domain_names", ()))
        with torch.no_grad():
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
                        z_scale=z_scale,
                        full_width=width,
                        full_height=height,
                    )
                )
                background = background_from_output_or_raw(output, tile_image)
                background_accum[tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]] += background
                background_count[tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]] += 1.0
        if was_training:
            model.train()
        background = background_accum / np.maximum(background_count, 1.0)
        emitters.sort(key=lambda item: float(item.probability), reverse=True)
        return RawInferenceResult(
            emitters=tuple(emitters[: int(max_emitters)]),
            background_mu=background,
            metadata={"domain": domain.name, "frame_window": frame_window, "source": "loc_infer_raw_tiff"},
        )

    return infer_fn


def iter_inference_tiles(height: int, width: int, *, tile_size: int, overlap_px: int) -> list[dict[str, int]]:
    tiles: list[dict[str, int]] = []
    xs = _tile_starts(width, tile_size, overlap_px)
    ys = _tile_starts(height, tile_size, overlap_px)
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


def spatial_integration(probability: torch.Tensor, *, raw_threshold: float, split_threshold: float) -> torch.Tensor:
    filt = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=probability.dtype,
        device=probability.device,
    ).view(1, 1, 3, 3)
    conv = torch.nn.functional.conv2d(probability.unsqueeze(1), filt, padding=1)
    clipped = torch.where(probability > float(raw_threshold), probability, torch.zeros_like(probability))
    pooled = torch.nn.functional.max_pool2d(clipped.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    max_mask = torch.eq(probability.unsqueeze(1), pooled)
    primary = max_mask.to(dtype=probability.dtype) * conv
    remainder = probability.unsqueeze(1) * (1.0 - max_mask.to(dtype=probability.dtype))
    secondary = torch.where(remainder > float(split_threshold), torch.ones_like(remainder), torch.zeros_like(remainder)) * conv
    return torch.clamp(primary + secondary, min=0.0, max=1.0).squeeze(1)


def emitters_from_active_smlm_tile(
    output: torch.Tensor,
    *,
    tile: Mapping[str, int],
    threshold: float,
    max_emitters: int,
    photon_scale: float | None,
    z_scale: float | None,
    full_width: int,
    full_height: int,
) -> list[InferredEmitter]:
    if not (output.ndim == 4 and int(output.shape[1]) == 10):
        return []
    decoded = output[0].detach().cpu().to(dtype=torch.float32).clone()
    probability = spatial_integration(decoded[0].unsqueeze(0), raw_threshold=0.3, split_threshold=0.6)[0]
    decoded[0] = probability
    scores = probability.reshape(-1)
    if scores.numel() == 0:
        return []
    values, indices = torch.topk(scores, k=min(int(max_emitters), int(scores.numel())))
    tile_width = int(probability.shape[-1])
    emitters: list[InferredEmitter] = []
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        if float(value) < float(threshold):
            continue
        row, col = divmod(int(flat_index), tile_width)
        x = float(tile["x0"]) + float(col) + float(decoded[2, row, col].item())
        y = float(tile["y0"]) + float(row) + float(decoded[3, row, col].item())
        if not (
            float(tile["keep_x0"]) <= x < float(tile["keep_x1"])
            and float(tile["keep_y0"]) <= y < float(tile["keep_y1"])
            and 0.0 <= x < float(full_width)
            and 0.0 <= y < float(full_height)
        ):
            continue
        photons = physical_photons(float(decoded[1, row, col].item()), photon_scale=photon_scale)
        z_nm = physical_z_nm(float(decoded[4, row, col].item()), z_scale=z_scale)
        emitters.append(
            InferredEmitter(
                probability=float(value),
                mu_xy_px=(x, y),
                sigma_xy_px=(float(decoded[6, row, col].item()), float(decoded[7, row, col].item())),
                mu_z_nm=z_nm,
                sigma_z_nm=abs(physical_z_nm(float(decoded[8, row, col].item()), z_scale=z_scale)),
                mu_photons=max(float(photons), 1e-6),
                sigma_photons=max(physical_photons(float(decoded[5, row, col].item()), photon_scale=photon_scale), 1e-6),
                cell_xy_px=(float(tile["x0"] + col), float(tile["y0"] + row),),
            )
        )
    return emitters


def physical_photons(value: float, *, photon_scale: float | None) -> float:
    return float(value) if photon_scale is None else float(value) * float(photon_scale)


def physical_z_nm(value: float, *, z_scale: float | None) -> float:
    if z_scale is None:
        return float(value)
    scale = abs(float(z_scale))
    scale_nm = scale * 1000.0 if scale <= 10.0 else scale
    return float(value) * scale_nm


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


def _tile_starts(size: int, tile_size: int, overlap_px: int) -> list[int]:
    if int(size) <= int(tile_size):
        return [0]
    stride = max(1, int(tile_size) - int(overlap_px))
    starts = list(range(0, int(size) - int(tile_size) + 1, stride))
    end = int(size) - int(tile_size)
    if starts[-1] != end:
        starts.append(end)
    return starts
