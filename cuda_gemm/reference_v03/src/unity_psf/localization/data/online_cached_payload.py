"""Cached sequence target precomputation and CPU storage."""

from __future__ import annotations

import torch
from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.runtime.profiling import time_block
from .online_types import OnlineBatchProviderConfig
from .online_conditioning import condition_vector_for_config
from .online_targets import detect_from_v03_targets
from .online_targets import finalize_pxyz_targets
from .online_targets import pxyz_target_order_tuple


def materialize_cached_sequence_payload_cpu(payload: dict[str, object]) -> None:
    """Move a simulated sequence and all tensor-derived views to pinned CPU memory."""
    can_pin = bool(torch.cuda.is_available())

    def convert(value: object) -> object:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device="cpu")
            if can_pin:
                tensor = tensor.pin_memory()
            return tensor
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    for key, value in tuple(payload.items()):
        if key != "metadata":
            payload[key] = convert(value)


def precompute_cached_sequence_payload(
    config: OnlineBatchProviderConfig,
    payload: dict[str, object],
    *,
    condition_provider: FullResZernikeConditioning,
    condition_source: str,
) -> None:
    window_size = int(config.channels)
    frames = payload.get("frames")
    pxyz = payload.get("pxyz")
    mask = payload.get("mask")
    bkg = payload.get("bkg")
    if not isinstance(frames, torch.Tensor) or not isinstance(pxyz, torch.Tensor) or not isinstance(mask, torch.Tensor) or not isinstance(bkg, torch.Tensor):
        return
    center_count = max(0, int(frames.shape[0]) - window_size + 1)
    if center_count <= 0:
        return
    center_start = window_size // 2
    window_frames = []
    detect_targets = []
    bkg_targets = []
    finalized_targets = []
    finalized_masks = []
    target_order: tuple[str, ...] | None = None
    with time_block("cached_sequence_precompute_windows"):
        for center_offset in range(center_count):
            center_idx = center_start + center_offset
            start = int(center_idx) - window_size // 2
            end = start + window_size
            target = pxyz[int(center_idx)]
            active = mask[int(center_idx)]
            window_frames.append(frames[start:end])
            detect_targets.append(detect_from_v03_targets(target, active, height=int(config.height), width=int(config.width)))
            bkg_targets.append(bkg[int(center_idx)])
            final_target, target_order = finalize_pxyz_targets(config, target[active])
            finalized_targets.append(final_target)
            finalized_masks.append(torch.ones((int(final_target.shape[0]),), dtype=torch.bool, device=final_target.device))
        payload["precomputed_center_start"] = int(center_start)
        payload["precomputed_window_frames"] = torch.stack(window_frames, dim=0).contiguous()
        payload["precomputed_detect"] = torch.stack(detect_targets, dim=0).contiguous()
        payload["precomputed_bkg"] = torch.stack(bkg_targets, dim=0).contiguous()
        payload["precomputed_pxyz"] = tuple(finalized_targets)
        payload["precomputed_mask"] = tuple(finalized_masks)
        payload["precomputed_target_order"] = target_order or pxyz_target_order_tuple(config)
    if str(config.conditioning_mode) == "film":
        with time_block("cached_sequence_precompute_condition"):
            field_origin = tuple(int(v) for v in payload["field_origin"])
            image_device = frames.device
            image_dtype = frames.dtype
            vector = condition_provider.condition_vector_from_xy(
                x0=int(field_origin[0]),
                y0=int(field_origin[1]),
                height=int(config.height),
                width=int(config.width),
                device=image_device,
                dtype=image_dtype,
            )
            payload["precomputed_condition_vector"] = condition_vector_for_config(config, vector).contiguous()
            payload["precomputed_condition_source"] = str(condition_source)
