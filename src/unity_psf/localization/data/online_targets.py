"""Target ordering and padding helpers for online localization batches."""

from __future__ import annotations

import torch

from unity_psf.localization.smlm_targets import (
    V03_PXYZ_TARGET_ORDER,
    normalize_pxyz_target_order,
    pxyz_target_order_tuple as canonical_pxyz_target_order_tuple,
    v03_pxyz_to_legacy_iwae,
)


def pad_pxyz_targets(targets: list[torch.Tensor], masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not targets:
        return torch.zeros((0, 0, 4), dtype=torch.float32), torch.zeros((0, 0), dtype=torch.bool)
    max_count = max(int(target.shape[0]) for target in targets)
    dtype, device = targets[0].dtype, targets[0].device
    padded_targets = torch.zeros((len(targets), max_count, 4), dtype=dtype, device=device)
    padded_masks = torch.zeros((len(targets), max_count), dtype=torch.bool, device=device)
    for sample_idx, (target, mask) in enumerate(zip(targets, masks, strict=True)):
        count = int(target.shape[0])
        if count > 0:
            padded_targets[sample_idx, :count] = target
            padded_masks[sample_idx, :count] = mask.to(dtype=torch.bool, device=device)
    return padded_targets, padded_masks


def finalize_pxyz_targets(config, target: torch.Tensor) -> tuple[torch.Tensor, tuple[str, ...]]:
    order = normalize_pxyz_target_order(config.pxyz_target_order)
    if order != "legacy_iwae":
        return target, V03_PXYZ_TARGET_ORDER
    converted = v03_pxyz_to_legacy_iwae(
        target,
        photon_scale=config.photon_scale,
        z_scale=config.z_scale,
    )
    return converted, canonical_pxyz_target_order_tuple(order)


def pxyz_target_order_tuple(config) -> tuple[str, ...]:
    return canonical_pxyz_target_order_tuple(config.pxyz_target_order)


def detect_from_v03_targets(target: torch.Tensor, mask: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    detect = torch.zeros((int(height), int(width)), dtype=torch.float32)
    if not bool(mask.any()):
        return detect
    active = target[mask]
    cols = torch.round(active[:, 0]).to(dtype=torch.long).clamp_(0, int(width) - 1)
    rows = torch.round(active[:, 1]).to(dtype=torch.long).clamp_(0, int(height) - 1)
    detect[rows, cols] = 1.0
    return detect
