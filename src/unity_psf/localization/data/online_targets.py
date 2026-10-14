"""Target ordering and padding helpers for online localization batches."""

from __future__ import annotations

import torch

from unity_psf.localization.smlm_targets import (
    SMLMTargetContract,
    normalize_pxyz_target_order,
    pxyz_target_order_tuple as canonical_pxyz_target_order_tuple,
)
from unity_psf.localization.scale_resolver import z_scale_to_nm


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
    contract = target_contract_for_config(config)
    canonical = contract.with_target_order("v03").to_canonical_physical(target)
    return contract.from_canonical_physical(canonical), canonical_pxyz_target_order_tuple(contract.target_order)


def target_contract_for_config(config) -> SMLMTargetContract:
    order = normalize_pxyz_target_order(config.pxyz_target_order)
    z_unit = getattr(config, "z_scale_unit", None)
    if z_unit is None and config.z_scale is not None:
        z_unit = "um" if float(config.z_scale) <= 10.0 else "nm"
    return SMLMTargetContract(
        target_order=order,
        photon_scale=config.photon_scale,
        z_scale=config.z_scale,
        z_scale_nm=None if config.z_scale is None else z_scale_to_nm(float(config.z_scale), unit=z_unit),
        z_unit=z_unit,
        z_activation="tanh",
    ).validate()


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
