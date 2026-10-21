from __future__ import annotations

from typing import Iterable

import torch

from ..evaluation.matcher import GreedyHungarianMatching
from ..generic.localizations import FrameLocSet


def targets_to_locset(
    pxyz_tar: torch.Tensor,
    mask_tar: torch.BoolTensor,
    *,
    loc_meta: FrameLocSet,
    phot_max: float,
    z_max: float,
) -> FrameLocSet:
    """Convert batched GMM targets back into a frame-indexed localization set."""
    if pxyz_tar.ndim != 3:
        raise ValueError(f"pxyz_tar must be (N, M, 4), got {tuple(pxyz_tar.shape)}")
    if mask_tar.shape != pxyz_tar.shape[:2]:
        raise ValueError(
            f"mask_tar must match leading target dims, got mask={tuple(mask_tar.shape)} target={tuple(pxyz_tar.shape)}"
        )

    ids: list[torch.Tensor] = []
    indf: list[torch.Tensor] = []
    xyz: list[torch.Tensor] = []
    phot: list[torch.Tensor] = []
    next_id = 0

    for batch_ix in range(int(pxyz_tar.shape[0])):
        valid_ix = torch.nonzero(mask_tar[batch_ix], as_tuple=False).squeeze(1)
        if valid_ix.numel() == 0:
            continue
        valid = pxyz_tar[batch_ix, valid_ix].detach().cpu().to(dtype=loc_meta.f_type)
        n_loc = int(valid.shape[0])
        ids.append(torch.arange(next_id, next_id + n_loc, dtype=loc_meta.i_type))
        indf.append(torch.full((n_loc,), batch_ix, dtype=loc_meta.i_type))
        xyz_batch = valid[:, 1:4].clone()
        xyz_batch[:, 2] *= float(z_max)
        xyz.append(xyz_batch)
        phot.append(valid[:, 0] * float(phot_max))
        next_id += n_loc

    loc = FrameLocSet(**loc_meta.meta)
    if not ids:
        return loc

    loc.init_load(
        id=torch.cat(ids, dim=0),
        indf=torch.cat(indf, dim=0),
        xyz=torch.cat(xyz, dim=0),
        phot=torch.cat(phot, dim=0),
    )
    return loc


def filter_by_probability(locs: FrameLocSet, threshold: float) -> FrameLocSet:
    """Keep only predictions whose probability is above the acceptance threshold."""
    if len(locs) == 0:
        return locs
    return locs[locs.prob > float(threshold)]


def matched_gt_and_remaining(
    accepted: FrameLocSet,
    remaining_gt: FrameLocSet,
    matcher: GreedyHungarianMatching,
) -> tuple[FrameLocSet, FrameLocSet]:
    """Return matcher-selected GT emitters and the remaining unmatched GT set."""
    if len(remaining_gt) == 0:
        return remaining_gt, remaining_gt
    if len(accepted) == 0:
        return FrameLocSet(**remaining_gt.meta), remaining_gt

    match = matcher.forward(accepted, remaining_gt)
    matched_gt = match.TP_match
    unmatched_gt = match.FN
    return matched_gt, unmatched_gt


def merge_round_predictions(
    rounds: Iterable[FrameLocSet],
    matcher: GreedyHungarianMatching,
) -> FrameLocSet:
    """Merge accepted predictions with earlier rounds taking precedence over later ones."""
    locsets = list(rounds)
    if not locsets:
        raise ValueError("merge_round_predictions requires at least one locset.")
    merged: FrameLocSet | None = None
    for locs in locsets:
        if len(locs) == 0:
            continue
        if merged is None or len(merged) == 0:
            merged = locs
            continue
        match = matcher.forward(locs, merged)
        keep = match.FP
        if len(keep) == 0:
            continue
        merged = FrameLocSet.loc_cat([merged, keep])

    if merged is None:
        return FrameLocSet(**locsets[0].meta)
    return merged


def subtract_center_render(window: torch.Tensor, center_render: torch.Tensor) -> torch.Tensor:
    """Subtract a rendered center-frame contribution from the middle input channel only."""
    if window.ndim != 4:
        raise ValueError(f"window must be (N, C, H, W), got {tuple(window.shape)}")
    if center_render.ndim != 3:
        raise ValueError(
            f"center_render must be (N, H, W), got {tuple(center_render.shape)}"
        )
    if window.shape[0] != center_render.shape[0] or window.shape[-2:] != center_render.shape[-2:]:
        raise ValueError(
            f"shape mismatch: window={tuple(window.shape)} center_render={tuple(center_render.shape)}"
        )
    center_ix = window.shape[1] // 2
    updated = window.clone()
    updated[:, center_ix] = updated[:, center_ix] - center_render
    return updated


def build_next_round_targets(
    accepted: FrameLocSet,
    pxyz_tar: torch.Tensor,
    mask_tar: torch.BoolTensor,
    *,
    loc_meta: FrameLocSet,
    matcher: GreedyHungarianMatching,
    phot_max: float,
    z_max: float,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, FrameLocSet]:
    """Remove matcher-hit GT emitters from the target tensors and rebuild the detect map."""
    if pxyz_tar.ndim != 3:
        raise ValueError(f"pxyz_tar must be (N, M, 4), got {tuple(pxyz_tar.shape)}")
    if mask_tar.shape != pxyz_tar.shape[:2]:
        raise ValueError(
            f"mask_tar must match leading target dims, got mask={tuple(mask_tar.shape)} target={tuple(pxyz_tar.shape)}"
        )

    matched_slot_mask = torch.zeros_like(mask_tar, dtype=torch.bool)
    matched_gt_parts: list[FrameLocSet] = []
    cam_pxszs = torch.as_tensor(matcher.cam_pxszs, dtype=loc_meta.f_type)

    for batch_ix in range(int(pxyz_tar.shape[0])):
        valid_ix = torch.nonzero(mask_tar[batch_ix], as_tuple=False).squeeze(1)
        if valid_ix.numel() == 0:
            continue
        accepted_frame = accepted.get_subframe((batch_ix, batch_ix + 1))
        if len(accepted_frame) == 0:
            continue

        gt_valid = pxyz_tar[batch_ix, valid_ix].detach().cpu().to(dtype=loc_meta.f_type)
        gt_xyz = gt_valid[:, 1:4].clone()
        gt_xyz[:, 2] *= float(z_max)

        xyz_out_nm = accepted_frame.xyz.to(dtype=loc_meta.f_type) * cam_pxszs[None, :]
        xyz_tar_nm = gt_xyz * cam_pxszs[None, :]
        filter_mask = matcher._filter(xyz_out_nm, xyz_tar_nm)
        _tp_idx, tp_match_idx, _tp_out_bool, _tp_tar_bool = matcher._match_kernel(
            xyz_out_nm,
            xyz_tar_nm,
            filter_mask,
        )
        if tp_match_idx.numel() == 0:
            continue

        matched_valid_ix = valid_ix[tp_match_idx.to(dtype=valid_ix.dtype)]
        matched_slot_mask[batch_ix, matched_valid_ix] = True

        matched_xyz = gt_xyz[tp_match_idx]
        matched_phot = gt_valid[tp_match_idx, 0] * float(phot_max)
        part = FrameLocSet(**loc_meta.meta)
        n_loc = int(tp_match_idx.numel())
        part.init_load(
            id=torch.arange(n_loc, dtype=loc_meta.i_type),
            indf=torch.full((n_loc,), batch_ix, dtype=loc_meta.i_type),
            xyz=matched_xyz,
            phot=matched_phot,
        )
        matched_gt_parts.append(part)

    next_mask = mask_tar & (~matched_slot_mask)
    next_pxyz = pxyz_tar.clone()
    next_pxyz[matched_slot_mask] = 0
    next_detect = torch.zeros(
        (pxyz_tar.shape[0], int(height), int(width)),
        dtype=pxyz_tar.dtype,
        device=pxyz_tar.device,
    )
    if bool(next_mask.any()):
        coords = next_pxyz[next_mask]
        batch_ix = torch.nonzero(next_mask, as_tuple=False)[:, 0]
        col = torch.floor(coords[:, 1] + 0.5).to(dtype=torch.int64)
        row = torch.floor(coords[:, 2] + 0.5).to(dtype=torch.int64)
        col = torch.clamp(col, 0, int(width) - 1)
        row = torch.clamp(row, 0, int(height) - 1)
        next_detect[batch_ix, row, col] = 1.0

    if matched_gt_parts:
        matched_gt = FrameLocSet.loc_cat(matched_gt_parts)
    else:
        matched_gt = FrameLocSet(**loc_meta.meta)
    return next_detect, next_pxyz, next_mask, matched_gt
