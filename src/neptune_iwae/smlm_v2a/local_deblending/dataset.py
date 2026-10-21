from __future__ import annotations

import torch

from ..generic.localizations import FrameLocSet
from .geometry import ConflictPatch


def crop_patch(tensor: torch.Tensor, *, x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError("tensor must have at least 2 spatial dimensions.")
    return tensor[..., y0:y1, x0:x1]


def encode_patch_targets(
    gt_locs: FrameLocSet,
    patch: ConflictPatch,
    *,
    max_targets: int,
    phot_max: float,
    z_max: float,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    if max_targets <= 0:
        raise ValueError("max_targets must be positive.")

    frame_locs = gt_locs.get_subframe((patch.frame_index, patch.frame_index + 1))
    if len(frame_locs) > 0:
        mask_patch = (
            (frame_locs.xyz[:, 0] >= patch.x0)
            & (frame_locs.xyz[:, 0] < patch.x1)
            & (frame_locs.xyz[:, 1] >= patch.y0)
            & (frame_locs.xyz[:, 1] < patch.y1)
        )
        frame_locs = frame_locs[mask_patch]

    pxyz_tar = torch.zeros((max_targets, 4), dtype=gt_locs.f_type)
    mask = torch.zeros((max_targets,), dtype=torch.bool)
    if len(frame_locs) == 0:
        return pxyz_tar, mask

    n_take = min(len(frame_locs), max_targets)
    xyz = frame_locs.xyz[:n_take]
    phot = frame_locs.phot[:n_take]
    pxyz_tar[:n_take, 0] = phot / max(float(phot_max), 1e-6)
    pxyz_tar[:n_take, 1] = xyz[:, 0] - float(patch.x0)
    pxyz_tar[:n_take, 2] = xyz[:, 1] - float(patch.y0)
    pxyz_tar[:n_take, 3] = xyz[:, 2] / max(float(z_max), 1e-6)
    mask[:n_take] = True
    return pxyz_tar, mask

