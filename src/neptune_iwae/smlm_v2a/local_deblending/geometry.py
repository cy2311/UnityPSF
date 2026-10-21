from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from ..generic.localizations import FrameLocSet


@dataclass(frozen=True)
class ConflictPatch:
    frame_index: int
    anchor_index: int
    proposal_indices: tuple[int, ...]
    x0: int
    y0: int
    x1: int
    y1: int
    patch_size: int


def _image_size_from_extent(extent: tuple[float, float]) -> int:
    return int(round(float(extent[1]) - float(extent[0])))


def _resolve_patch_bounds(center: float, patch_size: int, size_limit: int) -> tuple[int, int]:
    half = patch_size // 2
    start = int(round(float(center))) - half
    start = max(0, min(start, max(size_limit - patch_size, 0)))
    end = min(size_limit, start + patch_size)
    start = max(0, end - patch_size)
    return int(start), int(end)


def _connected_components(indices: List[int], xyz: torch.Tensor, radius_px: float) -> list[list[int]]:
    if not indices:
        return []
    xy = xyz[indices, :2]
    dist = torch.cdist(xy, xy, p=2)
    visited: set[int] = set()
    out: list[list[int]] = []
    for local_ix, global_ix in enumerate(indices):
        if local_ix in visited:
            continue
        stack = [local_ix]
        comp_local: list[int] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp_local.append(cur)
            neigh = torch.nonzero(dist[cur] <= float(radius_px), as_tuple=False).squeeze(1).tolist()
            for nb in neigh:
                if nb not in visited:
                    stack.append(nb)
        out.append([indices[i] for i in sorted(comp_local)])
    return out


def build_conflict_patches(
    proposals: FrameLocSet,
    *,
    conflict_radius_px: float,
    patch_size: int,
    min_component_size: int = 2,
) -> list[ConflictPatch]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if len(proposals) == 0:
        return []

    width = _image_size_from_extent(proposals.xextent)
    height = _image_size_from_extent(proposals.yextent)
    patches: list[ConflictPatch] = []
    active_frames = proposals.active_frames.tolist()
    for frame_index in active_frames:
        frame_mask = proposals.indf == int(frame_index)
        frame_indices = torch.nonzero(frame_mask, as_tuple=False).squeeze(1).tolist()
        comps = _connected_components(frame_indices, proposals.xyz, radius_px=float(conflict_radius_px))
        for comp in comps:
            if len(comp) < int(min_component_size):
                continue
            comp_t = torch.tensor(comp, dtype=torch.long)
            anchor_local_ix = int(torch.argmax(proposals.prob[comp_t]).item())
            anchor_index = int(comp[anchor_local_ix])
            anchor_xyz = proposals.xyz[anchor_index]
            x0, x1 = _resolve_patch_bounds(float(anchor_xyz[0]), int(patch_size), width)
            y0, y1 = _resolve_patch_bounds(float(anchor_xyz[1]), int(patch_size), height)
            patches.append(
                ConflictPatch(
                    frame_index=int(frame_index),
                    anchor_index=anchor_index,
                    proposal_indices=tuple(int(i) for i in comp),
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    patch_size=int(patch_size),
                )
            )
    return patches

