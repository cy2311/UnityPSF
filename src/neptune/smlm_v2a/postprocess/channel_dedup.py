from __future__ import annotations

import math

import torch

from ..generic.localizations import FrameLocSet


def _empty_like(reference: FrameLocSet) -> FrameLocSet:
    return FrameLocSet(**reference.meta)


def _slice_by_indices(locs: FrameLocSet, keep: list[int]) -> FrameLocSet:
    if not keep:
        return _empty_like(locs)
    return locs[torch.tensor(keep, dtype=torch.int64)]


def deduplicate_channel_locsets(
    left: FrameLocSet,
    right: FrameLocSet,
    *,
    lateral_tol_px: float = 1.0,
    axial_tol_px: float = 0.2,
) -> FrameLocSet:
    if not left.meta_isclose(right):
        raise RuntimeError("left/right locsets must share the same metadata.")
    if len(left) == 0:
        return right.clone()
    if len(right) == 0:
        return left.clone()

    right_keep = [True] * len(right)
    merged_parts = [left.clone()]

    left_frames = left.split_by_frame(loose=False)
    right_frames = right.split_by_frame(loose=False)
    right_by_frame = {int(frame.indf[0].item()): frame for frame in right_frames if len(frame) > 0}

    offset = 0
    for left_frame in left_frames:
        if len(left_frame) == 0:
            continue
        frame_index = int(left_frame.indf[0].item())
        right_frame = right_by_frame.get(frame_index)
        if right_frame is None or len(right_frame) == 0:
            offset += len(left_frame)
            continue
        for right_local_idx in range(len(right_frame)):
            right_xyz = right_frame.xyz[right_local_idx]
            right_prob = float(right_frame.prob[right_local_idx].item())
            matched_left = None
            for left_local_idx in range(len(left_frame)):
                left_xyz = left_frame.xyz[left_local_idx]
                dx = float(left_xyz[0].item() - right_xyz[0].item())
                dy = float(left_xyz[1].item() - right_xyz[1].item())
                dz = float(left_xyz[2].item() - right_xyz[2].item()) if left.ndim == 3 else 0.0
                lateral = math.sqrt(dx * dx + dy * dy)
                if lateral <= lateral_tol_px and abs(dz) <= axial_tol_px:
                    matched_left = left_local_idx
                    break
            if matched_left is None:
                continue
            global_right_idx = int(torch.nonzero(right.indf == frame_index, as_tuple=False)[right_local_idx].item())
            left_prob = float(left_frame.prob[matched_left].item())
            if right_prob > left_prob:
                global_left_idx = offset + matched_left
                keep = [idx for idx in range(len(merged_parts[0])) if idx != global_left_idx]
                merged_parts[0] = _slice_by_indices(merged_parts[0], keep)
                merged_parts.append(right_frame[right_local_idx])
                right_keep[global_right_idx] = False
            else:
                right_keep[global_right_idx] = False
        offset += len(left_frame)

    extra_right = _slice_by_indices(right, [idx for idx, keep in enumerate(right_keep) if keep])
    if len(extra_right) > 0:
        merged_parts.append(extra_right)
    return FrameLocSet.loc_cat(merged_parts)
