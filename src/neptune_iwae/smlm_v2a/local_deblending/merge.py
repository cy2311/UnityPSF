from __future__ import annotations

import torch

from ..evaluation.matcher import GreedyHungarianMatching
from ..generic.localizations import FrameLocSet
from .geometry import ConflictPatch


MERGE_STRATEGIES = {
    "replace_all",
    "keep_global_add_local_unmatched",
    "keep_high_conf_global_add_local_unmatched",
    "keep_anchor_global_local_fill",
}


def _proposal_inside_patch(locs: FrameLocSet, patch: ConflictPatch) -> torch.BoolTensor:
    return (
        (locs.indf == int(patch.frame_index))
        & (locs.xyz[:, 0] >= float(patch.x0))
        & (locs.xyz[:, 0] < float(patch.x1))
        & (locs.xyz[:, 1] >= float(patch.y0))
        & (locs.xyz[:, 1] < float(patch.y1))
    )


def _dedupe_by_matcher(locs: FrameLocSet, matcher: GreedyHungarianMatching) -> FrameLocSet:
    if len(locs) <= 1:
        return locs

    cam_pxszs = torch.as_tensor(matcher.cam_pxszs, dtype=locs.f_type)
    xyz_nm = locs.xyz.to(dtype=locs.f_type) * cam_pxszs[None, :]
    keep_mask = torch.zeros((len(locs),), dtype=torch.bool)

    frame_to_indices: dict[int, list[int]] = {}
    for global_ix, frame_index in enumerate(locs.indf.tolist()):
        frame_to_indices.setdefault(int(frame_index), []).append(int(global_ix))

    for frame_indices in frame_to_indices.values():
        kept_global_ix: list[int] = []
        for global_ix in frame_indices:
            if not kept_global_ix:
                keep_mask[global_ix] = True
                kept_global_ix.append(global_ix)
                continue

            kept_ix = torch.tensor(kept_global_ix, dtype=torch.long)
            filter_mask = matcher._filter(
                xyz_nm[global_ix : global_ix + 1],
                xyz_nm[kept_ix],
            )
            if bool(filter_mask.any().item()):
                continue

            keep_mask[global_ix] = True
            kept_global_ix.append(global_ix)

    return locs[keep_mask]


def merge_patch_predictions_with_strategy(
    global_locs: FrameLocSet,
    patch_predictions: list[tuple[ConflictPatch, FrameLocSet]],
    *,
    matcher: GreedyHungarianMatching,
    strategy: str,
    global_prob_threshold: float = 0.8,
) -> FrameLocSet:
    if strategy not in MERGE_STRATEGIES:
        raise ValueError(f"Unsupported merge strategy: {strategy!r}")
    if not patch_predictions:
        return global_locs

    keep_mask = torch.ones((len(global_locs),), dtype=torch.bool)
    anchor_keep_mask = torch.zeros((len(global_locs),), dtype=torch.bool)
    local_parts: list[FrameLocSet] = []

    for patch, local_pred in patch_predictions:
        inside_patch = _proposal_inside_patch(global_locs, patch)
        if strategy == "replace_all":
            keep_mask &= ~inside_patch
        elif strategy == "keep_global_add_local_unmatched":
            pass
        elif strategy == "keep_high_conf_global_add_local_unmatched":
            keep_mask &= ~(inside_patch & (global_locs.prob < float(global_prob_threshold)))
        elif strategy == "keep_anchor_global_local_fill":
            keep_mask &= ~inside_patch
            if 0 <= int(patch.anchor_index) < len(global_locs):
                anchor_keep_mask[int(patch.anchor_index)] = True

        if len(local_pred) > 0:
            local_parts.append(local_pred)

    if strategy == "keep_anchor_global_local_fill":
        keep_mask |= anchor_keep_mask

    survivors = global_locs[keep_mask]
    if strategy == "replace_all":
        ordered_parts = [*local_parts, survivors] if len(survivors) > 0 else list(local_parts)
    else:
        ordered_parts = ([survivors] if len(survivors) > 0 else []) + list(local_parts)

    if not ordered_parts:
        return FrameLocSet(**global_locs.meta)
    if len(ordered_parts) == 1:
        merged = ordered_parts[0]
    else:
        merged = FrameLocSet.loc_cat(ordered_parts)
    return _dedupe_by_matcher(merged, matcher)


def merge_patch_predictions(
    global_locs: FrameLocSet,
    patch_predictions: list[tuple[ConflictPatch, FrameLocSet]],
    *,
    matcher: GreedyHungarianMatching,
) -> FrameLocSet:
    return merge_patch_predictions_with_strategy(
        global_locs,
        patch_predictions,
        matcher=matcher,
        strategy="replace_all",
    )
