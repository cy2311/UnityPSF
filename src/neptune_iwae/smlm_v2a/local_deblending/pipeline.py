from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch.utils.data import Dataset

from ..generic.localizations import FrameLocSet
from ..residual_refinement.render import MeanPSFCenterRenderer
from .dataset import crop_patch, encode_patch_targets
from .geometry import ConflictPatch


@dataclass(frozen=True)
class PatchExample:
    patch: ConflictPatch
    x: torch.Tensor
    target: torch.Tensor
    mask: torch.BoolTensor


class PatchExampleDataset(Dataset):
    def __init__(self, examples: Sequence[PatchExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        example = self.examples[idx]
        return example.x, example.target, example.mask


def realign_patch(patch: ConflictPatch, frame_offset: int) -> ConflictPatch:
    return replace(patch, frame_index=int(patch.frame_index + frame_offset))


def targets_to_locset(
    pxyz_tar: torch.Tensor,
    mask_tar: torch.BoolTensor,
    *,
    loc_meta: FrameLocSet,
    phot_max: float,
    z_max: float,
    frame_offset: int = 0,
) -> FrameLocSet:
    if pxyz_tar.ndim != 3:
        raise ValueError(f"pxyz_tar must have shape (N, M, 4), got {tuple(pxyz_tar.shape)}")
    if mask_tar.ndim != 2:
        raise ValueError(f"mask_tar must have shape (N, M), got {tuple(mask_tar.shape)}")
    if pxyz_tar.shape[:2] != mask_tar.shape:
        raise ValueError(f"shape mismatch: pxyz_tar={tuple(pxyz_tar.shape)} mask_tar={tuple(mask_tar.shape)}")

    ids: list[torch.Tensor] = []
    indf: list[torch.Tensor] = []
    xyz: list[torch.Tensor] = []
    phot: list[torch.Tensor] = []
    next_id = 0

    pxyz_cpu = pxyz_tar.detach().cpu()
    mask_cpu = mask_tar.detach().cpu()
    for batch_ix in range(int(pxyz_cpu.shape[0])):
        valid = pxyz_cpu[batch_ix, mask_cpu[batch_ix]]
        if valid.numel() == 0:
            continue
        count = int(valid.shape[0])
        ids.append(torch.arange(next_id, next_id + count, dtype=loc_meta.i_type))
        indf.append(torch.full((count,), int(frame_offset + batch_ix), dtype=loc_meta.i_type))
        xyz.append(
            torch.stack(
                (
                    valid[:, 1].to(dtype=loc_meta.f_type),
                    valid[:, 2].to(dtype=loc_meta.f_type),
                    (valid[:, 3] * float(z_max)).to(dtype=loc_meta.f_type),
                ),
                dim=1,
            )
        )
        phot.append((valid[:, 0] * float(phot_max)).to(dtype=loc_meta.f_type))
        next_id += count

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


def build_anchor_render(
    proposals: FrameLocSet,
    patches: Sequence[ConflictPatch],
    *,
    renderer: MeanPSFCenterRenderer,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    device = torch.device(device)
    if not patches or len(proposals) == 0:
        return torch.zeros((batch_size, height, width), dtype=torch.float32, device=device)
    anchor_indices = [int(patch.anchor_index) for patch in patches]
    anchor_locs = proposals[anchor_indices]
    return renderer.render(
        anchor_locs,
        batch_size=int(batch_size),
        height=int(height),
        width=int(width),
        device=device,
    )


def build_patch_examples(
    frames: torch.Tensor,
    anchor_render: torch.Tensor,
    *,
    gt_locs: FrameLocSet,
    patches: Sequence[ConflictPatch],
    max_targets: int,
    phot_max: float,
    z_max: float,
    frame_offset: int = 0,
) -> list[PatchExample]:
    if frames.ndim != 4:
        raise ValueError(f"frames must have shape (N, C, H, W), got {tuple(frames.shape)}")
    if anchor_render.ndim != 3:
        raise ValueError(f"anchor_render must have shape (N, H, W), got {tuple(anchor_render.shape)}")
    if frames.shape[0] != anchor_render.shape[0] or frames.shape[-2:] != anchor_render.shape[-2:]:
        raise ValueError(
            f"shape mismatch: frames={tuple(frames.shape)} anchor_render={tuple(anchor_render.shape)}"
        )

    frames_cpu = frames.detach().cpu()
    anchor_cpu = anchor_render.detach().cpu()
    examples: list[PatchExample] = []
    for patch in patches:
        frame_index = int(patch.frame_index)
        raw_patch = crop_patch(frames_cpu[frame_index], x0=patch.x0, y0=patch.y0, x1=patch.x1, y1=patch.y1)
        anchor_patch = crop_patch(anchor_cpu[frame_index], x0=patch.x0, y0=patch.y0, x1=patch.x1, y1=patch.y1)
        target, mask = encode_patch_targets(
            gt_locs,
            patch,
            max_targets=int(max_targets),
            phot_max=float(phot_max),
            z_max=float(z_max),
        )
        examples.append(
            PatchExample(
                patch=realign_patch(patch, int(frame_offset)),
                x=torch.cat((raw_patch, anchor_patch.unsqueeze(0)), dim=0),
                target=target,
                mask=mask,
            )
        )
    return examples


def decode_patch_predictions(
    output: dict[str, torch.Tensor],
    *,
    patches: Sequence[ConflictPatch],
    loc_meta: FrameLocSet,
    phot_max: float,
    z_max: float,
    prob_threshold: float,
) -> list[FrameLocSet]:
    prob = output["prob"].detach().cpu()
    mu = output["mu"].detach().cpu()
    sig = output["sig"].detach().cpu()
    if prob.ndim != 2:
        raise ValueError(f"prob must have shape (N, K), got {tuple(prob.shape)}")
    if mu.shape[:2] != prob.shape or sig.shape[:2] != prob.shape:
        raise ValueError(
            f"shape mismatch: prob={tuple(prob.shape)} mu={tuple(mu.shape)} sig={tuple(sig.shape)}"
        )
    if int(prob.shape[0]) != len(patches):
        raise ValueError(f"batch mismatch: prob batch={int(prob.shape[0])} patches={len(patches)}")

    out: list[FrameLocSet] = []
    for batch_ix, patch in enumerate(patches):
        keep = prob[batch_ix] >= float(prob_threshold)
        loc = FrameLocSet(**loc_meta.meta)
        if not bool(keep.any()):
            out.append(loc)
            continue

        mu_keep = mu[batch_ix, keep]
        sig_keep = sig[batch_ix, keep]
        prob_keep = prob[batch_ix, keep].to(dtype=loc_meta.f_type)
        count = int(mu_keep.shape[0])
        xyz = torch.stack(
            (
                (mu_keep[:, 1] + float(patch.x0)).to(dtype=loc_meta.f_type),
                (mu_keep[:, 2] + float(patch.y0)).to(dtype=loc_meta.f_type),
                (mu_keep[:, 3] * float(z_max)).to(dtype=loc_meta.f_type),
            ),
            dim=1,
        )
        xyz_sig = torch.stack(
            (
                sig_keep[:, 1].to(dtype=loc_meta.f_type),
                sig_keep[:, 2].to(dtype=loc_meta.f_type),
                (sig_keep[:, 3] * float(z_max)).to(dtype=loc_meta.f_type),
            ),
            dim=1,
        )
        loc.init_load(
            id=torch.arange(count, dtype=loc_meta.i_type),
            prob=prob_keep,
            indf=torch.full((count,), int(patch.frame_index), dtype=loc_meta.i_type),
            xyz=xyz,
            phot=(mu_keep[:, 0] * float(phot_max)).to(dtype=loc_meta.f_type),
            xyz_sig=xyz_sig,
            phot_sig=(sig_keep[:, 0] * float(phot_max)).to(dtype=loc_meta.f_type),
        )
        out.append(loc)
    return out
