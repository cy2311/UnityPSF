"""Shared ROI projection data structures and small sampling helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from unity_psf.localization.posterior import DetectionPosteriorSamples
from unity_psf.localization.scale_resolver import resolve_physical_scales
from unity_psf.roi_library import ROIBank, ROIRecord


@dataclass(frozen=True)
class ROIProjectionUpdateContext:
    bank: ROIBank
    sampling_metrics: dict[str, object]
    raw_frames: torch.Tensor
    background: torch.Tensor
    samples: DetectionPosteriorSamples
    roi_origin_xy_px: torch.Tensor
    domain_names: list[str]
    loss_mask: torch.Tensor | None = None


def posterior_photon_scale(train_cfg: Mapping[str, Any]) -> float | None:
    return resolve_physical_scales(train_cfg).photon_scale


def posterior_z_scale(train_cfg: Mapping[str, Any]) -> float | None:
    return resolve_physical_scales(train_cfg).z_scale


def posterior_z_scale_nm(train_cfg: Mapping[str, Any]) -> float | None:
    return resolve_physical_scales(train_cfg).z_scale_nm


def posterior_bg_scale(train_cfg: Mapping[str, Any]) -> float:
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping) and "bg_max" in scaling:
        return float(scaling["bg_max"])
    normalization = train_cfg.get("normalization")
    if isinstance(normalization, Mapping) and "bg_scale" in normalization:
        return float(normalization["bg_scale"])
    return 1.0


def posterior_input_offset(train_cfg: Mapping[str, Any]) -> float:
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping) and "input_offset" in scaling:
        return float(scaling["input_offset"])
    normalization = train_cfg.get("normalization")
    if isinstance(normalization, Mapping) and "input_offset" in normalization:
        return float(normalization["input_offset"])
    return 0.0


def posterior_input_scale(train_cfg: Mapping[str, Any]) -> float:
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping) and "input_scale" in scaling:
        return float(scaling["input_scale"])
    normalization = train_cfg.get("normalization")
    if isinstance(normalization, Mapping) and "input_scale" in normalization:
        return float(normalization["input_scale"])
    return 1.0


def select_roi_window_frame(model_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    if isinstance(model_input, tuple):
        model_input = model_input[0]
    tensor = torch.as_tensor(model_input, dtype=torch.float32)
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim != 4:
        raise ValueError(f"ROI model_input must have shape (B,H,W) or (B,T,H,W), got {tuple(tensor.shape)}")
    return tensor[:, int(tensor.shape[1] // 2)]


def select_roi_bank_update_subset(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
    *,
    epoch: int,
) -> tuple[ROIBank, dict[str, object]]:
    records = tuple(bank.records)
    if not records:
        return (
            ROIBank(
                records=(),
                config=bank.config,
                metadata={**bank.metadata, "roi_bank_update_subset": True},
                empty_grid_cell_ids=bank.empty_grid_cell_ids,
                format_version=bank.format_version,
            ),
            {
                "sampling_rounds": 0,
                "sample_count": 0,
                "selected_sample_count": 0,
                "sampled_roi_count": 0,
                "sampled_record_count": 0,
                "selected_sampled_emitter_count": 0,
                "sampled_emitter_count": 0,
                "sampled_emitter_count_inner": 0,
                "sampled_emitter_count_total": 0,
                "target_projected_emitters": int(gamma_cfg.get("target_projected_emitters", 0)),
                "target_projected_emitters_overshoot": 0,
                "target_emitters_reached": False,
                "roi_groups_per_update": 0,
                "roi_groups_per_update_reached": False,
                "roi_groups_per_update_overshoot": 0,
                "sampling_stop_reason": "empty_roi_bank",
                "selected_roi_ids": [],
            },
        )

    target_emitters = int(gamma_cfg.get("target_projected_emitters", gamma_cfg.get("target_emitters", len(records))))
    max_rounds = max(1, int(gamma_cfg.get("max_sampling_rounds", 1)))
    seed = int(gamma_cfg.get("seed", 0))
    roi_groups_raw = gamma_cfg.get("roi_groups_per_update")
    roi_groups_per_update = None if roi_groups_raw is None else int(roi_groups_raw)

    selected_records: list[ROIRecord] = []
    sampled_emitters = 0
    rounds = 0
    sampled_roi_ids: set[int] = set()
    sampling_stop_reason = "max_sampling_rounds"
    for round_index in range(max_rounds):
        round_records = list(records)
        round_seed = seed + int(epoch) * 1009 + int(round_index)
        random.Random(round_seed).shuffle(round_records)
        rounds += 1
        for record in round_records:
            selected_records.append(record)
            sampled_roi_ids.add(int(record.roi_id))
            sampled_emitters += len(tuple(record.emitters))
            if roi_groups_per_update is not None and len(selected_records) >= roi_groups_per_update:
                sampling_stop_reason = "roi_groups_per_update"
                break
            if sampled_emitters >= target_emitters:
                sampling_stop_reason = "target_projected_emitters"
                break
        if roi_groups_per_update is not None and len(selected_records) >= roi_groups_per_update:
            break
        if sampled_emitters >= target_emitters:
            break

    subset = ROIBank(
        records=tuple(selected_records),
        config=bank.config,
        metadata={
            **bank.metadata,
            "roi_bank_update_subset": True,
            "roi_bank_update_epoch": int(epoch),
            "roi_bank_update_seed": int(seed),
        },
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    )
    roi_group_target = 0 if roi_groups_per_update is None else int(roi_groups_per_update)
    return subset, {
        "sampling_rounds": int(rounds),
        "sample_count": int(len(selected_records)),
        "selected_sample_count": int(len(selected_records)),
        "sampled_roi_count": int(len(sampled_roi_ids)),
        "sampled_record_count": int(len(selected_records)),
        "selected_sampled_emitter_count": int(sampled_emitters),
        "sampled_emitter_count": int(sampled_emitters),
        "sampled_emitter_count_inner": int(sampled_emitters),
        "sampled_emitter_count_total": int(sampled_emitters),
        "target_projected_emitters": int(target_emitters),
        "target_projected_emitters_overshoot": int(max(0, sampled_emitters - target_emitters)),
        "target_emitters_reached": bool(sampled_emitters >= target_emitters),
        "roi_groups_per_update": roi_group_target,
        "roi_groups_per_update_reached": bool(roi_groups_per_update is not None and len(selected_records) >= roi_groups_per_update),
        "roi_groups_per_update_overshoot": int(
            0 if roi_groups_per_update is None else max(0, len(selected_records) - roi_groups_per_update)
        ),
        "sampling_stop_reason": sampling_stop_reason,
        "selected_roi_ids": [int(record.roi_id) for record in selected_records],
    }


def model_supports_conditioned_input(model: torch.nn.Module) -> bool:
    module = model.module if hasattr(model, "module") and isinstance(model.module, torch.nn.Module) else model
    return hasattr(module, "condition_dim") or hasattr(module, "film_modulator") or hasattr(module, "experts")


def normal_sample(mean: torch.Tensor, sigma: torch.Tensor, *, generator: torch.Generator, stochastic: bool) -> torch.Tensor:
    if not bool(stochastic):
        return mean.clone()
    return mean + torch.randn(mean.shape, generator=generator, dtype=mean.dtype, device=mean.device) * sigma.clamp_min(0.0)


def normal_log_prob(value: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_safe = sigma.clamp_min(1e-12)
    variance = sigma_safe.square()
    return -0.5 * ((value - mean).square() / variance + torch.log(torch.as_tensor(2.0 * np.pi, dtype=value.dtype) * variance))


def inner_xy_mask(xy_px: torch.Tensor, *, roi_size_px: int, over_cut_px: int) -> torch.Tensor:
    if xy_px.numel() == 0:
        return torch.zeros((xy_px.shape[0],), dtype=torch.bool, device=xy_px.device)
    over = max(0, int(over_cut_px))
    if over <= 0:
        return torch.ones((xy_px.shape[0],), dtype=torch.bool, device=xy_px.device)
    roi = int(roi_size_px)
    if roi <= 2 * over:
        raise ValueError("roi_size_px must be larger than 2 * over_cut_px")
    return (
        (xy_px[:, 0] >= float(over))
        & (xy_px[:, 0] < float(roi - over))
        & (xy_px[:, 1] >= float(over))
        & (xy_px[:, 1] < float(roi - over))
    )


def stack_record_loss_masks(records: Sequence[ROIRecord]) -> torch.Tensor | None:
    masks = [record_loss_mask(record) for record in records]
    if not masks or all(mask is None for mask in masks):
        return None
    concrete_masks = []
    for record, mask in zip(records, masks, strict=True):
        if mask is None:
            shape = tuple(int(v) for v in np.asarray(record.background_smoothed).shape)
            concrete_masks.append(torch.ones(shape, dtype=torch.bool))
        else:
            concrete_masks.append(mask)
    return torch.stack(concrete_masks, dim=0)


def record_loss_mask(record: ROIRecord) -> torch.Tensor | None:
    summary = dict(record.summary or {})
    if "valid_core_size_px" not in summary or "valid_core_offset_xy_px" not in summary:
        return None
    height, width = (int(v) for v in np.asarray(record.background_smoothed).shape)
    core = int(summary["valid_core_size_px"])
    offset = summary["valid_core_offset_xy_px"]
    x0 = int(round(float(offset[0])))
    y0 = int(round(float(offset[1])))
    if core <= 0:
        raise ValueError("valid_core_size_px must be positive")
    if x0 < 0 or y0 < 0 or x0 + core > width or y0 + core > height:
        raise ValueError(f"valid core {(x0, y0, core)} exceeds ROI bounds {(width, height)} for roi_id={record.roi_id}")
    mask = torch.zeros((height, width), dtype=torch.bool)
    mask[y0 : y0 + core, x0 : x0 + core] = True
    return mask


def record_observed_frame(record: ROIRecord, *, frame_index: int | None) -> torch.Tensor:
    raw = torch.as_tensor(record.raw_frames_photon, dtype=torch.float32)
    if raw.ndim == 2:
        return raw
    if raw.ndim != 3:
        raise ValueError(f"ROI raw_frames_photon must have shape (T,H,W) or (H,W), got {tuple(raw.shape)}")
    if frame_index is None:
        frame_offset = int(raw.shape[0] // 2)
    else:
        frame_offset = int(frame_index) - int(record.frame_window[0])
    if not (0 <= frame_offset < int(raw.shape[0])):
        raise ValueError(
            f"ROI emitter frame_index={frame_index} is outside frame_window={tuple(record.frame_window)} "
            f"with {int(raw.shape[0])} raw frames"
        )
    return raw[frame_offset]


def record_observed_center_frame(record: ROIRecord) -> torch.Tensor:
    return record_observed_frame(record, frame_index=None)


__all__ = [
    "ROIProjectionUpdateContext",
    "inner_xy_mask",
    "model_supports_conditioned_input",
    "normal_log_prob",
    "normal_sample",
    "posterior_bg_scale",
    "posterior_input_offset",
    "posterior_input_scale",
    "posterior_photon_scale",
    "posterior_z_scale",
    "posterior_z_scale_nm",
    "record_loss_mask",
    "record_observed_center_frame",
    "record_observed_frame",
    "select_roi_bank_update_subset",
    "select_roi_window_frame",
    "stack_record_loss_masks",
]


def record_projection_tensors(bank: ROIBank) -> tuple[torch.Tensor, torch.Tensor, DetectionPosteriorSamples, torch.Tensor, list[str], torch.Tensor | None]:
    records = tuple(bank.records)
    if not records:
        raise ValueError("ROI projection requires at least one ROI record")

    sample_rows: list[tuple[ROIRecord, int, tuple[Any, ...]]] = []
    for record in records:
        emitters_by_frame: dict[int, list[Any]] = {}
        for emitter in record.emitters:
            emitters_by_frame.setdefault(int(emitter.frame_index), []).append(emitter)
        for frame_index in sorted(emitters_by_frame):
            sample_rows.append((record, int(frame_index), tuple(emitters_by_frame[frame_index])))

    if not sample_rows:
        raise ValueError("ROI projection requires persisted ROI emitter records; refusing empty-emitter ROI bank")

    raw_frames = torch.stack([record_observed_frame(record, frame_index=frame_index) for record, frame_index, _ in sample_rows], dim=0)
    background = torch.stack([torch.as_tensor(record.background_smoothed, dtype=torch.float32) for record, _, _ in sample_rows], dim=0)
    max_emitters = max(1, max(len(emitters) for _, _, emitters in sample_rows))
    xyzph = torch.zeros((len(sample_rows), max_emitters, 4), dtype=torch.float32)
    mask = torch.zeros((len(sample_rows), max_emitters), dtype=torch.bool)
    for batch_idx, (_record, _frame_index, emitters) in enumerate(sample_rows):
        for emitter_idx, emitter in enumerate(emitters[:max_emitters]):
            xyzph[batch_idx, emitter_idx] = torch.tensor(
                [
                    float(emitter.local_xy_px[0]),
                    float(emitter.local_xy_px[1]),
                    float(emitter.mu_z_nm),
                    max(float(emitter.mu_photons), 1e-6),
                ],
                dtype=torch.float32,
            )
            mask[batch_idx, emitter_idx] = True
    samples = DetectionPosteriorSamples(
        xyzph=xyzph,
        mask=mask,
        logits=torch.zeros(raw_frames.shape, dtype=torch.float32),
        metadata={
            "source": "roi_record_emitters",
            "roi_id": [int(record.roi_id) for record, _, _ in sample_rows],
            "frame_index": [int(frame_index) for _, frame_index, _ in sample_rows],
        },
    )
    origins = torch.as_tensor([record.roi_origin_xy_px for record, _, _ in sample_rows], dtype=torch.float32)
    domain_names = [str(record.domain_name) for record, _, _ in sample_rows]
    loss_mask = stack_record_loss_masks([record for record, _, _ in sample_rows])
    return raw_frames, background, samples, origins, domain_names, loss_mask
