"""Posterior sampling and projection contexts for high-fidelity Gamma updates."""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from unity_psf.localization.posterior import DetectionPosteriorSamples
from unity_psf.localization.roi_posterior_sampling import CurrentROILibraryPosteriorSampler, ROILibraryConditionBuilder, ROIPosteriorSamplingConfig, SampledEmitterSet
from unity_psf.roi_library import ROIBank, ROIRecord
from unity_psf.roi_library.loc_harvest import _build_inference_frame_proc
from unity_psf.runtime import profiling
from unity_psf.training.high_fidelity.raw_tiff_inference import model_device
from unity_psf.training.high_fidelity.roi_bank_source import camera_backward_from_training_config, posterior_bg_scale, posterior_input_offset, posterior_input_scale, posterior_photon_scale, posterior_z_scale, roi_conditioning_context
__all__ = ["ROIProjectionUpdateContext", "record_projection_tensors", "record_observed_center_frame", "record_observed_frame", "select_roi_window_frame", "select_roi_bank_update_subset", "sample_roi_bank_posterior_update", "sample_roi_bank_posterior_update_from_cached_emitters", "sample_roi_bank_posterior_update_from_current_model", "sample_roi_bank_importance_update_context", "z_range_nm_from_gamma_config"]
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
def sample_roi_bank_posterior_update(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
    *,
    epoch: int,
) -> tuple[ROIProjectionUpdateContext, dict[str, object]]:
    return sample_roi_bank_posterior_update_from_cached_emitters(
        bank,
        gamma_cfg,
        epoch=epoch,
    )
def sample_roi_bank_posterior_update_from_cached_emitters(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
    *,
    epoch: int,
) -> tuple[ROIProjectionUpdateContext, dict[str, object]]:
    with profiling.time_block("posterior_cached_setup"):
        records = tuple(bank.records)
        if not records:
            raise ValueError("posterior ROI-bank update requires at least one ROI record")
        target_emitters = int(gamma_cfg.get("target_projected_emitters", gamma_cfg.get("target_emitters", len(records))))
        max_rounds = max(1, int(gamma_cfg.get("max_sampling_rounds", 1)))
        seed = int(gamma_cfg.get("seed", 0))
        roi_groups_raw = gamma_cfg.get("roi_groups_per_update")
        roi_groups_per_update = None if roi_groups_raw is None else int(roi_groups_raw)
        sample_count = int(gamma_cfg.get("num_posterior_samples", 1))
        probability_threshold = float(gamma_cfg.get("posterior_probability_threshold", gamma_cfg.get("probability_threshold", 0.5)))
        stochastic_existence = bool(gamma_cfg.get("stochastic_existence", False))
        sample_continuous = bool(gamma_cfg.get("sample_continuous", True))
        roi_size_px = int(gamma_cfg.get("roi_size_px", 128))
        over_cut_px = int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0)))
        min_photons = float(gamma_cfg.get("min_photons", 1e-3))
        z_range_nm = z_range_nm_from_gamma_config(gamma_cfg)
    sample_rows: list[dict[str, Any]] = []
    sampled_emitters = 0
    total_emitters = 0
    boundary_dropped = 0
    empty_posterior_rows_dropped = 0
    sampled_roi_ids: set[int] = set()
    sampled_record_count = 0
    rounds = 0
    sampling_stop_reason = "max_sampling_rounds"
    group_base = 0
    for round_index in range(max_rounds):
        round_records = list(records)
        round_seed = seed + int(epoch) * 1009 + int(round_index)
        random.Random(round_seed).shuffle(round_records)
        rounds += 1
        for record_index, record in enumerate(round_records):
            with profiling.time_block("posterior_cached_record_sampling"):
                rows = _sample_record_posterior_rows(
                    record,
                    num_posterior_samples=sample_count,
                    probability_threshold=probability_threshold,
                    stochastic_existence=stochastic_existence,
                    sample_continuous=sample_continuous,
                    seed=round_seed + int(record_index) * 1000003,
                    roi_size_px=roi_size_px,
                    z_range_nm=z_range_nm,
                    min_photons=min_photons,
                    over_cut_px=over_cut_px,
                    posterior_group_id_base=group_base,
                )
            group_base += max(1, len({int(row["posterior_group_id"]) for row in rows}))
            empty_posterior_rows_dropped += sum(1 for row in rows if int(row["emitter_count"]) <= 0)
            rows = [row for row in rows if int(row["emitter_count"]) > 0]
            if not rows:
                continue
            sample_rows.extend(rows)
            sampled_record_count += 1
            sampled_roi_ids.add(int(record.roi_id))
            sampled_emitters += sum(int(row["emitter_count"]) for row in rows)
            total_emitters += sum(int(row["emitter_count_total"]) for row in rows)
            boundary_dropped += sum(int(row["boundary_emitter_dropped"]) for row in rows)
            if roi_groups_per_update is not None and sampled_record_count >= roi_groups_per_update:
                sampling_stop_reason = "roi_groups_per_update"
                break
            if sampled_emitters >= target_emitters:
                sampling_stop_reason = "target_projected_emitters"
                break
        if roi_groups_per_update is not None and sampled_record_count >= roi_groups_per_update:
            break
        if sampled_emitters >= target_emitters:
            break
    if not sample_rows:
        raise ValueError("posterior ROI-bank update selected no samples")
    with profiling.time_block("posterior_cached_context_build"):
        context = _posterior_rows_to_context(
            sample_rows,
            bank=bank,
            sampling_metrics={},
        )
    group_sizes: dict[int, int] = {}
    for row in sample_rows:
        group_sizes[int(row["posterior_group_id"])] = group_sizes.get(int(row["posterior_group_id"]), 0) + 1
    roi_group_target = 0 if roi_groups_per_update is None else int(roi_groups_per_update)
    metrics: dict[str, object] = {
        "sampling_rounds": int(rounds),
        "sample_count": int(len(sample_rows)),
        "selected_sample_count": int(len(sample_rows)),
        "sampled_roi_count": int(len(sampled_roi_ids)),
        "sampled_record_count": int(sampled_record_count),
        "selected_sampled_emitter_count": int(sampled_emitters),
        "sampled_emitter_count": int(sampled_emitters),
        "sampled_emitter_count_inner": int(sampled_emitters),
        "sampled_emitter_count_total": int(total_emitters),
        "boundary_emitter_dropped": int(boundary_dropped),
        "empty_posterior_rows_dropped": int(empty_posterior_rows_dropped),
        "target_projected_emitters": int(target_emitters),
        "target_projected_emitters_overshoot": int(max(0, sampled_emitters - target_emitters)),
        "target_emitters_reached": bool(sampled_emitters >= target_emitters),
        "roi_groups_per_update": roi_group_target,
        "roi_groups_per_update_reached": bool(roi_groups_per_update is not None and sampled_record_count >= roi_groups_per_update),
        "roi_groups_per_update_overshoot": int(0 if roi_groups_per_update is None else max(0, sampled_record_count - roi_groups_per_update)),
        "sampling_stop_reason": sampling_stop_reason,
        "selected_roi_ids": sorted(int(value) for value in sampled_roi_ids),
        "projection_sample_source": "roi_record_posterior_samples",
        "posterior_sampling_source": "roi_record_posterior_fields_3052_style",
        "posterior_group_count": int(len(group_sizes)),
        "posterior_group_size_min": int(min(group_sizes.values())) if group_sizes else 0,
        "posterior_group_size_max": int(max(group_sizes.values())) if group_sizes else 0,
        "posterior_group_size_mean": float(sum(group_sizes.values()) / max(1, len(group_sizes))),
        "posterior_log_q_available_count": int(sum(1 for row in sample_rows if row.get("log_q_h_given_x") is not None)),
        "num_posterior_samples": int(sample_count),
        "sample_continuous": bool(sample_continuous),
        "stochastic_existence": bool(stochastic_existence),
    }
    context = ROIProjectionUpdateContext(
        bank=context.bank,
        sampling_metrics=metrics,
        raw_frames=context.raw_frames,
        background=context.background,
        samples=context.samples,
        roi_origin_xy_px=context.roi_origin_xy_px,
        domain_names=context.domain_names,
        loss_mask=context.loss_mask,
    )
    return context, metrics
def sample_roi_bank_posterior_update_from_current_model(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
    step_index: int,
) -> tuple[ROIProjectionUpdateContext, dict[str, object]]:
    records = tuple(bank.records)
    if not records:
        raise ValueError("posterior ROI-bank update requires at least one ROI record")
    roi_conditioning = roi_conditioning_context(train_cfg)
    condition_builder = None
    supports_conditioned_input = _model_supports_conditioned_input(model)
    if roi_conditioning["providers"] and supports_conditioned_input:
        condition_builder = ROILibraryConditionBuilder(
            providers_by_domain=dict(roi_conditioning["providers"]),
            append_domain_onehot=bool(roi_conditioning["append_domain_onehot"]),
            domain_names=tuple(str(name) for name in roi_conditioning["domain_names"]),
            condition_feature_dim=roi_conditioning["condition_feature_dim"],
            condition_dim=roi_conditioning["condition_dim"],
        )
    normalization_cfg = train_cfg.get("normalization")
    frame_proc = _build_inference_frame_proc({"normalization": dict(normalization_cfg)}) if isinstance(normalization_cfg, Mapping) else None
    z_scale = posterior_z_scale(train_cfg)
    z_scale_nm = 1.0 if z_scale is None else (abs(float(z_scale)) * 1000.0 if abs(float(z_scale)) <= 10.0 else abs(float(z_scale)))
    sampler = CurrentROILibraryPosteriorSampler(
        model=model,
        device=model_device(model),
        config=ROIPosteriorSamplingConfig(
            probability_threshold=float(
                gamma_cfg.get("posterior_probability_threshold", gamma_cfg.get("probability_threshold", 0.5))
            ),
            candidate_probability_threshold=float(
                gamma_cfg.get("posterior_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))
            ),
            adjacent_probability_threshold=float(
                gamma_cfg.get("posterior_adjacent_probability_threshold", gamma_cfg.get("split_threshold", 0.6))
            ),
            num_posterior_samples=int(gamma_cfg.get("num_posterior_samples", 1)),
            target_projected_emitters=int(gamma_cfg.get("target_projected_emitters", gamma_cfg.get("target_emitters", len(records)))),
            roi_groups_per_update=(
                None if gamma_cfg.get("roi_groups_per_update") is None else int(gamma_cfg.get("roi_groups_per_update"))
            ),
            max_emitters_per_roi=(
                None if gamma_cfg.get("max_emitters_per_roi") is None else int(gamma_cfg.get("max_emitters_per_roi"))
            ),
            max_sampling_rounds=max(1, int(gamma_cfg.get("max_sampling_rounds", 1))),
            roi_size_px=int(gamma_cfg.get("roi_size_px", 128)),
            sample_continuous=bool(gamma_cfg.get("sample_continuous", True)),
            stochastic_existence=bool(gamma_cfg.get("stochastic_existence", False)),
            min_photons=float(gamma_cfg.get("min_photons", 1e-3)),
            z_range_nm=z_range_nm_from_gamma_config(gamma_cfg),
            batch_size=max(1, int(gamma_cfg.get("posterior_batch_size", gamma_cfg.get("batch_size", 8)))),
            background_smoothing_kernel=int(
                gamma_cfg.get("roi_bank_background_smoothing_kernel", gamma_cfg.get("background_smoothing_kernel", 3))
            ),
            input_offset=posterior_input_offset(train_cfg),
            input_scale=posterior_input_scale(train_cfg),
            photon_scale=posterior_photon_scale(train_cfg) or 1.0,
            z_scale_nm=float(z_scale_nm),
            background_scale=posterior_bg_scale(train_cfg),
            seed=int(gamma_cfg.get("seed", 0)),
            camera_backward=camera_backward_from_training_config(train_cfg),
            over_cut_px=int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0))),
        ),
        condition_builder=condition_builder,
        frame_proc=frame_proc,
    )
    with profiling.time_block("posterior_current_model_sampler_total"):
        samples, metrics = sampler(records, step_index=int(step_index))
    with profiling.time_block("posterior_current_model_filter_nonempty"):
        nonempty_samples = tuple(sample for sample in samples if int(sample.emitter_count) > 0)
        empty_posterior_rows_dropped = int(len(samples) - len(nonempty_samples))
    if not nonempty_samples:
        raise ValueError("posterior ROI-bank update selected no samples")
    nonempty_roi_ids = sorted({int(sample.roi_id) for sample in nonempty_samples})
    metrics = {
        **metrics,
        "sample_count": int(len(nonempty_samples)),
        "selected_sample_count": int(len(nonempty_samples)),
        "sampled_roi_count": int(len(nonempty_roi_ids)),
        "sampled_record_count": int(len(nonempty_roi_ids)),
        "selected_sampled_emitter_count": int(metrics.get("sampled_emitter_count", 0)),
        "selected_roi_ids": nonempty_roi_ids,
        "empty_posterior_rows_dropped": int(empty_posterior_rows_dropped),
        "posterior_log_q_available_count": int(sum(1 for sample in nonempty_samples if sample.log_q_h_given_x is not None)),
    }
    with profiling.time_block("posterior_current_model_context_build"):
        context = _sampled_emitter_sets_to_context(nonempty_samples, bank=bank, sampling_metrics=metrics)
    return context, metrics
def sample_roi_bank_importance_update_context(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
    step_index: int,
) -> tuple[ROIProjectionUpdateContext, dict[str, object]]:
    try:
        return sample_roi_bank_posterior_update_from_current_model(
            bank,
            gamma_cfg,
            model=model,
            train_cfg=train_cfg,
            step_index=int(step_index),
        )
    except ValueError as exc:
        if "selected no samples" not in str(exc):
            raise
    context, metrics = sample_roi_bank_posterior_update_from_cached_emitters(
        bank,
        gamma_cfg,
        epoch=int(step_index),
    )
    metrics = {
        **metrics,
        "posterior_sampling_fallback": "roi_record_emitters_after_empty_current_model",
        "current_model_empty_posterior": True,
    }
    return (
        ROIProjectionUpdateContext(
            bank=context.bank,
            sampling_metrics=metrics,
            raw_frames=context.raw_frames,
            background=context.background,
            samples=context.samples,
            roi_origin_xy_px=context.roi_origin_xy_px,
            domain_names=context.domain_names,
            loss_mask=context.loss_mask,
        ),
        metrics,
    )
def _sample_record_posterior_rows(
    record: ROIRecord,
    *,
    num_posterior_samples: int,
    probability_threshold: float,
    stochastic_existence: bool,
    sample_continuous: bool,
    seed: int,
    roi_size_px: int,
    z_range_nm: tuple[float, float] | None,
    min_photons: float,
    over_cut_px: int,
    posterior_group_id_base: int,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    emitters = tuple(record.emitters)
    if emitters:
        frame_index = torch.tensor([int(em.frame_index) for em in emitters], dtype=torch.int64)
        probability = torch.tensor([float(em.probability) for em in emitters], dtype=torch.float32)
        cell_xy = torch.tensor([em.cell_xy_px for em in emitters], dtype=torch.float32)
        mu_xy = torch.tensor([em.mu_xy_px for em in emitters], dtype=torch.float32)
        sigma_xy = torch.tensor([em.sigma_xy_px for em in emitters], dtype=torch.float32)
        mu_z = torch.tensor([float(em.mu_z_nm) for em in emitters], dtype=torch.float32)
        sigma_z = torch.tensor([float(em.sigma_z_nm) for em in emitters], dtype=torch.float32)
        mu_photons = torch.tensor([float(em.mu_photons) for em in emitters], dtype=torch.float32)
        sigma_photons = torch.tensor([float(em.sigma_photons) for em in emitters], dtype=torch.float32)
    else:
        frame_index = torch.empty((0,), dtype=torch.int64)
        probability = torch.empty((0,), dtype=torch.float32)
        cell_xy = torch.empty((0, 2), dtype=torch.float32)
        mu_xy = torch.empty((0, 2), dtype=torch.float32)
        sigma_xy = torch.empty((0, 2), dtype=torch.float32)
        mu_z = torch.empty((0,), dtype=torch.float32)
        sigma_z = torch.empty((0,), dtype=torch.float32)
        mu_photons = torch.empty((0,), dtype=torch.float32)
        sigma_photons = torch.empty((0,), dtype=torch.float32)
    active_base = probability >= float(probability_threshold)
    active_frames = torch.unique(frame_index[active_base], sorted=True)
    if active_frames.numel() == 0:
        active_frames = torch.as_tensor([int(record.frame_window[0] + (record.frame_window[1] - record.frame_window[0]) // 2)])
    frame_to_group = {int(frame_value): int(posterior_group_id_base) + offset for offset, frame_value in enumerate(active_frames.tolist())}
    roi_max = max(0.0, float(roi_size_px) - 1.0)
    rows: list[dict[str, Any]] = []
    for sample_index in range(int(num_posterior_samples)):
        active = active_base
        if bool(stochastic_existence) and probability.numel() > 0:
            active = active & (torch.rand(probability.shape, generator=generator) < probability.clamp(0.0, 1.0))
        for frame_value in active_frames.tolist():
            frame_mask = active & (frame_index == int(frame_value))
            xy = _normal_sample(mu_xy[frame_mask], sigma_xy[frame_mask], generator=generator, stochastic=sample_continuous).clamp(0.0, roi_max)
            z = _normal_sample(mu_z[frame_mask], sigma_z[frame_mask], generator=generator, stochastic=sample_continuous)
            if z_range_nm is not None:
                z = z.clamp(float(z_range_nm[0]), float(z_range_nm[1]))
            photons = _normal_sample(mu_photons[frame_mask], sigma_photons[frame_mask], generator=generator, stochastic=sample_continuous).clamp_min(float(min_photons))
            cells = cell_xy[frame_mask].clone()
            total = int(cells.shape[0])
            if bool(sample_continuous) and total > 0:
                log_q_values = (
                    _normal_log_prob(xy[:, 0], mu_xy[frame_mask][:, 0], sigma_xy[frame_mask][:, 0])
                    + _normal_log_prob(xy[:, 1], mu_xy[frame_mask][:, 1], sigma_xy[frame_mask][:, 1])
                    + _normal_log_prob(z, mu_z[frame_mask], sigma_z[frame_mask])
                    + _normal_log_prob(photons, mu_photons[frame_mask], sigma_photons[frame_mask])
                )
            else:
                log_q_values = torch.empty((total,), dtype=torch.float32)
            inner = _inner_xy_mask(cells, roi_size_px=roi_size_px, over_cut_px=over_cut_px)
            xy = xy[inner]
            z = z[inner]
            photons = photons[inner]
            cells = cells[inner]
            log_q_h_given_x = float(log_q_values[inner].sum().item()) if bool(sample_continuous) and total > 0 else None
            frame_offset = int(frame_value) - int(record.frame_window[0])
            rows.append(
                {
                    "record": record,
                    "sample_index": int(sample_index),
                    "frame_index": int(frame_value),
                    "frame_offset": int(frame_offset),
                    "posterior_group_id": int(frame_to_group[int(frame_value)]),
                    "xy": xy.detach().cpu(),
                    "z": z.detach().cpu(),
                    "photons": photons.detach().cpu(),
                    "cell_xy": cells.detach().cpu(),
                    "emitter_count": int(xy.shape[0]),
                    "emitter_count_total": int(total),
                    "boundary_emitter_dropped": int(max(0, total - int(xy.shape[0]))),
                    "log_q_h_given_x": log_q_h_given_x,
                }
            )
    return rows
def _posterior_rows_to_context(
    rows: list[dict[str, Any]],
    *,
    bank: ROIBank,
    sampling_metrics: dict[str, object],
) -> ROIProjectionUpdateContext:
    if not rows:
        raise ValueError("posterior rows must be non-empty")
    with profiling.time_block("posterior_rows_context_raw_stack"):
        raw_frames = torch.stack([record_observed_frame(row["record"], frame_index=int(row["frame_index"])) for row in rows], dim=0)
    with profiling.time_block("posterior_rows_context_background_stack"):
        background = torch.stack([torch.as_tensor(row["record"].background_smoothed, dtype=torch.float32) for row in rows], dim=0)
    with profiling.time_block("posterior_rows_context_xyzph_pack"):
        max_emitters = max(1, max(int(row["emitter_count"]) for row in rows))
        xyzph = torch.zeros((len(rows), max_emitters, 4), dtype=torch.float32)
        mask = torch.zeros((len(rows), max_emitters), dtype=torch.bool)
        for batch_idx, row in enumerate(rows):
            count = int(row["emitter_count"])
            if count <= 0:
                continue
            xyzph[batch_idx, :count, 0:2] = row["xy"]
            xyzph[batch_idx, :count, 2] = row["z"]
            xyzph[batch_idx, :count, 3] = row["photons"]
            mask[batch_idx, :count] = True
    samples = DetectionPosteriorSamples(
        xyzph=xyzph,
        mask=mask,
        logits=torch.zeros(raw_frames.shape, dtype=torch.float32),
        metadata={
            "source": "roi_record_posterior_samples",
            "log_q_h_given_x": [None if row.get("log_q_h_given_x") is None else float(row["log_q_h_given_x"]) for row in rows],
            "posterior_group_id": [int(row["posterior_group_id"]) for row in rows],
            "roi_id": [int(row["record"].roi_id) for row in rows],
            "frame_index": [int(row["frame_index"]) for row in rows],
        },
    )
    with profiling.time_block("posterior_rows_context_metadata"):
        origins = torch.as_tensor([row["record"].roi_origin_xy_px for row in rows], dtype=torch.float32)
        domain_names = [str(row["record"].domain_name) for row in rows]
        loss_mask = _stack_record_loss_masks([row["record"] for row in rows])
        selected_records = tuple({int(row["record"].roi_id): row["record"] for row in rows}.values())
    return ROIProjectionUpdateContext(
        bank=ROIBank(
            records=selected_records,
            config=bank.config,
            metadata={**bank.metadata, "roi_bank_update_subset": True, "roi_bank_projection_sample_source": "posterior_samples"},
            empty_grid_cell_ids=bank.empty_grid_cell_ids,
            format_version=bank.format_version,
        ),
        sampling_metrics=sampling_metrics,
        raw_frames=raw_frames,
        background=background,
        samples=samples,
        roi_origin_xy_px=origins,
        domain_names=domain_names,
        loss_mask=loss_mask,
    )
def _sampled_emitter_sets_to_context(
    samples: Sequence[SampledEmitterSet],
    *,
    bank: ROIBank,
    sampling_metrics: dict[str, object],
) -> ROIProjectionUpdateContext:
    if not samples:
        raise ValueError("posterior samples must be non-empty")
    records_by_id = {int(record.roi_id): record for record in bank.records}
    with profiling.time_block("posterior_samples_context_raw_stack"):
        raw_frames = torch.stack(
            [record_observed_frame(records_by_id[int(sample.roi_id)], frame_index=int(sample.frame_index)) for sample in samples],
            dim=0,
        )
    with profiling.time_block("posterior_samples_context_background_stack"):
        background = torch.stack(
            [
                (
                    torch.as_tensor(sample.background_smoothed, dtype=torch.float32)
                    if sample.background_smoothed is not None
                    else torch.as_tensor(records_by_id[int(sample.roi_id)].background_smoothed, dtype=torch.float32)
                )
                for sample in samples
            ],
            dim=0,
        )
    with profiling.time_block("posterior_samples_context_xyzph_pack"):
        max_emitters = max(1, max(int(sample.emitter_count) for sample in samples))
        xyzph = torch.zeros((len(samples), max_emitters, 4), dtype=torch.float32)
        mask = torch.zeros((len(samples), max_emitters), dtype=torch.bool)
        for batch_idx, sample in enumerate(samples):
            count = int(sample.emitter_count)
            if count <= 0:
                continue
            xyzph[batch_idx, :count, 0:2] = sample.xy_px
            xyzph[batch_idx, :count, 2] = sample.z_nm
            xyzph[batch_idx, :count, 3] = sample.photons
            mask[batch_idx, :count] = True
    detection_samples = DetectionPosteriorSamples(
        xyzph=xyzph,
        mask=mask,
        logits=torch.zeros(raw_frames.shape, dtype=torch.float32),
        metadata={
            "source": "roi_record_posterior_samples",
            "log_q_h_given_x": [sample.log_q_h_given_x for sample in samples],
            "posterior_group_id": [int(sample.metrics.get("posterior_group_id", 0)) for sample in samples],
            "roi_id": [int(sample.roi_id) for sample in samples],
            "frame_index": [int(sample.frame_index) for sample in samples],
        },
    )
    with profiling.time_block("posterior_samples_context_metadata"):
        origins = torch.as_tensor(
            [records_by_id[int(sample.roi_id)].roi_origin_xy_px for sample in samples],
            dtype=torch.float32,
        )
        domain_names = [str(sample.domain_name) for sample in samples]
        loss_mask = _stack_record_loss_masks([records_by_id[int(sample.roi_id)] for sample in samples])
        selected_records = tuple({records_by_id[int(sample.roi_id)].roi_id: records_by_id[int(sample.roi_id)] for sample in samples}.values())
    return ROIProjectionUpdateContext(
        bank=ROIBank(
            records=selected_records,
            config=bank.config,
            metadata={**bank.metadata, "roi_bank_update_subset": True, "roi_bank_projection_sample_source": "posterior_samples"},
            empty_grid_cell_ids=bank.empty_grid_cell_ids,
            format_version=bank.format_version,
        ),
        sampling_metrics=sampling_metrics,
        raw_frames=raw_frames,
        background=background,
        samples=detection_samples,
        roi_origin_xy_px=origins,
        domain_names=domain_names,
        loss_mask=loss_mask,
    )
def _model_supports_conditioned_input(model: torch.nn.Module) -> bool:
    module = model.module if hasattr(model, "module") and isinstance(model.module, torch.nn.Module) else model
    return hasattr(module, "condition_dim") or hasattr(module, "film_modulator") or hasattr(module, "experts")
def _normal_sample(mean: torch.Tensor, sigma: torch.Tensor, *, generator: torch.Generator, stochastic: bool) -> torch.Tensor:
    if not bool(stochastic):
        return mean.clone()
    return mean + torch.randn(mean.shape, generator=generator, dtype=mean.dtype, device=mean.device) * sigma.clamp_min(0.0)
def _normal_log_prob(value: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_safe = sigma.clamp_min(1e-12)
    variance = sigma_safe.square()
    return -0.5 * ((value - mean).square() / variance + torch.log(torch.as_tensor(2.0 * np.pi, dtype=value.dtype) * variance))
def _inner_xy_mask(xy_px: torch.Tensor, *, roi_size_px: int, over_cut_px: int) -> torch.Tensor:
    if xy_px.numel() == 0:
        return torch.zeros((xy_px.shape[0],), dtype=torch.bool, device=xy_px.device)
    over = max(0, int(over_cut_px))
    if over <= 0:
        return torch.ones((xy_px.shape[0],), dtype=torch.bool, device=xy_px.device)
    roi = int(roi_size_px)
    if roi <= 2 * over:
        raise ValueError("roi_size_px must be larger than 2 * over_cut_px")
    return (xy_px[:, 0] >= float(over)) & (xy_px[:, 0] < float(roi - over)) & (xy_px[:, 1] >= float(over)) & (xy_px[:, 1] < float(roi - over))
def _stack_record_loss_masks(records: Sequence[ROIRecord]) -> torch.Tensor | None:
    masks = [_record_loss_mask(record) for record in records]
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
def _record_loss_mask(record: ROIRecord) -> torch.Tensor | None:
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
def z_range_nm_from_gamma_config(gamma_cfg: Mapping[str, Any]) -> tuple[float, float] | None:
    raw = gamma_cfg.get("z_range_nm")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("train.roi_bank_gamma.z_range_nm must be a two-element range")
    return (float(raw[0]), float(raw[1]))
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
    loss_mask = _stack_record_loss_masks([record for record, _, _ in sample_rows])
    return raw_frames, background, samples, origins, domain_names, loss_mask
def record_observed_center_frame(record: ROIRecord) -> torch.Tensor:
    return record_observed_frame(record, frame_index=None)
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
