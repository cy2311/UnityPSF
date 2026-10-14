from __future__ import annotations

import random
from typing import Any, Mapping

import torch

from unity_psf.localization.roi_posterior_sampling import (
    CurrentROILibraryPosteriorSampler,
    ROILibraryConditionBuilder,
    ROIPosteriorSamplingConfig,
)
from unity_psf.roi_library import ROIBank
from unity_psf.roi_library.loc_harvest import build_inference_frame_proc
from unity_psf.runtime import profiling
from unity_psf.training.high_fidelity.gamma_projection import z_range_nm_from_gamma_cfg as _z_range_nm_from_gamma_cfg
from unity_psf.training.high_fidelity.roi_bank_source import (
    camera_backward_from_training_config as _camera_backward_from_training_config,
    roi_conditioning_context as _roi_conditioning_context,
)
from unity_psf.training.high_fidelity.roi_inference import model_device as _model_device
from unity_psf.training.high_fidelity.roi_projection import (
    ROIProjectionUpdateContext as _ROIProjectionUpdateContext,
    model_supports_conditioned_input as _model_supports_conditioned_input,
    posterior_bg_scale as _posterior_bg_scale,
    posterior_input_offset as _posterior_input_offset,
    posterior_input_scale as _posterior_input_scale,
    posterior_photon_scale as _posterior_photon_scale,
    posterior_z_scale_nm as _posterior_z_scale_nm,
)
from unity_psf.training.high_fidelity.posterior_context import (
    posterior_rows_to_context as _posterior_rows_to_context,
    sample_record_posterior_rows as _sample_record_posterior_rows,
    sampled_emitter_sets_to_context as _sampled_emitter_sets_to_context,
)


def sample_roi_bank_posterior_update(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
    *,
    epoch: int,
) -> tuple[_ROIProjectionUpdateContext, dict[str, object]]:
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
) -> tuple[_ROIProjectionUpdateContext, dict[str, object]]:
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
        z_range_nm = _z_range_nm_from_gamma_cfg(gamma_cfg)

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
    context = _ROIProjectionUpdateContext(
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
) -> tuple[_ROIProjectionUpdateContext, dict[str, object]]:
    records = tuple(bank.records)
    if not records:
        raise ValueError("posterior ROI-bank update requires at least one ROI record")
    roi_conditioning = _roi_conditioning_context(train_cfg)
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
    frame_proc = build_inference_frame_proc({"normalization": dict(normalization_cfg)}) if isinstance(normalization_cfg, Mapping) else None
    z_scale_nm = _posterior_z_scale_nm(train_cfg) or 1.0
    sampler = CurrentROILibraryPosteriorSampler(
        model=model,
        device=_model_device(model),
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
            z_range_nm=_z_range_nm_from_gamma_cfg(gamma_cfg),
            batch_size=max(1, int(gamma_cfg.get("posterior_batch_size", gamma_cfg.get("batch_size", 8)))),
            background_smoothing_kernel=int(
                gamma_cfg.get("roi_bank_background_smoothing_kernel", gamma_cfg.get("background_smoothing_kernel", 3))
            ),
            input_offset=_posterior_input_offset(train_cfg),
            input_scale=_posterior_input_scale(train_cfg),
            photon_scale=_posterior_photon_scale(train_cfg) or 1.0,
            z_scale_nm=float(z_scale_nm),
            background_scale=_posterior_bg_scale(train_cfg),
            seed=int(gamma_cfg.get("seed", 0)),
            camera_backward=_camera_backward_from_training_config(train_cfg),
            over_cut_px=int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0))),
            tile_inference_enabled=bool(gamma_cfg.get("roi_bank_tile_inference_enabled", False)),
            tile_size_px=int(gamma_cfg.get("roi_bank_tile_size_px", 64)),
            tile_stride_px=int(gamma_cfg.get("roi_bank_tile_stride_px", 32)),
            tile_core_size_px=int(gamma_cfg.get("roi_bank_tile_core_size_px", 48)),
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
) -> tuple[_ROIProjectionUpdateContext, dict[str, object]]:
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
        _ROIProjectionUpdateContext(
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
