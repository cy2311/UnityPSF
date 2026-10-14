"""Conversion from ROI posterior records into projection tensors."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from unity_psf.localization.posterior import DetectionPosteriorSamples
from unity_psf.localization.roi_posterior_sampling import SampledEmitterSet
from unity_psf.roi_library import ROIBank, ROIRecord
from unity_psf.runtime import profiling
from unity_psf.training.high_fidelity.roi_projection import (
    ROIProjectionUpdateContext,
    inner_xy_mask,
    normal_log_prob,
    normal_sample,
    record_observed_frame,
    stack_record_loss_masks,
)


def sample_record_posterior_rows(
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
            xy = normal_sample(mu_xy[frame_mask], sigma_xy[frame_mask], generator=generator, stochastic=sample_continuous).clamp(0.0, roi_max)
            z = normal_sample(mu_z[frame_mask], sigma_z[frame_mask], generator=generator, stochastic=sample_continuous)
            if z_range_nm is not None:
                z = z.clamp(float(z_range_nm[0]), float(z_range_nm[1]))
            photons = normal_sample(mu_photons[frame_mask], sigma_photons[frame_mask], generator=generator, stochastic=sample_continuous).clamp_min(float(min_photons))
            cells = cell_xy[frame_mask].clone()
            total = int(cells.shape[0])
            if bool(sample_continuous) and total > 0:
                log_q_values = (
                    normal_log_prob(xy[:, 0], mu_xy[frame_mask][:, 0], sigma_xy[frame_mask][:, 0])
                    + normal_log_prob(xy[:, 1], mu_xy[frame_mask][:, 1], sigma_xy[frame_mask][:, 1])
                    + normal_log_prob(z, mu_z[frame_mask], sigma_z[frame_mask])
                    + normal_log_prob(photons, mu_photons[frame_mask], sigma_photons[frame_mask])
                )
            else:
                log_q_values = torch.empty((total,), dtype=torch.float32)
            inner = inner_xy_mask(cells, roi_size_px=roi_size_px, over_cut_px=over_cut_px)
            xy, z, photons, cells = xy[inner], z[inner], photons[inner], cells[inner]
            log_q_h_given_x = float(log_q_values[inner].sum().item()) if bool(sample_continuous) and total > 0 else None
            rows.append(
                {
                    "record": record,
                    "sample_index": int(sample_index),
                    "frame_index": int(frame_value),
                    "frame_offset": int(frame_value) - int(record.frame_window[0]),
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


def posterior_rows_to_context(
    rows: list[dict[str, Any]], *, bank: ROIBank, sampling_metrics: dict[str, object]
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
        loss_mask = stack_record_loss_masks([row["record"] for row in rows])
        selected_records = tuple({int(row["record"].roi_id): row["record"] for row in rows}.values())
    return ROIProjectionUpdateContext(
        bank=ROIBank(records=selected_records, config=bank.config, metadata={**bank.metadata, "roi_bank_update_subset": True, "roi_bank_projection_sample_source": "posterior_samples"}, empty_grid_cell_ids=bank.empty_grid_cell_ids, format_version=bank.format_version),
        sampling_metrics=sampling_metrics,
        raw_frames=raw_frames,
        background=background,
        samples=samples,
        roi_origin_xy_px=origins,
        domain_names=domain_names,
        loss_mask=loss_mask,
    )


def sampled_emitter_sets_to_context(
    samples: Sequence[SampledEmitterSet], *, bank: ROIBank, sampling_metrics: dict[str, object]
) -> ROIProjectionUpdateContext:
    if not samples:
        raise ValueError("posterior samples must be non-empty")
    records_by_id = {int(record.roi_id): record for record in bank.records}
    with profiling.time_block("posterior_samples_context_raw_stack"):
        raw_frames = torch.stack([record_observed_frame(records_by_id[int(sample.roi_id)], frame_index=int(sample.frame_index)) for sample in samples], dim=0)
    with profiling.time_block("posterior_samples_context_background_stack"):
        background = torch.stack([torch.as_tensor(sample.background_smoothed if sample.background_smoothed is not None else records_by_id[int(sample.roi_id)].background_smoothed, dtype=torch.float32) for sample in samples], dim=0)
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
        origins = torch.as_tensor([records_by_id[int(sample.roi_id)].roi_origin_xy_px for sample in samples], dtype=torch.float32)
        domain_names = [str(sample.domain_name) for sample in samples]
        loss_mask = stack_record_loss_masks([records_by_id[int(sample.roi_id)] for sample in samples])
        selected_records = tuple({records_by_id[int(sample.roi_id)].roi_id: records_by_id[int(sample.roi_id)] for sample in samples}.values())
    return ROIProjectionUpdateContext(
        bank=ROIBank(records=selected_records, config=bank.config, metadata={**bank.metadata, "roi_bank_update_subset": True, "roi_bank_projection_sample_source": "posterior_samples"}, empty_grid_cell_ids=bank.empty_grid_cell_ids, format_version=bank.format_version),
        sampling_metrics=sampling_metrics,
        raw_frames=raw_frames,
        background=background,
        samples=detection_samples,
        roi_origin_xy_px=origins,
        domain_names=domain_names,
        loss_mask=loss_mask,
    )


__all__ = ["posterior_rows_to_context", "sample_record_posterior_rows", "sampled_emitter_sets_to_context"]
