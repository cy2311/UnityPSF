from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.localization.model import LocalizationModelOutput
from unity_psf.localization.smlm_output import SMLMOutputChannels
from unity_psf.roi_library import ROIRecord
from unity_psf.runtime.profiling import time_block


@dataclass(frozen=True)
class ROIPosteriorSamplingConfig:
    probability_threshold: float = 0.7
    candidate_probability_threshold: float = 0.3
    adjacent_probability_threshold: float = 0.6
    num_posterior_samples: int = 8
    target_projected_emitters: int = 5000
    roi_groups_per_update: int | None = None
    max_emitters_per_roi: int | None = None
    max_sampling_rounds: int = 20
    roi_size_px: int = 128
    sample_continuous: bool = True
    stochastic_existence: bool = False
    min_photons: float = 1e-3
    z_range_nm: tuple[float, float] | None = None
    batch_size: int = 8
    background_smoothing_kernel: int = 9
    input_offset: float = 0.0
    input_scale: float = 1.0
    photon_scale: float = 1.0
    z_scale_nm: float = 1.0
    background_scale: float = 1.0
    seed: int = 0
    camera_backward: dict[str, float] | None = None
    over_cut_px: int = 0


@dataclass(frozen=True)
class SampledEmitterSet:
    roi_id: int
    domain_name: str
    sample_index: int
    xy_px: torch.Tensor
    z_nm: torch.Tensor
    photons: torch.Tensor
    probability: torch.Tensor
    emitter_count: int
    frame_index: int = 0
    frame_offset: int = 0
    background_smoothed: torch.Tensor | None = None
    log_q_h_given_x: float | None = None
    cell_xy_px: torch.Tensor | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class ROILibraryConditionBuilder:
    def __init__(
        self,
        *,
        providers_by_domain: dict[str, FullResZernikeConditioning],
        append_domain_onehot: bool = False,
        domain_names: tuple[str, ...] | list[str] = (),
        condition_feature_dim: int | None = None,
        condition_dim: int | None = None,
    ) -> None:
        self.providers_by_domain = dict(providers_by_domain)
        self.append_domain_onehot = bool(append_domain_onehot)
        self.domain_names = tuple(str(name) for name in domain_names)
        self.domain_index_by_name = {name: idx for idx, name in enumerate(self.domain_names)}
        self.condition_feature_dim = None if condition_feature_dim is None else int(condition_feature_dim)
        self.condition_dim = None if condition_dim is None else int(condition_dim)

    def __call__(self, records: Sequence[ROIRecord], model_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = []
        height = int(model_input.shape[-2])
        width = int(model_input.shape[-1])
        dtype = model_input.dtype
        device = model_input.device
        for record in records:
            domain_name = str(record.domain_name)
            provider = self.providers_by_domain.get(domain_name)
            if provider is None:
                raise KeyError(f"No conditioning provider for ROI domain {domain_name!r}")
            x0, y0 = _condition_origin_for_record(record)
            condition = provider.condition_vector_from_xy(
                x0=x0,
                y0=y0,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
            )
            feature_dim = int(condition.shape[0]) if self.condition_feature_dim is None else self.condition_feature_dim
            if int(condition.shape[0]) >= feature_dim:
                condition = condition[:feature_dim].contiguous()
            else:
                matched = torch.zeros((feature_dim,), dtype=dtype, device=device)
                matched[: int(condition.shape[0])] = condition
                condition = matched
            if self.append_domain_onehot:
                onehot = torch.zeros((len(self.domain_names),), dtype=dtype, device=device)
                onehot[self.domain_index_by_name[domain_name]] = 1.0
                condition = torch.cat((condition, onehot), dim=0)
            if self.condition_dim is not None and int(condition.shape[0]) != self.condition_dim:
                raise ValueError(f"Expected condition_dim={self.condition_dim}, got {int(condition.shape[0])}")
            vectors.append(condition)
        return model_input, torch.stack(vectors, dim=0).contiguous()


class CurrentROILibraryPosteriorSampler:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device | str,
        config: ROIPosteriorSamplingConfig,
        condition_builder: ROILibraryConditionBuilder | None = None,
        frame_proc: Any | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.config = config
        self.condition_builder = condition_builder
        self.frame_proc = frame_proc

    def __call__(
        self,
        records: Sequence[ROIRecord],
        *,
        step_index: int,
    ) -> tuple[tuple[SampledEmitterSet, ...], dict[str, Any]]:
        batch_size = max(1, int(self.config.batch_size))
        all_samples: list[SampledEmitterSet] = []
        sampled_emitters = 0
        sampled_emitters_total = 0
        boundary_emitter_dropped = 0
        target_cap_emitter_dropped = 0
        density_cap_emitter_dropped = 0
        posterior_log_q_available_count = 0
        target_cap_group_skipped = 0
        target_cap_emitter_skipped_after_target = 0
        sampled_roi_ids: set[int] = set()
        sampled_record_count = 0
        posterior_group_id_base = 0
        sampling_stop_reason = "max_sampling_rounds"
        model_was_training = bool(self.model.training)
        generator = torch.Generator(device="cpu")
        sampler_seed = int(self.config.seed) + int(step_index) * 1009
        generator.manual_seed(sampler_seed)

        try:
            self.model.eval()
            with torch.no_grad():
                rounds_completed = 0
                for round_index in range(max(1, int(self.config.max_sampling_rounds))):
                    round_records = list(records)
                    random.Random(int(sampler_seed) + int(round_index)).shuffle(round_records)
                    rounds_completed = int(round_index) + 1
                    for start in range(0, len(round_records), batch_size):
                        batch_records = tuple(round_records[start : start + batch_size])
                        with time_block("posterior_current_model_raw_stack"):
                            frames_photon = torch.stack([_raw_window(record) for record in batch_records], dim=0)
                        with time_block("posterior_current_model_camera_forward"):
                            frames_adu = _camera_forward_adu(frames_photon, self.config.camera_backward)
                        with time_block("posterior_current_model_preprocess"):
                            if self.frame_proc is not None:
                                model_input: Any = _apply_frame_proc(frames_adu, self.frame_proc)
                            else:
                                model_input = _normalize_model_input(
                                    frames_adu,
                                    input_offset=float(self.config.input_offset),
                                    input_scale=float(self.config.input_scale),
                                )
                        if self.condition_builder is not None:
                            with time_block("posterior_current_model_condition"):
                                model_input = self.condition_builder(batch_records, model_input)
                        with time_block("posterior_current_model_infer"):
                            loc_output = self.model(_move_model_input(model_input, self.device))
                        with time_block("posterior_current_model_decode_sample"):
                            batch_samples = _posterior_samples_from_loc_output(
                                loc_output,
                                records=batch_records,
                                sample_index_base=len(all_samples),
                                posterior_group_id_base=int(posterior_group_id_base),
                                config=self.config,
                                generator=generator,
                            )
                        posterior_group_id_base += len(batch_records)
                        with time_block("posterior_current_model_take_groups"):
                            selected_batch_samples, cap_summary = _take_complete_posterior_groups(
                                batch_samples,
                                current_emitters=int(sampled_emitters),
                                target_emitters=int(self.config.target_projected_emitters),
                                current_groups=len({_posterior_group_id(sample) for sample in all_samples}),
                                max_groups=self.config.roi_groups_per_update,
                            )
                        target_cap_group_skipped += int(cap_summary["target_cap_group_skipped"])
                        target_cap_emitter_skipped_after_target += int(cap_summary["target_cap_emitter_skipped_after_target"])
                        all_samples.extend(selected_batch_samples)
                        sampled_record_count += len({int(sample.roi_id) for sample in selected_batch_samples})
                        sampled_roi_ids.update(int(sample.roi_id) for sample in selected_batch_samples)
                        sampled_emitters += sum(int(sample.emitter_count) for sample in selected_batch_samples)
                        sampled_emitters_total += sum(
                            int(sample.metrics.get("emitter_count_total", sample.emitter_count))
                            for sample in selected_batch_samples
                        )
                        boundary_emitter_dropped += sum(
                            int(sample.metrics.get("boundary_emitter_dropped", 0))
                            for sample in selected_batch_samples
                        )
                        target_cap_emitter_dropped += sum(
                            int(sample.metrics.get("target_cap_emitter_dropped", 0))
                            for sample in selected_batch_samples
                        )
                        density_cap_emitter_dropped += sum(
                            int(sample.metrics.get("density_cap_emitter_dropped", 0))
                            for sample in selected_batch_samples
                        )
                        posterior_log_q_available_count += sum(
                            1 for sample in selected_batch_samples if sample.log_q_h_given_x is not None
                        )
                        group_count_now = len({_posterior_group_id(sample) for sample in all_samples})
                        if (
                            self.config.roi_groups_per_update is not None
                            and group_count_now >= int(self.config.roi_groups_per_update)
                        ):
                            sampling_stop_reason = "roi_groups_per_update"
                            break
                        if sampled_emitters >= int(self.config.target_projected_emitters):
                            sampling_stop_reason = "target_projected_emitters"
                            break
                    group_count_now = len({_posterior_group_id(sample) for sample in all_samples})
                    if self.config.roi_groups_per_update is not None and group_count_now >= int(self.config.roi_groups_per_update):
                        break
                    if sampled_emitters >= int(self.config.target_projected_emitters):
                        break
        finally:
            self.model.train(model_was_training)

        group_sizes: dict[int, int] = {}
        for sample in all_samples:
            group_id = _posterior_group_id(sample)
            group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
        group_size_values = tuple(group_sizes.values())
        selected_roi_ids = sorted(int(value) for value in sampled_roi_ids)
        summary = {
            "sampling_rounds": int(rounds_completed if "rounds_completed" in locals() else 1),
            "sample_count": int(len(all_samples)),
            "sampled_emitter_count": int(sampled_emitters),
            "sampled_emitter_count_inner": int(sampled_emitters),
            "sampled_emitter_count_total": int(sampled_emitters_total),
            "boundary_emitter_dropped": int(boundary_emitter_dropped),
            "target_cap_emitter_dropped": int(target_cap_emitter_dropped),
            "density_cap_emitter_dropped": int(density_cap_emitter_dropped),
            "target_cap_group_skipped": int(target_cap_group_skipped),
            "target_cap_emitter_skipped_after_target": int(target_cap_emitter_skipped_after_target),
            "posterior_log_q_available_count": int(posterior_log_q_available_count),
            "posterior_group_count": int(len(group_sizes)),
            "posterior_group_size_min": int(min(group_size_values)) if group_size_values else 0,
            "posterior_group_size_max": int(max(group_size_values)) if group_size_values else 0,
            "posterior_group_size_mean": float(sum(group_size_values) / len(group_size_values)) if group_size_values else 0.0,
            "sampled_roi_count": int(len(sampled_roi_ids)),
            "sampled_record_count": int(sampled_record_count),
            "posterior_batch_size": int(batch_size),
            "target_projected_emitters": int(self.config.target_projected_emitters),
            "target_projected_emitters_overshoot": int(max(0, sampled_emitters - int(self.config.target_projected_emitters))),
            "target_emitters_reached": bool(sampled_emitters >= int(self.config.target_projected_emitters)),
            "roi_groups_per_update": 0
            if self.config.roi_groups_per_update is None
            else int(self.config.roi_groups_per_update),
            "roi_groups_per_update_reached": bool(
                self.config.roi_groups_per_update is not None
                and len(group_sizes) >= int(self.config.roi_groups_per_update)
            ),
            "roi_groups_per_update_overshoot": int(
                0
                if self.config.roi_groups_per_update is None
                else max(0, len(group_sizes) - int(self.config.roi_groups_per_update))
            ),
            "max_emitters_per_roi": 0
            if self.config.max_emitters_per_roi is None
            else int(self.config.max_emitters_per_roi),
            "sampling_stop_reason": str(sampling_stop_reason),
            "over_cut_px": int(self.config.over_cut_px),
            "posterior_sampling_source": "current_loc_network",
            "posterior_background_source": "current_loc_network_smoothed",
            "projection_sample_source": "roi_record_posterior_samples",
            "selected_roi_ids": selected_roi_ids,
        }
        return tuple(all_samples), summary


def _raw_window(record: ROIRecord) -> torch.Tensor:
    raw = torch.as_tensor(record.raw_frames_photon, dtype=torch.float32)
    if raw.ndim == 2:
        return raw.unsqueeze(0)
    if raw.ndim != 3:
        raise ValueError(f"raw_frames_photon must have shape (T,H,W) or (H,W), got {tuple(raw.shape)}")
    return raw


def _normalize_model_input(frames: torch.Tensor, *, input_offset: float, input_scale: float) -> torch.Tensor:
    return (frames.to(dtype=torch.float32) - float(input_offset)) / max(float(input_scale), 1e-6)


def _camera_forward_adu(frames_photon: torch.Tensor, params: dict[str, float] | None) -> torch.Tensor:
    if not params:
        return frames_photon.to(dtype=torch.float32)
    qe = float(params.get("qe", 1.0))
    e_per_adu = float(params.get("e_per_adu", 1.0))
    baseline = float(params.get("baseline", params.get("baseline_adu", 0.0)))
    spurious = float(params.get("spurious_charge", 0.0))
    em_gain = float(params.get("em_gain", 1.0))
    electrons = (frames_photon.to(dtype=torch.float32) * max(qe, 1e-12)) + spurious
    electrons = electrons * max(em_gain, 1e-12)
    return electrons / max(e_per_adu, 1e-12) + baseline


def _apply_frame_proc(frames: torch.Tensor, frame_proc: Any) -> torch.Tensor:
    if frames.ndim == 3:
        return frame_proc.forward(frames.to(dtype=torch.float32)).to(dtype=torch.float32)
    if frames.ndim != 4:
        raise ValueError(f"frames must have shape (T,H,W) or (B,T,H,W), got {tuple(frames.shape)}")
    processed = [frame_proc.forward(frames[index].to(dtype=torch.float32)).to(dtype=torch.float32) for index in range(int(frames.shape[0]))]
    return torch.stack(processed, dim=0).contiguous()


def _move_model_input(model_input: Any, device: torch.device) -> Any:
    if isinstance(model_input, torch.Tensor):
        return model_input.to(device=device, dtype=torch.float32)
    if isinstance(model_input, Sequence):
        return tuple(_move_model_input(item, device) for item in model_input)
    raise TypeError(f"Unsupported model_input type: {type(model_input)!r}")


def _smooth_background(bg: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    kernel = int(kernel_size)
    if kernel <= 1:
        return bg
    if kernel % 2 == 0:
        raise ValueError("background_smoothing_kernel must be odd")
    return F.avg_pool2d(bg[:, None], kernel_size=kernel, stride=1, padding=kernel // 2)[:, 0]


def _spatial_integrated_probability(
    prob: torch.Tensor,
    *,
    candidate_threshold: float,
    adjacent_threshold: float,
) -> torch.Tensor:
    if prob.ndim == 2:
        prob_batched = prob.unsqueeze(0)
        squeeze_batch = True
    elif prob.ndim == 3:
        prob_batched = prob
        squeeze_batch = False
    else:
        raise ValueError(f"prob must have shape (H,W) or (B,H,W), got {tuple(prob.shape)}")
    p = prob_batched.to(dtype=torch.float32)
    p = torch.where(torch.isfinite(p), p, torch.zeros_like(p)).clamp_min(0.0)
    p4 = p[:, None]
    clipped = torch.where(p4 > float(candidate_threshold), p4, torch.zeros_like(p4))
    local_max = F.max_pool2d(clipped, kernel_size=3, stride=1, padding=1)
    candidate_mask_1 = (p4 == local_max).to(dtype=p4.dtype)
    candidate_mask_2 = (p4 * (1.0 - candidate_mask_1) > float(adjacent_threshold)).to(dtype=p4.dtype)
    filt = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=p4.dtype,
        device=p4.device,
    ).view(1, 1, 3, 3)
    integrated = F.conv2d(p4, filt, padding=1)
    candidate_integrated = torch.clamp((candidate_mask_1 + candidate_mask_2) * integrated, 0.0, 1.0)[:, 0]
    return candidate_integrated[0] if squeeze_batch else candidate_integrated


def _normal_sample(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    *,
    generator: torch.Generator,
    stochastic: bool,
) -> torch.Tensor:
    if mean.numel() == 0 or not stochastic:
        return mean.clone()
    return mean + torch.randn(mean.shape, generator=generator, dtype=mean.dtype, device=mean.device) * sigma.clamp_min(0.0)


def _normal_log_prob(value: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_safe = sigma.clamp_min(1e-12)
    variance = sigma_safe.square()
    return -0.5 * ((value - mean).square() / variance + torch.log(torch.as_tensor(2.0 * math.pi, dtype=value.dtype) * variance))


def _inner_xy_mask(xy_px: torch.Tensor, *, roi_size_px: int, over_cut_px: int) -> torch.Tensor:
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


@dataclass(frozen=True)
class _PosteriorCellSelection:
    x_ix: torch.Tensor
    y_ix: torch.Tensor
    p_keep: torch.Tensor
    cell_xy: torch.Tensor
    mu_keep: torch.Tensor
    sig_keep: torch.Tensor
    emitter_count_total: int
    emitter_count_after_boundary: int
    boundary_dropped: int
    density_cap_dropped: int


def _select_posterior_cells(
    p_integrated: torch.Tensor,
    mu_batch: torch.Tensor,
    sig_batch: torch.Tensor,
    *,
    config: ROIPosteriorSamplingConfig,
) -> _PosteriorCellSelection:
    keep = torch.isfinite(p_integrated) & (p_integrated > float(config.probability_threshold))
    y_ix, x_ix = keep.nonzero(as_tuple=True)
    p_keep = p_integrated[y_ix, x_ix]
    emitter_count_total = int(p_keep.numel())
    if p_keep.numel() == 0:
        empty_ix = torch.empty((0,), dtype=torch.long)
        return _PosteriorCellSelection(
            x_ix=empty_ix,
            y_ix=empty_ix,
            p_keep=torch.empty((0,), dtype=torch.float32),
            cell_xy=torch.empty((0, 2), dtype=torch.float32),
            mu_keep=torch.empty((0, 4), dtype=torch.float32),
            sig_keep=torch.empty((0, 4), dtype=torch.float32),
            emitter_count_total=0,
            emitter_count_after_boundary=0,
            boundary_dropped=0,
            density_cap_dropped=0,
        )
    cell_xy = torch.stack((x_ix.to(dtype=torch.float32), y_ix.to(dtype=torch.float32)), dim=1)
    mu_keep = mu_batch[:, y_ix, x_ix].transpose(0, 1).contiguous()
    sig_keep = sig_batch[:, y_ix, x_ix].transpose(0, 1).contiguous()
    inner_mask = _inner_xy_mask(cell_xy, roi_size_px=int(config.roi_size_px), over_cut_px=int(config.over_cut_px))
    emitter_count_after_boundary = int(inner_mask.sum().item())
    boundary_dropped = int(max(0, emitter_count_total - emitter_count_after_boundary))
    x_ix = x_ix[inner_mask]
    y_ix = y_ix[inner_mask]
    p_keep = p_keep[inner_mask]
    cell_xy = cell_xy[inner_mask]
    mu_keep = mu_keep[inner_mask]
    sig_keep = sig_keep[inner_mask]
    density_cap_dropped = 0
    if config.max_emitters_per_roi is not None and int(config.max_emitters_per_roi) > 0 and int(p_keep.numel()) > int(config.max_emitters_per_roi):
        order = torch.argsort(p_keep, descending=True)[: int(config.max_emitters_per_roi)]
        density_cap_dropped = int(p_keep.numel() - int(config.max_emitters_per_roi))
        x_ix = x_ix[order]
        y_ix = y_ix[order]
        p_keep = p_keep[order]
        cell_xy = cell_xy[order]
        mu_keep = mu_keep[order]
        sig_keep = sig_keep[order]
    return _PosteriorCellSelection(
        x_ix=x_ix,
        y_ix=y_ix,
        p_keep=p_keep,
        cell_xy=cell_xy,
        mu_keep=mu_keep,
        sig_keep=sig_keep,
        emitter_count_total=emitter_count_total,
        emitter_count_after_boundary=emitter_count_after_boundary,
        boundary_dropped=boundary_dropped,
        density_cap_dropped=density_cap_dropped,
    )


def _extract_posterior_fields(
    loc_output: torch.Tensor | LocalizationModelOutput,
    *,
    background_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(loc_output, LocalizationModelOutput):
        prob = loc_output.probability.detach().cpu().to(dtype=torch.float32)
        mu = torch.stack(
            (
                loc_output.photons.detach().cpu().to(dtype=torch.float32),
                loc_output.xy_offset[:, 0].detach().cpu().to(dtype=torch.float32),
                loc_output.xy_offset[:, 1].detach().cpu().to(dtype=torch.float32),
                loc_output.z.detach().cpu().to(dtype=torch.float32),
            ),
            dim=1,
        )
        sig = torch.zeros_like(mu)
        return prob, mu, sig, None
    if not (isinstance(loc_output, torch.Tensor) and loc_output.ndim == 4 and int(loc_output.shape[1]) == SMLMOutputChannels.count):
        raise ValueError(
            "current ROI posterior sampling requires either LocalizationModelOutput or a 10-channel SMLM tensor, "
            f"got {type(loc_output)!r}"
        )
    out = loc_output.detach().cpu().to(dtype=torch.float32)
    prob = out[:, SMLMOutputChannels.p]
    mu = out[:, SMLMOutputChannels.pxyz_mu]
    sig = out[:, SMLMOutputChannels.pxyz_sigma].abs()
    bg = out[:, SMLMOutputChannels.bg] * float(background_scale)
    return prob, mu, sig, bg


def _posterior_samples_from_loc_output(
    loc_output: torch.Tensor | LocalizationModelOutput,
    *,
    records: Sequence[ROIRecord],
    sample_index_base: int,
    posterior_group_id_base: int,
    config: ROIPosteriorSamplingConfig,
    generator: torch.Generator,
) -> tuple[SampledEmitterSet, ...]:
    prob, mu, sig, bg = _extract_posterior_fields(loc_output, background_scale=float(config.background_scale))
    bg_smoothed = None if bg is None else _smooth_background(bg, kernel_size=int(config.background_smoothing_kernel)).detach().cpu()
    prob_integrated = _spatial_integrated_probability(
        prob,
        candidate_threshold=float(config.candidate_probability_threshold),
        adjacent_threshold=float(config.adjacent_probability_threshold),
    )
    roi_max = max(0.0, float(config.roi_size_px) - 1.0)
    posterior_sample_count = max(int(config.num_posterior_samples), 1)
    selections = tuple(
        _select_posterior_cells(
            prob_integrated[batch_index],
            mu[batch_index],
            sig[batch_index],
            config=config,
        )
        for batch_index in range(len(records))
    )
    samples: list[SampledEmitterSet] = []
    for posterior_sample_index in range(posterior_sample_count):
        for batch_index, record in enumerate(records):
            posterior_group_id = int(posterior_group_id_base) + int(batch_index)
            selection = selections[batch_index]
            x_ix = selection.x_ix
            y_ix = selection.y_ix
            p_keep = selection.p_keep
            emitter_count_total = int(selection.emitter_count_total)
            if p_keep.numel() > 0:
                photon_mu = selection.mu_keep[:, 0] * float(config.photon_scale)
                photon_sig = selection.sig_keep[:, 0] * float(config.photon_scale)
                photons = _normal_sample(
                    photon_mu,
                    photon_sig,
                    generator=generator,
                    stochastic=bool(config.sample_continuous),
                ).clamp_min(float(config.min_photons))
                x_mu = x_ix.to(dtype=torch.float32) + 0.5 + selection.mu_keep[:, 1]
                x_px = (
                    x_mu
                    + _normal_sample(
                        torch.zeros_like(selection.mu_keep[:, 1]),
                        selection.sig_keep[:, 1],
                        generator=generator,
                        stochastic=bool(config.sample_continuous),
                    )
                ).clamp(0.0, roi_max)
                y_mu = y_ix.to(dtype=torch.float32) + 0.5 + selection.mu_keep[:, 2]
                y_px = (
                    y_mu
                    + _normal_sample(
                        torch.zeros_like(selection.mu_keep[:, 2]),
                        selection.sig_keep[:, 2],
                        generator=generator,
                        stochastic=bool(config.sample_continuous),
                    )
                ).clamp(0.0, roi_max)
                z_mu = selection.mu_keep[:, 3] * float(config.z_scale_nm)
                z_sig = selection.sig_keep[:, 3] * float(config.z_scale_nm)
                z_nm = _normal_sample(
                    z_mu,
                    z_sig,
                    generator=generator,
                    stochastic=bool(config.sample_continuous),
                )
                if config.z_range_nm is not None:
                    z_nm = z_nm.clamp(float(config.z_range_nm[0]), float(config.z_range_nm[1]))
                xy = torch.stack((x_px, y_px), dim=1)
                if bool(config.sample_continuous):
                    log_q_values = (
                        _normal_log_prob(xy[:, 0], x_mu, selection.sig_keep[:, 1])
                        + _normal_log_prob(xy[:, 1], y_mu, selection.sig_keep[:, 2])
                        + _normal_log_prob(z_nm, z_mu, z_sig)
                        + _normal_log_prob(photons, photon_mu, photon_sig)
                    )
                    log_q_h_given_x = float(log_q_values.sum().item())
                else:
                    log_q_h_given_x = None
            else:
                xy = torch.empty((0, 2), dtype=torch.float32)
                z_nm = torch.empty((0,), dtype=torch.float32)
                photons = torch.empty((0,), dtype=torch.float32)
                log_q_h_given_x = 0.0 if bool(config.sample_continuous) else None
            frame_raw = torch.as_tensor(record.raw_frames_photon, dtype=torch.float32)
            frame_count = 1 if frame_raw.ndim == 2 else int(frame_raw.shape[0])
            frame_offset = int(frame_count // 2)
            frame_index = int(record.frame_window[0]) + frame_offset
            metrics = {
                "roi_id": int(record.roi_id),
                "posterior_sample_index": int(posterior_sample_index),
                "frame_index": int(frame_index),
                "frame_offset": int(frame_offset),
                "emitter_count": int(p_keep.numel()),
                "emitter_count_total": int(emitter_count_total),
                "emitter_count_inner": int(p_keep.numel()),
                "emitter_count_after_boundary": int(selection.emitter_count_after_boundary),
                "boundary_emitter_dropped": int(selection.boundary_dropped),
                "target_cap_emitter_dropped": int(selection.density_cap_dropped),
                "density_cap_emitter_dropped": int(selection.density_cap_dropped),
                "max_emitters_per_roi": 0
                if config.max_emitters_per_roi is None
                else int(config.max_emitters_per_roi),
                "over_cut_px": int(config.over_cut_px),
                "posterior_group_id": int(posterior_group_id),
                "posterior_group_sample_index": int(posterior_sample_index),
                "posterior_sampling_source": "current_loc_network",
                "posterior_delta_source": "integrated_probability",
            }
            if log_q_h_given_x is not None:
                metrics["log_q_h_given_x"] = float(log_q_h_given_x)
            samples.append(
                SampledEmitterSet(
                    roi_id=int(record.roi_id),
                    domain_name=str(record.domain_name),
                    sample_index=int(sample_index_base + len(samples)),
                    cell_xy_px=selection.cell_xy.detach().cpu(),
                    xy_px=xy.detach().cpu(),
                    z_nm=z_nm.detach().cpu(),
                    photons=photons.detach().cpu(),
                    probability=p_keep.detach().cpu(),
                    emitter_count=int(p_keep.numel()),
                    frame_index=frame_index,
                    frame_offset=frame_offset,
                    background_smoothed=(
                        torch.as_tensor(record.background_smoothed, dtype=torch.float32)
                        if bg_smoothed is None
                        else bg_smoothed[batch_index].detach().cpu()
                    ),
                    log_q_h_given_x=log_q_h_given_x,
                    metrics=metrics,
                )
            )
    return tuple(samples)


def _posterior_group_id(sample: SampledEmitterSet) -> int:
    metrics = getattr(sample, "metrics", {}) or {}
    group_id = metrics.get("posterior_group_id")
    if group_id is not None:
        return int(group_id)
    return int(sample.roi_id) * 1000003 + int(getattr(sample, "frame_offset", 0))


def _take_complete_posterior_groups(
    samples: Sequence[SampledEmitterSet],
    *,
    current_emitters: int,
    target_emitters: int,
    current_groups: int = 0,
    max_groups: int | None = None,
) -> tuple[tuple[SampledEmitterSet, ...], dict[str, int]]:
    groups: dict[int, list[SampledEmitterSet]] = {}
    group_order: list[int] = []
    for sample in samples:
        group_id = _posterior_group_id(sample)
        if group_id not in groups:
            groups[group_id] = []
            group_order.append(group_id)
        groups[group_id].append(sample)
    selected: list[SampledEmitterSet] = []
    emitters_now = int(current_emitters)
    groups_now = int(current_groups)
    skipped_group_ids: set[int] = set()
    skipped_emitters = 0
    stop_after_current_group = False
    for group_id in group_order:
        group_samples = groups[group_id]
        group_emitters = sum(int(sample.emitter_count) for sample in group_samples)
        if stop_after_current_group or emitters_now >= int(target_emitters):
            skipped_group_ids.add(group_id)
            skipped_emitters += group_emitters
            continue
        if max_groups is not None and groups_now >= int(max_groups):
            skipped_group_ids.add(group_id)
            skipped_emitters += group_emitters
            continue
        selected.extend(group_samples)
        emitters_now += group_emitters
        groups_now += 1
        if emitters_now >= int(target_emitters):
            stop_after_current_group = True
    return tuple(selected), {
        "target_cap_group_skipped": int(len(skipped_group_ids)),
        "target_cap_emitter_skipped_after_target": int(skipped_emitters),
    }


def _condition_origin_for_record(record: ROIRecord) -> tuple[int, int]:
    local = record.summary.get("domain_local_roi_origin_xy_px") if isinstance(record.summary, dict) else None
    if isinstance(local, (list, tuple)) and len(local) == 2:
        return int(round(float(local[0]))), int(round(float(local[1])))
    return int(round(float(record.roi_origin_xy_px[0]))), int(round(float(record.roi_origin_xy_px[1])))


__all__ = [
    "CurrentROILibraryPosteriorSampler",
    "ROILibraryConditionBuilder",
    "ROIPosteriorSamplingConfig",
    "SampledEmitterSet",
]
