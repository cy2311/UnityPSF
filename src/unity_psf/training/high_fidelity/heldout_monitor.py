from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from unity_psf.gamma_update import GammaProjectionObjective
from unity_psf.localization.posterior import DetectionPosteriorSamples, sample_detection_posterior
from unity_psf.localization.roi_batches import build_roi_batch_provider
from unity_psf.localization.scale_resolver import resolve_target_contract
from unity_psf.roi_library import ROIBank
from unity_psf.training.high_fidelity.posterior_workflow import sample_roi_bank_posterior_update_from_current_model
from unity_psf.training.high_fidelity.roi_bank_source import roi_conditioning_context as _roi_conditioning_context
from unity_psf.training.high_fidelity.roi_projection import (
    posterior_photon_scale as _posterior_photon_scale,
    posterior_z_scale as _posterior_z_scale,
    select_roi_window_frame as _select_roi_window_frame,
    stack_record_loss_masks as _stack_record_loss_masks,
)


@dataclass(frozen=True)
class HeldoutMonitorContext:
    bank: ROIBank
    mode: str
    split_source: str | None
    samples_mask_count: int
    raw_frames: torch.Tensor
    background: torch.Tensor
    samples: DetectionPosteriorSamples
    roi_origin_xy_px: torch.Tensor
    domain_names: list[str]
    loss_mask: torch.Tensor | None = None


def build_heldout_monitor_context(
    gamma_cfg: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    bank: ROIBank | None,
    objective: GammaProjectionObjective,
    max_emitters: int,
    mode: str,
    split_source: str | None,
    train_cfg: Mapping[str, Any],
) -> HeldoutMonitorContext | None:
    if bank is None:
        return None
    objective_mode = str(gamma_cfg.get("roi_bank_objective", gamma_cfg.get("objective_mode", "poisson_nll"))).strip().lower()
    if objective_mode == "importance_wake":
        heldout_cfg = dict(gamma_cfg)
        heldout_cfg["target_projected_emitters"] = int(gamma_cfg.get("heldout_target_projected_emitters", gamma_cfg.get("target_projected_emitters", 1000)))
        try:
            context, _metrics = sample_roi_bank_posterior_update_from_current_model(
                bank,
                heldout_cfg,
                model=model,
                train_cfg=train_cfg,
                step_index=int(gamma_cfg.get("heldout_seed_offset", 100000)),
            )
        except ValueError as exc:
            if "selected no samples" in str(exc):
                return None
            raise
        return HeldoutMonitorContext(
            bank=bank,
            mode=mode,
            split_source=split_source,
            samples_mask_count=int(context.samples.mask.sum().item()),
            raw_frames=context.raw_frames,
            background=context.background,
            samples=context.samples,
            roi_origin_xy_px=context.roi_origin_xy_px,
            domain_names=context.domain_names,
            loss_mask=context.loss_mask,
        )
    roi_conditioning = _roi_conditioning_context(train_cfg)
    loc_batch = build_roi_batch_provider(
        bank,
        batch_size=len(bank.records),
        seed=0,
        condition_providers_by_domain=roi_conditioning["providers"],
        append_domain_onehot=roi_conditioning["append_domain_onehot"],
        domain_names=roi_conditioning["domain_names"],
    )(epoch=1)[0].inputs
    samples = sample_detection_posterior(
        model=model,
        batch=loc_batch,
        threshold=float(gamma_cfg.get("posterior_threshold", -1.0)),
        max_emitters=max_emitters,
        seed=0,
        photon_scale=_posterior_photon_scale(train_cfg),
        z_scale=_posterior_z_scale(train_cfg),
        contract=resolve_target_contract(train_cfg),
        candidate_threshold=float(
            gamma_cfg.get("posterior_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))
        ),
        split_threshold=float(gamma_cfg.get("posterior_adjacent_probability_threshold", gamma_cfg.get("split_threshold", 0.6))),
    )
    raw_frames = _select_roi_window_frame(loc_batch.model_input)
    return HeldoutMonitorContext(
        bank=bank,
        mode=mode,
        split_source=split_source,
        samples_mask_count=int(samples.mask.sum().item()),
        raw_frames=raw_frames,
        background=loc_batch.bkg_tar,
        samples=samples,
        roi_origin_xy_px=torch.as_tensor(loc_batch.metadata["roi_origin_xy_px"], dtype=torch.float32),
        domain_names=list(loc_batch.metadata["domain_names"]),
        loss_mask=_stack_record_loss_masks(bank.records),
    )


def heldout_monitor_metrics(
    context: HeldoutMonitorContext | None,
    *,
    objective: GammaProjectionObjective,
    gamma_before: torch.Tensor,
    gamma_after: torch.Tensor,
    unavailable_mode: str = "not_configured",
    unavailable_split_source: str | None = None,
) -> dict[str, object]:
    if context is None:
        return {
            "heldout_available": False,
            "heldout_monitor_mode": unavailable_mode,
            "heldout_roi_count": 0,
            "heldout_sample_count": 0,
            "heldout_sampled_emitter_count": 0,
            "heldout_split_source": unavailable_split_source,
            "heldout_initial_loss": None,
            "heldout_final_loss": None,
            "heldout_loss_delta": None,
            "heldout_loss_delta_percent": None,
            "heldout_poisson_nll": None,
            "heldout_poisson_nll_full_roi": None,
        }
    initial_loss = objective(
        gamma=gamma_before,
        samples=context.samples,
        raw_frames=context.raw_frames,
        background=context.background,
        roi_origin_xy_px=context.roi_origin_xy_px,
        domain_names=context.domain_names,
        loss_mask=context.loss_mask,
    ).detach()
    final_loss = objective(
        gamma=gamma_after,
        samples=context.samples,
        raw_frames=context.raw_frames,
        background=context.background,
        roi_origin_xy_px=context.roi_origin_xy_px,
        domain_names=context.domain_names,
        loss_mask=context.loss_mask,
    ).detach()
    initial_value = float(initial_loss.item())
    final_value = float(final_loss.item())
    delta = final_value - initial_value
    delta_percent = None if initial_value == 0.0 else float(delta / initial_value * 100.0)
    return {
        "heldout_available": True,
        "heldout_monitor_mode": context.mode,
        "heldout_roi_count": len(context.bank.records),
        "heldout_sample_count": len(context.bank.records),
        "heldout_sampled_emitter_count": context.samples_mask_count,
        "heldout_split_source": context.split_source,
        "heldout_roi_ids": [int(record.roi_id) for record in context.bank.records],
        "heldout_initial_loss": initial_value,
        "heldout_final_loss": final_value,
        "heldout_loss_delta": delta,
        "heldout_loss_delta_percent": delta_percent,
        "heldout_poisson_nll": final_value,
        "heldout_poisson_nll_full_roi": final_value,
    }
