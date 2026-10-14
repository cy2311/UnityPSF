from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from unity_psf.gamma_update import (
    GammaProjectionObjective,
    GammaUpdateState,
    build_gamma_update_hook,
)
from unity_psf.localization.conditioning import ConditioningProviderStore
from unity_psf.roi_library import ROIBank, ROIBankBuildConfig, ROIRecord, load_roi_bank
from unity_psf.runtime import profiling
from unity_psf.training.channel_context import ChannelTrainingContext
from unity_psf.training.loop_types import TrainingRunEpochResult, TrainingStepResult
from unity_psf.training.high_fidelity.diagnostics import (
    artifact_group_metrics as _artifact_group_metrics,
    write_gamma_monitor_report as _write_gamma_monitor_report,
)
from unity_psf.training.high_fidelity.gamma_projection import (
    base_coeff_maps_from_gamma_cfg as _base_coeff_maps_from_gamma_cfg,
    base_coeff_maps_from_train_cfg as _base_coeff_maps_from_train_cfg,
    build_vector_roi_gamma_objective as _build_vector_roi_gamma_objective,
    export_gamma_feedback_coeff_maps as _export_gamma_feedback_coeff_maps,
    gamma_update_config as _gamma_update_config,
    select_single_channel_roi_split as _select_single_channel_roi_split,
    should_run_gamma_update as _should_run_gamma_update,
    should_write_gamma_artifacts as _should_write_gamma_artifacts,
    split_roi_bank_by_domain as _split_roi_bank_by_domain,
)
from unity_psf.training.high_fidelity.physical_state import write_current_physical_state as _write_current_physical_state
from unity_psf.training.high_fidelity.posterior_workflow import sample_roi_bank_importance_update_context
from unity_psf.training.high_fidelity.roi_bank_builder import auto_build_roi_bank as _auto_build_roi_bank_impl
from unity_psf.training.high_fidelity.roi_bank_source import (
    ROIBankSource as _ROIBankSource,
    camera_backward_from_training_config as _camera_backward_from_training_config,
    resolve_roi_bank_source as _resolve_roi_bank_source,
    roi_bank_source_metrics as _roi_bank_source_metrics,
    roi_conditioning_context as _roi_conditioning_context,
)
from unity_psf.training.high_fidelity.roi_projection import (
    ROIProjectionUpdateContext as _ROIProjectionUpdateContext,
    posterior_bg_scale as _posterior_bg_scale,
    posterior_input_offset as _posterior_input_offset,
    posterior_input_scale as _posterior_input_scale,
    posterior_photon_scale as _posterior_photon_scale,
    posterior_z_scale_nm as _posterior_z_scale_nm,
    record_projection_tensors as _record_projection_tensors,
    select_roi_bank_update_subset as _select_roi_bank_update_subset,
)
from unity_psf.training.high_fidelity.roi_split import (
    resolve_roi_gamma_banks,
)
from unity_psf.training.high_fidelity.roi_workflow import build_roi_bank_from_loc_harvest_source
from unity_psf.training.high_fidelity.workflow_context import DeferredGammaFeedbackCommitter
from unity_psf.training.high_fidelity.heldout_monitor import (
    build_heldout_monitor_context as _build_heldout_monitor_context,
    heldout_monitor_metrics as _heldout_monitor_metrics,
)


def build_roi_bank_gamma_hook(
    train_cfg: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_base_dir: Path,
    model: torch.nn.Module,
    layout,
    condition_store: ConditioningProviderStore | None = None,
    physical_context: ChannelTrainingContext | None = None,
):
    gamma_cfg = train_cfg.get("roi_bank_gamma")
    if not isinstance(gamma_cfg, Mapping) or gamma_cfg.get("enabled") is not True:
        return None
    roi_source = _resolve_roi_bank_source(gamma_cfg, train_cfg=train_cfg, config=config, config_base_dir=config_base_dir)
    if "roi_library_path" in gamma_cfg:
        selected_bank, heldout_bank, heldout_mode, split_source = resolve_roi_gamma_banks(
            load_roi_bank(str(gamma_cfg["roi_library_path"])),
            gamma_cfg,
        )
        return _build_domain_roi_bank_gamma_hooks(
            gamma_cfg,
            train_cfg=train_cfg,
            config=config,
            model=model,
            layout=layout,
            selected_bank=selected_bank,
            heldout_bank=heldout_bank,
            heldout_mode=heldout_mode,
            heldout_split_source=split_source,
            objective_source="roi_projection_hdf5",
            roi_library_path=str(gamma_cfg["roi_library_path"]),
            condition_store=condition_store,
            physical_context=physical_context,
        )
    elif roi_source is not None:
        return _build_lazy_auto_roi_projection_objective(
            gamma_cfg,
            model=model,
            roi_source=roi_source,
            layout=layout,
            train_cfg=train_cfg,
            config=config,
            condition_store=condition_store,
            physical_context=physical_context,
        )
    elif gamma_cfg.get("smoke_roi_library") is True:
        return _build_domain_roi_bank_gamma_hooks(
            gamma_cfg,
            train_cfg=train_cfg,
            config=config,
            model=model,
            layout=layout,
            selected_bank=_smoke_roi_bank(roi_size=int(gamma_cfg.get("roi_size_px", 8))),
            heldout_bank=None,
            heldout_mode="not_configured",
            heldout_split_source=None,
            objective_source="roi_projection_smoke",
            roi_library_path=None,
            roi_library_source="smoke",
            condition_store=condition_store,
            physical_context=physical_context,
        )
    objective = _build_vector_roi_gamma_objective(gamma_cfg, train_cfg=train_cfg, config=config, model=model)
    return build_gamma_update_hook(
        state=GammaUpdateState(gamma=objective.initial_gamma()),
        localizer=model,
        layout=layout,
        config=_gamma_update_config(gamma_cfg, train_cfg),
        objective_fn=lambda gamma: (gamma - 1.0).square().sum(),
        feedback_fn=_build_gamma_feedback_fn(
            objective,
            layout=layout,
            condition_store=condition_store,
            physical_context=physical_context,
        ),
    )


def gamma_hook_bindings(gamma_hook, gamma_cfg: object):
    if gamma_hook is None:
        return None, None
    if isinstance(gamma_cfg, Mapping) and gamma_cfg.get("start_batch") is not None:
        return None, gamma_hook
    return gamma_hook, None


def _build_domain_roi_bank_gamma_hooks(
    gamma_cfg: Mapping[str, Any],
    *,
    train_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
    model: torch.nn.Module,
    layout,
    selected_bank: ROIBank,
    heldout_bank: ROIBank | None,
    heldout_mode: str,
    heldout_split_source: str | None,
    objective_source: str,
    roi_library_path: str | None,
    roi_library_source: str | None = None,
    roi_bank_source: _ROIBankSource | None = None,
    condition_store: ConditioningProviderStore | None = None,
    physical_context: ChannelTrainingContext | None = None,
):
    update_config = _gamma_update_config(gamma_cfg, train_cfg)
    splits = _split_roi_bank_by_domain(selected_bank, heldout_bank)
    if physical_context is not None:
        splits = _select_single_channel_roi_split(splits, physical_context.channel.channel_id)
    hooks = []
    objectives: dict[str, GammaProjectionObjective] = {}
    latest_coeff_maps = {
        str(domain): str(path)
        for domain, path in _base_coeff_maps_from_gamma_cfg(gamma_cfg) or _base_coeff_maps_from_train_cfg(train_cfg)
    }
    domain_order = tuple(sorted(splits))
    deferred_committer = (
        DeferredGammaFeedbackCommitter(
            layout=layout,
            condition_store=condition_store,
            latest_coeff_maps=latest_coeff_maps,
        )
        if condition_store is not None and len(domain_order) > 1
        else None
    )
    final_domain = domain_order[-1] if domain_order else None
    for domain_name in domain_order:
        domain_selected, domain_heldout = splits[domain_name]
        domain_heldout_mode = (
            heldout_mode
            if domain_heldout is not None or heldout_mode == "disabled_insufficient_records"
            else "not_configured"
        )
        domain_heldout_split_source = (
            heldout_split_source
            if domain_heldout is not None or heldout_mode == "disabled_insufficient_records"
            else None
        )
        objective = _build_vector_roi_gamma_objective(gamma_cfg, train_cfg=train_cfg, config=config, model=model)
        objectives[domain_name] = objective
        objective_fn, metrics_fn, prepare_fn = _build_roi_projection_objective(
            gamma_cfg,
            model=model,
            bank=domain_selected,
            objective_source=objective_source,
            roi_library_path=roi_library_path,
            heldout_bank=domain_heldout,
            heldout_mode=domain_heldout_mode,
            heldout_split_source=domain_heldout_split_source,
            layout=layout,
            roi_library_source=roi_library_source,
            roi_bank_source=roi_bank_source,
            objective=objective,
            train_cfg=train_cfg,
        )
        state = GammaUpdateState(gamma=objective.initial_gamma())
        hooks.append(
            build_gamma_update_hook(
                state=state,
                localizer=model,
                layout=layout,
                config=update_config,
                objective_fn=objective_fn,
                prepare_fn=prepare_fn,
                metrics_fn=metrics_fn,
                feedback_fn=_build_gamma_feedback_fn(
                    objective,
                    gamma_cfg=gamma_cfg,
                    layout=layout,
                    condition_store=condition_store,
                    domain_names=(domain_name,),
                    latest_coeff_maps=latest_coeff_maps,
                    deferred_committer=deferred_committer,
                    commit_deferred=bool(deferred_committer is not None and domain_name == final_domain),
                    physical_context=physical_context,
                ),
            )
        )
    if not hooks:
        return None

    def hook(result: TrainingRunEpochResult | TrainingStepResult) -> None:
        for item in hooks:
            item(result)
        if deferred_committer is not None and deferred_committer.pending_domains:
            with profiling.time_block("deferred_gamma_feedback_commit_fallback"):
                deferred_committer.commit()
            profiling.drain()

    return hook


def _build_gamma_feedback_fn(
    objective: GammaProjectionObjective,
    *,
    gamma_cfg: Mapping[str, Any],
    layout,
    condition_store: ConditioningProviderStore | None,
    domain_names: tuple[str, ...] | None = None,
    latest_coeff_maps: dict[str, str] | None = None,
    deferred_committer: DeferredGammaFeedbackCommitter | None = None,
    commit_deferred: bool = False,
    physical_context: ChannelTrainingContext | None = None,
):
    if condition_store is None or not objective.base_maps_by_domain:
        return None

    def feedback_fn(gamma: torch.Tensor, result: TrainingRunEpochResult | TrainingStepResult, metrics: dict[str, object]) -> dict[str, object]:
        gamma_before = metrics.get("_gamma_before")
        if not torch.is_tensor(gamma_before):
            gamma_before = torch.zeros_like(gamma.detach())
        entries = _export_gamma_feedback_coeff_maps(
            objective,
            gamma=gamma,
            gamma_before=gamma_before,
            domain_names=domain_names,
            layout=layout,
            epoch=int(result.epoch),
            global_step=int(result.global_step),
            artifact_policy=str(gamma_cfg.get("artifact_retention_policy", "per_update")),
        )
        if deferred_committer is not None:
            store_entries = deferred_committer.stage(entries=entries, result=result)
            output: dict[str, object] = {
                "feedback_coeff_maps": {name: path for name, path in entries},
                "feedback_coeff_maps_all_domains": {str(item["name"]): str(item["coeff_maps_npz"]) for item in store_entries},
                "feedback_domain_names": [name for name, _path in entries],
                "feedback_deferred": True,
                "feedback_deferred_commit": False,
                "feedback_deferred_pending_domain_count": len(deferred_committer.pending_domains),
                "condition_store_version": condition_store.version,
            }
            if commit_deferred:
                with profiling.time_block("deferred_gamma_feedback_commit"):
                    output.update(deferred_committer.commit())
            return output
        if physical_context is not None:
            if len(entries) != 1:
                raise RuntimeError("single-channel physical context received multiple gamma coefficient maps")
            physical_context.update_coefficient_map(entries[0][1])
            store_entries = (
                {
                    "name": physical_context.channel.channel_id,
                    "coeff_maps_npz": str(physical_context.coefficient_map_path),
                },
            )
            state_path = physical_context.write_physical_state(
                source="gamma_feedback",
                epoch=int(result.epoch),
                global_step=int(result.global_step),
                condition_store_version=physical_context.condition_store.version,
            )
        elif latest_coeff_maps is not None:
            for name, path in entries:
                latest_coeff_maps[str(name)] = str(path)
            store_entries = tuple(
                {"name": name, "coeff_maps_npz": path}
                for name, path in sorted(latest_coeff_maps.items())
            )
        else:
            store_entries = tuple({"name": name, "coeff_maps_npz": path} for name, path in entries)
        if physical_context is None:
            condition_store.update_from_coeff_maps(store_entries)
            state_path = _write_current_physical_state(
                layout,
                coeff_maps=store_entries,
                source="gamma_feedback",
                epoch=int(result.epoch),
                global_step=int(result.global_step),
                condition_store_version=condition_store.version,
            )
        return {
            "feedback_coeff_maps": {name: path for name, path in entries},
            "feedback_coeff_maps_all_domains": {str(item["name"]): str(item["coeff_maps_npz"]) for item in store_entries},
            "feedback_domain_names": [name for name, _path in entries],
            "condition_store_version": condition_store.version,
            "physical_state_path": state_path,
        }

    return feedback_fn


def _build_lazy_auto_roi_projection_objective(
    gamma_cfg: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    roi_source: _ROIBankSource,
    layout,
    train_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
    condition_store: ConditioningProviderStore | None,
    physical_context: ChannelTrainingContext | None = None,
):
    cached: dict[str, Any] = {}

    def ensure_built():
        if "hook" not in cached:
            selected_bank, heldout_bank, heldout_mode, split_source = resolve_roi_gamma_banks(
                _auto_build_roi_bank(gamma_cfg, roi_source=roi_source, model=model, train_cfg=train_cfg),
                gamma_cfg,
            )
            cached["hook"] = _build_domain_roi_bank_gamma_hooks(
                gamma_cfg,
                train_cfg=train_cfg,
                config=config,
                model=model,
                layout=layout,
                selected_bank=selected_bank,
                heldout_bank=heldout_bank,
                heldout_mode=heldout_mode,
                heldout_split_source=split_source,
                objective_source="roi_projection_auto_built",
                roi_library_path=None,
                roi_library_source="auto_built",
                roi_bank_source=roi_source,
                condition_store=condition_store,
                physical_context=physical_context,
            )
        return cached["hook"]

    def hook(result: TrainingRunEpochResult | TrainingStepResult) -> None:
        if not _should_run_gamma_update(result, _gamma_update_config(gamma_cfg, train_cfg)):
            return
        built_hook = ensure_built()
        if built_hook is not None:
            built_hook(result)

    return hook


def roi_bank_gamma_route(
    train_cfg: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_base_dir: Path,
) -> dict[str, Any]:
    gamma_cfg = train_cfg.get("roi_bank_gamma")
    if not isinstance(gamma_cfg, Mapping) or gamma_cfg.get("enabled") is not True:
        return {"enabled": False}
    roi_source = _resolve_roi_bank_source(gamma_cfg, train_cfg=train_cfg, config=config, config_base_dir=config_base_dir)
    return {
        "enabled": True,
        "start_epoch": int(gamma_cfg.get("start_epoch", 1)),
        "stop_epoch": int(gamma_cfg.get("stop_epoch", train_cfg.get("epochs", 1))),
        "update_interval_epochs": int(gamma_cfg.get("update_interval_epochs", gamma_cfg.get("interval_epochs", 1))),
        **({"start_batch": int(gamma_cfg["start_batch"])} if "start_batch" in gamma_cfg else {}),
        **({"stop_batch": int(gamma_cfg["stop_batch"])} if "stop_batch" in gamma_cfg else {}),
        **(
            {"update_interval_batches": int(gamma_cfg["update_interval_batches"])}
            if "update_interval_batches" in gamma_cfg
            else {}
        ),
        "gamma_steps": int(gamma_cfg.get("gamma_steps", gamma_cfg.get("steps", 1))),
        "gamma_lr": float(gamma_cfg.get("gamma_lr", gamma_cfg.get("lr", 0.01))),
        "artifact_retention_policy": str(gamma_cfg.get("artifact_retention_policy", "per_update")),
        "diagnostic_interval_updates": int(gamma_cfg.get("diagnostic_interval_updates", gamma_cfg.get("artifact_interval_updates", 10))),
        "objective_source": _gamma_objective_source(gamma_cfg, roi_source=roi_source),
        **({"roi_library_source": "auto_built"} if roi_source is not None else {}),
        **(_roi_bank_source_metrics(roi_source) if roi_source is not None else {}),
        **(
            {"auto_build_source_path": str(gamma_cfg["auto_build_source_path"])}
            if gamma_cfg.get("auto_build_roi_bank") is True and "auto_build_source_path" in gamma_cfg
            else {}
        ),
        **({"roi_library_path": str(gamma_cfg["roi_library_path"])} if "roi_library_path" in gamma_cfg else {}),
        **(
            {"auto_heldout_min_rois": int(gamma_cfg["auto_heldout_min_rois"])}
            if "auto_heldout_min_rois" in gamma_cfg
            else {}
        ),
        **(
            {"auto_heldout_max_rois": int(gamma_cfg["auto_heldout_max_rois"])}
            if "auto_heldout_max_rois" in gamma_cfg
            else {}
        ),
        **(
            {"heldout_roi_library_path": str(gamma_cfg["heldout_roi_library_path"])}
            if "heldout_roi_library_path" in gamma_cfg
            else {}
        ),
    }


def _build_roi_projection_objective(
    gamma_cfg: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    bank: ROIBank,
    objective_source: str,
    roi_library_path: str | None,
    heldout_bank: ROIBank | None,
    heldout_mode: str,
    heldout_split_source: str | None,
    layout,
    roi_library_source: str | None = None,
    roi_bank_source: _ROIBankSource | None = None,
    objective: GammaProjectionObjective,
    train_cfg: Mapping[str, Any],
):
    over_cut = int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0)))
    max_emitters = int(gamma_cfg.get("num_posterior_samples", 2))
    objective_mode = str(gamma_cfg.get("roi_bank_objective", gamma_cfg.get("objective_mode", "poisson_nll"))).strip().lower()
    update_context: dict[str, _ROIProjectionUpdateContext] = {}
    heldout_context = _build_heldout_monitor_context(
        gamma_cfg,
        model=model,
        bank=heldout_bank,
        objective=objective,
        max_emitters=max_emitters,
        mode=heldout_mode,
        split_source=heldout_split_source,
        train_cfg=train_cfg,
    )

    def prepare_fn(result: TrainingRunEpochResult | TrainingStepResult) -> dict[str, object]:
        if objective_mode == "importance_wake":
            context, sampling_metrics = sample_roi_bank_importance_update_context(
                bank,
                gamma_cfg,
                model=model,
                train_cfg=train_cfg,
                step_index=int(result.global_step),
            )
            update_context["active"] = context
        else:
            selected_bank, sampling_metrics = _select_roi_bank_update_subset(bank, gamma_cfg, epoch=int(result.epoch))
            raw_frames, background, samples, roi_origin_xy_px, domain_names, loss_mask = _record_projection_tensors(selected_bank)
            update_context["active"] = _ROIProjectionUpdateContext(
                bank=selected_bank,
                sampling_metrics=sampling_metrics,
                raw_frames=raw_frames,
                background=background,
                samples=samples,
                roi_origin_xy_px=roi_origin_xy_px,
                domain_names=domain_names,
                loss_mask=loss_mask,
            )
        return sampling_metrics

    def active_context() -> _ROIProjectionUpdateContext:
        context = update_context.get("active")
        if context is None:
            if objective_mode == "importance_wake":
                context, _sampling_metrics = sample_roi_bank_importance_update_context(
                    bank,
                    gamma_cfg,
                    model=model,
                    train_cfg=train_cfg,
                    step_index=0,
                )
            else:
                selected_bank, sampling_metrics = _select_roi_bank_update_subset(bank, gamma_cfg, epoch=0)
                raw_frames, background, samples, roi_origin_xy_px, domain_names, loss_mask = _record_projection_tensors(selected_bank)
                context = _ROIProjectionUpdateContext(
                    bank=selected_bank,
                    sampling_metrics=sampling_metrics,
                    raw_frames=raw_frames,
                    background=background,
                    samples=samples,
                    roi_origin_xy_px=roi_origin_xy_px,
                    domain_names=domain_names,
                    loss_mask=loss_mask,
                )
            update_context["active"] = context
        return context

    def objective_fn(gamma: torch.Tensor) -> torch.Tensor:
        context = active_context()
        return objective(
            gamma=gamma,
            samples=context.samples,
            raw_frames=context.raw_frames,
            background=context.background,
            roi_origin_xy_px=context.roi_origin_xy_px,
            domain_names=context.domain_names,
            loss_mask=context.loss_mask,
        )

    def metrics_fn(
        gamma: torch.Tensor,
        selected_loss: torch.Tensor,
        gamma_before: torch.Tensor,
        result: TrainingRunEpochResult | TrainingStepResult,
        base_metrics: dict[str, object],
    ) -> dict[str, object]:
        context = active_context()
        active = context.samples.mask
        active_photons = context.samples.xyzph[..., 3][active]
        selected_projected_photons = float(active_photons.sum().item()) if active_photons.numel() else 0.0
        selected_sampled_emitters = int(active.sum().item())
        selected_step = int(gamma_cfg.get("gamma_steps", gamma_cfg.get("steps", 1)))
        metrics = {
            "objective_source": objective_source,
            **({"roi_library_source": roi_library_source} if roi_library_source is not None else {}),
            **(_roi_bank_source_metrics(roi_bank_source) if roi_bank_source is not None else {}),
            "roi_count": len(bank.records),
            "selected_roi_count": len(context.bank.records),
            "posterior_max_emitters": max_emitters,
            "over_cut_px": over_cut,
            "loss_mask_mode": "valid_core" if context.loss_mask is not None else "over_cut",
            "loss_mask_pixel_count": 0 if context.loss_mask is None else int(context.loss_mask.sum().item()),
            "posterior_active_emitters": selected_sampled_emitters,
            "best_step": selected_step,
            "selected_step": selected_step,
            "selected_poisson_nll": float(selected_loss.item()),
            "selected_sample_count": len(context.bank.records),
            "selected_sampled_emitter_count": selected_sampled_emitters,
            "selected_projected_photons": selected_projected_photons,
            "selected_background_mean": float(context.background.mean().item()),
            "projection_sample_source": str(context.sampling_metrics.get("projection_sample_source", "roi_record_emitters")),
            **context.sampling_metrics,
            **{f"objective_{key}": value for key, value in objective.last_metrics.items()},
            **_artifact_group_metrics(context.bank, roi_library_source=roi_library_source, objective_source=objective_source),
            **_heldout_monitor_metrics(
                heldout_context,
                objective=objective,
                gamma_before=gamma_before,
                gamma_after=gamma,
                unavailable_mode=heldout_mode,
                unavailable_split_source=heldout_split_source,
            ),
            "checkpoint_path": None,
            "report_path": None,
        }
        if roi_library_path is not None:
            metrics["roi_library_path"] = roi_library_path
        if _should_write_gamma_artifacts(gamma_cfg, result=result):
            metrics.update(
                _write_gamma_monitor_report(
                    layout,
                    {**base_metrics, **metrics},
                    result=result,
                    raw_frames=context.raw_frames,
                    background=context.background,
                    samples=context.samples,
                    gamma_before=gamma_before,
                    gamma=gamma,
                    objective=objective,
                    roi_origin_xy_px=context.roi_origin_xy_px,
                    domain_names=context.domain_names,
                )
            )
        else:
            metrics.update(
                {
                    "summary_path": None,
                    "report_path": None,
                    "diagnostic_png_path": None,
                    "diagnostic_observed_units": None,
                    "diagnostics_manifest_path": None,
                    "artifact_write_skipped": True,
                    "artifact_retention_policy": str(gamma_cfg.get("artifact_retention_policy", "per_update")),
                }
            )
        return metrics

    return objective_fn, metrics_fn, prepare_fn


def _auto_build_roi_bank(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: _ROIBankSource,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
) -> ROIBank:
    return _auto_build_roi_bank_impl(
        gamma_cfg,
        roi_source=roi_source,
        model=model,
        train_cfg=train_cfg,
        camera_backward_from_training_config=_camera_backward_from_training_config,
        posterior_photon_scale=_posterior_photon_scale,
        posterior_z_scale_nm=_posterior_z_scale_nm,
        roi_conditioning_context=_roi_conditioning_context,
        loc_harvest_builder=_auto_build_roi_bank_from_loc_harvest,
    )


def _auto_build_roi_bank_from_loc_harvest(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: _ROIBankSource,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
    config: ROIBankBuildConfig,
    roi_size: int,
) -> ROIBank:
    return build_roi_bank_from_loc_harvest_source(
        gamma_cfg,
        roi_source=roi_source,
        model=model,
        train_cfg=train_cfg,
        config=config,
        roi_size=roi_size,
        input_offset=_posterior_input_offset(train_cfg),
        input_scale=_posterior_input_scale(train_cfg),
        photon_scale=_posterior_photon_scale(train_cfg),
        z_scale=_posterior_z_scale_nm(train_cfg),
        bg_scale=_posterior_bg_scale(train_cfg),
        roi_conditioning_context=_roi_conditioning_context,
    )


def _smoke_roi_bank(*, roi_size: int) -> ROIBank:
    raw = np.full((3, int(roi_size), int(roi_size)), 0.2, dtype=np.float32)
    center = int(roi_size) // 2
    raw[:, center, center] = 8.0
    background = np.full((int(roi_size), int(roi_size)), 0.2, dtype=np.float32)
    record = ROIRecord(
        roi_id=0,
        domain_name="smoke",
        frame_window=(0, 3),
        roi_origin_xy_px=(0.0, 0.0),
        raw_frames_photon=raw,
        background_mu=background,
        background_smoothed=background,
        grid_cell_id=0,
        emitters=(),
        summary={"source": "smoke_roi_projection"},
    )
    return ROIBank(records=(record,), metadata={"source": "smoke_roi_projection"})


def _count_gamma_updates(layout) -> int:
    path = layout.metrics_dir / "gamma_update_metrics.jsonl"
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _gamma_objective_source(gamma_cfg: Mapping[str, Any], *, roi_source: _ROIBankSource | None = None) -> str:
    if "roi_library_path" in gamma_cfg:
        return "roi_projection_hdf5"
    if roi_source is not None or gamma_cfg.get("auto_build_roi_bank") is True:
        return "roi_projection_auto_built"
    if gamma_cfg.get("smoke_roi_library") is True:
        return "roi_projection_smoke"
    return "smoke_quadratic"
