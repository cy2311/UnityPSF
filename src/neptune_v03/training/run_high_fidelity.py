from __future__ import annotations

import argparse
import json
import random
import struct
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import tifffile

from neptune_v03.config import load_config
from neptune_v03.gamma_update import (
    GammaProjectionObjective,
    GammaProjectionObjectiveConfig,
    GammaUpdateConfig,
    GammaUpdateState,
    build_gamma_update_hook,
)
from neptune_v03.localization import build_localization_model_registry, build_localization_runtime_config
from neptune_v03.localization.conditioning import ConditioningProviderStore, FullResZernikeConditioning
from neptune_v03.localization.online import OnlineBatchProviderConfig, build_online_batch_provider
from neptune_v03.localization.posterior import DetectionPosteriorSamples, sample_detection_posterior
from neptune_v03.localization.roi_posterior_sampling import (
    CurrentROILibraryPosteriorSampler,
    ROILibraryConditionBuilder,
    ROIPosteriorSamplingConfig,
    SampledEmitterSet,
)
from neptune_v03.localization.roi_batches import build_roi_batch_provider
from neptune_v03.localization.training_adapter import LocalizationTrainBatch, localization_batch_to_device
from neptune_v03.optics.vector_psf import render_vector_psf_bank
from neptune_v03.optics.nat_field import full_roi_coeff_stack_torch
from neptune_v03.peak import PeakBootstrapConfig, run_peak_bootstrap_pipeline
from neptune_v03.roi_library import (
    InferredEmitter,
    ROIBank,
    ROIBankBuildConfig,
    ROIBankDomain,
    ROIRecord,
    RawInferenceResult,
    build_roi_bank_from_inference,
    load_roi_bank,
)
from neptune_v03.roi_library.loc_harvest import (
    LocHarvestConfig,
    _build_inference_frame_proc,
    build_roi_bank_from_loc_harvest,
)
from neptune_v03.runtime import ensure_run_layout, profiling, write_run_manifest, write_stage_status
from neptune_v03.training import build_trainer_runtime, train_epochs
from neptune_v03.training.localizer_eval import build_localizer_eval_provider, localizer_eval_route, make_legacy_localization_eval_loss
from neptune_v03.training.loop import TrainingRunEpochResult, TrainingStepResult


STAGE_NAME = "high_fidelity_localization"


@dataclass(frozen=True)
class _HeldoutMonitorContext:
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


@dataclass(frozen=True)
class _ROIBankSource:
    mode: str
    raw_path: str
    candidate_mode: str
    frame_range: tuple[int, int] | None
    domains: tuple[Mapping[str, Any], ...] | None
    alias: str | None = None


class _DeferredGammaFeedbackCommitter:
    """Publish multi-domain physical feedback once per update cycle."""

    def __init__(
        self,
        *,
        layout,
        condition_store: ConditioningProviderStore,
        latest_coeff_maps: dict[str, str],
    ) -> None:
        self.layout = layout
        self.condition_store = condition_store
        self.latest_coeff_maps = latest_coeff_maps
        self.pending_domains: set[str] = set()
        self.last_result: TrainingRunEpochResult | TrainingStepResult | None = None
        self.last_store_entries: tuple[dict[str, str], ...] = ()
        self.committed_this_cycle = False

    def stage(
        self,
        *,
        entries: tuple[tuple[str, str], ...],
        result: TrainingRunEpochResult | TrainingStepResult,
    ) -> tuple[dict[str, str], ...]:
        for name, path in entries:
            self.latest_coeff_maps[str(name)] = str(path)
            self.pending_domains.add(str(name))
        self.last_result = result
        self.last_store_entries = tuple(
            {"name": name, "coeff_maps_npz": path}
            for name, path in sorted(self.latest_coeff_maps.items())
        )
        self.committed_this_cycle = False
        return self.last_store_entries

    def commit(self) -> dict[str, object]:
        if not self.pending_domains:
            return {
                "feedback_deferred_commit_skipped": True,
                "feedback_deferred_pending_domain_count": 0,
                "condition_store_version": self.condition_store.version,
            }
        if self.last_result is None:
            raise RuntimeError("deferred gamma feedback has pending domains but no trigger result")
        if not self.last_store_entries:
            raise RuntimeError("deferred gamma feedback has pending domains but no coeff-map entries")
        self.condition_store.update_from_coeff_maps(self.last_store_entries)
        state_path = _write_current_physical_state(
            self.layout,
            coeff_maps=self.last_store_entries,
            source="gamma_feedback",
            epoch=int(self.last_result.epoch),
            global_step=int(self.last_result.global_step),
            condition_store_version=self.condition_store.version,
        )
        committed = sorted(self.pending_domains)
        self.pending_domains.clear()
        self.committed_this_cycle = True
        return {
            "feedback_deferred_commit": True,
            "feedback_deferred_committed_domains": committed,
            "feedback_deferred_committed_domain_count": len(committed),
            "feedback_deferred_pending_domain_count": 0,
            "feedback_coeff_maps_all_domains": {
                str(item["name"]): str(item["coeff_maps_npz"])
                for item in self.last_store_entries
            },
            "condition_store_version": self.condition_store.version,
            "physical_state_path": state_path,
        }


@dataclass(frozen=True)
class _ROIProjectionUpdateContext:
    bank: ROIBank
    sampling_metrics: dict[str, object]
    raw_frames: torch.Tensor
    background: torch.Tensor
    samples: DetectionPosteriorSamples
    roi_origin_xy_px: torch.Tensor
    domain_names: list[str]
    loss_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class _DomainPeakLayout:
    base: Any
    domain_name: str

    @property
    def run_dir(self):
        return self.base.run_dir

    @property
    def logs_dir(self):
        return self.base.logs_dir

    @property
    def cache_dir(self):
        return self.base.cache_dir

    @property
    def metadata_dir(self):
        return self.base.metadata_dir

    @property
    def checkpoints_dir(self):
        return self.base.checkpoints_dir

    @property
    def metrics_dir(self):
        return self.base.metrics_dir

    @property
    def artifacts_dir(self):
        return self.base.artifacts_dir

    def stage_dir(self, stage: str) -> Path:
        if str(stage) == "peak":
            return self.base.stage_dir("peak") / _path_token(self.domain_name)
        return self.base.stage_dir(stage)


def _condition_store_from_runtime_config(runtime_config: Mapping[str, Any]) -> ConditioningProviderStore | None:
    batch_provider = runtime_config.get("batch_provider")
    if not isinstance(batch_provider, Mapping):
        return None
    params = batch_provider.get("params")
    if not isinstance(params, Mapping):
        return None
    entries = params.get("dual_domain_coeff_maps")
    if not entries:
        return None
    if not isinstance(entries, (tuple, list)):
        return None
    return ConditioningProviderStore.from_coeff_maps(tuple(entries))


def _condition_store_batch_provider_overrides(condition_store: ConditioningProviderStore | None):
    if condition_store is None:
        return None

    def online_train_batch(params: dict[str, object]):
        return build_online_batch_provider(OnlineBatchProviderConfig(**params), condition_store=condition_store)

    return {"online_train_batch": online_train_batch}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neptune v0.3 high-fidelity localization training.")
    parser.add_argument("--config", required=True, help="Resolved YAML config.")
    parser.add_argument("--run-root", required=True, help="Directory containing the run directory.")
    parser.add_argument("--run-name", required=True, help="Relative run directory name.")
    parser.add_argument("--seed", type=int, default=0, help="Training batch seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config_base_dir = Path(args.config).resolve().parent
    layout = ensure_run_layout(Path(args.run_root), args.run_name, stage_names=("peak", STAGE_NAME))
    try:
        config = _run_peak_zmap_bootstrap_if_enabled(config, config_base_dir=config_base_dir, layout=layout)
        feedback_path = _feedback_map_path(config)
        train_cfg = _mapping(config.get("train"), "train")
        runtime_config = build_localization_runtime_config(
            config,
            config_base_dir=config_base_dir,
            seed=int(args.seed),
        )
        model_name = str(runtime_config["model"]["name"])
        condition_store = _condition_store_from_runtime_config(runtime_config)
        gamma_route = _roi_bank_gamma_route(train_cfg, config=config, config_base_dir=config_base_dir)
        localizer_eval = localizer_eval_route(train_cfg, config_base_dir=config_base_dir)
        write_run_manifest(
            layout,
            {
                "stage": STAGE_NAME,
                "config_path": str(args.config),
                "seed": int(args.seed),
                "model_name": model_name,
                "batch_provider": runtime_config["batch_provider"]["name"],
                "feedback_map_path": feedback_path,
                "resolved_contract": runtime_config.get("resolved_contract", {}),
                "roi_bank_gamma": gamma_route,
                "localizer_eval": localizer_eval,
            },
        )
        _write_initial_physical_state(layout, runtime_config)
        runtime = build_trainer_runtime(
            runtime_config,
            layout=layout,
            model_registry=build_localization_model_registry(),
            batch_provider_overrides=_condition_store_batch_provider_overrides(condition_store),
        )
        gamma_hook = _build_roi_bank_gamma_hook(
            train_cfg,
            config=config,
            config_base_dir=config_base_dir,
            model=runtime.model,
            layout=layout,
            condition_store=condition_store,
        )
        gamma_epoch_hook, gamma_batch_hook = _gamma_hook_bindings(gamma_hook, train_cfg.get("roi_bank_gamma"))
        eval_provider = build_localizer_eval_provider(
            train_cfg,
            config_base_dir=config_base_dir,
            root_config=config,
            condition_store=condition_store,
        )
        eval_loss_fn = make_legacy_localization_eval_loss(runtime.loss_fn, train_cfg, root_config=config) if eval_provider is not None else None
        results = train_epochs(
            model=runtime.model,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            batch_provider=runtime.batch_provider,
            layout=runtime.layout,
            config=runtime.config,
            on_epoch_end=gamma_epoch_hook,
            on_batch_end=gamma_batch_hook,
            loss_fn=runtime.loss_fn,
            eval_provider=eval_provider,
            eval_loss_fn=eval_loss_fn,
            checkpoint_extra_fn=_physical_checkpoint_extra_fn(layout),
        )
        gamma_updates = _count_gamma_updates(layout)
        best_checkpoint_path = layout.checkpoints_dir / "checkpoint_best.pt"
        write_stage_status(
            layout,
            STAGE_NAME,
            "completed",
            {
                "epochs": [result.epoch for result in results],
                "global_step": results[-1].global_step if results else 0,
                "training_budget_mode": "max_batches" if runtime.config.max_batches is not None else "epochs",
                "max_batches": runtime.config.max_batches,
                "model_name": model_name,
                "resolved_contract": runtime_config.get("resolved_contract", {}),
                "roi_bank_gamma": {**gamma_route, "updates": gamma_updates},
                "localizer_eval": {
                    **localizer_eval,
                    "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
                },
            },
        )
    except Exception as exc:
        write_stage_status(layout, STAGE_NAME, "failed", {"error": str(exc)})
        raise
    return 0


def _run_peak_zmap_bootstrap_if_enabled(
    config: Mapping[str, Any],
    *,
    config_base_dir: Path,
    layout,
) -> dict[str, Any]:
    train_cfg = _mapping(config.get("train"), "train")
    bootstrap_cfg = train_cfg.get("peak_zmap_bootstrap")
    if not isinstance(bootstrap_cfg, Mapping) or bootstrap_cfg.get("enabled") is not True:
        return dict(config)

    real_tiff_cfg = _mapping(train_cfg.get("real_tiff_wake"), "train.real_tiff_wake")
    raw_path = real_tiff_cfg.get("tiff_path")
    if raw_path is None:
        raw_path = _phase_retrieval_tiff_path(config)
    if raw_path is None:
        raise ValueError("train.peak_zmap_bootstrap.enabled=True requires train.real_tiff_wake.tiff_path")
    tiff_path = Path(_resolve_config_path(str(raw_path), base_dir=config_base_dir))

    domains = real_tiff_cfg.get("domains")
    if not isinstance(domains, list) or not domains:
        height = int(_mapping(train_cfg.get("online_generation"), "train.online_generation").get("height", 128))
        width = int(_mapping(train_cfg.get("online_generation"), "train.online_generation").get("width", 128))
        domains = [{"name": "left", "crop_left": 0, "crop_top": 0, "crop_width": width, "crop_height": height}]

    coeff_maps: list[dict[str, str]] = []
    peak_summaries: dict[str, Any] = {}
    for idx, domain_raw in enumerate(domains):
        domain = _mapping(domain_raw, "train.real_tiff_wake.domains[]")
        name = str(domain.get("name", f"domain{idx}"))
        peak_layout = _DomainPeakLayout(layout, name)
        result = run_peak_bootstrap_pipeline(
            layout=peak_layout,
            config=_peak_bootstrap_config(
                bootstrap_cfg,
                domain=domain,
                name=name,
                tiff_path=tiff_path,
            ),
        )
        coeff_path = Path(result.export["coeff_map_path"]).resolve()
        zmap_path = Path(result.export["zmap_path"]).resolve()
        coeff_maps.append({"name": name, "coeff_maps_npz": str(coeff_path)})
        peak_summaries[name] = {
            "summary_path": str(result.artifacts.summary_path),
            "coeff_map_path": str(coeff_path),
            "zmap_path": str(zmap_path),
            "selected_emitters": int(result.summary.selected_emitters),
            "kept_count": int(result.summary.kept_count),
        }

    updated = dict(config)
    updated_train = dict(train_cfg)
    online_cfg = dict(_mapping(updated_train.get("online_generation"), "train.online_generation"))
    online_cfg["dual_domain_coeff_maps"] = coeff_maps
    updated_train["online_generation"] = online_cfg
    updated["train"] = updated_train
    updated.setdefault("metadata", {})
    updated["metadata"] = {
        **dict(updated.get("metadata", {})),
        "peak_zmap_bootstrap": {
            "source": "raw_tiff_peak_bootstrap",
            "tiff_path": str(tiff_path),
            "domains": peak_summaries,
        },
    }
    return updated


def _peak_bootstrap_config(
    bootstrap_cfg: Mapping[str, Any],
    *,
    domain: Mapping[str, Any],
    name: str,
    tiff_path: Path,
) -> PeakBootstrapConfig:
    frame_start = int(bootstrap_cfg.get("frame_start", 0))
    frame_stop = int(bootstrap_cfg.get("frame_stop", 100))
    crop_left = int(domain.get("crop_left", 0))
    crop_top = int(domain.get("crop_top", 0))
    return PeakBootstrapConfig(
        sample=str(bootstrap_cfg.get("sample", "microtube")),
        side=name,
        frame_range=(frame_start, frame_stop),
        tiff_path=tiff_path,
        crop_x0=crop_left,
        crop_x1=crop_left + int(domain["crop_width"]),
        crop_y0=crop_top,
        crop_y1=crop_top + int(domain["crop_height"]),
        max_emitters=int(bootstrap_cfg.get("max_emitters", 1000)),
        target_selected_emitters=int(bootstrap_cfg.get("target_selected_emitters", 0)),
        min_distance_px=float(bootstrap_cfg.get("min_distance_px", 15.0)),
        gaussian_sigma_px=float(bootstrap_cfg.get("gaussian_sigma_px", 1.0)),
        threshold_sigma=float(bootstrap_cfg.get("threshold_sigma", 5.0)),
        patch_size_px=int(bootstrap_cfg.get("patch_size_px", 15)),
        nat_config_kind=str(bootstrap_cfg.get("nat_config_kind", "order1")),
        alternating_rounds=int(bootstrap_cfg.get("alternating_rounds", 3)),
        alternating_local_steps=int(bootstrap_cfg.get("alternating_local_steps", 2)),
        alternating_global_steps=int(bootstrap_cfg.get("alternating_global_steps", 2)),
        alternating_local_warmup_rounds=int(bootstrap_cfg.get("alternating_local_warmup_rounds", 0)),
        alternating_local_warmup_steps=int(bootstrap_cfg.get("alternating_local_warmup_steps", 0)),
        alternating_optimizer_kind=str(bootstrap_cfg.get("alternating_optimizer_kind", "lm")),
        global_projected_min_distance_px=float(bootstrap_cfg.get("global_projected_min_distance_px", 10.0)),
        spatial_balance_grid_px=int(bootstrap_cfg.get("spatial_balance_grid_px", 100)),
        spatial_balance_max_per_cell=int(bootstrap_cfg.get("spatial_balance_max_per_cell", 0)),
        max_patch_peak_distance_px=float(bootstrap_cfg.get("max_patch_peak_distance_px", 2.5)),
        max_secondary_peak_fraction=float(bootstrap_cfg.get("max_secondary_peak_fraction", 0.45)),
        min_center_peak_norm=float(bootstrap_cfg.get("min_center_peak_norm", 0.0)),
        min_signal_sum_norm=float(bootstrap_cfg.get("min_signal_sum_norm", 0.0)),
        ncc_threshold=float(bootstrap_cfg.get("ncc_threshold", 0.7)),
        freeze_initial_astig_standard=bool(bootstrap_cfg.get("freeze_initial_astig_standard", False)),
        freeze_defocus_zero_gauge=bool(bootstrap_cfg.get("freeze_defocus_zero_gauge", True)),
        vectorfit_astig_gauge=bool(bootstrap_cfg.get("vectorfit_astig_gauge", True)),
        vectorfit_astig_anchor_nm=bootstrap_cfg.get("vectorfit_astig_anchor_nm"),
        vectorfit_astig_anchor_mode=str(bootstrap_cfg.get("vectorfit_astig_anchor_mode", "init_only")),
        vectorfit_phasor_z_init=bool(bootstrap_cfg.get("vectorfit_phasor_z_init", True)),
        include_fixed_astig_baseline=bool(bootstrap_cfg.get("include_fixed_astig_baseline", False)),
    )


def _feedback_map_path(config: Mapping[str, Any]) -> str | None:
    train_cfg = _mapping(config.get("train"), "train")
    feedback_cfg = train_cfg.get("feedback")
    if isinstance(feedback_cfg, Mapping) and "map_path" in feedback_cfg:
        return str(feedback_cfg["map_path"])
    return None


def _build_roi_bank_gamma_hook(
    train_cfg: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_base_dir: Path,
    model: torch.nn.Module,
    layout,
    condition_store: ConditioningProviderStore | None = None,
):
    gamma_cfg = train_cfg.get("roi_bank_gamma")
    if not isinstance(gamma_cfg, Mapping) or gamma_cfg.get("enabled") is not True:
        return None
    roi_source = _resolve_roi_bank_source(gamma_cfg, train_cfg=train_cfg, config=config, config_base_dir=config_base_dir)
    if "roi_library_path" in gamma_cfg:
        selected_bank, heldout_bank, heldout_mode, split_source = _resolve_roi_gamma_banks(
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
        )
    objective = _build_vector_roi_gamma_objective(gamma_cfg, train_cfg=train_cfg, config=config, model=model)
    return build_gamma_update_hook(
        state=GammaUpdateState(gamma=objective.initial_gamma()),
        localizer=model,
        layout=layout,
        config=_gamma_update_config(gamma_cfg, train_cfg),
        objective_fn=lambda gamma: (gamma - 1.0).square().sum(),
        feedback_fn=_build_gamma_feedback_fn(objective, layout=layout, condition_store=condition_store),
    )


def _gamma_hook_bindings(gamma_hook, gamma_cfg: object):
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
):
    update_config = _gamma_update_config(gamma_cfg, train_cfg)
    splits = _split_roi_bank_by_domain(selected_bank, heldout_bank)
    hooks = []
    objectives: dict[str, GammaProjectionObjective] = {}
    latest_coeff_maps = {
        str(domain): str(path)
        for domain, path in _base_coeff_maps_from_gamma_cfg(gamma_cfg) or _base_coeff_maps_from_train_cfg(train_cfg)
    }
    domain_order = tuple(sorted(splits))
    deferred_committer = (
        _DeferredGammaFeedbackCommitter(
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
                    layout=layout,
                    condition_store=condition_store,
                    domain_names=(domain_name,),
                    latest_coeff_maps=latest_coeff_maps,
                    deferred_committer=deferred_committer,
                    commit_deferred=bool(deferred_committer is not None and domain_name == final_domain),
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


def _gamma_update_config(gamma_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> GammaUpdateConfig:
    return GammaUpdateConfig(
        start_epoch=int(gamma_cfg.get("start_epoch", 1)),
        stop_epoch=int(gamma_cfg.get("stop_epoch", train_cfg.get("epochs", 1))),
        update_interval_epochs=int(gamma_cfg.get("update_interval_epochs", gamma_cfg.get("interval_epochs", 1))),
        lr=float(gamma_cfg.get("gamma_lr", gamma_cfg.get("lr", 0.01))),
        steps=int(gamma_cfg.get("gamma_steps", gamma_cfg.get("steps", 1))),
        start_batch=None if gamma_cfg.get("start_batch") is None else int(gamma_cfg["start_batch"]),
        stop_batch=None if gamma_cfg.get("stop_batch") is None else int(gamma_cfg["stop_batch"]),
        update_interval_batches=None
        if gamma_cfg.get("update_interval_batches") is None
        else int(gamma_cfg["update_interval_batches"]),
        optimizer=str(gamma_cfg.get("gamma_optimizer", gamma_cfg.get("optimizer", "adam"))),
        heldout_accept_policy=str(gamma_cfg.get("heldout_accept_policy", "monitor")),
        checkpoint_policy=str(gamma_cfg.get("checkpoint_policy", gamma_cfg.get("selected_checkpoint_policy", "final_step"))),
    )


def _split_roi_bank_by_domain(selected_bank: ROIBank, heldout_bank: ROIBank | None = None) -> dict[str, tuple[ROIBank, ROIBank | None]]:
    selected_by_domain: dict[str, list[ROIRecord]] = {}
    for record in selected_bank.records:
        selected_by_domain.setdefault(str(record.domain_name), []).append(record)
    heldout_by_domain: dict[str, list[ROIRecord]] = {}
    if heldout_bank is not None:
        for record in heldout_bank.records:
            heldout_by_domain.setdefault(str(record.domain_name), []).append(record)
    splits: dict[str, tuple[ROIBank, ROIBank | None]] = {}
    for domain_name, records in selected_by_domain.items():
        domain_selected = ROIBank(
            records=tuple(records),
            config=selected_bank.config,
            metadata={**selected_bank.metadata, "domain_split": domain_name},
            empty_grid_cell_ids=selected_bank.empty_grid_cell_ids,
            format_version=selected_bank.format_version,
        )
        domain_heldout_records = tuple(heldout_by_domain.get(domain_name, ()))
        domain_heldout = None
        if domain_heldout_records:
            source = heldout_bank if heldout_bank is not None else selected_bank
            domain_heldout = ROIBank(
                records=domain_heldout_records,
                config=source.config,
                metadata={**source.metadata, "domain_split": domain_name},
                empty_grid_cell_ids=source.empty_grid_cell_ids,
                format_version=source.format_version,
            )
        splits[domain_name] = (domain_selected, domain_heldout)
    return splits


def _build_gamma_feedback_fn(
    objective: GammaProjectionObjective,
    *,
    layout,
    condition_store: ConditioningProviderStore | None,
    domain_names: tuple[str, ...] | None = None,
    latest_coeff_maps: dict[str, str] | None = None,
    deferred_committer: _DeferredGammaFeedbackCommitter | None = None,
    commit_deferred: bool = False,
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
        if latest_coeff_maps is not None:
            for name, path in entries:
                latest_coeff_maps[str(name)] = str(path)
            store_entries = tuple(
                {"name": name, "coeff_maps_npz": path}
                for name, path in sorted(latest_coeff_maps.items())
            )
        else:
            store_entries = tuple({"name": name, "coeff_maps_npz": path} for name, path in entries)
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


def _write_initial_physical_state(layout, runtime_config: Mapping[str, Any]) -> str | None:
    entries = _runtime_dual_domain_coeff_maps(runtime_config)
    if not entries:
        return None
    return _write_current_physical_state(
        layout,
        coeff_maps=entries,
        source="initial",
        epoch=None,
        global_step=None,
        condition_store_version=None,
    )


def _runtime_dual_domain_coeff_maps(runtime_config: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    batch_provider = runtime_config.get("batch_provider")
    if not isinstance(batch_provider, Mapping):
        return ()
    params = batch_provider.get("params")
    if not isinstance(params, Mapping):
        return ()
    entries = params.get("dual_domain_coeff_maps")
    if not isinstance(entries, (tuple, list)):
        return ()
    output = []
    for idx, item in enumerate(entries):
        if not isinstance(item, Mapping):
            continue
        path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
        if path is None:
            continue
        output.append({"name": str(item.get("name", f"domain{idx}")), "coeff_maps_npz": str(path)})
    return tuple(output)


def _write_current_physical_state(
    layout,
    *,
    coeff_maps,
    source: str,
    epoch: int | None,
    global_step: int | None,
    condition_store_version: int | None,
) -> str:
    entries = tuple({"name": str(item["name"]), "coeff_maps_npz": str(item["coeff_maps_npz"])} for item in coeff_maps)
    payload = {
        "source": str(source),
        "epoch": None if epoch is None else int(epoch),
        "global_step": None if global_step is None else int(global_step),
        "condition_store_version": None if condition_store_version is None else int(condition_store_version),
        "coeff_maps": list(entries),
    }
    path = layout.metadata_dir / "current_physical_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = layout.metadata_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["current_physical_state"] = payload
        manifest["current_physical_coeff_maps"] = list(entries)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _physical_checkpoint_extra_fn(layout):
    def checkpoint_extra() -> dict[str, object]:
        state = _read_current_physical_state(layout)
        if state is None:
            return {}
        return {
            "physical_state": state,
            "physical_coeff_maps": state.get("coeff_maps", []),
        }

    return checkpoint_extra


def _read_current_physical_state(layout) -> dict[str, object] | None:
    path = layout.metadata_dir / "current_physical_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _export_gamma_feedback_coeff_maps(
    objective: GammaProjectionObjective,
    *,
    gamma: torch.Tensor,
    gamma_before: torch.Tensor | None = None,
    domain_names: tuple[str, ...] | list[str] | None = None,
    layout,
    epoch: int,
    global_step: int,
) -> tuple[tuple[str, str], ...]:
    after_stack = full_roi_coeff_stack_torch(
        gamma.detach().to(device=objective.device, dtype=torch.float32),
        objective.nat_config,
        dtype=torch.float32,
        device=objective.device,
    )
    before_gamma = torch.zeros_like(gamma.detach()) if gamma_before is None else gamma_before.detach()
    before_stack = full_roi_coeff_stack_torch(
        before_gamma.to(device=objective.device, dtype=torch.float32),
        objective.nat_config,
        dtype=torch.float32,
        device=objective.device,
    )
    delta_maps = after_stack.maps_nm - before_stack.maps_nm
    output: list[tuple[str, str]] = []
    domains = objective.base_maps_by_domain or {"default": torch.zeros_like(delta_maps)}
    if domain_names is not None:
        wanted = {str(name) for name in domain_names}
        domains = {name: maps for name, maps in domains.items() if str(name) in wanted}
    for domain_name, base_maps in domains.items():
        maps = (base_maps.to(device=objective.device, dtype=torch.float32) + delta_maps).detach().cpu().numpy()
        path = (
            layout.artifacts_dir
            / "roi_bank_gamma"
            / f"step_{int(global_step):08d}"
            / "feedback"
            / _path_token(str(domain_name))
            / "coeff_maps.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            zernike_maps_nm=maps.astype(np.float32),
            mode_order=np.asarray(after_stack.mode_order, dtype=np.int64),
            gamma_delta_abs_max_nm=np.asarray(float(delta_maps.detach().abs().max().cpu().item()), dtype=np.float32),
            gamma_delta_abs_mean_nm=np.asarray(float(delta_maps.detach().abs().mean().cpu().item()), dtype=np.float32),
        )
        output.append((str(domain_name), str(path)))
    return tuple(output)


def _build_lazy_auto_roi_projection_objective(
    gamma_cfg: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    roi_source: _ROIBankSource,
    layout,
    train_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
    condition_store: ConditioningProviderStore | None,
):
    cached: dict[str, Any] = {}

    def ensure_built():
        if "hook" not in cached:
            selected_bank, heldout_bank, heldout_mode, split_source = _resolve_roi_gamma_banks(
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
            )
        return cached["hook"]

    def hook(result: TrainingRunEpochResult | TrainingStepResult) -> None:
        if not _should_run_gamma_update(result, _gamma_update_config(gamma_cfg, train_cfg)):
            return
        built_hook = ensure_built()
        if built_hook is not None:
            built_hook(result)

    return hook


def _should_run_gamma_update(result: TrainingRunEpochResult | TrainingStepResult, config: GammaUpdateConfig) -> bool:
    if config.start_batch is not None:
        batch = int(result.global_step)
        if batch < int(config.start_batch):
            return False
        if config.stop_batch is not None and batch > int(config.stop_batch):
            return False
        return (batch - int(config.start_batch)) % int(config.update_interval_batches or 1) == 0
    if isinstance(result, TrainingStepResult):
        return False
    epoch = int(result.epoch)
    if epoch < int(config.start_epoch) or epoch > int(config.stop_epoch):
        return False
    return (epoch - int(config.start_epoch)) % int(config.update_interval_epochs) == 0


def _build_vector_roi_gamma_objective(
    gamma_cfg: Mapping[str, Any],
    *,
    train_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
    model: torch.nn.Module,
) -> GammaProjectionObjective:
    optical_cfg = config.get("optical") if isinstance(config.get("optical"), Mapping) else {}
    simulation_cfg = config.get("simulation") if isinstance(config.get("simulation"), Mapping) else {}
    psf_cfg = simulation_cfg.get("psf") if isinstance(simulation_cfg.get("psf"), Mapping) else {}
    sim_vector_cfg = psf_cfg.get("vector") if isinstance(psf_cfg.get("vector"), Mapping) else {}
    device = next(model.parameters()).device
    online_cfg = train_cfg.get("online_generation") if isinstance(train_cfg.get("online_generation"), Mapping) else {}
    nat_wake_cfg = train_cfg.get("nat_wake") if isinstance(train_cfg.get("nat_wake"), Mapping) else {}
    bootstrap_cfg = train_cfg.get("peak_zmap_bootstrap") if isinstance(train_cfg.get("peak_zmap_bootstrap"), Mapping) else {}
    roi_size = int(gamma_cfg.get("roi_size_px", online_cfg.get("width", 128)))
    image_size_x = int(gamma_cfg.get("image_size_x", nat_wake_cfg.get("image_size_x", roi_size)))
    image_size_y = int(gamma_cfg.get("image_size_y", nat_wake_cfg.get("image_size_y", roi_size)))
    return GammaProjectionObjective(
        GammaProjectionObjectiveConfig(
            image_size_x=image_size_x,
            image_size_y=image_size_y,
            pixel_size_x_nm=float(gamma_cfg.get("pixel_size_x_nm", optical_cfg.get("pixel_size_nm_x", 95.0))),
            pixel_size_y_nm=float(gamma_cfg.get("pixel_size_y_nm", optical_cfg.get("pixel_size_nm_y", 95.0))),
            patch_size_px=int(gamma_cfg.get("patch_size_px", nat_wake_cfg.get("patch_size_px", 25))),
            npupil=int(gamma_cfg.get("npupil", sim_vector_cfg.get("npupil", 64))),
            NA=float(gamma_cfg.get("NA", optical_cfg.get("NA", 1.4))),
            wavelength_nm=float(gamma_cfg.get("wavelength_nm", optical_cfg.get("wavelength_nm", 660.0))),
            refmed=float(gamma_cfg.get("refmed", sim_vector_cfg.get("refmed", optical_cfg.get("n_medium", 1.518)))),
            refcov=float(gamma_cfg.get("refcov", sim_vector_cfg.get("refcov", optical_cfg.get("n_medium", 1.518)))),
            refimm=float(gamma_cfg.get("refimm", sim_vector_cfg.get("refimm", optical_cfg.get("n_medium", 1.518)))),
            objstage0=float(gamma_cfg.get("objstage0", sim_vector_cfg.get("objstage0", 0.0))),
            otf_rescale_xy=tuple(float(v) for v in gamma_cfg.get("otf_rescale_xy", sim_vector_cfg.get("otf_rescale_xy", (0.0, 0.0)))),
            renderer_batch_size=int(gamma_cfg.get("renderer_batch_size", sim_vector_cfg.get("batch_size", 64))),
            over_cut_px=int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0))),
            base_coeff_maps=_base_coeff_maps_from_gamma_cfg(gamma_cfg) or _base_coeff_maps_from_train_cfg(train_cfg),
            objective_mode=str(gamma_cfg.get("roi_bank_objective", gamma_cfg.get("objective_mode", "poisson_nll"))),
            projection_sample_batch_size=int(
                gamma_cfg.get("roi_bank_projection_sample_batch_size", gamma_cfg.get("projection_sample_batch_size", 16))
            ),
            projection_emitter_chunk_size=int(
                gamma_cfg.get(
                    "roi_bank_projection_emitter_chunk_size",
                    gamma_cfg.get("projection_emitter_chunk_size", nat_wake_cfg.get("roi_bank_projection_emitter_chunk_size", 1024)),
                )
            ),
            nat_config_kind=str(gamma_cfg.get("nat_config_kind", bootstrap_cfg.get("nat_config_kind", "order1"))),
        ),
        device=device,
    )


def _base_coeff_maps_from_gamma_cfg(gamma_cfg: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    entries = gamma_cfg.get("base_coeff_maps")
    if entries is None:
        return ()
    return _normalize_base_coeff_maps(entries)


def _base_coeff_maps_from_train_cfg(train_cfg: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    online_cfg = train_cfg.get("online_generation") if isinstance(train_cfg.get("online_generation"), Mapping) else {}
    return _normalize_base_coeff_maps(online_cfg.get("dual_domain_coeff_maps", ()))


def _normalize_base_coeff_maps(entries: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(entries, (list, tuple)):
        return ()
    result = []
    for item in entries:
        if isinstance(item, Mapping):
            name = str(item.get("name", f"domain_{len(result)}"))
            path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            name = str(item[0])
            path = item[1]
        else:
            continue
        if path:
            result.append((name, str(path)))
    return tuple(result)


def _posterior_photon_scale(train_cfg: Mapping[str, Any]) -> float | None:
    normalization = train_cfg.get("normalization")
    if isinstance(normalization, Mapping) and "photon_scale" in normalization:
        return float(normalization["photon_scale"])
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping):
        for key in ("photon_max", "phot_max", "ph_scale"):
            if key in scaling:
                return float(scaling[key])
    return None


def _posterior_z_scale(train_cfg: Mapping[str, Any]) -> float | None:
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping) and "z_max" in scaling:
        return float(scaling["z_max"])
    return None


def _posterior_bg_scale(train_cfg: Mapping[str, Any]) -> float:
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping) and "bg_max" in scaling:
        return float(scaling["bg_max"])
    normalization = train_cfg.get("normalization")
    if isinstance(normalization, Mapping) and "bg_scale" in normalization:
        return float(normalization["bg_scale"])
    return 1.0


def _posterior_input_offset(train_cfg: Mapping[str, Any]) -> float:
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping) and "input_offset" in scaling:
        return float(scaling["input_offset"])
    normalization = train_cfg.get("normalization")
    if isinstance(normalization, Mapping) and "input_offset" in normalization:
        return float(normalization["input_offset"])
    return 0.0


def _posterior_input_scale(train_cfg: Mapping[str, Any]) -> float:
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping) and "input_scale" in scaling:
        return float(scaling["input_scale"])
    normalization = train_cfg.get("normalization")
    if isinstance(normalization, Mapping) and "input_scale" in normalization:
        return float(normalization["input_scale"])
    return 1.0


def _select_roi_window_frame(model_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    if isinstance(model_input, tuple):
        model_input = model_input[0]
    tensor = torch.as_tensor(model_input, dtype=torch.float32)
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim != 4:
        raise ValueError(f"ROI model_input must have shape (B,H,W) or (B,T,H,W), got {tuple(tensor.shape)}")
    return tensor[:, int(tensor.shape[1] // 2)]


def _select_roi_bank_update_subset(
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


def _sample_roi_bank_posterior_update(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
    *,
    epoch: int,
) -> tuple[_ROIProjectionUpdateContext, dict[str, object]]:
    return _sample_roi_bank_posterior_update_from_cached_emitters(
        bank,
        gamma_cfg,
        epoch=epoch,
    )


def _sample_roi_bank_posterior_update_from_cached_emitters(
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


def _sample_roi_bank_posterior_update_from_current_model(
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
        )
    normalization_cfg = train_cfg.get("normalization")
    frame_proc = _build_inference_frame_proc({"normalization": dict(normalization_cfg)}) if isinstance(normalization_cfg, Mapping) else None
    z_scale = _posterior_z_scale(train_cfg)
    z_scale_nm = 1.0 if z_scale is None else (abs(float(z_scale)) * 1000.0 if abs(float(z_scale)) <= 10.0 else abs(float(z_scale)))
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


def _sample_roi_bank_importance_update_context(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
    step_index: int,
) -> tuple[_ROIProjectionUpdateContext, dict[str, object]]:
    try:
        return _sample_roi_bank_posterior_update_from_current_model(
            bank,
            gamma_cfg,
            model=model,
            train_cfg=train_cfg,
            step_index=int(step_index),
        )
    except ValueError as exc:
        if "selected no samples" not in str(exc):
            raise
    context, metrics = _sample_roi_bank_posterior_update_from_cached_emitters(
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
) -> _ROIProjectionUpdateContext:
    if not rows:
        raise ValueError("posterior rows must be non-empty")
    with profiling.time_block("posterior_rows_context_raw_stack"):
        raw_frames = torch.stack([_record_observed_frame(row["record"], frame_index=int(row["frame_index"])) for row in rows], dim=0)
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
    return _ROIProjectionUpdateContext(
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
) -> _ROIProjectionUpdateContext:
    if not samples:
        raise ValueError("posterior samples must be non-empty")
    records_by_id = {int(record.roi_id): record for record in bank.records}
    with profiling.time_block("posterior_samples_context_raw_stack"):
        raw_frames = torch.stack(
            [_record_observed_frame(records_by_id[int(sample.roi_id)], frame_index=int(sample.frame_index)) for sample in samples],
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
    return _ROIProjectionUpdateContext(
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


def _z_range_nm_from_gamma_cfg(gamma_cfg: Mapping[str, Any]) -> tuple[float, float] | None:
    raw = gamma_cfg.get("z_range_nm")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("train.roi_bank_gamma.z_range_nm must be a two-element range")
    return (float(raw[0]), float(raw[1]))


def _roi_bank_gamma_route(
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
            context, sampling_metrics = _sample_roi_bank_importance_update_context(
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
                context, _sampling_metrics = _sample_roi_bank_importance_update_context(
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
        return metrics

    return objective_fn, metrics_fn, prepare_fn


def _record_projection_tensors(bank: ROIBank) -> tuple[torch.Tensor, torch.Tensor, DetectionPosteriorSamples, torch.Tensor, list[str], torch.Tensor | None]:
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

    raw_frames = torch.stack([_record_observed_frame(record, frame_index=frame_index) for record, frame_index, _ in sample_rows], dim=0)
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


def _record_observed_center_frame(record: ROIRecord) -> torch.Tensor:
    return _record_observed_frame(record, frame_index=None)


def _record_observed_frame(record: ROIRecord, *, frame_index: int | None) -> torch.Tensor:
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


def _write_gamma_monitor_report(
    layout,
    metrics: Mapping[str, object],
    *,
    result: TrainingRunEpochResult | TrainingStepResult,
    raw_frames: torch.Tensor,
    background: torch.Tensor,
    samples: DetectionPosteriorSamples,
    gamma_before: torch.Tensor | None = None,
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    roi_origin_xy_px: torch.Tensor | None = None,
    domain_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    report_dir = (
        layout.artifacts_dir
        / "roi_bank_gamma"
        / f"step_{int(result.global_step):08d}"
        / f"source_{_path_token(metrics.get('artifact_source_group', 'unknown'))}"
        / f"domain_{_path_token(metrics.get('artifact_domain_group', 'unknown'))}"
    )
    summary_path = report_dir / "gamma_alternation_summary.json"
    report_path = report_dir / "gamma_update_monitor.md"
    raw_tiff_png_path = report_dir / "raw_tiff_adu_vs_recon.png"
    observed_png_path = report_dir / "observed_photons_vs_recon.png"
    diagnostics_manifest_path = report_dir / "diagnostics" / "diagnostics_manifest.json"
    resolved_checkpoint_path = result.checkpoint_path
    if resolved_checkpoint_path is None:
        resolved_checkpoint_path = layout.checkpoints_dir / "checkpoint_latest.pt"
    checkpoint_path = None if resolved_checkpoint_path is None else str(resolved_checkpoint_path)
    reconstruction = objective.render_reconstruction(
        background=background[0],
        samples=samples,
        batch_index=0,
        gamma=gamma,
        roi_origin_xy_px=roi_origin_xy_px,
        domain_names=domain_names,
    )
    raw_tiff_frame = _raw_tiff_adu_frame_for_diagnostic(
        metrics,
        samples=samples,
        roi_origin_xy_px=roi_origin_xy_px,
        domain_names=domain_names,
        fallback_shape=raw_frames[0].shape,
    )
    diagnostic_path = raw_tiff_png_path if raw_tiff_frame is not None else observed_png_path
    diagnostic_units = "raw_tiff_adu" if raw_tiff_frame is not None else "camera_corrected_photons"
    diagnostic_frame = raw_tiff_frame if raw_tiff_frame is not None else raw_frames[0]
    _write_raw_vs_reconstruction_png(
        diagnostic_path,
        raw_frame=diagnostic_frame,
        reconstruction=reconstruction,
    )
    if raw_tiff_frame is not None:
        _write_raw_vs_reconstruction_png(
            observed_png_path,
            raw_frame=raw_frames[0],
            reconstruction=reconstruction,
        )
    payload = dict(metrics)
    diagnostic_rel = str(diagnostic_path.relative_to(layout.run_dir))
    diagnostics_manifest_rel = str(diagnostics_manifest_path.relative_to(layout.run_dir))
    payload.update(
        {
            "epoch": int(result.epoch),
            "steps_completed": int(metrics["steps"]) if "steps" in metrics else None,
            "checkpoint_path": checkpoint_path,
            "diagnostic_png_path": diagnostic_rel,
            "diagnostic_observed_units": diagnostic_units,
            "diagnostics_manifest_path": diagnostics_manifest_rel,
        }
    )
    _write_diagnostics_manifest(
        diagnostics_manifest_path,
        payload,
        summary_path=summary_path,
        report_path=report_path,
        raw_tiff_vs_recon_path=raw_tiff_png_path if raw_tiff_frame is not None else None,
        observed_vs_recon_path=observed_png_path,
        raw_frame=diagnostic_frame,
        observed_photons_frame=raw_frames[0],
        observed_photons_frames=raw_frames,
        reconstruction=reconstruction,
        gamma_before=gamma_before,
        gamma=gamma,
        objective=objective,
        samples=samples,
        background=background,
        roi_origin_xy_px=roi_origin_xy_px,
        domain_names=domain_names,
        layout=layout,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_gamma_monitor_markdown(payload), encoding="utf-8")
    return {
        "summary_path": str(summary_path.relative_to(layout.run_dir)),
        "report_path": str(report_path.relative_to(layout.run_dir)),
        "checkpoint_path": checkpoint_path,
        "diagnostic_png_path": diagnostic_rel,
        "diagnostic_observed_units": diagnostic_units,
        "diagnostics_manifest_path": diagnostics_manifest_rel,
    }


def _render_gamma_monitor_markdown(payload: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Gamma Update Monitor",
            "",
            f"- epoch: {payload.get('epoch')}",
            f"- best_step: {payload.get('best_step')}",
            f"- selected_poisson_nll: {payload.get('selected_poisson_nll')}",
            f"- selected_sampled_emitter_count: {payload.get('selected_sampled_emitter_count')}",
            f"- selected_projected_photons: {payload.get('selected_projected_photons')}",
            f"- selected_background_mean: {payload.get('selected_background_mean')}",
            f"- heldout_available: {payload.get('heldout_available')}",
            f"- heldout_monitor_mode: {payload.get('heldout_monitor_mode')}",
            f"- heldout_initial_loss: {payload.get('heldout_initial_loss')}",
            f"- heldout_final_loss: {payload.get('heldout_final_loss')}",
            f"- heldout_loss_delta: {payload.get('heldout_loss_delta')}",
            f"- diagnostic_png_path: {payload.get('diagnostic_png_path')}",
            "",
        ]
    )


def _write_diagnostics_manifest(
    path: Path,
    payload: Mapping[str, object],
    *,
    summary_path: Path,
    report_path: Path,
    raw_tiff_vs_recon_path: Path | None,
    observed_vs_recon_path: Path,
    raw_frame: torch.Tensor,
    observed_photons_frame: torch.Tensor,
    observed_photons_frames: torch.Tensor | None = None,
    reconstruction: torch.Tensor,
    gamma_before: torch.Tensor | None = None,
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    samples: DetectionPosteriorSamples | None = None,
    background: torch.Tensor | None = None,
    roi_origin_xy_px: torch.Tensor | None = None,
    domain_names: list[str] | tuple[str, ...] | None = None,
    layout,
) -> None:
    def rel(item: Path) -> str:
        return str(item.relative_to(layout.run_dir))

    diagnostics_dir = path.parent
    zmap = _write_zmap_delta_summary(
        diagnostics_dir / "zmap_before_after",
        payload=payload,
        gamma=gamma,
        objective=objective,
        layout=layout,
    )
    fixed_roi = _write_fixed_roi_recon_smoke(
        diagnostics_dir / "fixed_roi_recon",
        payload=payload,
        raw_frame=observed_photons_frame,
        reconstruction=reconstruction,
        layout=layout,
    )
    raw_patch = _write_raw_tiff_patch_recon_smoke(
        diagnostics_dir / "raw_tiff_patch_recon",
        payload=payload,
        raw_frame=raw_frame,
        reconstruction=reconstruction,
        layout=layout,
    )
    raw_patch_montage = None
    if samples is not None and background is not None:
        raw_patch_montage = _write_raw_tiff_patch_recon_montage(
            diagnostics_dir / "raw_tiff_patch_recon_montage",
            payload=payload,
            samples=samples,
            background=background,
            gamma=gamma,
            objective=objective,
            roi_origin_xy_px=roi_origin_xy_px,
            domain_names=domain_names,
            observed_photons_frames=observed_photons_frames,
            layout=layout,
        )
    observed_patch = _write_observed_photons_patch_recon_smoke(
        diagnostics_dir / "observed_photons_patch_recon",
        payload=payload,
        observed_frame=observed_photons_frame,
        reconstruction=reconstruction,
        layout=layout,
    )
    model_triplet = None
    if gamma_before is not None and samples is not None and background is not None:
        initial_reconstruction = objective.render_reconstruction(
            background=background[0],
            samples=samples,
            batch_index=0,
            gamma=gamma_before,
            roi_origin_xy_px=roi_origin_xy_px,
            domain_names=domain_names,
        )
        model_triplet = _write_raw_initial_latest_triplet(
            diagnostics_dir / "raw_initial_latest_triplet",
            payload=payload,
            raw_frame=raw_frame,
            initial_reconstruction=initial_reconstruction,
            latest_reconstruction=reconstruction,
            layout=layout,
        )
    psf_grid = _write_vector_psf_shape_grid(
        diagnostics_dir / "psf_shape_grid",
        payload=payload,
        gamma=gamma,
        objective=objective,
        layout=layout,
    )
    manifest = {
        "schema_version": "roi_gamma_diagnostics_manifest.v1",
        "epoch": int(payload.get("epoch", 0)),
        "artifact_source_group": payload.get("artifact_source_group"),
        "artifact_domain_group": payload.get("artifact_domain_group"),
        "selected_domain_names": payload.get("selected_domain_names", []),
        "diagnostics": {
            "compact_monitor": {
                "status": "available",
                "summary_path": rel(summary_path),
                "report_path": rel(report_path),
            },
            **(
                {
                    "raw_tiff_adu_vs_recon": {
                        "status": "available",
                        "png_path": rel(raw_tiff_vs_recon_path),
                    }
                }
                if raw_tiff_vs_recon_path is not None
                else {}
            ),
            "observed_photons_vs_recon": {
                "status": "available",
                "png_path": rel(observed_vs_recon_path),
            },
            "zmap_before_after": zmap,
            "fixed_roi_recon": fixed_roi,
            "raw_tiff_patch_recon": raw_patch,
            **({"raw_tiff_patch_recon_montage": raw_patch_montage} if raw_patch_montage is not None else {}),
            "observed_photons_patch_recon": observed_patch,
            **({"raw_initial_latest_triplet": model_triplet} if model_triplet is not None else {}),
            "psf_shape_grid": psf_grid,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _raw_tiff_adu_frame_for_diagnostic(
    payload: Mapping[str, object],
    *,
    samples: DetectionPosteriorSamples,
    roi_origin_xy_px: torch.Tensor | None,
    domain_names: list[str] | tuple[str, ...] | None,
    fallback_shape: torch.Size | tuple[int, ...],
    batch_index: int = 0,
) -> torch.Tensor | None:
    raw_path = payload.get("roi_bank_raw_path")
    if raw_path is None:
        return None
    frame_indices = samples.metadata.get("frame_index") if isinstance(samples.metadata, Mapping) else None
    if not isinstance(frame_indices, (list, tuple)) or not frame_indices:
        return None
    if roi_origin_xy_px is None or int(roi_origin_xy_px.numel()) < 2:
        return None
    shape = tuple(int(v) for v in fallback_shape)
    if len(shape) != 2:
        return None
    height, width = shape
    try:
        sample_index = int(batch_index)
        if sample_index < 0 or sample_index >= len(frame_indices):
            return None
        frame_index = int(frame_indices[sample_index])
        origin = roi_origin_xy_px.detach().cpu().to(dtype=torch.float32)
        if origin.ndim == 2:
            if sample_index >= int(origin.shape[0]):
                return None
            x0 = int(round(float(origin[sample_index, 0].item())))
            y0 = int(round(float(origin[sample_index, 1].item())))
        else:
            x0 = int(round(float(origin[0].item())))
            y0 = int(round(float(origin[1].item())))
        domain = str(domain_names[sample_index]).strip().lower() if domain_names and sample_index < len(domain_names) else ""
        with tifffile.TiffFile(str(raw_path)) as tif:
            frame = np.asarray(tif.series[0].asarray(key=frame_index), dtype=np.float32)
        if frame.ndim != 2:
            frame = np.squeeze(frame)
        if frame.ndim != 2:
            return None
        x_offset = _diagnostic_domain_x_offset(domain, frame_width=int(frame.shape[1]), x0=x0, crop_width=width)
        crop = frame[y0 : y0 + height, x_offset + x0 : x_offset + x0 + width]
        if crop.shape != (height, width):
            return None
        return torch.as_tensor(np.ascontiguousarray(crop), dtype=torch.float32)
    except Exception:
        return None


def _diagnostic_domain_x_offset(domain: str, *, frame_width: int, x0: int, crop_width: int) -> int:
    if domain in {"right", "r", "domain_right"}:
        half_width = int(frame_width) // 2
        if int(x0) >= half_width:
            return 0
        if int(x0) + int(crop_width) <= half_width:
            return half_width
    return 0


def _write_zmap_delta_summary(
    path: Path,
    *,
    payload: Mapping[str, object],
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    layout,
) -> dict[str, object]:
    stack = objective.nat_config
    zmap = torch.zeros((len(stack.aberrations), 8, 8), dtype=torch.float32)
    xs = torch.linspace(0.5, float(objective.config.image_size_x) - 0.5, 8)
    ys = torch.linspace(0.5, float(objective.config.image_size_y) - 0.5, 8)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    roixy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    domain_name = _diagnostic_domain_name(payload, objective)
    coeffs = objective.coefficients_at(gamma=gamma, full_xy_px=roixy, domain_name=domain_name).detach().cpu()
    zmap = coeffs.reshape(8, 8, len(stack.aberrations)).permute(2, 0, 1).contiguous()
    mode_delta = zmap.abs().mean(dim=(1, 2))
    dominant = int(torch.argmax(mode_delta).item()) if mode_delta.numel() else 0
    png_path = path / "zmap_delta_vector_nat.png"
    summary_path = path / "delta_gamma_physical_zmap_before_after_summary.json"
    _write_grayscale_png(png_path, _tile_frames_uint8([zmap[index] for index in range(min(3, int(zmap.shape[0])))]))
    summary = {
        "schema_version": "roi_gamma_vector_nat_zmap_delta.v1",
        "epoch": int(payload.get("epoch", 0)),
        "artifact_domain_group": payload.get("artifact_domain_group"),
        "delta_abs_mean": float(zmap.abs().mean().item()),
        "delta_abs_max": float(zmap.abs().max().item()),
        "dominant_delta_mode": dominant,
        "mode_order": [list(mode) for mode in stack.aberrations],
        "gamma_size": int(gamma.numel()),
        "coeff_domain": domain_name,
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_fixed_roi_recon_smoke(
    path: Path,
    *,
    payload: Mapping[str, object],
    raw_frame: torch.Tensor,
    reconstruction: torch.Tensor,
    layout,
) -> dict[str, object]:
    residual = (raw_frame.detach().cpu() - reconstruction.detach().cpu()).abs()
    png_path = path / "fixed_roi_recon_smoke.png"
    summary_path = path / "fixed_roi_recon_summary.json"
    _write_grayscale_png(png_path, _tile_frames_uint8([raw_frame, reconstruction, residual]))
    rms = float(torch.sqrt((residual.to(dtype=torch.float32).square()).mean()).item())
    summary = {
        "schema_version": "roi_gamma_fixed_roi_recon_smoke.v1",
        "epoch": int(payload.get("epoch", 0)),
        "selected_roi_count": int(payload.get("roi_count", 0)),
        "rendered_count": 1,
        "poisson_nll": _poisson_nll_value(raw_frame, reconstruction),
        "rms": rms,
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_raw_tiff_patch_recon_smoke(
    path: Path,
    *,
    payload: Mapping[str, object],
    raw_frame: torch.Tensor,
    reconstruction: torch.Tensor,
    layout,
) -> dict[str, object]:
    residual = raw_frame.detach().cpu().to(dtype=torch.float32) - reconstruction.detach().cpu().to(dtype=torch.float32)
    png_path = path / "raw_tiff_patch_recon_smoke.png"
    summary_path = path / "raw_tiff_patch_recon_summary.json"
    _write_grayscale_png(png_path, _tile_frames_uint8([raw_frame, reconstruction, residual.abs()]))
    summary = {
        "schema_version": "roi_gamma_raw_tiff_patch_recon_smoke.v1",
        "epoch": int(payload.get("epoch", 0)),
        "selected_patch_count": 1,
        "observed_units": "raw_tiff_adu",
        "note": "Raw TIFF ADU and reconstruction are not in the same physical units; this diagnostic is visual only.",
        "ncc": _ncc_value(raw_frame, reconstruction),
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_raw_tiff_patch_recon_montage(
    path: Path,
    *,
    payload: Mapping[str, object],
    samples: DetectionPosteriorSamples,
    background: torch.Tensor,
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    roi_origin_xy_px: torch.Tensor | None,
    domain_names: list[str] | tuple[str, ...] | None,
    observed_photons_frames: torch.Tensor | None,
    layout,
) -> dict[str, object] | None:
    batch_count = int(samples.xyzph.shape[0])
    if batch_count <= 0:
        return None
    requested = int(payload.get("diagnostic_raw_tiff_patch_recon_montage_count", 5) or 5)
    rendered_count = max(1, min(requested, batch_count))
    if rendered_count == 1:
        indices = [0]
    else:
        indices = sorted({int(round(v)) for v in np.linspace(0, batch_count - 1, num=rendered_count)})

    rows: list[np.ndarray] = []
    rendered_indices: list[int] = []
    ncc_values: list[float] = []
    roi_ids: list[object] = []
    frame_indices: list[object] = []
    for index in indices:
        bkg = torch.as_tensor(background, dtype=torch.float32)
        frame_shape = tuple(int(v) for v in bkg[index].shape[-2:]) if bkg.ndim == 3 else tuple(int(v) for v in bkg.shape[-2:])
        raw_frame = _raw_tiff_adu_frame_for_diagnostic(
            payload,
            samples=samples,
            roi_origin_xy_px=roi_origin_xy_px,
            domain_names=domain_names,
            fallback_shape=frame_shape,
            batch_index=index,
        )
        if raw_frame is None:
            continue
        reconstruction = objective.render_reconstruction(
            background=background[index] if torch.as_tensor(background).ndim == 3 else background,
            samples=samples,
            batch_index=index,
            gamma=gamma,
            roi_origin_xy_px=roi_origin_xy_px,
            domain_names=domain_names,
        )
        corrected = None
        if observed_photons_frames is not None:
            observed_all = torch.as_tensor(observed_photons_frames, dtype=torch.float32)
            if observed_all.ndim == 3 and index < int(observed_all.shape[0]):
                corrected = observed_all[index]
        if corrected is None:
            corrected = raw_frame
        residual = raw_frame.detach().cpu().to(dtype=torch.float32) - reconstruction.detach().cpu().to(dtype=torch.float32)
        rows.append(_tile_frames_uint8([raw_frame, corrected, reconstruction, residual.abs()]))
        rendered_indices.append(index)
        ncc_values.append(_ncc_value(raw_frame, reconstruction))
        roi_ids.append(_metadata_item(samples.metadata.get("roi_id") if isinstance(samples.metadata, Mapping) else None, index))
        frame_indices.append(_metadata_item(samples.metadata.get("frame_index") if isinstance(samples.metadata, Mapping) else None, index))

    if not rows:
        return None
    spacer = np.full((4, int(rows[0].shape[1])), 255, dtype=np.uint8)
    canvas_parts: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        if row_index:
            canvas_parts.append(spacer)
        canvas_parts.append(row)
    canvas = np.concatenate(canvas_parts, axis=0)
    png_path = path / "raw_tiff_patch_recon_montage.png"
    summary_path = path / "raw_tiff_patch_recon_montage_summary.json"
    _write_grayscale_png(png_path, canvas)
    summary = {
        "schema_version": "roi_gamma_raw_tiff_patch_recon_montage.v1",
        "epoch": int(payload.get("epoch", 0)),
        "selected_patch_count": int(len(rendered_indices)),
        "observed_units": "raw_tiff_adu",
        "note": "Each row is raw TIFF ADU | corrected camera photon | reconstruction | absolute residual. Raw TIFF ADU and reconstruction are not in the same physical units; this diagnostic is visual only.",
        "batch_indices": rendered_indices,
        "roi_ids": roi_ids,
        "frame_indices": frame_indices,
        "ncc": ncc_values,
        "panels": ["raw_tiff_adu", "corrected_camera_photon", "reconstruction", "absolute_residual"],
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _metadata_item(value: object, index: int) -> object:
    if isinstance(value, torch.Tensor):
        if index < int(value.numel()):
            item = value.reshape(-1)[index].item()
            if isinstance(item, (bool, np.bool_)):
                return bool(item)
            if isinstance(item, (int, np.integer)):
                return int(item)
            if isinstance(item, (float, np.floating)):
                return int(item) if float(item).is_integer() else float(item)
            return item
        return None
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if index < int(flat.size):
            item = flat[index].item()
            if isinstance(item, (bool, np.bool_)):
                return bool(item)
            if isinstance(item, (int, np.integer)):
                return int(item)
            if isinstance(item, (float, np.floating)):
                return int(item) if float(item).is_integer() else float(item)
            return item
        return None
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return None


def _write_observed_photons_patch_recon_smoke(
    path: Path,
    *,
    payload: Mapping[str, object],
    observed_frame: torch.Tensor,
    reconstruction: torch.Tensor,
    layout,
) -> dict[str, object]:
    residual = observed_frame.detach().cpu().to(dtype=torch.float32) - reconstruction.detach().cpu().to(dtype=torch.float32)
    png_path = path / "observed_photons_patch_recon_smoke.png"
    summary_path = path / "observed_photons_patch_recon_summary.json"
    _write_grayscale_png(png_path, _tile_frames_uint8([observed_frame, reconstruction, residual.abs()]))
    summary = {
        "schema_version": "roi_gamma_observed_photons_patch_recon_smoke.v1",
        "epoch": int(payload.get("epoch", 0)),
        "selected_patch_count": 1,
        "observed_units": "camera_corrected_photons",
        "poisson_nll": _poisson_nll_value(observed_frame, reconstruction),
        "mse": float(residual.square().mean().item()),
        "ncc": _ncc_value(observed_frame, reconstruction),
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_raw_initial_latest_triplet(
    path: Path,
    *,
    payload: Mapping[str, object],
    raw_frame: torch.Tensor,
    initial_reconstruction: torch.Tensor,
    latest_reconstruction: torch.Tensor,
    layout,
) -> dict[str, object]:
    delta = latest_reconstruction.detach().cpu().to(dtype=torch.float32) - initial_reconstruction.detach().cpu().to(dtype=torch.float32)
    png_path = path / "raw_initial_latest_triplet.png"
    summary_path = path / "raw_initial_latest_triplet_summary.json"
    _write_grayscale_png(png_path, _tile_frames_uint8([raw_frame, initial_reconstruction, latest_reconstruction]))
    summary = {
        "schema_version": "roi_gamma_raw_initial_latest_triplet.v1",
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "selected_roi_count": int(payload.get("roi_count", 0)),
        "rendered_count": 1,
        "initial_poisson_nll": _poisson_nll_value(raw_frame, initial_reconstruction),
        "latest_poisson_nll": _poisson_nll_value(raw_frame, latest_reconstruction),
        "latest_minus_initial_rms": float(torch.sqrt(delta.square().mean()).item()),
        "png_path": str(png_path.relative_to(layout.run_dir)),
        "panels": ["raw_frame", "initial_physical_model_projection", "latest_updated_physical_model_projection"],
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_vector_psf_shape_grid(
    path: Path,
    *,
    payload: Mapping[str, object],
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    layout,
) -> dict[str, object]:
    z_values = torch.tensor([-600.0, 0.0, 600.0], dtype=torch.float32)
    center_xy = torch.tensor(
        [[float(objective.config.image_size_x) * 0.5, float(objective.config.image_size_y) * 0.5]],
        dtype=torch.float32,
        device=objective.device,
    )
    domain_name = _diagnostic_domain_name(payload, objective)
    center_coeffs = objective.coefficients_at(gamma=gamma, full_xy_px=center_xy, domain_name=domain_name)
    coeffs = center_coeffs.expand(3, -1).contiguous()
    coeffs_rad = coeffs * (2.0 * np.pi / max(float(objective.config.wavelength_nm), 1e-6)) * objective.ctx.normfac[None, :]
    psf = render_vector_psf_bank(
        objective.ctx,
        coeffs_rad,
        z_values.to(device=objective.device) * 1e-9,
        out_size=int(objective.config.patch_size_px),
        batch_size=int(objective.config.renderer_batch_size),
        return_torch=True,
    ).detach().cpu()
    png_path = path / "vector_psf_zstack.png"
    summary_path = path / "psf_shape_grid_summary.json"
    _write_grayscale_png(png_path, _tile_frames_uint8([psf[index] for index in range(int(psf.shape[0]))]))
    summary = {
        "schema_version": "roi_gamma_vector_psf_zstack.v1",
        "epoch": int(payload.get("epoch", 0)),
        "psf_sum": [float(value) for value in psf.sum(dim=(1, 2)).tolist()],
        "z_nm": [float(value) for value in z_values.tolist()],
        "center_coeffs_nm": [float(value) for value in center_coeffs.detach().cpu().reshape(-1).tolist()],
        "coeff_domain": domain_name,
        "renderer": "vector_psf",
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _diagnostic_domain_name(payload: Mapping[str, object], objective: GammaProjectionObjective) -> str | None:
    domain = payload.get("artifact_domain_group")
    if isinstance(domain, str) and domain not in {"multi", "unknown"}:
        return domain
    names = payload.get("selected_domain_names")
    if isinstance(names, list) and names:
        return str(names[0])
    if objective.base_maps_by_domain:
        return next(iter(objective.base_maps_by_domain))
    return None


def _artifact_group_metrics(
    bank: ROIBank,
    *,
    roi_library_source: str | None,
    objective_source: str,
) -> dict[str, object]:
    domains = sorted({str(record.domain_name) for record in bank.records})
    domain_group = domains[0] if len(domains) == 1 else "multi"
    source_group = roi_library_source or objective_source
    return {
        "artifact_source_group": source_group,
        "artifact_domain_group": domain_group,
        "selected_domain_names": domains,
    }


def _path_token(value: object) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    token = "_".join(part for part in "".join(chars).split("_") if part)
    return token or "unknown"


def _write_raw_vs_reconstruction_png(path: Path, *, raw_frame: torch.Tensor, reconstruction: torch.Tensor) -> None:
    raw = _to_uint8(raw_frame.detach().cpu())
    recon = _to_uint8(reconstruction.detach().cpu())
    canvas = np.concatenate([raw, recon], axis=1)
    _write_grayscale_png(path, canvas)


def _to_uint8(frame: torch.Tensor) -> np.ndarray:
    array = frame.numpy().astype(np.float32)
    low = float(np.min(array))
    high = float(np.max(array))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)


def _tile_frames_uint8(frames: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate([_to_uint8(frame.detach().cpu()) for frame in frames], axis=1)


def _poisson_nll_value(raw_frame: torch.Tensor, reconstruction: torch.Tensor) -> float:
    raw = raw_frame.detach().cpu().to(dtype=torch.float32).clamp_min(0.0)
    recon = reconstruction.detach().cpu().to(dtype=torch.float32).clamp_min(1e-6)
    return float((recon - raw * torch.log(recon)).mean().item())


def _ncc_value(raw_frame: torch.Tensor, reconstruction: torch.Tensor) -> float:
    raw = raw_frame.detach().cpu().to(dtype=torch.float32)
    recon = reconstruction.detach().cpu().to(dtype=torch.float32)
    raw_centered = raw - raw.mean()
    recon_centered = recon - recon.mean()
    denom = torch.sqrt(raw_centered.square().sum().clamp_min(1e-8) * recon_centered.square().sum().clamp_min(1e-8))
    return float(((raw_centered * recon_centered).sum() / denom).item())


def _write_grayscale_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width = int(image.shape[0]), int(image.shape[1])
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw_rows)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _auto_build_roi_bank(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: _ROIBankSource,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
) -> ROIBank:
    roi_size = int(gamma_cfg.get("roi_size_px", 8))
    frame_range = roi_source.frame_range
    camera_backward = _camera_backward_from_training_config(train_cfg)
    config = ROIBankBuildConfig(
        roi_size_px=roi_size,
        window_size=int(gamma_cfg.get("roi_bank_window_size", 3)),
        frame_range=frame_range,
        grid_shape=tuple(int(v) for v in gamma_cfg.get("roi_bank_grid_shape", (2, 2))),
        max_rois=int(
            gamma_cfg.get(
                "roi_bank_max_rois",
                gamma_cfg.get("roi_library_max_rois", gamma_cfg.get("target_projected_emitters", 2)),
            )
        ),
        target_emitters=int(gamma_cfg.get("target_projected_emitters", 2)),
        candidate_probability_threshold=float(
            gamma_cfg.get("roi_bank_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))
        ),
        probability_threshold=float(gamma_cfg.get("roi_bank_probability_threshold", gamma_cfg.get("probability_threshold", 0.5))),
        max_overlap_fraction=float(gamma_cfg.get("roi_bank_max_overlap_fraction", 0.95)),
        seed=int(gamma_cfg.get("seed", 0)),
        background_smoothing_kernel=int(
            gamma_cfg.get("background_smoothing_kernel", gamma_cfg.get("roi_bank_background_smoothing_kernel", 3))
        ),
        camera_backward=camera_backward,
        over_cut_px=int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0))),
        origin_mode=str(gamma_cfg.get("roi_bank_origin_mode", gamma_cfg.get("origin_mode", "emitter_centered"))),
        origin_stride_px=(
            None
            if gamma_cfg.get("roi_bank_origin_stride_px", gamma_cfg.get("origin_stride_px")) is None
            else int(gamma_cfg.get("roi_bank_origin_stride_px", gamma_cfg.get("origin_stride_px")))
        ),
        valid_core_size_px=(
            None
            if gamma_cfg.get("roi_bank_valid_core_size_px", gamma_cfg.get("valid_core_size_px")) is None
            else int(gamma_cfg.get("roi_bank_valid_core_size_px", gamma_cfg.get("valid_core_size_px")))
        ),
    )
    if _uses_model_infer(roi_source):
        return _auto_build_roi_bank_from_loc_harvest(
            gamma_cfg,
            roi_source=roi_source,
            model=model,
            train_cfg=train_cfg,
            config=config,
            roi_size=roi_size,
        )
    bank = build_roi_bank_from_inference(
        raw_frames_photon=roi_source.raw_path,
        domains=_auto_build_domains(gamma_cfg, roi_size=roi_size, roi_source=roi_source),
        infer_fn=_roi_bank_infer_fn(
            gamma_cfg,
            roi_source=roi_source,
            model=model,
            window_size=int(config.window_size),
            train_cfg=train_cfg,
        ),
        config=config,
    )
    return ROIBank(
        records=bank.records,
        config=bank.config,
        metadata={
            **bank.metadata,
            "roi_bank_source_alias": roi_source.alias,
            "roi_bank_candidate_mode": roi_source.candidate_mode,
            "roi_bank_infer_source": "bright_pixel",
            "roi_bank_observed_units": "camera_corrected_photons" if camera_backward is not None else "raw_input",
            **_roi_bank_camera_backward_metadata(bank.metadata, camera_backward),
        },
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
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
    harvest_cfg = LocHarvestConfig(
        raw_path=roi_source.raw_path,
        domains=_auto_build_domains(gamma_cfg, roi_size=roi_size, roi_source=roi_source),
        roi_bank_config=config,
        input_offset=_posterior_input_offset(train_cfg),
        input_scale=_posterior_input_scale(train_cfg),
        photon_scale=_posterior_photon_scale(train_cfg) or 1.0,
        z_scale=_posterior_z_scale(train_cfg) or 1.0,
        bg_scale=_posterior_bg_scale(train_cfg),
        normalization_config={"normalization": dict(train_cfg.get("normalization", {}))}
        if isinstance(train_cfg.get("normalization"), Mapping)
        else None,
        candidate_probability_threshold=float(
            gamma_cfg.get(
                "posterior_candidate_probability_threshold",
                gamma_cfg.get("roi_bank_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3)),
            )
        ),
        probability_threshold=float(gamma_cfg.get("roi_bank_probability_threshold", gamma_cfg.get("probability_threshold", 0.5))),
        split_threshold=float(gamma_cfg.get("posterior_adjacent_probability_threshold", gamma_cfg.get("split_threshold", 0.6))),
        phot_min=float(gamma_cfg.get("roi_bank_phot_min", gamma_cfg.get("phot_min", 100.0))),
        sigma_max_px=float(gamma_cfg.get("roi_bank_sigma_max_px", gamma_cfg.get("sigma_max_px", 2.5))),
        tile_size_px=int(gamma_cfg.get("roi_bank_spatial_tile_size", 128)),
        overlap_px=int(gamma_cfg.get("roi_bank_spatial_overlap_px", 16)),
        max_emitters_per_window=(
            None
            if gamma_cfg.get("roi_bank_infer_max_emitters") is None
            else int(gamma_cfg.get("roi_bank_infer_max_emitters"))
        ),
    )
    bank = build_roi_bank_from_loc_harvest(
        model=model,
        config=harvest_cfg,
        condition_context=_roi_conditioning_context(train_cfg),
    )
    return ROIBank(
        records=bank.records,
        config=bank.config,
        metadata={
            **bank.metadata,
            "roi_bank_source_alias": roi_source.alias,
            "roi_bank_candidate_mode": roi_source.candidate_mode,
            "roi_bank_infer_source": "loc_harvest_raw_tiff",
            "roi_bank_harvest_channel_order": "p,photons,x,y,z,photons_sigma,x_sigma,y_sigma,z_sigma,bg",
            "roi_bank_observed_units": "camera_corrected_photons" if config.camera_backward is not None else "raw_input",
            **_roi_bank_camera_backward_metadata(bank.metadata, config.camera_backward),
        },
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    )


def _roi_bank_camera_backward_metadata(
    bank_metadata: Mapping[str, Any],
    configured: Mapping[str, Any] | None,
) -> dict[str, object]:
    if not configured:
        return {}
    resolved = bank_metadata.get("camera_backward") if isinstance(bank_metadata, Mapping) else None
    return {"camera_backward": resolved if isinstance(resolved, Mapping) else dict(configured)}


def _camera_backward_from_training_config(train_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    camera_cfg = train_cfg.get("camera") if isinstance(train_cfg.get("camera"), Mapping) else {}
    normalization = train_cfg.get("normalization") if isinstance(train_cfg.get("normalization"), Mapping) else {}
    params: dict[str, Any] = {}
    for key in ("qe", "e_per_adu", "em_gain", "spurious_charge"):
        if key in camera_cfg:
            params[key] = float(camera_cfg[key])
        elif key in normalization:
            params[key] = float(normalization[key])
    if str(camera_cfg.get("baseline_mode", "")).strip():
        params["baseline_mode"] = str(camera_cfg["baseline_mode"])
        if "baseline_percentile" in camera_cfg:
            params["baseline_percentile"] = float(camera_cfg["baseline_percentile"])
        if "baseline_frame_range" in camera_cfg:
            frame_range = camera_cfg["baseline_frame_range"]
            if isinstance(frame_range, (list, tuple)) and len(frame_range) == 2:
                params["baseline_frame_range"] = [int(frame_range[0]), int(frame_range[1])]
        if "baseline_by_domain" in camera_cfg and isinstance(camera_cfg["baseline_by_domain"], Mapping):
            params["baseline_by_domain"] = {str(key): float(value) for key, value in camera_cfg["baseline_by_domain"].items()}
    if "baseline" in camera_cfg:
        params["baseline"] = float(camera_cfg["baseline"])
    elif "baseline_adu" in camera_cfg:
        params["baseline"] = float(camera_cfg["baseline_adu"])
    elif "baseline" in normalization:
        params["baseline"] = float(normalization["baseline"])
    elif "baseline_adu" in normalization:
        params["baseline"] = float(normalization["baseline_adu"])
    if not params:
        return None
    resolved = {
        "baseline": float(params.get("baseline", 0.0)),
        "e_per_adu": float(params.get("e_per_adu", 1.0)),
        "em_gain": float(params.get("em_gain", 1.0)),
        "qe": float(params.get("qe", 1.0)),
        "spurious_charge": float(params.get("spurious_charge", 0.0)),
    }
    for key in ("baseline_mode", "baseline_percentile", "baseline_frame_range", "baseline_by_domain"):
        if key in params:
            resolved[key] = params[key]
    return resolved


def _roi_conditioning_context(train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    online_cfg = train_cfg.get("online_generation") if isinstance(train_cfg.get("online_generation"), Mapping) else {}
    providers: dict[str, FullResZernikeConditioning] = {}
    entries = online_cfg.get("dual_domain_coeff_maps", ())
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name", f"domain{len(providers)}"))
            path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
            if path:
                providers[name] = FullResZernikeConditioning.from_npz(str(path))
    return {
        "providers": providers or None,
        "append_domain_onehot": bool(online_cfg.get("append_domain_onehot", False)),
        "domain_names": tuple(providers.keys()),
    }


def _roi_bank_infer_fn(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: _ROIBankSource,
    model: torch.nn.Module,
    window_size: int,
    train_cfg: Mapping[str, Any],
):
    if not _uses_model_infer(roi_source):
        return _bright_pixel_infer_fn
    return _model_raw_tiff_infer_fn(
        model=model,
        threshold=float(
            gamma_cfg.get("roi_bank_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))
        ),
        max_emitters=int(
            gamma_cfg.get(
                "roi_bank_infer_max_emitters",
                gamma_cfg.get("num_posterior_samples", gamma_cfg.get("target_projected_emitters", 2)),
            )
        ),
        seed=int(gamma_cfg.get("seed", 0)),
        expected_channels=int(gamma_cfg.get("roi_bank_infer_channels", window_size)),
        photon_scale=_posterior_photon_scale(train_cfg),
        z_scale=_posterior_z_scale(train_cfg),
        condition_context=_roi_conditioning_context(train_cfg),
    )


def _uses_model_infer(roi_source: _ROIBankSource) -> bool:
    return roi_source.alias == "loc_infer_raw_tiff" or roi_source.candidate_mode == "dense_tile_temporal"


def _auto_build_domains(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_size: int,
    roi_source: _ROIBankSource,
) -> tuple[ROIBankDomain, ...]:
    raw_domains = roi_source.domains if roi_source.domains is not None else gamma_cfg.get("auto_build_domains")
    if raw_domains is None:
        size = int(gamma_cfg.get("auto_build_domain_size_px", max(roi_size + 4, roi_size)))
        return (ROIBankDomain("auto", crop_left=0, crop_top=0, crop_width=size, crop_height=size),)
    domains = []
    for item in raw_domains:
        mapping = _mapping(item, "auto_build_domains[]")
        domains.append(
            ROIBankDomain(
                str(mapping.get("name", "auto")),
                crop_left=int(mapping.get("crop_left", 0)),
                crop_top=int(mapping.get("crop_top", 0)),
                crop_width=int(mapping["crop_width"]),
                crop_height=int(mapping["crop_height"]),
            )
        )
    return tuple(domains)


def _resolve_roi_bank_source(
    gamma_cfg: Mapping[str, Any],
    *,
    train_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
    config_base_dir: Path,
) -> _ROIBankSource | None:
    source_cfg = gamma_cfg.get("roi_bank_source")
    if isinstance(source_cfg, Mapping):
        mode = str(source_cfg.get("mode", "auto_build"))
        if mode not in {"auto_build", "loc_infer_raw_tiff"}:
            return None
        raw_path = source_cfg.get("raw_path", source_cfg.get("tiff_path"))
        if raw_path is None:
            raise ValueError("train.roi_bank_gamma.roi_bank_source requires raw_path for auto_build mode")
        frame_range = source_cfg.get("frame_range", gamma_cfg.get("roi_bank_frame_range"))
        roi_size = source_cfg.get("roi_size_px")
        if roi_size is not None:
            gamma_cfg = {**gamma_cfg, "roi_size_px": roi_size}
        return _ROIBankSource(
            mode="auto_build",
            raw_path=_resolve_config_path(str(raw_path), base_dir=config_base_dir),
            candidate_mode=str(source_cfg.get("candidate_mode", gamma_cfg.get("roi_bank_candidate_mode", "bright_pixel"))),
            frame_range=_frame_range_tuple(frame_range),
            domains=_domains_tuple(source_cfg.get("domains")),
            alias=None if mode == "auto_build" else mode,
        )
    if gamma_cfg.get("auto_build_roi_bank") is True and "auto_build_source_path" in gamma_cfg:
        return _ROIBankSource(
            mode="auto_build",
            raw_path=_resolve_config_path(str(gamma_cfg["auto_build_source_path"]), base_dir=config_base_dir),
            candidate_mode=str(gamma_cfg.get("roi_bank_candidate_mode", "bright_pixel")),
            frame_range=_frame_range_tuple(gamma_cfg.get("roi_bank_frame_range")),
            domains=_domains_tuple(gamma_cfg.get("auto_build_domains")),
        )
    if gamma_cfg.get("auto_build_roi_bank") is True and source_cfg == "loc_infer_raw_tiff":
        real_tiff_cfg = _mapping(train_cfg.get("real_tiff_wake"), "train.real_tiff_wake")
        raw_path = real_tiff_cfg.get("tiff_path")
        if raw_path is None:
            raw_path = _phase_retrieval_tiff_path(config)
        if raw_path is None:
            raise ValueError("loc_infer_raw_tiff ROI-bank source requires train.real_tiff_wake.tiff_path")
        return _ROIBankSource(
            mode="auto_build",
            raw_path=_resolve_config_path(str(raw_path), base_dir=config_base_dir),
            candidate_mode=str(gamma_cfg.get("roi_bank_candidate_mode", "bright_pixel")),
            frame_range=_frame_range_tuple(gamma_cfg.get("roi_bank_frame_range")),
            domains=_domains_tuple(real_tiff_cfg.get("domains")),
            alias="loc_infer_raw_tiff",
        )
    return None


def _phase_retrieval_tiff_path(config: Mapping[str, Any]) -> str | None:
    phase_cfg = config.get("phase_retrieval")
    if not isinstance(phase_cfg, Mapping):
        return None
    input_cfg = phase_cfg.get("input")
    if isinstance(input_cfg, Mapping) and input_cfg.get("tiff_path") is not None:
        return str(input_cfg["tiff_path"])
    return None


def _frame_range_tuple(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("ROI-bank frame_range must contain exactly two values")
    return int(value[0]), int(value[1])


def _domains_tuple(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if value is None:
        return None
    return tuple(_mapping(item, "roi_bank_source.domains[]") for item in value)


def _resolve_config_path(value: str, *, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _roi_bank_source_metrics(source: _ROIBankSource) -> dict[str, object]:
    return {
        "roi_bank_source_mode": source.mode,
        "roi_bank_raw_path": source.raw_path,
        "roi_bank_candidate_mode": source.candidate_mode,
        **({"roi_bank_source_alias": source.alias} if source.alias is not None else {}),
        **({"roi_bank_frame_range": list(source.frame_range)} if source.frame_range is not None else {}),
    }


def _bright_pixel_infer_fn(
    *,
    domain: ROIBankDomain,
    frame_window: tuple[int, int],
    raw_domain_frames_photon: np.ndarray,
) -> RawInferenceResult:
    projection = np.asarray(raw_domain_frames_photon, dtype=np.float32).mean(axis=0)
    flat_indices = np.argsort(projection.reshape(-1))[::-1][:2]
    height, width = projection.shape
    emitters = []
    for flat_index in flat_indices:
        y, x = divmod(int(flat_index), int(width))
        emitters.append(
            InferredEmitter(
                probability=0.9,
                mu_xy_px=(float(x), float(y)),
                sigma_xy_px=(0.3, 0.3),
                mu_z_nm=0.0,
                sigma_z_nm=10.0,
                mu_photons=float(max(projection[y, x], 1.0)),
                sigma_photons=1.0,
                cell_xy_px=(float(x), float(y)),
            )
        )
    return RawInferenceResult(
        emitters=tuple(emitters),
        background_mu=np.full((height, width), float(np.median(projection)), dtype=np.float32),
        metadata={"domain": domain.name, "frame_window": frame_window},
    )


def _model_raw_tiff_infer_fn(
    *,
    model: torch.nn.Module,
    threshold: float,
    max_emitters: int,
    seed: int,
    expected_channels: int,
    photon_scale: float | None,
    z_scale: float | None,
    condition_context: dict[str, Any] | None = None,
):
    condition_context = condition_context or {"providers": None, "append_domain_onehot": False, "domain_names": ()}

    def infer_fn(
        *,
        domain: ROIBankDomain,
        frame_window: tuple[int, int],
        raw_domain_frames_photon: np.ndarray,
    ) -> RawInferenceResult:
        raw = np.asarray(raw_domain_frames_photon, dtype=np.float32)
        if raw.ndim != 3:
            raise ValueError(f"raw TIFF inference window must have shape (T,H,W), got {raw.shape}")
        if int(raw.shape[0]) != int(expected_channels):
            raise ValueError(
                "raw TIFF inference window channel count must match model input channels: "
                f"got {int(raw.shape[0])}, expected {int(expected_channels)}"
            )
        _, height, width = raw.shape
        image = torch.as_tensor(raw, dtype=torch.float32)
        device = _model_device(model)
        was_training = model.training
        model.eval()
        emitters = []
        background_accum = np.zeros((height, width), dtype=np.float32)
        background_count = np.zeros((height, width), dtype=np.float32)
        tile_size = min(128, height, width)
        overlap = 16 if tile_size > 32 else 0
        providers = condition_context.get("providers")
        provider = None if providers is None else providers.get(str(domain.name))
        append_domain_onehot = bool(condition_context.get("append_domain_onehot", False))
        domain_names = tuple(str(name) for name in condition_context.get("domain_names", ()))
        with torch.no_grad():
            for tile in _iter_inference_tiles(height, width, tile_size=tile_size, overlap_px=overlap):
                tile_image = image[:, tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]].unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                )
                model_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = tile_image
                if provider is not None:
                    condition = provider.condition_vector_from_xy(
                        x0=int(tile["x0"]),
                        y0=int(tile["y0"]),
                        height=int(tile["y1"] - tile["y0"]),
                        width=int(tile["x1"] - tile["x0"]),
                        device=device,
                        dtype=tile_image.dtype,
                    )
                    if append_domain_onehot:
                        onehot = torch.zeros((len(domain_names),), dtype=tile_image.dtype, device=device)
                        onehot[list(domain_names).index(str(domain.name))] = 1.0
                        condition = torch.cat((condition, onehot), dim=0)
                    model_input = (tile_image, condition.unsqueeze(0))
                output = model(model_input)
                output = output.detach().to(dtype=torch.float32)
                tile_emitters = _emitters_from_active_smlm_tile(
                    output,
                    tile=tile,
                    threshold=float(threshold),
                    max_emitters=int(max_emitters),
                    photon_scale=photon_scale,
                    z_scale=z_scale,
                    frame_window=frame_window,
                    full_width=width,
                    full_height=height,
                )
                emitters.extend(tile_emitters)
                bg = _background_from_output_or_raw(output, tile_image)
                background_accum[tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]] += bg
                background_count[tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]] += 1.0
        if was_training:
            model.train()
        background = background_accum / np.maximum(background_count, 1.0)
        emitters.sort(key=lambda item: float(item.probability), reverse=True)
        emitters = emitters[: int(max_emitters)]
        return RawInferenceResult(
            emitters=tuple(emitters),
            background_mu=background,
            metadata={"domain": domain.name, "frame_window": frame_window, "source": "loc_infer_raw_tiff"},
        )

    return infer_fn


def _tile_starts(size: int, tile_size: int, overlap_px: int) -> list[int]:
    if int(size) <= int(tile_size):
        return [0]
    stride = max(1, int(tile_size) - int(overlap_px))
    starts = list(range(0, int(size) - int(tile_size) + 1, stride))
    end = int(size) - int(tile_size)
    if starts[-1] != end:
        starts.append(end)
    return starts


def _iter_inference_tiles(height: int, width: int, *, tile_size: int, overlap_px: int) -> list[dict[str, int]]:
    tiles = []
    xs = _tile_starts(width, tile_size, overlap_px)
    ys = _tile_starts(height, tile_size, overlap_px)
    left_margin = int(overlap_px) // 2
    right_margin = int(overlap_px) - left_margin
    for iy, y0 in enumerate(ys):
        y1 = min(y0 + int(tile_size), height)
        keep_y0 = y0 if iy == 0 else y0 + left_margin
        keep_y1 = y1 if iy == len(ys) - 1 else y1 - right_margin
        for ix, x0 in enumerate(xs):
            x1 = min(x0 + int(tile_size), width)
            keep_x0 = x0 if ix == 0 else x0 + left_margin
            keep_x1 = x1 if ix == len(xs) - 1 else x1 - right_margin
            tiles.append(
                {
                    "x0": int(x0),
                    "x1": int(x1),
                    "y0": int(y0),
                    "y1": int(y1),
                    "keep_x0": int(keep_x0),
                    "keep_x1": int(keep_x1),
                    "keep_y0": int(keep_y0),
                    "keep_y1": int(keep_y1),
                }
            )
    return tiles


def _emitters_from_active_smlm_tile(
    output: torch.Tensor,
    *,
    tile: dict[str, int],
    threshold: float,
    max_emitters: int,
    photon_scale: float | None,
    z_scale: float | None,
    frame_window: tuple[int, int],
    full_width: int,
    full_height: int,
) -> list[InferredEmitter]:
    if not (output.ndim == 4 and int(output.shape[1]) == 10):
        return []
    out = output[0].detach().cpu().to(dtype=torch.float32).clone()
    p = _spatial_integration(out[0].unsqueeze(0), raw_th=0.3, split_th=0.6)[0]
    out[0] = p
    scores = p.reshape(-1)
    if scores.numel() == 0:
        return []
    k = min(int(max_emitters), int(scores.numel()))
    values, indices = torch.topk(scores, k=k)
    tile_w = int(p.shape[-1])
    emitters = []
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        if float(value) < float(threshold):
            continue
        row = int(flat_index) // tile_w
        col = int(flat_index) % tile_w
        x = float(tile["x0"]) + float(col) + float(out[2, row, col].item())
        y = float(tile["y0"]) + float(row) + float(out[3, row, col].item())
        if not (
            float(tile["keep_x0"]) <= x < float(tile["keep_x1"])
            and float(tile["keep_y0"]) <= y < float(tile["keep_y1"])
            and 0.0 <= x < float(full_width)
            and 0.0 <= y < float(full_height)
        ):
            continue
        photons = _physical_photons(float(out[1, row, col].item()), photon_scale=photon_scale)
        z_nm = _physical_z_nm(float(out[4, row, col].item()), z_scale=z_scale)
        emitters.append(
            InferredEmitter(
                probability=float(value),
                mu_xy_px=(x, y),
                sigma_xy_px=(float(out[6, row, col].item()), float(out[7, row, col].item())),
                mu_z_nm=z_nm,
                sigma_z_nm=abs(_physical_z_nm(float(out[8, row, col].item()), z_scale=z_scale)),
                mu_photons=max(float(photons), 1e-6),
                sigma_photons=max(_physical_photons(float(out[5, row, col].item()), photon_scale=photon_scale), 1e-6),
                cell_xy_px=(float(tile["x0"] + col), float(tile["y0"] + row)),
            )
        )
    return emitters


def _spatial_integration(p: torch.Tensor, *, raw_th: float, split_th: float) -> torch.Tensor:
    diag = 0.0
    filt = torch.tensor(
        [[diag, 1.0, diag], [1.0, 1.0, 1.0], [diag, 1.0, diag]],
        dtype=p.dtype,
        device=p.device,
    ).view(1, 1, 3, 3)
    conv = torch.nn.functional.conv2d(p.unsqueeze(1), filt, padding=1)
    p_clip = torch.where(p > float(raw_th), p, torch.zeros_like(p))
    pool = torch.nn.functional.max_pool2d(p_clip.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    max_mask1 = torch.eq(p.unsqueeze(1), pool)
    p_ps1 = max_mask1.to(dtype=p.dtype) * conv
    p_copy = p.unsqueeze(1) * (1.0 - max_mask1.to(dtype=p.dtype))
    max_mask2 = torch.where(p_copy > float(split_th), torch.ones_like(p_copy), torch.zeros_like(p_copy))
    p_ps2 = max_mask2 * conv
    return torch.clamp(p_ps1 + p_ps2, min=0.0, max=1.0).squeeze(1)


def _physical_photons(value: float, *, photon_scale: float | None) -> float:
    return float(value) if photon_scale is None else float(value) * float(photon_scale)


def _physical_z_nm(value: float, *, z_scale: float | None) -> float:
    if z_scale is None:
        return float(value)
    scale = abs(float(z_scale))
    scale_nm = scale * 1000.0 if scale <= 10.0 else scale
    return float(value) * scale_nm


def _background_from_output_or_raw(output: torch.Tensor, tile_image: torch.Tensor) -> np.ndarray:
    if output.ndim == 4 and int(output.shape[1]) == 10:
        return output[0, 9].detach().cpu().numpy().astype(np.float32, copy=False)
    projection = tile_image[0].detach().cpu().mean(dim=0).numpy().astype(np.float32, copy=False)
    return np.full(projection.shape, float(np.median(projection)), dtype=np.float32)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_roi_gamma_banks(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
) -> tuple[ROIBank, ROIBank | None, str, str | None]:
    if "heldout_roi_library_path" in gamma_cfg:
        return bank, load_roi_bank(str(gamma_cfg["heldout_roi_library_path"])), "fixed_samples", "configured_hdf5"
    heldout_count = _auto_heldout_count(gamma_cfg, total_count=len(bank.records))
    if heldout_count <= 0:
        if _auto_heldout_requested(gamma_cfg):
            return bank, None, "disabled_insufficient_records", "auto_from_roi_library_insufficient_records"
        return bank, None, "not_configured", None
    records_by_domain: dict[str, list[ROIRecord]] = {}
    for record in bank.records:
        records_by_domain.setdefault(str(record.domain_name), []).append(record)
    selected_records: list[ROIRecord] = []
    heldout_records: list[ROIRecord] = []
    if len(records_by_domain) <= 1:
        split_at = len(bank.records) - heldout_count
        selected_records = list(bank.records[:split_at])
        heldout_records = list(bank.records[split_at:])
    else:
        for domain_records in records_by_domain.values():
            domain_heldout_count = _auto_heldout_count(gamma_cfg, total_count=len(domain_records))
            if domain_heldout_count <= 0:
                selected_records.extend(domain_records)
                continue
            split_at = len(domain_records) - domain_heldout_count
            selected_records.extend(domain_records[:split_at])
            heldout_records.extend(domain_records[split_at:])
    if not selected_records:
        raise ValueError("auto held-out split must leave at least one selected ROI")
    if not heldout_records:
        return bank, None, "disabled_insufficient_records", "auto_from_roi_library_insufficient_records"
    selected_bank = ROIBank(
        records=selected_records,
        config=bank.config,
        metadata={**bank.metadata, "heldout_split": "selected"},
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    )
    heldout_bank = ROIBank(
        records=heldout_records,
        config=bank.config,
        metadata={**bank.metadata, "heldout_split": "heldout"},
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    )
    return selected_bank, heldout_bank, "auto_split_fixed_samples", "auto_from_roi_library"


def _auto_heldout_requested(gamma_cfg: Mapping[str, Any]) -> bool:
    return int(gamma_cfg.get("auto_heldout_min_rois", 0)) > 0 or int(gamma_cfg.get("auto_heldout_max_rois", 0)) > 0


def _auto_heldout_count(gamma_cfg: Mapping[str, Any], *, total_count: int) -> int:
    min_rois = int(gamma_cfg.get("auto_heldout_min_rois", 0))
    max_rois = int(gamma_cfg.get("auto_heldout_max_rois", min_rois))
    if min_rois <= 0 and max_rois <= 0:
        return 0
    return min(max(min_rois, 0), max(max_rois, 0), max(int(total_count) - 1, 0))


def _build_heldout_monitor_context(
    gamma_cfg: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    bank: ROIBank | None,
    objective: GammaProjectionObjective,
    max_emitters: int,
    mode: str,
    split_source: str | None,
    train_cfg: Mapping[str, Any],
) -> _HeldoutMonitorContext | None:
    if bank is None:
        return None
    objective_mode = str(gamma_cfg.get("roi_bank_objective", gamma_cfg.get("objective_mode", "poisson_nll"))).strip().lower()
    if objective_mode == "importance_wake":
        heldout_cfg = dict(gamma_cfg)
        heldout_cfg["target_projected_emitters"] = int(gamma_cfg.get("heldout_target_projected_emitters", gamma_cfg.get("target_projected_emitters", 1000)))
        try:
            context, _metrics = _sample_roi_bank_posterior_update_from_current_model(
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
        return _HeldoutMonitorContext(
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
        candidate_threshold=float(
            gamma_cfg.get("posterior_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))
        ),
        split_threshold=float(gamma_cfg.get("posterior_adjacent_probability_threshold", gamma_cfg.get("split_threshold", 0.6))),
    )
    raw_frames = _select_roi_window_frame(loc_batch.model_input)
    return _HeldoutMonitorContext(
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


def _heldout_monitor_metrics(
    context: _HeldoutMonitorContext | None,
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


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
