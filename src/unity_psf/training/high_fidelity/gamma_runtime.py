"""Gamma update lifecycle for high-fidelity localization training."""
from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from unity_psf.gamma_update import GammaProjectionObjective, GammaProjectionObjectiveConfig, GammaUpdateConfig, GammaUpdateState, build_gamma_update_hook
from unity_psf.localization.conditioning import ConditioningProviderStore
from unity_psf.localization.posterior import DetectionPosteriorSamples, sample_detection_posterior
from unity_psf.localization.roi_batches import build_roi_batch_provider
from unity_psf.optics.nat_field import full_roi_coeff_stack_torch
from unity_psf.roi_library import ROIBank, ROIRecord, load_roi_bank
from unity_psf.runtime import profiling
from unity_psf.training.channel_context import ChannelTrainingContext
from unity_psf.training.high_fidelity.diagnostics import _artifact_group_metrics, _path_token, _write_gamma_monitor_report
from unity_psf.training.high_fidelity.physical_state import _write_current_physical_state
from unity_psf.training.high_fidelity.roi_bank_source import ROIBankSource, auto_build_roi_bank, posterior_photon_scale, posterior_z_scale, resolve_roi_bank_source, roi_bank_source_metrics, roi_conditioning_context
from unity_psf.training.high_fidelity.roi_posterior import _stack_record_loss_masks, ROIProjectionUpdateContext, record_projection_tensors, sample_roi_bank_importance_update_context, sample_roi_bank_posterior_update_from_current_model, select_roi_bank_update_subset, select_roi_window_frame
from unity_psf.training.loop import TrainingRunEpochResult, TrainingStepResult
__all__ = ["build_roi_bank_gamma_hook", "build_vector_roi_gamma_objective", "count_gamma_updates", "gamma_hook_bindings", "gamma_objective_source", "gamma_update_config", "merge_zernike_delta_maps", "roi_bank_gamma_route", "select_single_channel_roi_split", "should_run_gamma_update", "HeldoutMonitorContext", "DeferredGammaFeedbackCommitter"]
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
class DeferredGammaFeedbackCommitter:
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
    roi_source = resolve_roi_bank_source(gamma_cfg, train_cfg=train_cfg, config=config, config_base_dir=config_base_dir)
    if "roi_library_path" in gamma_cfg:
        selected_bank, heldout_bank, heldout_mode, split_source = resolve_roi_gamma_banks(
            load_roi_bank(str(gamma_cfg["roi_library_path"])),
            gamma_cfg,
        )
        return build_domain_roi_bank_gamma_hooks(
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
        return build_lazy_auto_roi_projection_objective(
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
        return build_domain_roi_bank_gamma_hooks(
            gamma_cfg,
            train_cfg=train_cfg,
            config=config,
            model=model,
            layout=layout,
            selected_bank=smoke_roi_bank(roi_size=int(gamma_cfg.get("roi_size_px", 8))),
            heldout_bank=None,
            heldout_mode="not_configured",
            heldout_split_source=None,
            objective_source="roi_projection_smoke",
            roi_library_path=None,
            roi_library_source="smoke",
            condition_store=condition_store,
            physical_context=physical_context,
        )
    objective = build_vector_roi_gamma_objective(gamma_cfg, train_cfg=train_cfg, config=config, model=model)
    return build_gamma_update_hook(
        state=GammaUpdateState(gamma=objective.initial_gamma()),
        localizer=model,
        layout=layout,
        config=gamma_update_config(gamma_cfg, train_cfg),
        objective_fn=lambda gamma: (gamma - 1.0).square().sum(),
        feedback_fn=build_gamma_feedback_fn(
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
def build_domain_roi_bank_gamma_hooks(
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
    roi_bank_source: ROIBankSource | None = None,
    condition_store: ConditioningProviderStore | None = None,
    physical_context: ChannelTrainingContext | None = None,
):
    update_config = gamma_update_config(gamma_cfg, train_cfg)
    splits = split_roi_bank_by_domain(selected_bank, heldout_bank)
    if physical_context is not None:
        splits = select_single_channel_roi_split(splits, physical_context.channel.channel_id)
    hooks = []
    objectives: dict[str, GammaProjectionObjective] = {}
    latest_coeff_maps = {
        str(domain): str(path)
        for domain, path in base_coeff_maps_from_gamma_config(gamma_cfg) or base_coeff_maps_from_training_config(train_cfg)
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
        objective = build_vector_roi_gamma_objective(gamma_cfg, train_cfg=train_cfg, config=config, model=model)
        objectives[domain_name] = objective
        objective_fn, metrics_fn, prepare_fn = build_roi_projection_objective(
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
                feedback_fn=build_gamma_feedback_fn(
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
def gamma_update_config(gamma_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> GammaUpdateConfig:
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
def should_write_gamma_artifacts(gamma_cfg: Mapping[str, Any], *, result: TrainingRunEpochResult | TrainingStepResult) -> bool:
    policy = str(gamma_cfg.get("artifact_retention_policy", "per_update")).lower()
    if policy not in {"compact", "compact_latest", "latest"}:
        return True
    interval = int(gamma_cfg.get("diagnostic_interval_updates", gamma_cfg.get("artifact_interval_updates", 10)) or 0)
    start_epoch = int(gamma_cfg.get("start_epoch", 1))
    stop_epoch = int(gamma_cfg.get("stop_epoch", 0) or 0)
    update_interval = max(1, int(gamma_cfg.get("update_interval_epochs", gamma_cfg.get("interval_epochs", 1))))
    epoch = int(result.epoch)
    if stop_epoch > 0 and epoch >= stop_epoch:
        return True
    if interval <= 0:
        return False
    update_index = max(1, ((epoch - start_epoch) // update_interval) + 1)
    return update_index == 1 or update_index % interval == 0
def split_roi_bank_by_domain(selected_bank: ROIBank, heldout_bank: ROIBank | None = None) -> dict[str, tuple[ROIBank, ROIBank | None]]:
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
def select_single_channel_roi_split(
    splits: Mapping[str, tuple[ROIBank, ROIBank | None]],
    channel_id: str,
) -> dict[str, tuple[ROIBank, ROIBank | None]]:
    """Bind one channel context to exactly one ROI-bank domain."""
    if not splits:
        return {}
    selected = splits.get(str(channel_id))
    if selected is None:
        if len(splits) != 1:
            raise ValueError(
                "single-channel physical context has no matching ROI-bank domain: "
                f"channel={channel_id!r}, domains={sorted(splits)!r}"
            )
        selected = next(iter(splits.values()))
    return {str(channel_id): selected}
def build_gamma_feedback_fn(
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
        entries = export_gamma_feedback_coeff_maps(
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
def merge_zernike_delta_maps(
    base_maps: np.ndarray,
    *,
    base_mode_order: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    delta_maps: np.ndarray | torch.Tensor,
    delta_mode_order: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> np.ndarray:
    merged = np.asarray(base_maps, dtype=np.float32).copy()
    delta = torch.as_tensor(delta_maps, dtype=torch.float32).detach().cpu().numpy()
    base_order = tuple((int(n), int(m)) for n, m in base_mode_order)
    update_order = tuple((int(n), int(m)) for n, m in delta_mode_order)
    if merged.ndim != 3 or delta.ndim != 3 or merged.shape[1:] != delta.shape[1:]:
        raise ValueError("base and delta Zernike maps must have matching (H,W) dimensions")
    if merged.shape[0] != len(base_order) or delta.shape[0] != len(update_order):
        raise ValueError("Zernike map channels must match their mode order")
    base_indices = {mode: index for index, mode in enumerate(base_order)}
    for delta_index, mode in enumerate(update_order):
        if mode not in base_indices:
            raise ValueError(f"gamma update mode {mode!r} is missing from the base Zernike map")
        merged[base_indices[mode]] += delta[delta_index]
    return merged
def export_gamma_feedback_coeff_maps(
    objective: GammaProjectionObjective,
    *,
    gamma: torch.Tensor,
    gamma_before: torch.Tensor | None = None,
    domain_names: tuple[str, ...] | list[str] | None = None,
    layout,
    epoch: int,
    global_step: int,
    artifact_policy: str = "per_update",
) -> tuple[tuple[str, str], ...]:
    gamma_t = gamma.detach().to(device=objective.device, dtype=torch.float32)
    before_gamma = torch.zeros_like(gamma.detach()) if gamma_before is None else gamma_before.detach()
    before_gamma_t = before_gamma.to(device=objective.device, dtype=torch.float32)
    stack_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]] = {}
    mode_order: list[tuple[int, int]] | None = None
    output: list[tuple[str, str]] = []
    full_base_maps: dict[str, tuple[np.ndarray, list[tuple[int, int]]]] = {}
    if bool(objective.config.preserve_full_base_modes):
        for domain_name, source_path in objective.config.base_coeff_maps:
            with np.load(source_path, allow_pickle=False) as payload:
                full_base_maps[str(domain_name)] = (
                    np.asarray(payload["zernike_maps_nm"], dtype=np.float32),
                    [tuple(int(value) for value in row) for row in np.asarray(payload["mode_order"], dtype=np.int64)],
                )
    domains = objective.base_maps_by_domain
    if not domains:
        default_shape = (int(objective.nat_config.img_size_y), int(objective.nat_config.img_size_x))
        domains = {"default": torch.zeros((len(objective.nat_config.aberrations), *default_shape), device=objective.device)}
    if domain_names is not None:
        wanted = {str(name) for name in domain_names}
        domains = {name: maps for name, maps in domains.items() if str(name) in wanted}
    for domain_name, base_maps in domains.items():
        base_maps_t = base_maps.to(device=objective.device, dtype=torch.float32)
        if base_maps_t.ndim != 3:
            raise ValueError(f"base coeff maps for domain {domain_name!r} must have shape (C,H,W), got {tuple(base_maps_t.shape)}")
        if int(base_maps_t.shape[0]) != len(objective.nat_config.aberrations):
            raise ValueError(
                f"base coeff maps for domain {domain_name!r} have {int(base_maps_t.shape[0])} modes, "
                f"expected {len(objective.nat_config.aberrations)}"
            )
        shape_hw = (int(base_maps_t.shape[1]), int(base_maps_t.shape[2]))
        stacks = stack_cache.get(shape_hw)
        if stacks is None:
            domain_nat_config = replace(
                objective.nat_config,
                img_size_y=int(shape_hw[0]),
                img_size_x=int(shape_hw[1]),
            )
            after_stack = full_roi_coeff_stack_torch(
                gamma_t,
                domain_nat_config,
                dtype=torch.float32,
                device=objective.device,
            )
            before_stack = full_roi_coeff_stack_torch(
                before_gamma_t,
                domain_nat_config,
                dtype=torch.float32,
                device=objective.device,
            )
            delta_maps = after_stack.maps_nm - before_stack.maps_nm
            stacks = (after_stack.maps_nm, delta_maps, after_stack.mode_order)
            stack_cache[shape_hw] = stacks
            if mode_order is None:
                mode_order = after_stack.mode_order
        after_maps, delta_maps, update_mode_order = stacks
        if bool(objective.config.preserve_full_base_modes):
            source_maps, source_order = full_base_maps[str(domain_name)]
            maps = merge_zernike_delta_maps(
                source_maps,
                base_mode_order=source_order,
                delta_maps=after_maps,
                delta_mode_order=update_mode_order,
            )
            output_mode_order = source_order
        else:
            maps = (base_maps_t + after_maps).detach().cpu().numpy()
            output_mode_order = mode_order or list(objective.nat_config.aberrations)
        if str(artifact_policy).lower() in {"compact", "compact_latest", "latest"}:
            path = (
                layout.artifacts_dir
                / "roi_bank_gamma"
                / "latest"
                / "feedback"
                / _path_token(str(domain_name))
                / "coeff_maps.npz"
            )
        else:
            path = (
                layout.artifacts_dir
                / "roi_bank_gamma"
                / f"step_{int(global_step):08d}"
                / "feedback"
                / _path_token(str(domain_name))
                / "coeff_maps.npz"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary_path,
                zernike_maps_nm=maps.astype(np.float32),
                mode_order=np.asarray(output_mode_order, dtype=np.int64),
                gamma_values_nm=gamma_t.detach().cpu().numpy().astype(np.float32),
                gamma_before_values_nm=before_gamma_t.detach().cpu().numpy().astype(np.float32),
                physical_state_semantics=np.asarray("base_plus_cumulative_gamma"),
                gamma_delta_abs_max_nm=np.asarray(float(delta_maps.detach().abs().max().cpu().item()), dtype=np.float32),
                gamma_delta_abs_mean_nm=np.asarray(float(delta_maps.detach().abs().mean().cpu().item()), dtype=np.float32),
            )
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        output.append((str(domain_name), str(path)))
    return tuple(output)
def build_lazy_auto_roi_projection_objective(
    gamma_cfg: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    roi_source: ROIBankSource,
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
                auto_build_roi_bank(gamma_cfg, roi_source=roi_source, model=model, train_cfg=train_cfg),
                gamma_cfg,
            )
            cached["hook"] = build_domain_roi_bank_gamma_hooks(
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
        if not should_run_gamma_update(result, gamma_update_config(gamma_cfg, train_cfg)):
            return
        built_hook = ensure_built()
        if built_hook is not None:
            built_hook(result)
    return hook
def should_run_gamma_update(result: TrainingRunEpochResult | TrainingStepResult, config: GammaUpdateConfig) -> bool:
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
def build_vector_roi_gamma_objective(
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
            zemit0=(
                None
                if gamma_cfg.get("zemit0", online_cfg.get("zemit0", sim_vector_cfg.get("zemit0"))) is None
                else float(gamma_cfg.get("zemit0", online_cfg.get("zemit0", sim_vector_cfg.get("zemit0"))))
            ),
            otf_rescale_xy=tuple(float(v) for v in gamma_cfg.get("otf_rescale_xy", sim_vector_cfg.get("otf_rescale_xy", (0.0, 0.0)))),
            renderer_batch_size=int(gamma_cfg.get("renderer_batch_size", sim_vector_cfg.get("batch_size", 64))),
            over_cut_px=int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0))),
            base_coeff_maps=base_coeff_maps_from_gamma_config(gamma_cfg) or base_coeff_maps_from_training_config(train_cfg),
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
            pupil_carrier_complex_npz=(
                None
                if gamma_cfg.get("pupil_carrier_complex_npz", online_cfg.get("pupil_carrier_complex_npz")) is None
                else str(gamma_cfg.get("pupil_carrier_complex_npz", online_cfg.get("pupil_carrier_complex_npz")))
            ),
            preserve_full_base_modes=bool(gamma_cfg.get("preserve_full_base_modes", False)),
        ),
        device=device,
    )
def base_coeff_maps_from_gamma_config(gamma_cfg: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    entries = gamma_cfg.get("base_coeff_maps")
    if entries is None:
        return ()
    return normalize_base_coeff_maps(entries)
def base_coeff_maps_from_training_config(train_cfg: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    online_cfg = train_cfg.get("online_generation") if isinstance(train_cfg.get("online_generation"), Mapping) else {}
    return normalize_base_coeff_maps(online_cfg.get("dual_domain_coeff_maps", ()))
def normalize_base_coeff_maps(entries: Any) -> tuple[tuple[str, str], ...]:
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
def roi_bank_gamma_route(
    train_cfg: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_base_dir: Path,
) -> dict[str, Any]:
    gamma_cfg = train_cfg.get("roi_bank_gamma")
    if not isinstance(gamma_cfg, Mapping) or gamma_cfg.get("enabled") is not True:
        return {"enabled": False}
    roi_source = resolve_roi_bank_source(gamma_cfg, train_cfg=train_cfg, config=config, config_base_dir=config_base_dir)
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
        "objective_source": gamma_objective_source(gamma_cfg, roi_source=roi_source),
        **({"roi_library_source": "auto_built"} if roi_source is not None else {}),
        **(roi_bank_source_metrics(roi_source) if roi_source is not None else {}),
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
def build_roi_projection_objective(
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
    roi_bank_source: ROIBankSource | None = None,
    objective: GammaProjectionObjective,
    train_cfg: Mapping[str, Any],
):
    over_cut = int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0)))
    max_emitters = int(gamma_cfg.get("num_posterior_samples", 2))
    objective_mode = str(gamma_cfg.get("roi_bank_objective", gamma_cfg.get("objective_mode", "poisson_nll"))).strip().lower()
    update_context: dict[str, ROIProjectionUpdateContext] = {}
    heldout_context = build_heldout_monitor_context(
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
            selected_bank, sampling_metrics = select_roi_bank_update_subset(bank, gamma_cfg, epoch=int(result.epoch))
            raw_frames, background, samples, roi_origin_xy_px, domain_names, loss_mask = record_projection_tensors(selected_bank)
            update_context["active"] = ROIProjectionUpdateContext(
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
    def active_context() -> ROIProjectionUpdateContext:
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
                selected_bank, sampling_metrics = select_roi_bank_update_subset(bank, gamma_cfg, epoch=0)
                raw_frames, background, samples, roi_origin_xy_px, domain_names, loss_mask = record_projection_tensors(selected_bank)
                context = ROIProjectionUpdateContext(
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
            **(roi_bank_source_metrics(roi_bank_source) if roi_bank_source is not None else {}),
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
            **heldout_monitor_metrics(
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
        if should_write_gamma_artifacts(gamma_cfg, result=result):
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
def resolve_roi_gamma_banks(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
) -> tuple[ROIBank, ROIBank | None, str, str | None]:
    if "heldout_roi_library_path" in gamma_cfg:
        return bank, load_roi_bank(str(gamma_cfg["heldout_roi_library_path"])), "fixed_samples", "configured_hdf5"
    heldout_count = auto_heldout_count(gamma_cfg, total_count=len(bank.records))
    if heldout_count <= 0:
        if auto_heldout_requested(gamma_cfg):
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
            domain_heldout_count = auto_heldout_count(gamma_cfg, total_count=len(domain_records))
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
def auto_heldout_requested(gamma_cfg: Mapping[str, Any]) -> bool:
    return int(gamma_cfg.get("auto_heldout_min_rois", 0)) > 0 or int(gamma_cfg.get("auto_heldout_max_rois", 0)) > 0
def auto_heldout_count(gamma_cfg: Mapping[str, Any], *, total_count: int) -> int:
    min_rois = int(gamma_cfg.get("auto_heldout_min_rois", 0))
    max_rois = int(gamma_cfg.get("auto_heldout_max_rois", min_rois))
    if min_rois <= 0 and max_rois <= 0:
        return 0
    return min(max(min_rois, 0), max(max_rois, 0), max(int(total_count) - 1, 0))
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
    roi_conditioning = roi_conditioning_context(train_cfg)
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
        photon_scale=posterior_photon_scale(train_cfg),
        z_scale=posterior_z_scale(train_cfg),
        candidate_threshold=float(
            gamma_cfg.get("posterior_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))
        ),
        split_threshold=float(gamma_cfg.get("posterior_adjacent_probability_threshold", gamma_cfg.get("split_threshold", 0.6))),
    )
    raw_frames = select_roi_window_frame(loc_batch.model_input)
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
def smoke_roi_bank(*, roi_size: int) -> ROIBank:
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
def count_gamma_updates(layout) -> int:
    path = layout.metrics_dir / "gamma_update_metrics.jsonl"
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def gamma_objective_source(gamma_cfg: Mapping[str, Any], *, roi_source: ROIBankSource | None = None) -> str:
    if "roi_library_path" in gamma_cfg:
        return "roi_projection_hdf5"
    if roi_source is not None or gamma_cfg.get("auto_build_roi_bank") is True:
        return "roi_projection_auto_built"
    if gamma_cfg.get("smoke_roi_library") is True:
        return "roi_projection_smoke"
    return "smoke_quadratic"
