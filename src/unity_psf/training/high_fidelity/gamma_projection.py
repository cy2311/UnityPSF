"""Gamma projection configuration, domain splitting, and artifact export."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from unity_psf.gamma_update import GammaProjectionObjective, GammaProjectionObjectiveConfig, GammaUpdateConfig
from unity_psf.optics.nat_field import full_roi_coeff_stack_torch
from unity_psf.roi_library import ROIBank, ROIRecord
from unity_psf.training.high_fidelity.diagnostics import path_token as _path_token
from unity_psf.training.loop_types import TrainingRunEpochResult, TrainingStepResult


def gamma_update_config(gamma_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> GammaUpdateConfig:
    return GammaUpdateConfig(
        start_epoch=int(gamma_cfg.get("start_epoch", 1)),
        stop_epoch=int(gamma_cfg.get("stop_epoch", train_cfg.get("epochs", 1))),
        update_interval_epochs=int(gamma_cfg.get("update_interval_epochs", gamma_cfg.get("interval_epochs", 1))),
        lr=float(gamma_cfg.get("gamma_lr", gamma_cfg.get("lr", 0.01))),
        steps=int(gamma_cfg.get("gamma_steps", gamma_cfg.get("steps", 1))),
        start_batch=None if gamma_cfg.get("start_batch") is None else int(gamma_cfg["start_batch"]),
        stop_batch=None if gamma_cfg.get("stop_batch") is None else int(gamma_cfg["stop_batch"]),
        update_interval_batches=None if gamma_cfg.get("update_interval_batches") is None else int(gamma_cfg["update_interval_batches"]),
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
        heldout_records = tuple(heldout_by_domain.get(domain_name, ()))
        domain_heldout = None
        if heldout_records:
            source = heldout_bank if heldout_bank is not None else selected_bank
            domain_heldout = ROIBank(
                records=heldout_records,
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
    del epoch
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
            domain_nat_config = replace(objective.nat_config, img_size_y=shape_hw[0], img_size_x=shape_hw[1])
            after_stack = full_roi_coeff_stack_torch(gamma_t, domain_nat_config, dtype=torch.float32, device=objective.device)
            before_stack = full_roi_coeff_stack_torch(before_gamma_t, domain_nat_config, dtype=torch.float32, device=objective.device)
            stacks = (after_stack.maps_nm, after_stack.maps_nm - before_stack.maps_nm, after_stack.mode_order)
            stack_cache[shape_hw] = stacks
            if mode_order is None:
                mode_order = after_stack.mode_order
        after_maps, delta_maps, update_mode_order = stacks
        if bool(objective.config.preserve_full_base_modes):
            source_maps, source_order = full_base_maps[str(domain_name)]
            maps = merge_zernike_delta_maps(source_maps, base_mode_order=source_order, delta_maps=after_maps, delta_mode_order=update_mode_order)
            output_mode_order = source_order
        else:
            maps = (base_maps_t + after_maps).detach().cpu().numpy()
            output_mode_order = mode_order or list(objective.nat_config.aberrations)
        if str(artifact_policy).lower() in {"compact", "compact_latest", "latest"}:
            path = layout.artifacts_dir / "roi_bank_gamma" / "latest" / "feedback" / _path_token(str(domain_name)) / "coeff_maps.npz"
        else:
            path = layout.artifacts_dir / "roi_bank_gamma" / f"step_{int(global_step):08d}" / "feedback" / _path_token(str(domain_name)) / "coeff_maps.npz"
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


def normalize_base_coeff_maps(entries: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(entries, (list, tuple)):
        return ()
    result: list[tuple[str, str]] = []
    for item in entries:
        if isinstance(item, Mapping):
            name = str(item.get("name", f"domain_{len(result)}"))
            path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            name, path = str(item[0]), item[1]
        else:
            continue
        if path:
            result.append((name, str(path)))
    return tuple(result)


def base_coeff_maps_from_gamma_cfg(gamma_cfg: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    entries = gamma_cfg.get("base_coeff_maps")
    return () if entries is None else normalize_base_coeff_maps(entries)


def base_coeff_maps_from_train_cfg(train_cfg: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    online_cfg = train_cfg.get("online_generation") if isinstance(train_cfg.get("online_generation"), Mapping) else {}
    return normalize_base_coeff_maps(online_cfg.get("dual_domain_coeff_maps", ()))


def z_range_nm_from_gamma_cfg(gamma_cfg: Mapping[str, Any]) -> tuple[float, float] | None:
    raw = gamma_cfg.get("z_range_nm")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("train.roi_bank_gamma.z_range_nm must be a two-element range")
    return float(raw[0]), float(raw[1])


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
            base_coeff_maps=base_coeff_maps_from_gamma_cfg(gamma_cfg) or base_coeff_maps_from_train_cfg(train_cfg),
            objective_mode=str(gamma_cfg.get("roi_bank_objective", gamma_cfg.get("objective_mode", "poisson_nll"))),
            projection_sample_batch_size=int(gamma_cfg.get("roi_bank_projection_sample_batch_size", gamma_cfg.get("projection_sample_batch_size", 16))),
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
            fixed_astig_anchor_nm=(
                None
                if gamma_cfg.get("fixed_astig_anchor_nm", online_cfg.get("fixed_astig_anchor_nm")) is None
                else float(gamma_cfg.get("fixed_astig_anchor_nm", online_cfg.get("fixed_astig_anchor_nm")))
            ),
        ),
        device=device,
    )


__all__ = [
    "base_coeff_maps_from_gamma_cfg",
    "base_coeff_maps_from_train_cfg",
    "build_vector_roi_gamma_objective",
    "export_gamma_feedback_coeff_maps",
    "gamma_update_config",
    "merge_zernike_delta_maps",
    "normalize_base_coeff_maps",
    "select_single_channel_roi_split",
    "should_run_gamma_update",
    "should_write_gamma_artifacts",
    "split_roi_bank_by_domain",
    "z_range_nm_from_gamma_cfg",
]
