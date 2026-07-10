from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import torch

from neptune_v03.localization.legacy_decode import (
    LegacyEmitterSet,
    decode_liteloc_eval_emitters,
    decode_legacy_targets,
    evaluate_legacy_localizations,
)
from neptune_v03.localization.conditioning import ConditioningProviderStore
from neptune_v03.localization.materialized_eval import MaterializedDatasetEvalConfig, build_materialized_dataset_eval_provider
from neptune_v03.localization.online import OnlineBatchProviderConfig, build_online_batch_provider
from neptune_v03.localization.runtime_config import build_localization_runtime_config
from neptune_v03.localization.training_adapter import LocalizationTrainBatch, localization_batch_to_device
from neptune_v03.training.loop import TrainingBatch


def localizer_eval_route(
    train_cfg: Mapping[str, Any],
    *,
    config_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    eval_cfg = train_cfg.get("eval")
    if not isinstance(eval_cfg, Mapping) or eval_cfg.get("enabled") is not True:
        return {"enabled": False, "source": None}
    source = str(eval_cfg.get("source", "online_generation"))
    if source in {"materialized_dataset", "materialized_microtube"}:
        return _materialized_eval_route(
            eval_cfg,
            train_cfg,
            source_alias=None if source == "materialized_dataset" else source,
            config_base_dir=None if config_base_dir is None else Path(config_base_dir),
        )
    if source != "online_generation":
        raise ValueError("train.eval.source must be online_generation or materialized_dataset")
    online_cfg = _mapping(train_cfg.get("online_generation"), "train.online_generation")
    batch_count = int(eval_cfg.get("batch_count", 1))
    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("batch_size", online_cfg.get("batch_size", 1))))
    if batch_count <= 0:
        raise ValueError("train.eval.batch_count must be positive")
    if batch_size <= 0:
        raise ValueError("train.eval.batch_size must be positive")
    return {
        "enabled": True,
        "source": source,
        "seed": int(eval_cfg.get("seed", int(online_cfg.get("seed", 0)) + 100000)),
        "batch_count": batch_count,
        "batch_size": batch_size,
    }


def build_localizer_eval_provider(
    train_cfg: Mapping[str, Any],
    *,
    config_base_dir: str | Path | None = None,
    root_config: Mapping[str, Any] | None = None,
    condition_store: ConditioningProviderStore | None = None,
):
    route = localizer_eval_route(train_cfg, config_base_dir=config_base_dir)
    if route["enabled"] is not True:
        return None
    if route["source"] == "materialized_dataset":
        return build_materialized_dataset_eval_provider(
            MaterializedDatasetEvalConfig(
                source_path=str(route["source_path"]),
                dataset_id=str(route["dataset_id"]),
                sample_id=str(route["sample_id"]),
                batch_size=int(route["batch_size"]),
                batch_count=int(route["batch_count"]),
                frame_range=route["frame_range"],
                crop=_mapping(route["crop"], "train.eval.crop"),
                heldout_split=_mapping(route["heldout_split"], "train.eval.heldout_split"),
            )
        )
    runtime_root = {"train": train_cfg} if root_config is None else root_config
    runtime_config = build_localization_runtime_config(
        runtime_root,
        config_base_dir=config_base_dir,
        seed=int(route["seed"]),
    )
    provider_cfg = _mapping(runtime_config.get("batch_provider"), "batch_provider")
    if provider_cfg.get("name") != "online_train_batch":
        raise ValueError("train.eval.source=online_generation requires train.online_generation")
    params = dict(_mapping(provider_cfg.get("params"), "batch_provider.params"))
    params.update(
        {
            "batch_size": int(route["batch_size"]),
            "seed": int(route["seed"]),
            "steps_per_epoch": int(route["batch_count"]),
        }
    )
    field_names = {field.name for field in fields(OnlineBatchProviderConfig)}
    provider = build_online_batch_provider(
        OnlineBatchProviderConfig(**{key: value for key, value in params.items() if key in field_names}),
        condition_store=condition_store,
    )
    if condition_store is not None:
        return lambda: list(provider(0))
    fixed_batches = tuple(provider(0))
    return lambda: list(fixed_batches)


def make_legacy_localization_eval_loss(
    base_loss_fn,
    train_cfg: Mapping[str, Any],
    *,
    root_config: Mapping[str, Any] | None = None,
):
    scaling_cfg = train_cfg.get("scaling") if isinstance(train_cfg.get("scaling"), Mapping) else {}
    loss_cfg = train_cfg.get("loss") if isinstance(train_cfg.get("loss"), Mapping) else {}
    eval_cfg = train_cfg.get("eval") if isinstance(train_cfg.get("eval"), Mapping) else {}
    photon_scale = float(scaling_cfg.get("photon_max", 1.0)) if "photon_max" in scaling_cfg else None
    z_scale = float(scaling_cfg.get("z_max", 1.0)) if "z_max" in scaling_cfg else None
    target_order = str(_mapping(loss_cfg.get("params", {}), "train.loss.params").get("target_order", "legacy_iwae")) if "params" in loss_cfg else "legacy_iwae"
    if "dist_tol_xy_nm" in eval_cfg:
        dist_tol_xy_nm = float(eval_cfg["dist_tol_xy_nm"])
        dist_tol_xy_px = None
    elif "dist_tolr_nm" in eval_cfg:
        dist_tol_xy_nm = float(eval_cfg["dist_tolr_nm"])
        dist_tol_xy_px = None
    elif "dist_tolr" in eval_cfg:
        dist_tol_xy_nm = float(eval_cfg["dist_tolr"])
        dist_tol_xy_px = None
    elif "dist_tol_xy_px" in eval_cfg:
        dist_tol_xy_nm = None
        dist_tol_xy_px = float(eval_cfg["dist_tol_xy_px"])
    else:
        dist_tol_xy_nm = 250.0
        dist_tol_xy_px = None
    dist_tol_z_nm = eval_cfg.get("dist_tol_z_nm", eval_cfg.get("dist_tolz_nm", eval_cfg.get("dist_tolz", 500.0)))
    dist_tol_z_nm = None if dist_tol_z_nm is None else float(dist_tol_z_nm)
    match_dims = int(eval_cfg.get("match_dims", 3))
    online_cfg = train_cfg.get("online_generation") if isinstance(train_cfg.get("online_generation"), Mapping) else {}
    optical_cfg = root_config.get("optical") if isinstance(root_config, Mapping) and isinstance(root_config.get("optical"), Mapping) else {}
    pixel_size_nm_x = float(eval_cfg.get("pixel_size_nm_x", online_cfg.get("pixel_size_nm_x", optical_cfg.get("pixel_size_nm_x", 1.0))))
    pixel_size_nm_y = float(eval_cfg.get("pixel_size_nm_y", online_cfg.get("pixel_size_nm_y", optical_cfg.get("pixel_size_nm_y", 1.0))))
    eval_loss_state: dict[str, object] = {"pred_sets": [], "target_sets": [], "batch_offset": 0}

    def eval_loss_fn(model: torch.nn.Module, batch: TrainingBatch) -> torch.Tensor:
        loss = base_loss_fn(model, batch)
        eval_loss_fn.last_metrics = dict(getattr(base_loss_fn, "last_metrics", {}) or {})
        loc_batch = batch.inputs
        if not isinstance(loc_batch, LocalizationTrainBatch):
            return loss
        loc_batch = localization_batch_to_device(loc_batch, _model_device(model))
        with torch.no_grad():
            y_out = model(loc_batch.model_input)
        if not isinstance(y_out, torch.Tensor) or y_out.ndim != 4 or int(y_out.shape[1]) != 10:
            return loss
        pred = decode_liteloc_eval_emitters(
            y_out,
            photon_scale=photon_scale,
            z_scale=z_scale,
        )
        target = decode_legacy_targets(
            loc_batch.pxyz_tar,
            loc_batch.mask_tar,
            target_order=target_order,
            photon_scale=photon_scale,
            z_scale=z_scale,
        )
        metrics = evaluate_legacy_localizations(
            pred,
            target,
            dist_tol_xy_px=dist_tol_xy_px,
            dist_tol_xy_nm=dist_tol_xy_nm,
            dist_tol_z_nm=dist_tol_z_nm,
            pixel_size_nm_x=pixel_size_nm_x,
            pixel_size_nm_y=pixel_size_nm_y,
            match_dims=match_dims,
        )
        batch_offset = int(eval_loss_state["batch_offset"])
        batch_size = _model_input_batch_size(loc_batch.model_input)
        eval_loss_state["pred_sets"].append(_offset_emitter_set(pred, batch_offset))
        eval_loss_state["target_sets"].append(_offset_emitter_set(target, batch_offset))
        eval_loss_state["batch_offset"] = batch_offset + batch_size
        eval_loss_fn.last_metrics.update(metrics.to_dict())
        eval_loss_fn.last_metrics["decode_contract"] = "liteloc_evalmetric_nms_v1"
        if dist_tol_xy_nm is not None:
            eval_loss_fn.last_metrics["legacy_eval_dist_tol_xy_nm"] = dist_tol_xy_nm
        if dist_tol_xy_px is not None:
            eval_loss_fn.last_metrics["legacy_eval_dist_tol_xy_px"] = dist_tol_xy_px
        if dist_tol_z_nm is not None:
            eval_loss_fn.last_metrics["legacy_eval_dist_tol_z_nm"] = dist_tol_z_nm
        eval_loss_fn.last_metrics["legacy_eval_match_dims"] = float(match_dims)
        eval_loss_fn.last_metrics["legacy_eval_pixel_size_nm_x"] = pixel_size_nm_x
        eval_loss_fn.last_metrics["legacy_eval_pixel_size_nm_y"] = pixel_size_nm_y
        return loss

    def reset_eval_metrics() -> None:
        eval_loss_state["pred_sets"] = []
        eval_loss_state["target_sets"] = []
        eval_loss_state["batch_offset"] = 0

    def aggregate_eval_metrics() -> dict[str, float]:
        pred_sets = list(eval_loss_state["pred_sets"])
        target_sets = list(eval_loss_state["target_sets"])
        if not pred_sets or not target_sets:
            return dict(eval_loss_fn.last_metrics)
        metrics = evaluate_legacy_localizations(
            _concat_emitter_sets(pred_sets),
            _concat_emitter_sets(target_sets),
            dist_tol_xy_px=dist_tol_xy_px,
            dist_tol_xy_nm=dist_tol_xy_nm,
            dist_tol_z_nm=dist_tol_z_nm,
            pixel_size_nm_x=pixel_size_nm_x,
            pixel_size_nm_y=pixel_size_nm_y,
            match_dims=match_dims,
        ).to_dict()
        merged = dict(eval_loss_fn.last_metrics)
        merged.update(metrics)
        merged["legacy_eval_batch_count"] = float(len(pred_sets))
        return merged

    eval_loss_fn.last_metrics = {}
    eval_loss_fn.reset_eval_metrics = reset_eval_metrics
    eval_loss_fn.aggregate_eval_metrics = aggregate_eval_metrics
    return eval_loss_fn


def _offset_emitter_set(emitters: LegacyEmitterSet, batch_offset: int) -> LegacyEmitterSet:
    if int(batch_offset) == 0:
        return emitters
    return LegacyEmitterSet(
        batch_index=emitters.batch_index + int(batch_offset),
        probability=emitters.probability,
        xyz_px_nm=emitters.xyz_px_nm,
        photons=emitters.photons,
        sigma_xy_px=emitters.sigma_xy_px,
    )


def _concat_emitter_sets(items: list[LegacyEmitterSet]) -> LegacyEmitterSet:
    nonempty = [item for item in items if int(item.batch_index.numel()) > 0]
    if not nonempty:
        return LegacyEmitterSet(
            batch_index=torch.empty((0,), dtype=torch.long),
            probability=torch.empty((0,), dtype=torch.float32),
            xyz_px_nm=torch.empty((0, 3), dtype=torch.float32),
            photons=torch.empty((0,), dtype=torch.float32),
            sigma_xy_px=torch.empty((0, 2), dtype=torch.float32),
        )
    return LegacyEmitterSet(
        batch_index=torch.cat([item.batch_index for item in nonempty], dim=0),
        probability=torch.cat([item.probability for item in nonempty], dim=0),
        xyz_px_nm=torch.cat([item.xyz_px_nm for item in nonempty], dim=0),
        photons=torch.cat([item.photons for item in nonempty], dim=0),
        sigma_xy_px=torch.cat([item.sigma_xy_px for item in nonempty], dim=0),
    )


def _model_input_batch_size(model_input: object) -> int:
    if torch.is_tensor(model_input):
        return int(model_input.shape[0])
    if isinstance(model_input, (tuple, list)) and model_input:
        first = model_input[0]
        if torch.is_tensor(first):
            return int(first.shape[0])
    raise TypeError("Localization eval model_input must be a tensor or a non-empty tensor tuple/list")


def _materialized_eval_route(
    eval_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    *,
    source_alias: str | None,
    config_base_dir: Path | None,
) -> dict[str, Any]:
    batch_count = int(eval_cfg.get("batch_count", 1))
    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 1)))
    if batch_count <= 0:
        raise ValueError("train.eval.batch_count must be positive")
    if batch_size <= 0:
        raise ValueError("train.eval.batch_size must be positive")
    route = {
        "enabled": True,
        "source": "materialized_dataset",
        "seed": None,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "dataset_id": str(eval_cfg["dataset_id"]),
        "sample_id": str(eval_cfg["sample_id"]),
        "source_path": _resolve_path(str(eval_cfg["source_path"]), base_dir=config_base_dir),
        "frame_range": [int(item) for item in eval_cfg["frame_range"]],
        "crop": dict(_mapping(eval_cfg["crop"], "train.eval.crop")),
        "heldout_split": dict(_mapping(eval_cfg["heldout_split"], "train.eval.heldout_split")),
    }
    if source_alias is not None:
        route["source_alias"] = source_alias
    return route


def _resolve_path(value: str, *, base_dir: Path | None) -> str:
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str((base_dir / path).resolve())


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
