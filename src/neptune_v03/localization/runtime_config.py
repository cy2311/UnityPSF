from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from neptune_v03.localization.smlm_targets import V03_PXYZ_TARGET_ORDER


_ACTIVE_SMLM_LOSS_PARAM_KEYS = frozenset(
    {
        "detection_weight",
        "pxyz_weight",
        "background_weight",
        "sigma_weight",
        "photon_scale",
        "z_scale",
        "z_activation",
        "sigma_min",
    }
)
_LEGACY_GMM_LOSS_PARAM_KEYS = frozenset({"gmm_target_chunk", "gmm_component_chunk", "gmm_backend"})
_ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS = frozenset(
    {
        "xyoffset",
        "ch_weight",
        "photon_scale",
        "z_scale",
        "disable_attr",
        "gmm_target_chunk",
        "gmm_component_chunk",
        "gmm_backend",
        "target_order",
        "eps",
    }
)


def build_localization_runtime_config(
    config: Mapping[str, Any],
    *,
    config_base_dir: str | Path | None = None,
    model_name: str | None = None,
    model_params: Mapping[str, Any] | None = None,
    optimizer_name: str | None = None,
    optimizer_params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    train_cfg = _mapping(config.get("train"), "train")
    microtube_cfg = train_cfg.get("microtube_tiff")
    if isinstance(microtube_cfg, Mapping) and microtube_cfg.get("enabled") is True:
        batch_provider = _microtube_tiff_provider_config(train_cfg, microtube_cfg, seed=seed)
    else:
        online_cfg = _mapping(train_cfg.get("online_generation"), "train.online_generation")
        if online_cfg.get("enabled", True) is not True:
            raise ValueError("train.online_generation.enabled must be true")
        batch_provider = _online_provider_config(config, train_cfg, online_cfg, config_base_dir=config_base_dir, seed=seed)
    resolved_model_name, resolved_model_params = resolve_localization_model_config(config)
    if model_name is not None:
        resolved_model_name = str(model_name)
        resolved_model_params = {}
    resolved_model_params.update(dict(model_params or {}))
    resolved_optimizer_name, resolved_optimizer_params = _resolve_training_optimizer_config(
        config,
        train_cfg,
        optimizer_name=optimizer_name,
        optimizer_params=optimizer_params,
    )

    loss = _loss_config(config, train_cfg, model_name=resolved_model_name)
    runtime_config = {
        "device": str(train_cfg.get("device", "cpu")),
        "model": {"name": resolved_model_name, "params": resolved_model_params},
        "optimizer": {"name": resolved_optimizer_name, "params": resolved_optimizer_params},
        "batch_provider": batch_provider,
        "loss": loss,
        "epochs": {"start": 1, "stop": int(train_cfg.get("epochs", 1))},
        "resolved_contract": _resolved_contract(
            model_name=resolved_model_name,
            model_params=resolved_model_params,
            batch_provider=batch_provider,
            loss=loss,
            legacy_loss_params=_legacy_gmm_loss_params(train_cfg.get("loss")),
            training_runtime=_training_runtime_contract(
                config,
                train_cfg,
                optimizer_name=resolved_optimizer_name,
                optimizer_params=resolved_optimizer_params,
            ),
        ),
    }
    if train_cfg.get("max_batches") is not None:
        runtime_config["max_batches"] = int(train_cfg["max_batches"])
    feedback_cfg = train_cfg.get("feedback")
    if isinstance(feedback_cfg, Mapping) and "map_path" in feedback_cfg:
        runtime_config["feedback"] = {"map_path": str(feedback_cfg["map_path"])}
    return runtime_config


def resolve_localization_model_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    train_cfg = _mapping(config.get("train"), "train")
    microtube_cfg = train_cfg.get("microtube_tiff")
    if isinstance(microtube_cfg, Mapping) and microtube_cfg.get("enabled") is True:
        return "production_localizer", {"in_channels": int(microtube_cfg.get("channels", 3))}

    online_cfg = train_cfg.get("online_generation")
    if not isinstance(online_cfg, Mapping):
        return "production_localizer", {"in_channels": 3}

    channels = int(online_cfg.get("channels", 3))
    conditioning_mode = str(online_cfg.get("conditioning_mode", "channels"))
    expert_mode = str(online_cfg.get("expert_mode", ""))
    if conditioning_mode != "film" or expert_mode != "soft_moe":
        return "production_localizer", {"in_channels": channels}

    _validate_condition_dimensions(online_cfg, soft_moe=True)
    model_params: dict[str, Any] = {
        "nch_in": channels,
        "condition_feature_dim": _condition_feature_dim(online_cfg),
        "domain_count": int(online_cfg.get("domain_count", 2)),
        "film_hidden_dim": int(online_cfg.get("film_hidden_dim", 32)),
        "depth_shared": 2,
        "depth_union": 2,
        "nfeatures_init": 48,
        "nfeatures_inter": None,
        "norm_start_level": -1,
        "norm_groups": 0,
        "activation": "ELU",
        "dropout_start_level": None,
        "p_dropout": -0.1,
        "pool_mode": "StrideConv",
        "upsample_mode": "nearest",
        "inter_activation": None,
        "norm_head_groups": 0,
        "final_activation": "ELU",
        "kaiming_normal": True,
        "depthwise": True,
    }
    model_params["condition_dim"] = _condition_dim(online_cfg)
    overrides = _localization_overrides(config, train_cfg)
    for key in (
        "depthwise",
        "depth_shared",
        "depth_union",
        "nfeatures_init",
        "nfeatures_inter",
        "norm_start_level",
        "norm_groups",
        "activation",
        "dropout_start_level",
        "p_dropout",
        "pool_mode",
        "upsample_mode",
        "inter_activation",
        "norm_head_groups",
        "final_activation",
        "disabled_attr",
        "kaiming_normal",
        "z_mu_activation",
        "film_hidden_dim",
    ):
        if key in overrides:
            model_params[key] = _normalize_z_activation(overrides[key]) if key == "z_mu_activation" else overrides[key]
    return "active_smlm_soft_moe_double_unet", model_params


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _loss_config(root_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any], *, model_name: str) -> dict[str, Any]:
    loss_cfg = train_cfg.get("loss")
    if isinstance(loss_cfg, Mapping) and "name" in loss_cfg:
        name = str(loss_cfg["name"])
        if name == "active_smlm_gmm_loss":
            return {
                "name": name,
                "params": _active_smlm_gmm_loss_params(train_cfg, _mapping(loss_cfg.get("params", {}), "train.loss.params")),
            }
        return {
            "name": name,
            "params": dict(_mapping(loss_cfg.get("params", {}), "train.loss.params")),
        }
    if str(model_name).startswith("active_smlm_"):
        raw_loss_keys = set(loss_cfg) if isinstance(loss_cfg, Mapping) else set()
        legacy_gmm_params = _legacy_gmm_loss_params(loss_cfg)
        if legacy_gmm_params and raw_loss_keys <= _LEGACY_GMM_LOSS_PARAM_KEYS:
            return {"name": "active_smlm_gmm_loss", "params": _active_smlm_gmm_loss_params(train_cfg, legacy_gmm_params)}
        params = _active_smlm_loss_params(root_cfg, train_cfg, loss_cfg)
        return {"name": "active_smlm_loss", "params": params}
    return {"name": "localization_mse", "params": {}}


def _active_smlm_loss_params(root_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any], loss_cfg: Any) -> dict[str, Any]:
    raw_params = dict(loss_cfg) if isinstance(loss_cfg, Mapping) else {}
    unknown = sorted(set(raw_params) - _ACTIVE_SMLM_LOSS_PARAM_KEYS - _LEGACY_GMM_LOSS_PARAM_KEYS)
    if unknown:
        raise ValueError(f"unknown active_smlm_loss params: {unknown}")
    params = {key: raw_params[key] for key in _ACTIVE_SMLM_LOSS_PARAM_KEYS if key in raw_params}
    if "photon_scale" not in params:
        scaling_cfg = train_cfg.get("scaling")
        normalization_cfg = train_cfg.get("normalization")
        if isinstance(scaling_cfg, Mapping) and "photon_max" in scaling_cfg:
            params["photon_scale"] = float(scaling_cfg["photon_max"])
        elif isinstance(normalization_cfg, Mapping) and "photon_scale" in normalization_cfg:
            params["photon_scale"] = float(normalization_cfg["photon_scale"])
    if "z_scale" not in params:
        scaling_cfg = train_cfg.get("scaling")
        if isinstance(scaling_cfg, Mapping) and "z_max" in scaling_cfg:
            params["z_scale"] = float(scaling_cfg["z_max"])
        else:
            train_params_cfg = train_cfg.get("train_params")
            if isinstance(train_params_cfg, Mapping) and "z_max" in train_params_cfg:
                params["z_scale"] = float(train_params_cfg["z_max"])
    if "z_activation" not in params:
        overrides = _localization_overrides(root_cfg, train_cfg)
        if "z_mu_activation" in overrides:
            params["z_activation"] = _normalize_z_activation(overrides["z_mu_activation"])
    elif params["z_activation"] is not None:
        params["z_activation"] = _normalize_z_activation(params["z_activation"])
    return params


def _active_smlm_gmm_loss_params(train_cfg: Mapping[str, Any], raw_params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(raw_params) - _ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS)
    if unknown:
        raise ValueError(f"unknown active_smlm_gmm_loss params: {unknown}")
    params = {key: raw_params[key] for key in _ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS if key in raw_params}
    if "photon_scale" not in params:
        scaling_cfg = train_cfg.get("scaling")
        normalization_cfg = train_cfg.get("normalization")
        if isinstance(scaling_cfg, Mapping) and "photon_max" in scaling_cfg:
            params["photon_scale"] = float(scaling_cfg["photon_max"])
        elif isinstance(normalization_cfg, Mapping) and "photon_scale" in normalization_cfg:
            params["photon_scale"] = float(normalization_cfg["photon_scale"])
    if "z_scale" not in params:
        scaling_cfg = train_cfg.get("scaling")
        if isinstance(scaling_cfg, Mapping) and "z_max" in scaling_cfg:
            params["z_scale"] = float(scaling_cfg["z_max"])
        else:
            train_params_cfg = train_cfg.get("train_params")
            if isinstance(train_params_cfg, Mapping) and "z_max" in train_params_cfg:
                params["z_scale"] = float(train_params_cfg["z_max"])
    params.setdefault("target_order", "legacy_iwae")
    return params


def _legacy_gmm_loss_params(loss_cfg: Any) -> dict[str, Any]:
    if not isinstance(loss_cfg, Mapping) or "name" in loss_cfg:
        return {}
    return {key: loss_cfg[key] for key in sorted(_LEGACY_GMM_LOSS_PARAM_KEYS) if key in loss_cfg}


def _resolved_contract(
    *,
    model_name: str,
    model_params: Mapping[str, Any],
    batch_provider: Mapping[str, Any],
    loss: Mapping[str, Any],
    legacy_loss_params: Mapping[str, Any],
    training_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    provider_params = _mapping(batch_provider.get("params"), "batch_provider.params")
    loss_params = dict(_mapping(loss.get("params", {}), "loss.params"))
    contract: dict[str, Any] = {
        "model": {
            "name": str(model_name),
            "output": "smlm_10ch" if str(model_name).startswith("active_smlm_") else "localization",
        },
        "batch_provider": {
            "name": str(batch_provider["name"]),
            "pxyz_target_order": loss_params.get("target_order", V03_PXYZ_TARGET_ORDER),
        },
        "loss": {
            "name": str(loss["name"]),
            "params": loss_params,
            "legacy_params": dict(legacy_loss_params),
        },
        "training_runtime": dict(training_runtime),
    }
    if str(model_name).startswith("active_smlm_"):
        contract["model"]["z_mu_activation"] = str(model_params.get("z_mu_activation", loss_params.get("z_activation", "tanh")))
    for key in (
        "batch_strategy",
        "sequence_count",
        "condition_dim",
        "condition_feature_dim",
        "domain_count",
        "append_domain_onehot",
    ):
        if key in provider_params:
            contract["batch_provider"][key] = provider_params[key]
    if provider_params.get("dual_domain_coeff_maps"):
        contract["batch_provider"]["dual_domain_coeff_maps"] = provider_params["dual_domain_coeff_maps"]
    return contract


def _training_runtime_contract(
    root_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    *,
    optimizer_name: str,
    optimizer_params: Mapping[str, Any],
) -> dict[str, Any]:
    online_cfg = train_cfg.get("online_generation")
    grad_clip_norm = None
    amp_enabled = False
    amp_dtype = None
    if isinstance(online_cfg, Mapping):
        if "grad_clip_norm" in online_cfg:
            grad_clip_norm = float(online_cfg["grad_clip_norm"])
        amp_enabled = bool(online_cfg.get("amp_enabled", False))
        if "amp_dtype" in online_cfg:
            amp_dtype = str(online_cfg["amp_dtype"])
    legacy_optimizer = _legacy_optimizer_contract(root_cfg, train_cfg)
    if legacy_optimizer is not None and _optimizer_matches_legacy(optimizer_name, optimizer_params, legacy_optimizer):
        legacy_optimizer = {**legacy_optimizer, "active": True, "inactive_reason": None}
    scheduler = _scheduler_contract(root_cfg)
    contract = {
        "optimizer": {"name": str(optimizer_name), "params": dict(optimizer_params)},
        "legacy_optimizer": legacy_optimizer,
        "scheduler": scheduler,
        "grad_clip": {"configured_norm": grad_clip_norm, "active": grad_clip_norm is not None},
        "amp": {
            "configured": amp_enabled,
            "dtype": amp_dtype,
            "active": amp_enabled,
            "inactive_reason": None if amp_enabled else "not_configured",
        },
    }
    if train_cfg.get("max_batches") is not None:
        contract["max_batches"] = int(train_cfg["max_batches"])
    elif legacy_optimizer is not None or scheduler.get("legacy_source") is not None:
        contract["max_batches"] = None
    return contract


def _resolve_training_optimizer_config(
    root_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    *,
    optimizer_name: str | None,
    optimizer_params: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    explicit_params = dict(optimizer_params or {})
    if optimizer_name is not None:
        return str(optimizer_name), explicit_params

    legacy_optimizer = _legacy_optimizer_contract(root_cfg, train_cfg)
    if legacy_optimizer is not None:
        params = dict(_mapping(legacy_optimizer["params"], "legacy_optimizer.params"))
        params.update(explicit_params)
        return str(legacy_optimizer["name"]).lower(), params

    if not explicit_params and train_cfg.get("learning_rate") is not None:
        explicit_params["lr"] = float(train_cfg["learning_rate"])
    return "sgd", explicit_params


def _legacy_optimizer_contract(root_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    overrides = root_cfg.get("smlm_overrides")
    if not isinstance(overrides, Mapping) or not overrides.get("optimizer"):
        return None
    online_cfg = train_cfg.get("online_generation")
    params: dict[str, Any] = {
        "lr": _legacy_optimizer_lr(overrides, train_cfg, online_cfg),
        "weight_decay": _legacy_optimizer_weight_decay(overrides, online_cfg),
    }
    return {
        "name": str(overrides["optimizer"]),
        "params": params,
        "active": False,
        "inactive_reason": "optimizer_runtime_not_wired",
        "legacy_source": "smlm_overrides",
    }


def _optimizer_matches_legacy(
    optimizer_name: str,
    optimizer_params: Mapping[str, Any],
    legacy_optimizer: Mapping[str, Any],
) -> bool:
    if str(optimizer_name).lower() != str(legacy_optimizer["name"]).lower():
        return False
    legacy_params = _mapping(legacy_optimizer["params"], "legacy_optimizer.params")
    return dict(optimizer_params) == dict(legacy_params)


def _legacy_optimizer_lr(overrides: Mapping[str, Any], train_cfg: Mapping[str, Any], online_cfg: Any) -> float:
    if isinstance(online_cfg, Mapping) and online_cfg.get("optimizer_lr") is not None:
        return float(online_cfg["optimizer_lr"])
    if train_cfg.get("learning_rate") is not None:
        return float(train_cfg["learning_rate"])
    return float(overrides.get("optimizer_lr", 0.0006))


def _legacy_optimizer_weight_decay(overrides: Mapping[str, Any], online_cfg: Any) -> float:
    if isinstance(online_cfg, Mapping) and online_cfg.get("weight_decay") is not None:
        return float(online_cfg["weight_decay"])
    return float(overrides.get("optimizer_weight_decay", 0.1))


def _scheduler_contract(root_cfg: Mapping[str, Any]) -> dict[str, Any]:
    overrides = root_cfg.get("smlm_overrides")
    if not isinstance(overrides, Mapping) or not overrides.get("lr_scheduler"):
        return {
            "name": "none",
            "active": False,
            "step_unit": None,
            "params": {},
            "inactive_reason": "not_configured",
        }
    name = str(overrides["lr_scheduler"])
    params: dict[str, Any] = {}
    if overrides.get("lr_step_size") is not None:
        params["step_size"] = int(overrides["lr_step_size"])
    if overrides.get("lr_gamma") is not None:
        params["gamma"] = float(overrides["lr_gamma"])
    step_unit = _scheduler_step_unit(overrides.get("lr_step_unit"))
    active = name.lower() == "steplr" and step_unit in {"optimizer_step", "epoch"}
    return {
        "name": name,
        "active": active,
        "step_unit": step_unit,
        "params": params,
        "inactive_reason": None if active else "scheduler_runtime_not_wired",
        "legacy_source": "smlm_overrides",
    }


def _scheduler_step_unit(value: Any) -> str:
    unit = str(value or "epoch").strip().lower()
    if unit in {"optimizer_step", "step", "batch", "iteration", "iter"}:
        return "optimizer_step"
    return "epoch"


def _localization_overrides(root_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    root_overrides = root_cfg.get("localization_overrides")
    if isinstance(root_overrides, Mapping):
        overrides.update(root_overrides)
    train_overrides = train_cfg.get("localization_overrides")
    if isinstance(train_overrides, Mapping):
        overrides.update(train_overrides)
    return overrides


def _normalize_z_activation(value: Any) -> str:
    z_activation = str(value).strip().lower()
    if z_activation not in {"tanh", "sigmoid"}:
        raise ValueError("z_mu_activation must be 'tanh' or 'sigmoid'")
    return z_activation


def _online_provider_config(
    config: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    config_base_dir: str | Path | None,
    seed: int,
) -> dict[str, Any]:
    batch_size = int(train_cfg.get("batch_size", online_cfg.get("batch_size", 1)))
    soft_moe = str(online_cfg.get("conditioning_mode", "channels")) == "film" and str(online_cfg.get("expert_mode", "")) == "soft_moe"
    _validate_condition_dimensions(online_cfg, soft_moe=soft_moe)
    dual_domain_coeff_maps = _dual_domain_coeff_maps(
        online_cfg.get("dual_domain_coeff_maps", ()),
        base_dir=None if config_base_dir is None else Path(config_base_dir),
    )
    if dual_domain_coeff_maps and int(online_cfg.get("domain_count", 2)) != len(dual_domain_coeff_maps):
        raise ValueError("domain_count must match dual_domain_coeff_maps length")
    simulation_cfg = _optional_mapping(config.get("simulation"), "simulation") or {}
    psf_cfg = _optional_mapping(simulation_cfg.get("psf"), "simulation.psf") or {}
    vector_cfg = _optional_mapping(psf_cfg.get("vector"), "simulation.psf.vector") or {}
    optical_cfg = _optional_mapping(config.get("optical"), "optical") or {}
    height = int(online_cfg.get("height", 128))
    width = int(online_cfg.get("width", 128))
    pixel_size_nm_x = float(online_cfg.get("pixel_size_nm_x", optical_cfg.get("pixel_size_nm_x", 101.11)))
    pixel_size_nm_y = float(online_cfg.get("pixel_size_nm_y", optical_cfg.get("pixel_size_nm_y", 98.83)))
    scaling_cfg = _optional_mapping(train_cfg.get("scaling"), "train.scaling") or {}
    sim_ranges = _online_simulator_range_params(
        config,
        online_cfg,
        train_cfg=train_cfg,
        height=height,
        width=width,
        pixel_size_nm_x=pixel_size_nm_x,
        pixel_size_nm_y=pixel_size_nm_y,
    )
    return {
        "name": "online_train_batch",
        "params": {
            "batch_size": batch_size,
            "channels": int(online_cfg.get("channels", 3)),
            "height": height,
            "width": width,
            "emitters_per_sample": int(online_cfg.get("emitters_per_sample", 8)),
            "seed": int(seed),
            "steps_per_epoch": int(online_cfg.get("steps_per_epoch", 1)),
            "background": float(online_cfg.get("background", 0.0)),
            "signal": float(online_cfg.get("signal", 1.0)),
            "simulation_backend": str(online_cfg.get("simulation_backend", "native")),
            "simulation_output_device": str(online_cfg.get("simulation_output_device", "cpu")),
            "cached_window_order": str(online_cfg.get("cached_window_order", "auto")),
            "cached_window_max_gpu_sequences": int(online_cfg.get("cached_window_max_gpu_sequences", 0)),
            "psf_type": str(online_cfg.get("psf_type", psf_cfg.get("psf_type", "vector"))),
            "pixel_size_nm_x": pixel_size_nm_x,
            "pixel_size_nm_y": pixel_size_nm_y,
            "wavelength_nm": float(online_cfg.get("wavelength_nm", optical_cfg.get("wavelength_nm", 660.0))),
            "na": float(online_cfg.get("NA", online_cfg.get("na", optical_cfg.get("NA", 1.4)))),
            "npupil": int(online_cfg.get("npupil", vector_cfg.get("npupil", 128))),
            "vector_psf_size": int(online_cfg.get("vector_psf_size", (online_cfg.get("lut_simulation") or {}).get("psf_size", vector_cfg.get("psf_size", 51)))),
            "vector_batch_size": int(online_cfg.get("vector_batch_size", vector_cfg.get("batch_size", 96))),
            "lut_field_stride": int((online_cfg.get("lut_simulation") or {}).get("field_stride", online_cfg.get("lut_field_stride", 16))),
            "lut_z_steps": int(online_cfg.get("nat_grid_z_steps", (online_cfg.get("lut_simulation") or {}).get("z_steps", online_cfg.get("lut_z_steps", 41)))),
            "lut_subpixel_bins": int((online_cfg.get("lut_simulation") or {}).get("subpixel_bins", online_cfg.get("lut_subpixel_bins", 1))),
            "lut_field_mode": str((online_cfg.get("lut_simulation") or {}).get("field_mode", online_cfg.get("lut_field_mode", "roi_origin"))),
            "lut_storage_dtype": str((online_cfg.get("lut_simulation") or {}).get("storage_dtype", online_cfg.get("lut_storage_dtype", "fp32"))),
            "field_origin_sampling_mode": str(online_cfg.get("field_origin_sampling_mode", "grid")),
            "field_origin_stride_px": int(online_cfg.get("field_origin_stride_px", online_cfg.get("field_origin_stride", 40))),
            **sim_ranges,
            "conditioning_mode": str(online_cfg.get("conditioning_mode", "channels")),
            "nat_simulation_mode": str(online_cfg.get("nat_simulation_mode", "tile_center")),
            "nat_grid_size": _grid_size(online_cfg.get("nat_grid_size", 32)),
            "nat_grid_z_steps": int(online_cfg.get("nat_grid_z_steps", 41)),
            "append_domain_onehot": _append_domain_onehot(online_cfg, soft_moe=soft_moe),
            "condition_feature_dim": _condition_feature_dim(online_cfg),
            "condition_dim": _condition_dim(online_cfg),
            "domain_count": int(online_cfg.get("domain_count", 2)),
            "domain_balance_mode": str(online_cfg.get("domain_balance_mode", "fixed")),
            "dual_domain_coeff_maps": dual_domain_coeff_maps,
            "batch_strategy": str(online_cfg.get("batch_strategy", "triplet")),
            "sequence_window_chunks": int(online_cfg.get("sequence_window_chunks", 1)),
            "sequence_count": int(online_cfg.get("sequence_count", 64)),
            "camera_qe": float((config.get("camera") or {}).get("qe", online_cfg.get("camera_qe", 0.9))),
            "camera_spurious_charge": float(
                (config.get("camera") or {}).get("spurious_charge", online_cfg.get("camera_spurious_charge", 0.002))
            ),
            "camera_baseline": float((config.get("camera") or {}).get("baseline", online_cfg.get("camera_baseline", 398.6))),
            "camera_e_per_adu": float((config.get("camera") or {}).get("e_per_adu", online_cfg.get("camera_e_per_adu", 1.020784562122306))),
            "pxyz_target_order": str(online_cfg.get("pxyz_target_order", "legacy_iwae")),
            "photon_scale": (
                None
                if "photon_max" not in scaling_cfg
                else float(scaling_cfg["photon_max"])
            ),
            "z_scale": (
                None
                if "z_max" not in scaling_cfg
                else float(scaling_cfg["z_max"])
            ),
        },
    }


def _microtube_tiff_provider_config(
    train_cfg: Mapping[str, Any],
    microtube_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "name": "microtube_tiff_train_batch",
        "params": {
            "tiff_path": str(microtube_cfg["tiff_path"]),
            "batch_size": int(train_cfg.get("batch_size", microtube_cfg.get("batch_size", 1))),
            "channels": int(microtube_cfg.get("channels", 3)),
            "height": None if microtube_cfg.get("height") is None else int(microtube_cfg["height"]),
            "width": None if microtube_cfg.get("width") is None else int(microtube_cfg["width"]),
            "steps_per_epoch": int(microtube_cfg.get("steps_per_epoch", 1)),
            "frame_start": int(microtube_cfg.get("frame_start", 0)),
            "frame_stop": None if microtube_cfg.get("frame_stop") is None else int(microtube_cfg["frame_stop"]),
            "crop_top": int(microtube_cfg.get("crop_top", 0)),
            "crop_left": int(microtube_cfg.get("crop_left", 0)),
            "seed": int(seed),
            "calibration": dict(_mapping(microtube_cfg.get("calibration", {}), "train.microtube_tiff.calibration")),
            "normalization": dict(_mapping(microtube_cfg.get("normalization", {}), "train.microtube_tiff.normalization")),
        },
    }


def _online_simulator_range_params(
    config: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    train_cfg: Mapping[str, Any],
    height: int,
    width: int,
    pixel_size_nm_x: float,
    pixel_size_nm_y: float,
) -> dict[str, Any]:
    simulation_cfg = _optional_mapping(config.get("simulation"), "simulation") or {}
    emitter_cfg = _optional_mapping(simulation_cfg.get("emitter"), "simulation.emitter") or {}
    scaling_cfg = _optional_mapping(train_cfg.get("scaling"), "train.scaling") or {}
    online_lut_cfg = _optional_mapping(online_cfg.get("lut_simulation"), "train.online_generation.lut_simulation") or {}

    params: dict[str, Any] = {}
    background_range = _range_from_config(
        online_cfg.get("background_range", simulation_cfg.get("background_uniform")),
        label="background_range",
    )
    if background_range is not None:
        params["background_range"] = background_range
    if "background_scale" in online_cfg:
        params["background_scale"] = float(online_cfg["background_scale"])
    elif "bg_max" in scaling_cfg:
        params["background_scale"] = float(scaling_cfg["bg_max"])

    z_range = _range_from_config(online_cfg.get("z_range", emitter_cfg.get("z_range")), label="z_range")
    if z_range is None and "z_max" in scaling_cfg:
        z_max = float(scaling_cfg["z_max"])
        z_range = (-z_max, z_max)
    if z_range is not None:
        params["z_range"] = z_range

    photon_range = _range_from_config(
        online_cfg.get("photon_range", emitter_cfg.get("intensity_clip")),
        label="photon_range",
    )
    if photon_range is None and "photon_max" in scaling_cfg:
        photon_range = (0.0, float(scaling_cfg["photon_max"]))
    if photon_range is not None:
        params["photon_range"] = photon_range

    intensity_mu_sig = _pair_from_config(
        online_cfg.get("photon_mean_sigma", emitter_cfg.get("intensity_mu_sig")),
        label="photon_mean_sigma",
    )
    if intensity_mu_sig is not None:
        params["photon_mean"] = float(intensity_mu_sig[0])
        params["photon_sigma"] = float(intensity_mu_sig[1])
    elif "photon_mean" in online_cfg and "photon_sigma" in online_cfg:
        params["photon_mean"] = float(online_cfg["photon_mean"])
        params["photon_sigma"] = float(online_cfg["photon_sigma"])

    density = _online_density_um2(
        online_cfg,
        simulation_cfg,
        emitter_cfg,
        frames_per_sample=int(simulation_cfg.get("frames_per_sample", online_cfg.get("channels", 3))),
        height=int(height),
        width=int(width),
        pixel_size_nm_x=float(pixel_size_nm_x),
        pixel_size_nm_y=float(pixel_size_nm_y),
    )
    if density is not None:
        params["emitter_density_um2"] = float(density)
    if "lifetime_avg" in online_cfg:
        params["lifetime_avg"] = float(online_cfg["lifetime_avg"])
    elif "lifetime_avg" in emitter_cfg:
        params["lifetime_avg"] = float(emitter_cfg["lifetime_avg"])
    if "warmup_frames" in online_cfg:
        params["warmup_frames"] = float(online_cfg["warmup_frames"])
    elif "warmup_frames" in online_lut_cfg:
        params["warmup_frames"] = float(online_lut_cfg["warmup_frames"])

    return params


def _online_density_um2(
    online_cfg: Mapping[str, Any],
    simulation_cfg: Mapping[str, Any],
    emitter_cfg: Mapping[str, Any],
    *,
    frames_per_sample: int,
    height: int,
    width: int,
    pixel_size_nm_x: float,
    pixel_size_nm_y: float,
) -> float | None:
    for source in (online_cfg, emitter_cfg, simulation_cfg):
        if "emitter_density_um2" in source:
            return float(source["emitter_density_um2"])
        if "density_um2" in source:
            return float(source["density_um2"])
        if "density" in source:
            return float(source["density"])
    num_emitters = simulation_cfg.get("num_emitters")
    if num_emitters is not None:
        lifetime_avg = float(emitter_cfg.get("lifetime_avg", 1.0))
        active = float(num_emitters) * (lifetime_avg + 1.0) / (float(frames_per_sample) + 6.0 * lifetime_avg)
        area_um2 = (
            int(width)
            * float(pixel_size_nm_x)
            / 1000.0
            * int(height)
            * float(pixel_size_nm_y)
            / 1000.0
        )
        return active / max(area_um2, 1e-9)
    return None


def _range_from_config(value: Any, *, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    lo, hi = float(value[0]), float(value[1])
    if hi < lo:
        raise ValueError(f"{label} max must be greater than or equal to min")
    return lo, hi


def _pair_from_config(value: Any, *, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    return float(value[0]), float(value[1])


def _optional_mapping(value: Any, label: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, label)


def _grid_size(value: Any) -> int | tuple[int, int]:
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    return int(value)


def _condition_feature_dim(online_cfg: Mapping[str, Any]) -> int:
    if "condition_feature_dim" in online_cfg:
        return int(online_cfg["condition_feature_dim"])
    if "condition_dim" in online_cfg:
        domain_terms = int(online_cfg.get("domain_count", 2)) if bool(online_cfg.get("append_domain_onehot", False)) else 0
        return max(0, int(online_cfg["condition_dim"]) - domain_terms)
    return 8


def _condition_dim(online_cfg: Mapping[str, Any]) -> int:
    if "condition_dim" in online_cfg:
        return int(online_cfg["condition_dim"])
    soft_moe = str(online_cfg.get("conditioning_mode", "channels")) == "film" and str(online_cfg.get("expert_mode", "")) == "soft_moe"
    domain_terms = int(online_cfg.get("domain_count", 2)) if _append_domain_onehot(online_cfg, soft_moe=soft_moe) else 0
    return _condition_feature_dim(online_cfg) + domain_terms


def _append_domain_onehot(online_cfg: Mapping[str, Any], *, soft_moe: bool) -> bool:
    if "append_domain_onehot" in online_cfg:
        return bool(online_cfg["append_domain_onehot"])
    return bool(soft_moe)


def _validate_condition_dimensions(online_cfg: Mapping[str, Any], *, soft_moe: bool) -> None:
    if not soft_moe:
        return
    if "condition_feature_dim" not in online_cfg or "condition_dim" not in online_cfg:
        return
    expected = int(online_cfg["condition_feature_dim"]) + int(online_cfg.get("domain_count", 2))
    if int(online_cfg["condition_dim"]) != expected:
        raise ValueError("condition_dim must equal condition_feature_dim + domain_count for soft_moe film conditioning")


def _dual_domain_coeff_maps(value: Any, *, base_dir: Path | None = None) -> tuple[dict[str, str], ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("dual_domain_coeff_maps must be a list")
    maps: list[dict[str, str]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("dual_domain_coeff_maps entries must be mappings")
        name = str(item.get("name", f"domain{idx}"))
        path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
        if path is None:
            raise ValueError("dual_domain_coeff_maps entries must include coeff_maps_npz, alternating_coeff_maps_npz, or path")
        maps.append({"name": name, "coeff_maps_npz": _resolve_path(str(path), base_dir=base_dir)})
    return tuple(maps)


def _resolve_path(value: str, *, base_dir: Path | None) -> str:
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str((base_dir / path).resolve())
