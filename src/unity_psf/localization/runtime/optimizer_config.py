"""Optimizer, scheduler, and training-runtime contract resolution."""

from __future__ import annotations

from typing import Any, Mapping


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def resolve_training_optimizer_config(
    root_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    *,
    optimizer_name: str | None,
    optimizer_params: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    explicit_params = dict(optimizer_params or {})
    if optimizer_name is not None:
        return str(optimizer_name), explicit_params

    configured_optimizer = train_cfg.get("optimizer")
    if isinstance(configured_optimizer, Mapping):
        if not configured_optimizer.get("name"):
            raise ValueError("train.optimizer.name is required")
        params = dict(_mapping(configured_optimizer.get("params", {}), "train.optimizer.params"))
        params.update(explicit_params)
        return str(configured_optimizer["name"]).lower(), params

    legacy_optimizer = legacy_optimizer_contract(root_cfg, train_cfg)
    if legacy_optimizer is not None:
        params = dict(_mapping(legacy_optimizer["params"], "legacy_optimizer.params"))
        params.update(explicit_params)
        return str(legacy_optimizer["name"]).lower(), params

    if not explicit_params and train_cfg.get("learning_rate") is not None:
        explicit_params["lr"] = float(train_cfg["learning_rate"])
    return "sgd", explicit_params


def legacy_optimizer_contract(root_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    overrides = root_cfg.get("smlm_overrides")
    if not isinstance(overrides, Mapping) or not overrides.get("optimizer"):
        return None
    online_cfg = train_cfg.get("online_generation")
    return {
        "name": str(overrides["optimizer"]),
        "params": {
            "lr": _legacy_optimizer_lr(overrides, train_cfg, online_cfg),
            "weight_decay": _legacy_optimizer_weight_decay(overrides, online_cfg),
        },
        "active": False,
        "inactive_reason": "optimizer_runtime_not_wired",
        "legacy_source": "smlm_overrides",
    }


def optimizer_matches_legacy(
    optimizer_name: str,
    optimizer_params: Mapping[str, Any],
    legacy_optimizer: Mapping[str, Any],
) -> bool:
    if str(optimizer_name).lower() != str(legacy_optimizer["name"]).lower():
        return False
    return dict(optimizer_params) == dict(_mapping(legacy_optimizer["params"], "legacy_optimizer.params"))


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


def scheduler_contract(root_cfg: Mapping[str, Any]) -> dict[str, Any]:
    overrides = root_cfg.get("smlm_overrides")
    if not isinstance(overrides, Mapping) or not overrides.get("lr_scheduler"):
        return {"name": "none", "active": False, "step_unit": None, "params": {}, "inactive_reason": "not_configured"}
    name = str(overrides["lr_scheduler"])
    params: dict[str, Any] = {}
    if overrides.get("lr_step_size") is not None:
        params["step_size"] = int(overrides["lr_step_size"])
    if overrides.get("lr_gamma") is not None:
        params["gamma"] = float(overrides["lr_gamma"])
    step_unit = scheduler_step_unit(overrides.get("lr_step_unit"))
    active = name.lower() == "steplr" and step_unit in {"optimizer_step", "epoch"}
    return {
        "name": name,
        "active": active,
        "step_unit": step_unit,
        "params": params,
        "inactive_reason": None if active else "scheduler_runtime_not_wired",
        "legacy_source": "smlm_overrides",
    }


def scheduler_step_unit(value: Any) -> str:
    unit = str(value or "epoch").strip().lower()
    return "optimizer_step" if unit in {"optimizer_step", "step", "batch", "iteration", "iter"} else "epoch"


def localization_overrides(root_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if isinstance(root_cfg.get("localization_overrides"), Mapping):
        overrides.update(root_cfg["localization_overrides"])
    if isinstance(train_cfg.get("localization_overrides"), Mapping):
        overrides.update(train_cfg["localization_overrides"])
    return overrides


def normalize_z_activation(value: Any) -> str:
    z_activation = str(value).strip().lower()
    if z_activation not in {"tanh", "sigmoid"}:
        raise ValueError("z_mu_activation must be 'tanh' or 'sigmoid'")
    return z_activation


def training_runtime_contract(
    root_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    *,
    optimizer_name: str,
    optimizer_params: Mapping[str, Any],
) -> dict[str, Any]:
    online_cfg = train_cfg.get("online_generation")
    grad_clip_norm = float(online_cfg["grad_clip_norm"]) if isinstance(online_cfg, Mapping) and "grad_clip_norm" in online_cfg else None
    amp_enabled = bool(online_cfg.get("amp_enabled", False)) if isinstance(online_cfg, Mapping) else False
    amp_dtype = str(online_cfg["amp_dtype"]) if isinstance(online_cfg, Mapping) and "amp_dtype" in online_cfg else None
    legacy_optimizer = legacy_optimizer_contract(root_cfg, train_cfg)
    if legacy_optimizer is not None and optimizer_matches_legacy(optimizer_name, optimizer_params, legacy_optimizer):
        legacy_optimizer = {**legacy_optimizer, "active": True, "inactive_reason": None}
    scheduler = scheduler_contract(root_cfg)
    contract = {
        "optimizer": {"name": str(optimizer_name), "params": dict(optimizer_params)},
        "legacy_optimizer": legacy_optimizer,
        "scheduler": scheduler,
        "grad_clip": {"configured_norm": grad_clip_norm, "active": grad_clip_norm is not None},
        "amp": {"configured": amp_enabled, "dtype": amp_dtype, "active": amp_enabled, "inactive_reason": None if amp_enabled else "not_configured"},
    }
    if train_cfg.get("max_batches") is not None:
        contract["max_batches"] = int(train_cfg["max_batches"])
    elif legacy_optimizer is not None or scheduler.get("legacy_source") is not None:
        contract["max_batches"] = None
    return contract
