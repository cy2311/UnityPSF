from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .optimizer_config import (
    resolve_training_optimizer_config as _resolve_training_optimizer_config,
    training_runtime_contract as _training_runtime_contract,
)
from .conditioning_config import (
    expert_type as _expert_type,
    is_single_channel_runtime as _is_single_channel_runtime,
)
from .paths import (
    mapping as _mapping,
)
from .contracts import _resolved_contract, _runtime_modality_contract
from .loss_config import _legacy_gmm_loss_params, _loss_config
from .model_config import _resolve_localization_model_config
from .provider_config import _microtube_tiff_provider_config, _online_provider_config


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
    _validate_runtime_schema(config)
    train_cfg = _mapping(config.get("train"), "train")
    resolved_model_name, resolved_model_params = _resolve_localization_model_config(config)
    if model_name is not None:
        resolved_model_name = str(model_name)
        resolved_model_params = {}
    resolved_model_params.update(dict(model_params or {}))
    microtube_cfg = train_cfg.get("microtube_tiff")
    if isinstance(microtube_cfg, Mapping) and microtube_cfg.get("enabled") is True:
        batch_provider = _microtube_tiff_provider_config(train_cfg, microtube_cfg, seed=seed)
    elif _expert_type(train_cfg) == "double_helix":
        online_cfg = _mapping(train_cfg.get("online_generation"), "train.online_generation")
        batch_provider = _online_provider_config(config, train_cfg, online_cfg, config_base_dir=config_base_dir, seed=seed, single_channel=True, model_params=resolved_model_params)
        batch_provider["name"] = "dh_online_direct_xyz_batch"
    elif isinstance(train_cfg.get("dh_raw_tiff"), Mapping) and train_cfg["dh_raw_tiff"].get("enabled") is True:
        dh_cfg = train_cfg["dh_raw_tiff"]
        batch_provider = {"name": "dh_raw_tiff_train_batch", "params": {**dict(dh_cfg), "enabled": None}}
        batch_provider["params"].pop("enabled", None)
    else:
        online_cfg = _mapping(train_cfg.get("online_generation"), "train.online_generation")
        if online_cfg.get("enabled", True) is not True:
            raise ValueError("train.online_generation.enabled must be true")
        batch_provider = _online_provider_config(
            config,
            train_cfg,
            online_cfg,
            config_base_dir=config_base_dir,
            seed=seed,
            single_channel=_is_single_channel_runtime(train_cfg),
            model_params=resolved_model_params,
        )
    resolved_optimizer_name, resolved_optimizer_params = _resolve_training_optimizer_config(
        config,
        train_cfg,
        optimizer_name=optimizer_name,
        optimizer_params=optimizer_params,
    )

    loss = _loss_config(config, train_cfg, model_name=resolved_model_name)
    modality_contract = _runtime_modality_contract(
        train_cfg,
        model_name=resolved_model_name,
    )
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
            modality_contract=modality_contract,
            training_runtime=_training_runtime_contract(
                config,
                train_cfg,
                optimizer_name=resolved_optimizer_name,
                optimizer_params=resolved_optimizer_params,
            ),
        ),
    }
    runtime_config.update(modality_contract)
    if train_cfg.get("max_batches") is not None:
        runtime_config["max_batches"] = int(train_cfg["max_batches"])
    feedback_cfg = train_cfg.get("feedback")
    if isinstance(feedback_cfg, Mapping) and "map_path" in feedback_cfg:
        runtime_config["feedback"] = {"map_path": str(feedback_cfg["map_path"])}
    return runtime_config


def _validate_runtime_schema(config: Mapping[str, Any]) -> None:
    schema_version = config.get("schema_version")
    if schema_version is None:
        return
    if schema_version not in {"0.4", "unitypsf.instance_training.v1"}:
        raise ValueError(f"unsupported localization runtime schema: {schema_version!r}")
