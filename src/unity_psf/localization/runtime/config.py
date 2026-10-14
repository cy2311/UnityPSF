"""Compatibility facade for localization runtime configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from unity_psf.localization.scale_resolver import PhysicalScaleSpec, resolve_physical_scales, resolve_target_contract
from .conditioning_config import (
    expert_type,
    is_single_channel_runtime,
)
from .contract_config import resolved_contract
from .loss_config import (
    legacy_gmm_loss_params,
    resolve_loss_config,
)
from .model_config import resolve_localization_model_config
from .modality_config import runtime_modality_contract
from .optimizer_config import (
    resolve_training_optimizer_config,
    training_runtime_contract,
)
from .paths import mapping
from .physical_config import pupil_carrier_complex as _pupil_carrier_complex
from .provider_config import (
    microtube_tiff_provider_config,
    online_provider_config as resolve_online_provider_config,
)
from .types import RuntimeConfig, RuntimeLossConfig as RuntimeLossConfig, RuntimeProviderConfig

from .conditioning_config import (
    append_domain_onehot_enabled as _append_domain_onehot,
    condition_dim as _condition_dim,
    condition_feature_dim as _condition_feature_dim,
    has_explicit_astigmatism_condition_contract as _has_explicit_astigmatism_condition_contract,
    is_astigmatism_expert as _is_astigmatism_expert,
    is_emitter_2d_expert as _is_emitter_2d_expert,
    single_channel_online_config as _single_channel_online_config,
    validate_condition_dimensions as _validate_condition_dimensions,
    astigmatism_condition_dim as _astigmatism_condition_dim,
    astigmatism_condition_fields as _astigmatism_condition_fields,
    emitter_2d_condition_fields as _emitter_2d_condition_fields,
)
from .loss_config import (
    active_smlm_loss_params as _active_smlm_loss_params,
    active_smlm_gmm_loss_params as _active_smlm_gmm_loss_params,
    ACTIVE_SMLM_LOSS_PARAM_KEYS as _ACTIVE_SMLM_LOSS_PARAM_KEYS,
    SCALE_METADATA_KEYS as _SCALE_METADATA_KEYS,
    LEGACY_GMM_LOSS_PARAM_KEYS as _LEGACY_GMM_LOSS_PARAM_KEYS,
    ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS as _ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS,
)
from .modality_config import (
    input_frame_spec as _input_frame_spec,
)
from .simulator_range_config import (
    online_simulator_range_params as _online_simulator_range_params,
    online_density_um2 as _online_density_um2,
)
from .paths import (
    grid_size as _grid_size,
    optional_mapping as _optional_mapping,
    pair_from_config as _pair_from_config,
    range_from_config as _range_from_config,
    resolve_path as _resolve_path,
)
from .physical_config import (
    dual_domain_coeff_maps as _dual_domain_coeff_maps,
    select_single_channel_coeff_maps as _select_single_channel_coeff_maps,
)
from .optimizer_config import (
    localization_overrides as _localization_overrides,
    normalize_z_activation as _normalize_z_activation,
)
from unity_psf.contracts.modality import ChannelLayout as ChannelLayout, ExpertInstanceSpec as ExpertInstanceSpec, InputFrameSpec as InputFrameSpec
from unity_psf.localization.smlm_targets import V03_PXYZ_TARGET_ORDER as V03_PXYZ_TARGET_ORDER


def online_provider_config(
    config: Mapping[str, Any], train_cfg: Mapping[str, Any], online_cfg: Mapping[str, Any],
    *, config_base_dir: str | Path | None, seed: int, single_channel: bool = False,
    model_params: Mapping[str, Any] | None = None, scales: PhysicalScaleSpec | None = None,
) -> RuntimeProviderConfig:
    return resolve_online_provider_config(
        config, train_cfg, online_cfg, config_base_dir=config_base_dir, seed=seed,
        single_channel=single_channel, model_params=model_params, scales=scales,
        pupil_carrier_resolver=_pupil_carrier_complex,
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
) -> RuntimeConfig:
    train_cfg = mapping(config.get("train"), "train")
    resolved_model_name, resolved_model_params = resolve_localization_model_config(config)
    if model_name is not None:
        resolved_model_name, resolved_model_params = str(model_name), {}
    resolved_model_params.update(dict(model_params or {}))
    scales = resolve_physical_scales(train_cfg)
    microtube_cfg = train_cfg.get("microtube_tiff")
    if isinstance(microtube_cfg, Mapping) and microtube_cfg.get("enabled") is True:
        batch_provider = microtube_tiff_provider_config(train_cfg, microtube_cfg, seed=seed)
    else:
        online_cfg = mapping(train_cfg.get("online_generation"), "train.online_generation")
        if expert_type(train_cfg) != "double_helix" and online_cfg.get("enabled", True) is not True:
            raise ValueError("train.online_generation.enabled must be true")
        batch_provider = online_provider_config(
            config, train_cfg, online_cfg, config_base_dir=config_base_dir, seed=seed,
            single_channel=expert_type(train_cfg) == "double_helix" or is_single_channel_runtime(train_cfg),
            model_params=resolved_model_params, scales=scales,
        )
    resolved_optimizer_name, resolved_optimizer_params = resolve_training_optimizer_config(
        config, train_cfg, optimizer_name=optimizer_name, optimizer_params=optimizer_params
    )
    loss = resolve_loss_config(config, train_cfg, model_name=resolved_model_name, scales=scales)
    target_contract = resolve_target_contract(train_cfg, target_order=loss["params"].get("target_order"), scales=scales)
    modality_contract = runtime_modality_contract(train_cfg, model_name=resolved_model_name)
    runtime_config: dict[str, object] = {
        "device": str(train_cfg.get("device", "cpu")),
        "model": {"name": resolved_model_name, "params": resolved_model_params},
        "optimizer": {"name": resolved_optimizer_name, "params": resolved_optimizer_params},
        "batch_provider": batch_provider, "loss": loss,
        "epochs": {"start": 1, "stop": int(train_cfg.get("epochs", 1))},
        "resolved_contract": resolved_contract(
            model_name=resolved_model_name, model_params=resolved_model_params,
            batch_provider=batch_provider, loss=loss,
            legacy_loss_params=legacy_gmm_loss_params(train_cfg.get("loss")),
            modality_contract=modality_contract, scales=scales, target_contract=target_contract,
            training_runtime=training_runtime_contract(config, train_cfg, optimizer_name=resolved_optimizer_name, optimizer_params=resolved_optimizer_params),
        ),
    }
    runtime_config.update(modality_contract)
    if train_cfg.get("max_batches") is not None:
        runtime_config["max_batches"] = int(train_cfg["max_batches"])
    feedback_cfg = train_cfg.get("feedback")
    if isinstance(feedback_cfg, Mapping) and "map_path" in feedback_cfg:
        runtime_config["feedback"] = {"map_path": str(feedback_cfg["map_path"])}
    return cast(RuntimeConfig, runtime_config)


__all__ = [
    "build_localization_runtime_config", "resolve_localization_model_config",
    '_append_domain_onehot', '_condition_dim', '_condition_feature_dim',
    '_has_explicit_astigmatism_condition_contract', '_is_astigmatism_expert', '_is_emitter_2d_expert',
    '_single_channel_online_config', '_validate_condition_dimensions', '_astigmatism_condition_dim',
    '_astigmatism_condition_fields', '_emitter_2d_condition_fields', '_active_smlm_loss_params',
    '_active_smlm_gmm_loss_params', '_ACTIVE_SMLM_LOSS_PARAM_KEYS', '_SCALE_METADATA_KEYS',
    '_LEGACY_GMM_LOSS_PARAM_KEYS', '_ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS', '_input_frame_spec',
    '_online_simulator_range_params', '_online_density_um2', '_grid_size',
    '_optional_mapping', '_pair_from_config', '_range_from_config',
    '_resolve_path', '_dual_domain_coeff_maps', '_select_single_channel_coeff_maps',
    '_localization_overrides', '_normalize_z_activation',
]

_expert_type = expert_type
_is_single_channel_runtime = is_single_channel_runtime
_resolved_contract = resolved_contract
_legacy_gmm_loss_params = legacy_gmm_loss_params
_loss_config = resolve_loss_config
_runtime_modality_contract = runtime_modality_contract
_resolve_training_optimizer_config = resolve_training_optimizer_config
_training_runtime_contract = training_runtime_contract
_mapping = mapping
_microtube_tiff_provider_config = microtube_tiff_provider_config
_provider_config = resolve_online_provider_config
_online_provider_config = online_provider_config
