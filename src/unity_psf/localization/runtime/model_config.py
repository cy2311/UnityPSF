from __future__ import annotations

from typing import Any, Mapping

from unity_psf.contracts.modality import InputFrameSpec
from .conditioning_config import condition_dim as _condition_dim, condition_feature_dim as _condition_feature_dim, expert_type as _expert_type, is_single_channel_runtime as _is_single_channel_runtime, single_channel_online_config as _single_channel_online_config, validate_condition_dimensions as _validate_condition_dimensions
from .contracts import _input_frame_spec
from .optimizer_config import localization_overrides as _localization_overrides, normalize_z_activation as _normalize_z_activation
from .paths import mapping as _mapping


def _resolve_localization_model_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    train_cfg = _mapping(config.get("train"), "train")
    microtube_cfg = train_cfg.get("microtube_tiff")
    if isinstance(microtube_cfg, Mapping) and microtube_cfg.get("enabled") is True:
        return "production_localizer", {"in_channels": int(microtube_cfg.get("channels", 3))}
    dh_cfg = train_cfg.get("dh_raw_tiff")
    if _expert_type(train_cfg) == "double_helix":
        model_cfg = train_cfg.get("model")
        raw_model_params = model_cfg.get("params", {}) if isinstance(model_cfg, Mapping) else {}
        model_params = raw_model_params if isinstance(raw_model_params, Mapping) else {}
        online_cfg = train_cfg.get("online_generation")
        effective_online_cfg = (
            _single_channel_online_config(online_cfg)
            if isinstance(online_cfg, Mapping) and _is_single_channel_runtime(train_cfg)
            else online_cfg if isinstance(online_cfg, Mapping) else {}
        )
        return "double_helix_expert", {
            "in_channels": int(model_params.get("in_channels", 3)),
            "feature_channels": int(model_params.get("feature_channels", 32)),
            "condition_dim": _condition_dim(effective_online_cfg),
            "domain_count": int(effective_online_cfg.get("domain_count", 1)),
        }

    online_cfg = train_cfg.get("online_generation")
    if not isinstance(online_cfg, Mapping):
        return "production_localizer", {"in_channels": 3}

    expert_cfg = train_cfg.get("expert")
    expert_type = _expert_type(train_cfg)
    if isinstance(expert_cfg, Mapping) and expert_type in {"astigmatism", "emitter_2d"}:
        input_spec = _input_frame_spec(train_cfg, online_cfg)
        model_cfg = train_cfg.get("model")
        model_name = f"{expert_type}_expert"
        default_fields = (
            ["zernike_0", "zernike_1", "field_x", "field_y"]
            if expert_type == "astigmatism"
            else ["field_x", "field_y"]
        )
        model_params: dict[str, Any] = {
            "nch_in": input_spec.input_frame_channels,
            "condition_fields": default_fields,
            "film_hidden_dim": int(online_cfg.get("film_hidden_dim", 32)),
        }
        if isinstance(model_cfg, Mapping):
            raw_params = model_cfg.get("params", model_cfg)
            if isinstance(raw_params, Mapping):
                model_params.update(dict(raw_params))
            if "name" in model_cfg and str(model_cfg["name"]) != model_name:
                raise ValueError(f"train.model.name must be {model_name!r} for a {expert_type} expert runtime")
        model_params["nch_in"] = input_spec.input_frame_channels
        fields = model_params.get("condition_fields", default_fields)
        if not isinstance(fields, (list, tuple)) or not fields:
            raise ValueError(f"{expert_type} expert condition_fields must be a non-empty sequence")
        if "condition_dim" not in model_params:
            model_params["condition_dim"] = len(fields)
        if int(model_params["condition_dim"]) != len(fields):
            raise ValueError(f"{expert_type} expert condition_dim must equal condition_fields length")
        if expert_type == "emitter_2d":
            disabled = model_params.get("disabled_attr", [3])
            disabled_values = (int(disabled),) if isinstance(disabled, int) else tuple(int(item) for item in disabled)
            if 3 not in disabled_values:
                raise ValueError("emitter_2d expert must disable the z attribute")
            model_params["disabled_attr"] = list(disabled_values)
        return model_name, model_params

    channels = int(online_cfg.get("channels", 3))
    conditioning_mode = str(online_cfg.get("conditioning_mode", "channels"))
    expert_mode = str(online_cfg.get("expert_mode", ""))
    if conditioning_mode != "film" or expert_mode != "soft_moe":
        return "production_localizer", {"in_channels": channels}

    effective_online_cfg = (
        _single_channel_online_config(online_cfg)
        if _is_single_channel_runtime(train_cfg)
        else online_cfg
    )
    _validate_condition_dimensions(effective_online_cfg, soft_moe=True)
    model_params: dict[str, Any] = {
        "nch_in": channels,
        "condition_feature_dim": _condition_feature_dim(effective_online_cfg),
        "domain_count": int(effective_online_cfg.get("domain_count", 2)),
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
    model_params["condition_dim"] = _condition_dim(effective_online_cfg)
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
