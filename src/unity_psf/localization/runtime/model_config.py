"""Localization model configuration resolution."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, cast

from .conditioning_config import (
    condition_dim,
    condition_feature_dim,
    expert_type as resolve_expert_type,
    is_single_channel_runtime,
    single_channel_online_config,
    validate_condition_dimensions,
)
from .optimizer_config import (
    localization_overrides,
    normalize_z_activation,
)
from .paths import mapping
from .modality_config import input_frame_spec


def resolve_localization_model_config(config: Mapping[str, Any]) -> tuple[str, dict[str, object]]:
    train_cfg = mapping(config.get("train"), "train")
    microtube_cfg = train_cfg.get("microtube_tiff")
    if isinstance(microtube_cfg, Mapping) and microtube_cfg.get("enabled") is True:
        return "production_localizer", {"in_channels": int(microtube_cfg.get("channels", 3))}
    if resolve_expert_type(train_cfg) == "double_helix":
        model_cfg = train_cfg.get("model")
        raw_model_params = model_cfg.get("params", {}) if isinstance(model_cfg, Mapping) else {}
        dh_model_params: Mapping[str, object] = raw_model_params if isinstance(raw_model_params, Mapping) else {}
        online_cfg = train_cfg.get("online_generation")
        effective_online_cfg = (
            single_channel_online_config(online_cfg)
            if isinstance(online_cfg, Mapping) and is_single_channel_runtime(train_cfg)
            else online_cfg if isinstance(online_cfg, Mapping) else {}
        )
        input_normalization = str(
            dh_model_params.get("input_normalization", "identity")
        ).strip().lower()
        if input_normalization == "lunar":
            background_range = effective_online_cfg.get("background_range")
            photon_range = effective_online_cfg.get("photon_range")
            if not isinstance(background_range, (list, tuple)) or len(background_range) != 2:
                raise ValueError(
                    "LUNAR input normalization requires online_generation.background_range"
                )
            if not isinstance(photon_range, (list, tuple)) or len(photon_range) != 2:
                raise ValueError(
                    "LUNAR input normalization requires online_generation.photon_range"
                )
            normalization_config = {
                "name": "lunar",
                "background_range": [float(value) for value in background_range],
                "photon_range": [float(value) for value in photon_range],
            }
        elif input_normalization == "identity":
            normalization_config = None
        else:
            raise ValueError(
                "double_helix input_normalization must be 'lunar' or 'identity'"
            )
        defaults = {
            "nch_in": int(cast(int | str | float, dh_model_params.get("in_channels", 3))),
            "depth_shared": 1,
            "depth_union": 1,
            "nfeatures_init": 32,
            "nfeatures_inter": None,
            "norm_start_level": 0,
            "norm_groups": 0,
            "activation": "ELU",
            "dropout_start_level": None,
            "p_dropout": 0.1,
            "pool_mode": "StrideConv",
            "upsample_mode": "bilinear",
            "inter_activation": "ELU",
            "norm_head_groups": 0,
            "final_activation": "ELU",
            "kaiming_normal": True,
            "depthwise": True,
            "z_mu_activation": "tanh",
        }
        resolved = {
            key: dh_model_params.get(key, default)
            for key, default in defaults.items()
        }
        resolved["domain_count"] = int(effective_online_cfg.get("domain_count", 2))
        resolved["condition_dim"] = int(effective_online_cfg.get("condition_dim", dh_model_params.get("condition_dim", 0)))
        if str(effective_online_cfg.get("psf_type", "")).strip().lower() == "ppg_direct":
            ppg_fields = (
                "field_x",
                "field_y",
                "ppg_pca_0",
                "ppg_pca_1",
                "ppg_pca_2",
                "ppg_pca_3",
            )
            resolved.update(
                {
                    "condition_fields": list(ppg_fields),
                    "condition_dim": len(ppg_fields),
                    "film_hidden_dim": int(effective_online_cfg.get("film_hidden_dim", 32)),
                    "ppg_map_path": str(effective_online_cfg.get("ppg_map_path", "")),
                }
            )
        resolved["input_normalization"] = normalization_config
        return "double_helix_expert", resolved

    online_cfg = train_cfg.get("online_generation")
    if not isinstance(online_cfg, Mapping):
        return "production_localizer", {"in_channels": 3}

    expert_cfg = train_cfg.get("expert")
    expert_type = resolve_expert_type(train_cfg)
    if isinstance(expert_cfg, Mapping) and expert_type in {"astigmatism", "emitter_2d"}:
        input_spec = input_frame_spec(train_cfg, online_cfg)
        model_cfg = train_cfg.get("model")
        model_name = f"{expert_type}_expert"
        default_fields = (
            ["zernike_0", "zernike_1", "field_x", "field_y"]
            if expert_type == "astigmatism"
            else ["field_x", "field_y"]
        )
        expert_model_params: dict[str, object] = {
            "nch_in": input_spec.input_frame_channels,
            "condition_fields": default_fields,
            "film_hidden_dim": int(online_cfg.get("film_hidden_dim", 32)),
        }
        if isinstance(model_cfg, Mapping):
            raw_params = model_cfg.get("params", model_cfg)
            if isinstance(raw_params, Mapping):
                expert_model_params.update(dict(raw_params))
            if "name" in model_cfg and str(model_cfg["name"]) != model_name:
                raise ValueError(f"train.model.name must be {model_name!r} for a {expert_type} expert runtime")
        expert_model_params["nch_in"] = input_spec.input_frame_channels
        fields = expert_model_params.get("condition_fields", default_fields)
        if not isinstance(fields, (list, tuple)) or not fields:
            raise ValueError(f"{expert_type} expert condition_fields must be a non-empty sequence")
        if "condition_dim" not in expert_model_params:
            expert_model_params["condition_dim"] = len(fields)
        if int(cast(int | str | float, expert_model_params["condition_dim"])) != len(fields):
            raise ValueError(f"{expert_type} expert condition_dim must equal condition_fields length")
        if expert_type == "emitter_2d":
            disabled = expert_model_params.get("disabled_attr", [3])
            if isinstance(disabled, int):
                disabled_values: tuple[int, ...] = (disabled,)
            else:
                disabled_values = tuple(int(item) for item in cast(Iterable[Any], disabled))
            if 3 not in disabled_values:
                raise ValueError("emitter_2d expert must disable the z attribute")
            expert_model_params["disabled_attr"] = list(disabled_values)
        return model_name, expert_model_params

    channels = int(online_cfg.get("channels", 3))
    conditioning_mode = str(online_cfg.get("conditioning_mode", "channels"))
    expert_mode = str(online_cfg.get("expert_mode", ""))
    if conditioning_mode != "film" or expert_mode != "soft_moe":
        return "production_localizer", {"in_channels": channels}

    effective_online_cfg = (
        single_channel_online_config(online_cfg)
        if is_single_channel_runtime(train_cfg)
        else online_cfg
    )
    validate_condition_dimensions(effective_online_cfg, soft_moe=True)
    soft_moe_params: dict[str, object] = {
        "nch_in": channels,
        "condition_feature_dim": condition_feature_dim(effective_online_cfg),
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
    soft_moe_params["condition_dim"] = condition_dim(effective_online_cfg)
    overrides = localization_overrides(config, train_cfg)
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
            soft_moe_params[key] = normalize_z_activation(overrides[key]) if key == "z_mu_activation" else overrides[key]
    return "active_smlm_soft_moe_double_unet", soft_moe_params
