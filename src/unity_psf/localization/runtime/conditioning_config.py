"""Conditioning and single-channel runtime contract normalization."""

from __future__ import annotations

from typing import Any, Mapping


def is_single_channel_runtime(train_cfg: Mapping[str, Any]) -> bool:
    layout_cfg = train_cfg.get("channel_layout")
    if not isinstance(layout_cfg, Mapping):
        return False
    channels = layout_cfg.get("channels", layout_cfg.get("measurement_channels"))
    if not isinstance(channels, (list, tuple)) or len(channels) != 1:
        return False
    channel = channels[0]
    channel_id = channel.get("id", channel.get("channel_id")) if isinstance(channel, Mapping) else channel
    expert_cfg = train_cfg.get("expert")
    if not isinstance(expert_cfg, Mapping):
        return False
    expert_channel = expert_cfg.get("channel_id", expert_cfg.get("instance_id"))
    return expert_channel is not None and str(expert_channel) == str(channel_id)


def single_channel_online_config(online_cfg: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(online_cfg)
    soft_moe = is_soft_moe(online_cfg)
    feature_dim = condition_feature_dim(online_cfg)
    append_domain_onehot = append_domain_onehot_enabled(online_cfg, soft_moe=soft_moe)
    resolved.update(
        {
            "domain_count": 1,
            "condition_feature_dim": feature_dim,
            "append_domain_onehot": append_domain_onehot,
            "condition_dim": feature_dim + (1 if append_domain_onehot else 0),
        }
    )
    return resolved


def is_soft_moe(online_cfg: Mapping[str, Any]) -> bool:
    return str(online_cfg.get("conditioning_mode", "channels")) == "film" and str(online_cfg.get("expert_mode", "")) == "soft_moe"


def expert_type(train_cfg: Mapping[str, Any]) -> str:
    expert_cfg = train_cfg.get("expert")
    if not isinstance(expert_cfg, Mapping):
        return ""
    value = str(expert_cfg.get("expert_type", expert_cfg.get("name", ""))).strip().lower().replace("-", "_")
    aliases = {
        "2d": "emitter_2d",
        "emitter": "emitter_2d",
        "emitter2d": "emitter_2d",
        "emitter_2d_expert": "emitter_2d",
        "astig": "astigmatism",
        "astigmatism_expert": "astigmatism",
    }
    return aliases.get(value, value)


def is_astigmatism_expert(train_cfg: Mapping[str, Any]) -> bool:
    return expert_type(train_cfg) == "astigmatism"


def is_emitter_2d_expert(train_cfg: Mapping[str, Any]) -> bool:
    return expert_type(train_cfg) == "emitter_2d"


def has_explicit_astigmatism_condition_contract(train_cfg: Mapping[str, Any]) -> bool:
    model_cfg = train_cfg.get("model")
    if not isinstance(model_cfg, Mapping):
        return False
    raw_params = model_cfg.get("params", model_cfg)
    return isinstance(raw_params, Mapping) and ("condition_fields" in raw_params or "condition_dim" in raw_params)


def condition_feature_dim(online_cfg: Mapping[str, Any]) -> int:
    if "condition_feature_dim" in online_cfg:
        return int(online_cfg["condition_feature_dim"])
    if "condition_dim" in online_cfg:
        domain_terms = int(online_cfg.get("domain_count", 2)) if bool(online_cfg.get("append_domain_onehot", False)) else 0
        return max(0, int(online_cfg["condition_dim"]) - domain_terms)
    return 8


def condition_dim(online_cfg: Mapping[str, Any]) -> int:
    if "condition_dim" in online_cfg:
        return int(online_cfg["condition_dim"])
    domain_terms = int(online_cfg.get("domain_count", 2)) if append_domain_onehot_enabled(online_cfg, soft_moe=is_soft_moe(online_cfg)) else 0
    return condition_feature_dim(online_cfg) + domain_terms


def append_domain_onehot_enabled(online_cfg: Mapping[str, Any], *, soft_moe: bool) -> bool:
    return bool(online_cfg["append_domain_onehot"]) if "append_domain_onehot" in online_cfg else bool(soft_moe)


def validate_condition_dimensions(online_cfg: Mapping[str, Any], *, soft_moe: bool) -> None:
    if not soft_moe or not append_domain_onehot_enabled(online_cfg, soft_moe=soft_moe):
        return
    if "condition_feature_dim" not in online_cfg or "condition_dim" not in online_cfg:
        return
    expected = int(online_cfg["condition_feature_dim"]) + int(online_cfg.get("domain_count", 2))
    if int(online_cfg["condition_dim"]) != expected:
        raise ValueError("condition_dim must equal condition_feature_dim + domain_count for soft_moe film conditioning")
