"""Joint-training configuration parsing and instance binding."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from unity_psf.contracts import ChannelLayout, JointExpertKey


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_joint_config(
    path: Path,
    *,
    expected_execution: str | None = "round_robin",
) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _mapping(value, "joint config")
    if config.get("schema_version") != "unitypsf.joint_training.v1":
        raise ValueError("unsupported joint training schema")
    execution = str(config.get("execution", "round_robin"))
    if expected_execution is not None and execution != expected_execution:
        raise ValueError(f"this entrypoint requires execution: {expected_execution}")
    return config


def instance_specs(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = config.get("instances")
    if not isinstance(raw, list) or not raw:
        raise ValueError("joint config instances must be a non-empty list")
    specs: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        spec = _mapping(item, "instance")
        if "key" not in spec or "config" not in spec:
            raise ValueError("every joint instance requires key and config")
        key = JointExpertKey.parse(str(spec["key"])).storage_key
        if key in specs:
            raise ValueError(f"duplicate joint instance key {key!r}")
        if JointExpertKey.parse(key).modality.value not in {
            "emitter_2d",
            "astigmatism",
            "double_helix",
        }:
            raise ValueError(f"unsupported trainable modality in joint config: {key!r}")
        specs[key] = spec
    modalities = {JointExpertKey.parse(key).modality.value for key in specs}
    if not {"emitter_2d", "astigmatism"}.issubset(modalities):
        raise ValueError("joint config must include emitter_2d and astigmatism instances")
    return specs


def bind_instance(
    config: Mapping[str, Any],
    key: str,
    *,
    device: str | None,
) -> dict[str, Any]:
    bound = deepcopy(dict(config))
    train = dict(_mapping(bound.get("train"), "train"))
    parsed = JointExpertKey.parse(key)
    layout = ChannelLayout.from_value(train.get("channel_layout", {"channels": ["main"]}))
    matching = [channel for channel in layout.channels if channel.channel_id == parsed.channel_id]
    if matching:
        channel = matching[0]
    elif len(layout.channels) == 1:
        template = layout.channels[0]
        channel = type(template)(
            channel_id=parsed.channel_id,
            crop=template.crop,
            anchor_profile=template.anchor_profile,
            calibration_ref=template.calibration_ref,
        )
    else:
        raise ValueError(f"config has no measurement channel {parsed.channel_id!r}")
    channel_value = {
        "id": channel.channel_id,
        "crop": channel.crop,
        "anchor_profile": channel.anchor_profile,
        "calibration_ref": channel.calibration_ref,
    }
    train["channel_layout"] = {
        "channels": [{name: value for name, value in channel_value.items() if value is not None}],
        **({"frame_size": list(layout.frame_size)} if layout.frame_size is not None else {}),
    }
    train["expert"] = {
        **dict(_mapping(train.get("expert", {}), "train.expert")),
        "expert_type": parsed.modality.value,
        "instance_id": parsed.channel_id,
        "channel_id": parsed.channel_id,
    }
    if device is not None:
        train["device"] = device
    bound["train"] = train
    return bound


__all__ = ["bind_instance", "instance_specs", "load_joint_config"]
