"""Runtime modality and input-frame contract resolution."""

from __future__ import annotations

from typing import Any, Mapping

from unity_psf.contracts.modality import ChannelLayout, ExpertInstanceSpec, InputFrameSpec
from .paths import mapping


def input_frame_spec(train_cfg: Mapping[str, Any], online_cfg: Mapping[str, Any]) -> InputFrameSpec:
    raw_spec = train_cfg.get("input_frame_spec")
    if raw_spec is None:
        raw_spec = {"input_frame_channels": int(online_cfg.get("channels", 3)), "frame_order": "temporal"}
    spec = InputFrameSpec.from_value(raw_spec)
    if "channels" in online_cfg and int(online_cfg["channels"]) != spec.input_frame_channels:
        raise ValueError("train.online_generation.channels must match train.input_frame_spec.input_frame_channels")
    return spec




def runtime_modality_contract(
    train_cfg: Mapping[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    expert_cfg = train_cfg.get("expert")
    if not isinstance(expert_cfg, Mapping):
        return {}
    if str(model_name) == "double_helix_expert":
        layout = ChannelLayout.from_value(train_cfg.get("channel_layout", {"channels": ["main"]}))
        instance = ExpertInstanceSpec.from_value(
            {
                "expert_type": "double_helix",
                "instance_id": expert_cfg.get("instance_id", expert_cfg.get("channel_id", "main")),
                "channel_id": expert_cfg.get("channel_id", expert_cfg.get("instance_id", "main")),
                "prototype_ref": expert_cfg.get("prototype_ref", expert_cfg.get("prototype_checkpoint")),
            }
        )
        if instance.channel_id not in layout.channel_ids:
            raise ValueError(f"expert instance channel_id {instance.channel_id!r} is absent from channel_layout")
        return {"input_frame_spec": {"input_frame_channels": 3, "frame_order": "temporal"}, "channel_layout": {"channels": [{"channel_id": c.channel_id, "crop": None if c.crop is None else list(c.crop), "anchor_profile": c.anchor_profile, "calibration_ref": c.calibration_ref} for c in layout.channels], "frame_size": list(layout.frame_size) if layout.frame_size is not None else [96, 96]}, "expert_instance": {"expert_type": instance.expert_type.value, "instance_id": instance.instance_id, "channel_id": instance.channel_id, "prototype_ref": instance.prototype_ref}}
    online_cfg = mapping(train_cfg.get("online_generation"), "train.online_generation")
    frame_spec = input_frame_spec(train_cfg, online_cfg)
    raw_layout = train_cfg.get("channel_layout", {"channels": ["main"]})
    layout = ChannelLayout.from_value(raw_layout)
    if layout.input_instances != 1:
        if str(model_name) in {"astigmatism_expert", "emitter_2d_expert"}:
            raise ValueError(f"{model_name} runtime requires exactly one measurement channel")
        return {}
    expert_type = str(expert_cfg.get("expert_type", expert_cfg.get("name", ""))).strip().lower()
    if not expert_type:
        return {}
    if expert_type == "astigmatism_expert":
        expert_type = "astigmatism"
    if expert_type in {"emitter_2d_expert", "2d", "emitter", "emitter2d"}:
        expert_type = "emitter_2d"
    instance = ExpertInstanceSpec.from_value(
        {
            "expert_type": expert_type,
            "instance_id": expert_cfg.get("instance_id", "main"),
            "channel_id": expert_cfg.get("channel_id", "main"),
            "prototype_ref": expert_cfg.get("prototype_ref", expert_cfg.get("prototype_checkpoint")),
        }
    )
    if instance.channel_id not in layout.channel_ids:
        raise ValueError(f"expert instance channel_id {instance.channel_id!r} is absent from channel_layout")
    channels = []
    for channel in layout.channels:
        channels.append(
            {
                "channel_id": channel.channel_id,
                "crop": None if channel.crop is None else list(channel.crop),
                "anchor_profile": channel.anchor_profile,
                "calibration_ref": channel.calibration_ref,
            }
        )
    instance_value = {
        "expert_type": instance.expert_type.value,
        "instance_id": instance.instance_id,
        "channel_id": instance.channel_id,
        "prototype_ref": instance.prototype_ref,
    }
    layout_value: dict[str, Any] = {"channels": channels}
    if layout.frame_size is not None:
        layout_value["frame_size"] = list(layout.frame_size)
    return {
        "input_frame_spec": {
            "input_frame_channels": frame_spec.input_frame_channels,
            "frame_order": list(frame_spec.frame_order) if isinstance(frame_spec.frame_order, tuple) else frame_spec.frame_order,
        },
        "channel_layout": layout_value,
        "expert_instance": instance_value,
    }
