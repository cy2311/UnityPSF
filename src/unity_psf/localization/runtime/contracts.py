from __future__ import annotations

from typing import Any, Mapping

from unity_psf.contracts.modality import ChannelLayout, ExpertInstanceSpec, InputFrameSpec
from unity_psf.localization.smlm_targets import V03_PXYZ_TARGET_ORDER
from .paths import mapping as _mapping


def _resolved_contract(
    *,
    model_name: str,
    model_params: Mapping[str, Any],
    batch_provider: Mapping[str, Any],
    loss: Mapping[str, Any],
    legacy_loss_params: Mapping[str, Any],
    modality_contract: Mapping[str, Any],
    training_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    provider_params = _mapping(batch_provider.get("params"), "batch_provider.params")
    loss_params = dict(_mapping(loss.get("params", {}), "loss.params"))
    contract: dict[str, Any] = {
        "model": {
            "name": str(model_name),
            "output": "smlm_10ch"
            if str(model_name).startswith("active_smlm_") or str(model_name) in {"astigmatism_expert", "emitter_2d_expert"}
            else "localization",
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
        "condition_fields",
        "domain_count",
        "append_domain_onehot",
    ):
        if key in provider_params:
            contract["batch_provider"][key] = provider_params[key]
    if provider_params.get("dual_domain_coeff_maps"):
        contract["batch_provider"]["dual_domain_coeff_maps"] = provider_params["dual_domain_coeff_maps"]
    contract.update(dict(modality_contract))
    return contract


def _input_frame_spec(train_cfg: Mapping[str, Any], online_cfg: Mapping[str, Any]) -> InputFrameSpec:
    raw_spec = train_cfg.get("input_frame_spec")
    if raw_spec is None:
        raw_spec = {"input_frame_channels": int(online_cfg.get("channels", 3)), "frame_order": "temporal"}
    spec = InputFrameSpec.from_value(raw_spec)  # type: ignore[arg-type]
    if "channels" in online_cfg and int(online_cfg["channels"]) != spec.input_frame_channels:
        raise ValueError("train.online_generation.channels must match train.input_frame_spec.input_frame_channels")
    return spec


def _astigmatism_condition_dim(
    train_cfg: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    model_params: Mapping[str, Any] | None = None,
) -> int:
    if model_params is not None and model_params.get("condition_dim") is not None:
        return int(model_params["condition_dim"])
    return len(_astigmatism_condition_fields(train_cfg, online_cfg, model_params=model_params))


def _astigmatism_condition_fields(
    train_cfg: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    model_params: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    if model_params is not None and model_params.get("condition_fields") is not None:
        fields = model_params["condition_fields"]
        if not isinstance(fields, (list, tuple)) or not fields:
            raise ValueError("astigmatism expert condition_fields must be a non-empty sequence")
        return tuple(str(field) for field in fields)
    model_cfg = train_cfg.get("model")
    if isinstance(model_cfg, Mapping):
        raw_params = model_cfg.get("params", model_cfg)
        if isinstance(raw_params, Mapping) and raw_params.get("condition_fields") is not None:
            fields = raw_params["condition_fields"]
            if not isinstance(fields, (list, tuple)) or not fields:
                raise ValueError("astigmatism expert condition_fields must be a non-empty sequence")
            return tuple(str(field) for field in fields)
    if "condition_fields" in online_cfg:
        fields = online_cfg["condition_fields"]
        if not isinstance(fields, (list, tuple)) or not fields:
            raise ValueError("astigmatism online condition_fields must be a non-empty sequence")
        return tuple(str(field) for field in fields)
    return ("zernike_0", "zernike_1", "field_x", "field_y")


def _emitter_2d_condition_fields(
    train_cfg: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    model_params: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    if model_params is not None and model_params.get("condition_fields") is not None:
        fields = model_params["condition_fields"]
    else:
        model_cfg = train_cfg.get("model")
        raw_params = model_cfg.get("params", model_cfg) if isinstance(model_cfg, Mapping) else {}
        fields = raw_params.get("condition_fields", online_cfg.get("condition_fields", ("field_x", "field_y"))) if isinstance(raw_params, Mapping) else ("field_x", "field_y")
    if not isinstance(fields, (list, tuple)) or not fields:
        raise ValueError("emitter_2d expert condition_fields must be a non-empty sequence")
    return tuple(str(field) for field in fields)


def _runtime_modality_contract(
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
    online_cfg = _mapping(train_cfg.get("online_generation"), "train.online_generation")
    frame_spec = _input_frame_spec(train_cfg, online_cfg)
    raw_layout = train_cfg.get("channel_layout", {"channels": ["main"]})
    layout = ChannelLayout.from_value(raw_layout)  # type: ignore[arg-type]
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
