"""Checkpoint metadata schema and serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from .modality import InputFrameSpec, MeasurementChannelSpec, PSFModality


CHECKPOINT_SCHEMA_VERSION = "unity_psf.checkpoint.v2"
LEGACY_CHECKPOINT_SCHEMA_VERSION = "unity_psf.checkpoint.v1"
LEGACY_TRAINING_SCHEMA_VERSION = "unity_psf.checkpoint.legacy.training"
LEGACY_STATE_DICT_SCHEMA_VERSION = "unity_psf.checkpoint.legacy.state_dict"
DEFAULT_EXPERTS = tuple(item.value for item in PSFModality)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return normalized


def _serialize_frame_spec(value: InputFrameSpec) -> dict[str, Any]:
    order = list(value.frame_order) if isinstance(value.frame_order, tuple) else value.frame_order
    return {"input_frame_channels": value.input_frame_channels, "frame_order": order}


def _serialize_channel_spec(value: MeasurementChannelSpec | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "channel_id": value.channel_id,
        "crop": list(value.crop) if value.crop is not None else None,
        "anchor_profile": value.anchor_profile,
        "calibration_ref": value.calibration_ref,
    }


@dataclass(frozen=True)
class CheckpointMetadata:
    """Portable metadata for a prototype or channel-bound checkpoint."""

    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    model_family: str = "UnityPSF"
    model_name: str = "psf_moe"
    checkpoint_role: str = "prototype"
    expert_type: PSFModality | str | None = None
    model_config: Mapping[str, Any] = field(default_factory=dict)
    input_frame_spec: InputFrameSpec = field(default_factory=InputFrameSpec)
    instance_id: str | None = None
    channel_spec: MeasurementChannelSpec | None = None
    condition_schema: Mapping[str, Any] = field(default_factory=dict)
    parent_checkpoint_hash: str | None = None
    code_version: str = "0.3.0"
    experts: tuple[str, ...] = DEFAULT_EXPERTS
    shared_feature_channels: int = 32
    router_mode: str = "hard"

    def __post_init__(self) -> None:
        schema = str(self.schema_version)
        allowed = {
            CHECKPOINT_SCHEMA_VERSION,
            LEGACY_CHECKPOINT_SCHEMA_VERSION,
            LEGACY_TRAINING_SCHEMA_VERSION,
            LEGACY_STATE_DICT_SCHEMA_VERSION,
        }
        if schema not in allowed:
            raise ValueError(f"unsupported checkpoint schema {schema!r}")
        object.__setattr__(self, "schema_version", schema)
        if schema in {CHECKPOINT_SCHEMA_VERSION, LEGACY_CHECKPOINT_SCHEMA_VERSION}:
            if self.model_family != "UnityPSF":
                raise ValueError(f"unsupported model family {self.model_family!r}")
            parsed = tuple(PSFModality.parse(item).value for item in self.experts)
            if len(set(parsed)) != len(parsed):
                raise ValueError("checkpoint experts must be unique")
            object.__setattr__(self, "experts", parsed)
            if int(self.shared_feature_channels) <= 0:
                raise ValueError("shared_feature_channels must be positive")
            if self.router_mode not in {"hard", "soft"}:
                raise ValueError("router_mode must be 'hard' or 'soft'")
            object.__setattr__(self, "shared_feature_channels", int(self.shared_feature_channels))
        if not isinstance(self.model_config, Mapping):
            raise TypeError("model_config must be a mapping")
        if not isinstance(self.condition_schema, Mapping):
            raise TypeError("condition_schema must be a mapping")
        if not isinstance(self.input_frame_spec, InputFrameSpec):
            object.__setattr__(self, "input_frame_spec", InputFrameSpec.from_value(self.input_frame_spec))
        if self.expert_type is not None:
            object.__setattr__(self, "expert_type", PSFModality.parse(self.expert_type))
        if self.channel_spec is not None and not isinstance(self.channel_spec, MeasurementChannelSpec):
            object.__setattr__(self, "channel_spec", MeasurementChannelSpec.from_value(self.channel_spec))
        if self.parent_checkpoint_hash is not None:
            object.__setattr__(
                self,
                "parent_checkpoint_hash",
                _validate_sha256(self.parent_checkpoint_hash, field_name="parent_checkpoint_hash"),
            )
        if schema != CHECKPOINT_SCHEMA_VERSION:
            return
        if self.checkpoint_role not in {"prototype", "instance"}:
            raise ValueError("checkpoint_role must be 'prototype' or 'instance'")
        if self.checkpoint_role == "instance":
            if not self.instance_id or not str(self.instance_id).strip():
                raise ValueError("instance checkpoint requires instance_id")
            if self.channel_spec is None:
                raise ValueError("instance checkpoint requires channel_spec")
            if self.parent_checkpoint_hash is None:
                raise ValueError("instance checkpoint requires parent_checkpoint_hash")
        if self.instance_id is not None and not str(self.instance_id).strip():
            raise ValueError("instance_id must be non-empty when provided")
        if self.parent_checkpoint_hash is not None and self.checkpoint_role == "prototype":
            raise ValueError("prototype checkpoint cannot carry parent_checkpoint_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "checkpoint_role": self.checkpoint_role,
            "expert_type": self.expert_type.value if isinstance(self.expert_type, PSFModality) else self.expert_type,
            "model_config": dict(self.model_config),
            "input_frame_spec": _serialize_frame_spec(self.input_frame_spec),
            "instance_id": self.instance_id,
            "channel_spec": _serialize_channel_spec(self.channel_spec),
            "condition_schema": dict(self.condition_schema),
            "parent_checkpoint_hash": self.parent_checkpoint_hash,
            "code_version": self.code_version,
            "experts": list(self.experts),
            "shared_feature_channels": self.shared_feature_channels,
            "router_mode": self.router_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointMetadata":
        schema = str(value.get("schema_version", ""))
        if schema == LEGACY_CHECKPOINT_SCHEMA_VERSION:
            required = {"schema_version", "model_family", "model_name", "experts"}
            missing = sorted(required.difference(value))
            if missing:
                raise ValueError(f"checkpoint metadata is missing fields: {', '.join(missing)}")
            return cls(
                schema_version=schema,
                model_family=str(value["model_family"]),
                model_name=str(value["model_name"]),
                experts=tuple(str(item) for item in value["experts"]),
                shared_feature_channels=int(value.get("shared_feature_channels", 32)),
                router_mode=str(value.get("router_mode", "hard")),
            )
        if schema != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint metadata schema {schema!r}")
        channel_value = value.get("channel_spec")
        return cls(
            schema_version=schema,
            model_family=str(value.get("model_family", "UnityPSF")),
            model_name=str(value.get("model_name", "psf_moe")),
            checkpoint_role=str(value.get("checkpoint_role", "prototype")),
            expert_type=value.get("expert_type"),
            model_config=value.get("model_config", {}),
            input_frame_spec=InputFrameSpec.from_value(value.get("input_frame_spec", {})),
            instance_id=value.get("instance_id"),
            channel_spec=None if channel_value is None else MeasurementChannelSpec.from_value(channel_value),
            condition_schema=value.get("condition_schema", {}),
            parent_checkpoint_hash=value.get("parent_checkpoint_hash"),
            code_version=str(value.get("code_version", "0.3.0")),
            experts=tuple(str(item) for item in value.get("experts", DEFAULT_EXPERTS)),
            shared_feature_channels=int(value.get("shared_feature_channels", 32)),
            router_mode=str(value.get("router_mode", "hard")),
        )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointMetadata",
    "DEFAULT_EXPERTS",
    "LEGACY_CHECKPOINT_SCHEMA_VERSION",
    "LEGACY_STATE_DICT_SCHEMA_VERSION",
    "LEGACY_TRAINING_SCHEMA_VERSION",
]
