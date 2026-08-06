"""Modality-routed joint checkpoint contract for UnityPSF."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import torch

from .joint_checkpoint import load_joint_checkpoint as _load_legacy_joint_checkpoint
from .joint_checkpoint import logical_sha256
from .modality import InputFrameSpec, MeasurementChannelSpec, PSFModality


MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION = "unity_psf.joint_checkpoint.v2"
LEGACY_JOINT_CHECKPOINT_SCHEMA_VERSION = "unity_psf.joint_checkpoint.v1"
_ROLES = frozenset({"release", "resume"})
_ROUTER = {"type": "deterministic", "mode": "hard_top1", "key": "modality"}


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if ":" in normalized:
        raise ValueError(f"{field_name} cannot contain ':'")
    return normalized


def _frame_spec_dict(value: InputFrameSpec) -> dict[str, Any]:
    order = value.frame_order
    return {
        "input_frame_channels": value.input_frame_channels,
        "frame_order": list(order) if isinstance(order, tuple) else order,
    }


@dataclass(frozen=True)
class ModalityJointCheckpointMetadata:
    """Identity and capability declaration for a modality-routed checkpoint."""

    checkpoint_role: str = "release"
    supported_modalities: tuple[PSFModality | str, ...] = (
        PSFModality.EMITTER_2D,
        PSFModality.ASTIGMATISM,
    )
    supported_channels_per_modality: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    schema_version: str = MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION
    model_family: str = "UnityPSF"
    model_name: str = "UnityPSF"
    router_mode: str = "hard_top1"
    code_version: str = "0.4.0"

    def __post_init__(self) -> None:
        if self.schema_version != MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported modality joint checkpoint schema {self.schema_version!r}")
        if self.model_family != "UnityPSF" or self.model_name != "UnityPSF":
            raise ValueError("joint checkpoint model identity must be UnityPSF")
        role = str(self.checkpoint_role).strip().lower()
        if role not in _ROLES:
            raise ValueError("checkpoint_role must be 'release' or 'resume'")
        if self.router_mode != "hard_top1":
            raise ValueError("modality joint checkpoint supports only hard_top1 routing")
        modalities = tuple(PSFModality.parse(item) for item in self.supported_modalities)
        if not modalities or len(modalities) != len(set(modalities)):
            raise ValueError("supported_modalities must be non-empty and unique")
        channels: dict[str, tuple[str, ...]] = {}
        for raw_modality, raw_channels in self.supported_channels_per_modality.items():
            modality = PSFModality.parse(raw_modality).value
            values = tuple(MeasurementChannelSpec.from_value(item).channel_id for item in raw_channels)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"supported channels for {modality!r} must be non-empty and unique")
            channels[modality] = values
        if channels and set(channels) != {item.value for item in modalities}:
            raise ValueError("supported_channels_per_modality must cover every supported modality")
        object.__setattr__(self, "checkpoint_role", role)
        object.__setattr__(self, "supported_modalities", modalities)
        object.__setattr__(self, "supported_channels_per_modality", channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "checkpoint_role": self.checkpoint_role,
            "supported_modalities": [item.value for item in self.supported_modalities],
            "supported_channels_per_modality": {
                key: list(value) for key, value in self.supported_channels_per_modality.items()
            },
            "router_mode": self.router_mode,
            "code_version": self.code_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModalityJointCheckpointMetadata":
        raw_channels = value.get("supported_channels_per_modality", {})
        if not isinstance(raw_channels, Mapping):
            raise ValueError("supported_channels_per_modality must be a mapping")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            model_family=str(value.get("model_family", "")),
            model_name=str(value.get("model_name", "")),
            checkpoint_role=str(value.get("checkpoint_role", "")),
            supported_modalities=tuple(value.get("supported_modalities", ())),
            supported_channels_per_modality={
                str(key): tuple(channels) for key, channels in raw_channels.items()
            },
            router_mode=str(value.get("router_mode", "")),
            code_version=str(value.get("code_version", "")),
        )


@dataclass(frozen=True)
class ModalityChannelState:
    """Physical and calibration state owned by one measurement channel."""

    channel_id: str
    physical_state: Mapping[str, Any] = field(default_factory=dict)
    calibration: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_id",
            MeasurementChannelSpec.from_value(self.channel_id).channel_id,
        )
        for field_name in ("physical_state", "calibration", "provenance"):
            if not isinstance(getattr(self, field_name), Mapping):
                raise TypeError(f"{field_name} must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_state": dict(self.physical_state),
            "calibration": dict(self.calibration),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, channel_id: str, value: Mapping[str, Any]) -> "ModalityChannelState":
        return cls(
            channel_id=channel_id,
            physical_state=value.get("physical_state", {}),
            calibration=value.get("calibration", {}),
            provenance=value.get("provenance", {}),
        )


@dataclass(frozen=True)
class ModalityExpertState:
    """One complete network plus independently owned measurement channel states."""

    modality: PSFModality | str
    model_class: str
    model_config: Mapping[str, Any]
    input_frame_spec: InputFrameSpec | Mapping[str, Any]
    condition_schema: Mapping[str, Any]
    model_state_dict: Mapping[str, Any]
    channel_states: tuple[ModalityChannelState, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", PSFModality.parse(self.modality))
        object.__setattr__(self, "model_class", _identifier(self.model_class, field_name="model_class"))
        if not isinstance(self.input_frame_spec, InputFrameSpec):
            object.__setattr__(self, "input_frame_spec", InputFrameSpec.from_value(self.input_frame_spec))
        channels = tuple(
            item if isinstance(item, ModalityChannelState) else ModalityChannelState(**item)
            for item in self.channel_states
        )
        channel_ids = tuple(item.channel_id for item in channels)
        if not channels or len(channel_ids) != len(set(channel_ids)):
            raise ValueError("channel_states must be non-empty and unique")
        object.__setattr__(self, "channel_states", channels)
        for field_name in ("model_config", "condition_schema", "model_state_dict", "provenance"):
            if not isinstance(getattr(self, field_name), Mapping):
                raise TypeError(f"{field_name} must be a mapping")
        if not self.model_state_dict:
            raise ValueError("model_state_dict must not be empty")

    def expert_dict(self) -> dict[str, Any]:
        return {
            "expert_type": self.modality.value,
            "model_class": self.model_class,
            "model_config": dict(self.model_config),
            "input_frame_spec": _frame_spec_dict(self.input_frame_spec),
            "condition_schema": dict(self.condition_schema),
            "model_state_dict": dict(self.model_state_dict),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_payload(
        cls,
        modality: str,
        expert: Mapping[str, Any],
        channels: Mapping[str, Any],
    ) -> "ModalityExpertState":
        parsed = PSFModality.parse(modality)
        if expert.get("expert_type") != parsed.value:
            raise ValueError(f"expert registry key {modality!r} conflicts with embedded metadata")
        return cls(
            modality=parsed,
            model_class=expert.get("model_class"),
            model_config=expert.get("model_config", {}),
            input_frame_spec=expert.get("input_frame_spec", {}),
            condition_schema=expert.get("condition_schema", {}),
            model_state_dict=expert.get("model_state_dict", {}),
            channel_states=tuple(
                ModalityChannelState.from_dict(channel_id, value)
                for channel_id, value in channels.items()
            ),
            provenance=expert.get("provenance", {}),
        )


def _integrity_table(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": "sha256-logical-v1",
        "root_sha256": logical_sha256(payload),
        "expert_sha256": {
            key: logical_sha256(value) for key, value in payload["experts"].items()
        },
        "channel_state_sha256": {
            modality: {key: logical_sha256(value) for key, value in channels.items()}
            for modality, channels in payload["channel_states"].items()
        },
    }


def build_modality_joint_checkpoint(
    *,
    metadata: ModalityJointCheckpointMetadata,
    experts: Iterable[ModalityExpertState],
    provenance: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a v2 checkpoint with exactly one network per PSF modality."""

    if not isinstance(metadata, ModalityJointCheckpointMetadata):
        raise TypeError("metadata must be ModalityJointCheckpointMetadata")
    registry: dict[str, Any] = {}
    channel_states: dict[str, Any] = {}
    for expert in experts:
        if not isinstance(expert, ModalityExpertState):
            raise TypeError("experts must contain ModalityExpertState values")
        key = expert.modality.value
        if key in registry:
            raise ValueError(f"duplicate modality expert {key!r}")
        registry[key] = expert.expert_dict()
        channel_states[key] = {item.channel_id: item.to_dict() for item in expert.channel_states}
    if set(registry) != {item.value for item in metadata.supported_modalities}:
        raise ValueError("supported_modalities must exactly match the expert registry")
    channel_inventory = {key: list(value) for key, value in channel_states.items()}
    metadata_value = metadata.to_dict()
    if metadata.supported_channels_per_modality:
        declared = {key: list(value) for key, value in metadata.supported_channels_per_modality.items()}
        if declared != channel_inventory:
            raise ValueError("supported channel metadata must match channel_states")
    metadata_value["supported_channels_per_modality"] = channel_inventory
    payload: dict[str, Any] = {
        "checkpoint_schema": MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata_value,
        "router": dict(_ROUTER),
        "experts": registry,
        "channel_states": channel_states,
        "provenance": dict(provenance or {}),
    }
    if metadata.checkpoint_role == "release":
        if training_state is not None:
            raise ValueError("release checkpoint cannot carry training_state")
    elif not isinstance(training_state, Mapping) or set(training_state) != set(registry):
        raise ValueError("resume checkpoint requires training_state for every modality")
    else:
        payload["training_state"] = dict(training_state)
    payload["integrity"] = _integrity_table(payload)
    validate_modality_joint_checkpoint(payload)
    return payload


def validate_modality_joint_checkpoint(
    payload: Mapping[str, Any],
    *,
    verify_integrity: bool = True,
) -> ModalityJointCheckpointMetadata:
    """Validate modality routing, nested channels, role semantics, and hashes."""

    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported or missing modality joint checkpoint schema")
    metadata_value = payload.get("metadata")
    experts = payload.get("experts")
    channels = payload.get("channel_states")
    if not isinstance(metadata_value, Mapping) or not isinstance(experts, Mapping) or not isinstance(channels, Mapping):
        raise ValueError("modality joint checkpoint requires metadata, experts, and channel_states mappings")
    if payload.get("router") != _ROUTER:
        raise ValueError("modality joint checkpoint router must use modality hard_top1 routing")
    metadata = ModalityJointCheckpointMetadata.from_dict(metadata_value)
    if set(experts) != set(channels) or set(experts) != {item.value for item in metadata.supported_modalities}:
        raise ValueError("expert, channel state, and supported modality registries must match")
    parsed = []
    for modality, expert in experts.items():
        channel_value = channels.get(modality)
        if not isinstance(expert, Mapping) or not isinstance(channel_value, Mapping):
            raise ValueError("expert and channel state entries must be mappings")
        if any(not isinstance(key, str) or not isinstance(value, Mapping) for key, value in channel_value.items()):
            raise ValueError("channel state registries must map string IDs to mappings")
        parsed.append(ModalityExpertState.from_payload(modality, expert, channel_value))
    inventory = {item.modality.value: tuple(channel.channel_id for channel in item.channel_states) for item in parsed}
    if inventory != dict(metadata.supported_channels_per_modality):
        raise ValueError("supported channel metadata must match nested channel states")
    training_state = payload.get("training_state")
    if metadata.checkpoint_role == "release" and training_state is not None:
        raise ValueError("release checkpoint cannot carry training_state")
    if metadata.checkpoint_role == "resume" and (
        not isinstance(training_state, Mapping) or set(training_state) != set(experts)
    ):
        raise ValueError("resume checkpoint requires training_state for every modality")
    if verify_integrity:
        integrity = payload.get("integrity")
        unsigned = {key: value for key, value in payload.items() if key != "integrity"}
        if not isinstance(integrity, Mapping) or integrity != _integrity_table(unsigned):
            raise ValueError("modality joint checkpoint integrity hash mismatch")
    return metadata


def save_modality_joint_checkpoint(
    path: str | Path,
    *,
    metadata: ModalityJointCheckpointMetadata,
    experts: Iterable[ModalityExpertState],
    provenance: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    payload = build_modality_joint_checkpoint(
        metadata=metadata,
        experts=experts,
        provenance=provenance,
        training_state=training_state,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        load_modality_joint_checkpoint(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_modality_joint_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("joint checkpoint file must contain a mapping")
    validate_modality_joint_checkpoint(payload)
    return dict(payload)


def load_legacy_joint_checkpoint_read_only(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Validate and return v1 without modifying or rewriting the source file."""

    return _load_legacy_joint_checkpoint(path, map_location=map_location)


def detect_joint_checkpoint_schema(path: str | Path) -> str:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("checkpoint_schema"), str):
        raise ValueError("joint checkpoint schema is missing")
    return payload["checkpoint_schema"]


__all__ = [
    "LEGACY_JOINT_CHECKPOINT_SCHEMA_VERSION",
    "MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION",
    "ModalityChannelState",
    "ModalityExpertState",
    "ModalityJointCheckpointMetadata",
    "build_modality_joint_checkpoint",
    "detect_joint_checkpoint_schema",
    "load_legacy_joint_checkpoint_read_only",
    "load_modality_joint_checkpoint",
    "save_modality_joint_checkpoint",
    "validate_modality_joint_checkpoint",
]
