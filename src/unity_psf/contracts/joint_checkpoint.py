"""Self-contained checkpoint contract for one UnityPSF model."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .checkpoint import CheckpointMetadata, load_checkpoint
from .modality import InputFrameSpec, MeasurementChannelSpec, PSFModality


JOINT_CHECKPOINT_SCHEMA_VERSION = "unity_psf.joint_checkpoint.v1"
_ROLES = frozenset({"release", "resume"})


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if ":" in normalized:
        raise ValueError(f"{field_name} cannot contain ':'")
    return normalized


@dataclass(frozen=True, order=True)
class JointExpertKey:
    """Unique address of one independently owned expert/channel instance."""

    modality: PSFModality | str
    channel_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", PSFModality.parse(self.modality))
        object.__setattr__(self, "channel_id", _identifier(self.channel_id, field_name="channel_id"))

    @property
    def storage_key(self) -> str:
        return f"{self.modality.value}:{self.channel_id}"

    @classmethod
    def parse(cls, value: "JointExpertKey | str") -> "JointExpertKey":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or value.count(":") != 1:
            raise ValueError("expert instance key must use 'modality:channel_id'")
        modality, channel_id = value.split(":", 1)
        return cls(modality=modality, channel_id=channel_id)


@dataclass(frozen=True)
class JointCheckpointMetadata:
    """Immutable identity and capability declaration for a joint checkpoint."""

    checkpoint_role: str = "release"
    supported_modalities: tuple[PSFModality | str, ...] = (
        PSFModality.EMITTER_2D,
        PSFModality.ASTIGMATISM,
    )
    schema_version: str = JOINT_CHECKPOINT_SCHEMA_VERSION
    model_family: str = "UnityPSF"
    model_name: str = "UnityPSF"
    router_mode: str = "hard"
    code_version: str = "0.4.0"

    def __post_init__(self) -> None:
        if self.schema_version != JOINT_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported joint checkpoint schema {self.schema_version!r}")
        if self.model_family != "UnityPSF" or self.model_name != "UnityPSF":
            raise ValueError("joint checkpoint model identity must be UnityPSF")
        role = str(self.checkpoint_role).strip().lower()
        if role not in _ROLES:
            raise ValueError("checkpoint_role must be 'release' or 'resume'")
        if self.router_mode != "hard":
            raise ValueError("joint checkpoint v1 supports only deterministic hard routing")
        modalities = tuple(PSFModality.parse(item) for item in self.supported_modalities)
        if not modalities or len(modalities) != len(set(modalities)):
            raise ValueError("supported_modalities must be non-empty and unique")
        object.__setattr__(self, "checkpoint_role", role)
        object.__setattr__(self, "supported_modalities", modalities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "checkpoint_role": self.checkpoint_role,
            "supported_modalities": [item.value for item in self.supported_modalities],
            "router_mode": self.router_mode,
            "code_version": self.code_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JointCheckpointMetadata":
        return cls(
            schema_version=str(value.get("schema_version", "")),
            model_family=str(value.get("model_family", "")),
            model_name=str(value.get("model_name", "")),
            checkpoint_role=str(value.get("checkpoint_role", "")),
            supported_modalities=tuple(value.get("supported_modalities", ())),
            router_mode=str(value.get("router_mode", "")),
            code_version=str(value.get("code_version", "")),
        )


@dataclass(frozen=True)
class JointExpertState:
    """Complete inference state for one expert/channel instance."""

    key: JointExpertKey
    instance_id: str
    model_class: str
    model_config: Mapping[str, Any]
    input_frame_spec: InputFrameSpec | Mapping[str, Any]
    condition_schema: Mapping[str, Any]
    model_state_dict: Mapping[str, Any]
    physical_state: Mapping[str, Any] = field(default_factory=dict)
    calibration: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.key, JointExpertKey):
            object.__setattr__(self, "key", JointExpertKey.parse(self.key))
        object.__setattr__(self, "instance_id", _identifier(self.instance_id, field_name="instance_id"))
        object.__setattr__(self, "model_class", _identifier(self.model_class, field_name="model_class"))
        if not isinstance(self.input_frame_spec, InputFrameSpec):
            object.__setattr__(self, "input_frame_spec", InputFrameSpec.from_value(self.input_frame_spec))
        for field_name in (
            "model_config",
            "condition_schema",
            "model_state_dict",
            "physical_state",
            "calibration",
            "provenance",
        ):
            if not isinstance(getattr(self, field_name), Mapping):
                raise TypeError(f"{field_name} must be a mapping")
        if not self.model_state_dict:
            raise ValueError("model_state_dict must not be empty")

    def to_dict(self) -> dict[str, Any]:
        frame_order = self.input_frame_spec.frame_order
        return {
            "expert_type": self.key.modality.value,
            "channel_id": self.key.channel_id,
            "instance_id": self.instance_id,
            "model_class": self.model_class,
            "model_config": dict(self.model_config),
            "input_frame_spec": {
                "input_frame_channels": self.input_frame_spec.input_frame_channels,
                "frame_order": list(frame_order) if isinstance(frame_order, tuple) else frame_order,
            },
            "condition_schema": dict(self.condition_schema),
            "model_state_dict": dict(self.model_state_dict),
            "physical_state": dict(self.physical_state),
            "calibration": dict(self.calibration),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, key: str, value: Mapping[str, Any]) -> "JointExpertState":
        parsed_key = JointExpertKey.parse(key)
        if value.get("expert_type") != parsed_key.modality.value or value.get("channel_id") != parsed_key.channel_id:
            raise ValueError(f"expert registry key {key!r} conflicts with embedded metadata")
        return cls(
            key=parsed_key,
            instance_id=value.get("instance_id"),
            model_class=value.get("model_class"),
            model_config=value.get("model_config", {}),
            input_frame_spec=value.get("input_frame_spec", {}),
            condition_schema=value.get("condition_schema", {}),
            model_state_dict=value.get("model_state_dict", {}),
            physical_state=value.get("physical_state", {}),
            calibration=value.get("calibration", {}),
            provenance=value.get("provenance", {}),
        )


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
    if value is None:
        digest.update(b"none")
    elif isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        digest.update(f"int:{value}".encode("ascii"))
    elif isinstance(value, float):
        digest.update(f"float:{value.hex()}".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
    elif isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}:".encode("ascii"))
        digest.update(value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f"ndarray:{array.dtype}:{array.shape}:".encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, np.generic):
        _update_digest(digest, value.item())
    elif isinstance(value, Mapping):
        if any(not isinstance(key, (str, int)) for key in value):
            raise TypeError("checkpoint mapping keys must be strings or integers")
        digest.update(f"mapping:{len(value)}:".encode("ascii"))
        for key in sorted(
            value,
            key=lambda item: (0, item) if isinstance(item, str) else (1, item),
        ):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"sequence:{len(value)}:".encode("ascii"))
        for item in value:
            _update_digest(digest, item)
    else:
        raise TypeError(f"unsupported checkpoint value type {type(value)!r}")


def logical_sha256(value: Any) -> str:
    """Hash nested checkpoint values independently of torch serialization details."""

    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _integrity_table(payload_without_integrity: Mapping[str, Any]) -> dict[str, Any]:
    experts = payload_without_integrity["experts"]
    return {
        "algorithm": "sha256-logical-v1",
        "root_sha256": logical_sha256(payload_without_integrity),
        "expert_sha256": {key: logical_sha256(value) for key, value in experts.items()},
    }


def build_joint_checkpoint(
    *,
    metadata: JointCheckpointMetadata,
    experts: Iterable[JointExpertState],
    router: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one complete UnityPSF checkpoint payload."""

    if not isinstance(metadata, JointCheckpointMetadata):
        raise TypeError("metadata must be JointCheckpointMetadata")
    registry: dict[str, Any] = {}
    for expert in experts:
        if not isinstance(expert, JointExpertState):
            raise TypeError("experts must contain JointExpertState values")
        key = expert.key.storage_key
        if key in registry:
            raise ValueError(f"duplicate expert instance key {key!r}")
        registry[key] = expert.to_dict()
    if not registry:
        raise ValueError("joint checkpoint requires at least one expert instance")

    registry_modalities = {JointExpertKey.parse(key).modality for key in registry}
    if registry_modalities != set(metadata.supported_modalities):
        raise ValueError("supported_modalities must exactly match modalities present in the expert registry")
    selected_router = dict(router or {"type": "deterministic", "mode": "hard"})
    if selected_router.get("mode") != "hard":
        raise ValueError("joint checkpoint v1 router must use hard mode")

    payload: dict[str, Any] = {
        "checkpoint_schema": JOINT_CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata.to_dict(),
        "router": selected_router,
        "experts": registry,
        "provenance": dict(provenance or {}),
    }
    if metadata.checkpoint_role == "release":
        if training_state is not None:
            raise ValueError("release checkpoint cannot carry training_state")
    else:
        if not isinstance(training_state, Mapping) or set(training_state) != set(registry):
            raise ValueError("resume checkpoint requires training_state for every expert instance")
        payload["training_state"] = dict(training_state)
    payload["integrity"] = _integrity_table(payload)
    validate_joint_checkpoint(payload)
    return payload


def validate_joint_checkpoint(
    payload: Mapping[str, Any],
    *,
    verify_integrity: bool = True,
) -> JointCheckpointMetadata:
    """Validate structure, capabilities, role semantics, and nested hashes."""

    if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != JOINT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported or missing joint checkpoint schema")
    metadata_value = payload.get("metadata")
    experts_value = payload.get("experts")
    router_value = payload.get("router")
    if not isinstance(metadata_value, Mapping) or not isinstance(experts_value, Mapping):
        raise ValueError("joint checkpoint requires metadata and experts mappings")
    if not isinstance(router_value, Mapping) or router_value.get("mode") != "hard":
        raise ValueError("joint checkpoint requires a hard router mapping")
    metadata = JointCheckpointMetadata.from_dict(metadata_value)
    if not experts_value:
        raise ValueError("joint checkpoint requires at least one expert instance")
    parsed_experts = []
    for key, value in experts_value.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise ValueError("expert registry must map string keys to mappings")
        parsed_experts.append(JointExpertState.from_dict(key, value))
    if len({item.key for item in parsed_experts}) != len(parsed_experts):
        raise ValueError("duplicate expert instance key")
    present_modalities = {item.key.modality for item in parsed_experts}
    if present_modalities != set(metadata.supported_modalities):
        raise ValueError("supported_modalities must exactly match modalities present in the expert registry")

    training_state = payload.get("training_state")
    if metadata.checkpoint_role == "release" and training_state is not None:
        raise ValueError("release checkpoint cannot carry training_state")
    if metadata.checkpoint_role == "resume":
        if not isinstance(training_state, Mapping) or set(training_state) != set(experts_value):
            raise ValueError("resume checkpoint requires training_state for every expert instance")

    if verify_integrity:
        integrity = payload.get("integrity")
        if not isinstance(integrity, Mapping) or integrity.get("algorithm") != "sha256-logical-v1":
            raise ValueError("joint checkpoint integrity table is missing or unsupported")
        unsigned = {key: value for key, value in payload.items() if key != "integrity"}
        expected = _integrity_table(unsigned)
        if integrity != expected:
            raise ValueError("joint checkpoint integrity hash mismatch")
    return metadata


def save_joint_checkpoint(
    path: str | Path,
    *,
    metadata: JointCheckpointMetadata,
    experts: Iterable[JointExpertState],
    router: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically persist a checkpoint after a disk round-trip validation."""

    payload = build_joint_checkpoint(
        metadata=metadata,
        experts=experts,
        router=router,
        provenance=provenance,
        training_state=training_state,
    )
    return save_joint_checkpoint_payload(path, payload)


def save_joint_checkpoint_payload(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically persist an already assembled and validated joint payload."""

    validate_joint_checkpoint(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        load_joint_checkpoint(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_joint_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load one joint checkpoint and verify its complete logical payload."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("joint checkpoint file must contain a mapping")
    validate_joint_checkpoint(payload)
    return dict(payload)


def assemble_joint_checkpoint(
    instance_checkpoints: Iterable[str | Path],
    *,
    metadata: JointCheckpointMetadata,
    physical_states: Mapping[str, Mapping[str, Any]] | None = None,
    calibrations: Mapping[str, Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Import task-1-10 v2 instance checkpoints into the joint schema."""

    experts = []
    for source in instance_checkpoints:
        source_path = Path(source)
        payload = load_checkpoint(source_path, map_location="cpu")
        checkpoint_metadata = CheckpointMetadata.from_dict(payload["metadata"])
        if checkpoint_metadata.checkpoint_role != "instance" or checkpoint_metadata.expert_type is None:
            raise ValueError(f"assembler requires a v2 instance checkpoint: {source_path}")
        channel_spec = checkpoint_metadata.channel_spec
        if not isinstance(channel_spec, MeasurementChannelSpec):
            raise ValueError(f"instance checkpoint is missing channel metadata: {source_path}")
        key = JointExpertKey(checkpoint_metadata.expert_type, channel_spec.channel_id)
        model_state = payload.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            raise ValueError(f"instance checkpoint is missing model_state_dict: {source_path}")
        experts.append(
            JointExpertState(
                key=key,
                instance_id=str(checkpoint_metadata.instance_id),
                model_class={
                    PSFModality.EMITTER_2D: "unity_psf.models.psf_moe.experts.emitter_2d.Emitter2DExpert",
                    PSFModality.ASTIGMATISM: "unity_psf.models.psf_moe.experts.astigmatism.AstigmatismExpert",
                    PSFModality.DOUBLE_HELIX: "unity_psf.models.psf_moe.experts.double_helix.DoubleHelixExpert",
                }[key.modality],
                model_config=checkpoint_metadata.model_config,
                input_frame_spec=checkpoint_metadata.input_frame_spec,
                condition_schema=checkpoint_metadata.condition_schema,
                model_state_dict=model_state,
                physical_state=dict((physical_states or {}).get(key.storage_key, payload.get("physical_state", {}))),
                calibration=dict((calibrations or {}).get(key.storage_key, payload.get("calibration", {}))),
                provenance={"source_checkpoint": str(source_path), **dict(payload.get("provenance", {}))},
            )
        )
    return build_joint_checkpoint(
        metadata=metadata,
        experts=experts,
        provenance=provenance,
        training_state=training_state,
    )


__all__ = [
    "JOINT_CHECKPOINT_SCHEMA_VERSION",
    "JointCheckpointMetadata",
    "JointExpertKey",
    "JointExpertState",
    "assemble_joint_checkpoint",
    "build_joint_checkpoint",
    "load_joint_checkpoint",
    "logical_sha256",
    "save_joint_checkpoint",
    "save_joint_checkpoint_payload",
    "validate_joint_checkpoint",
]
