"""Independent channel-bound instances created from UnityPSF prototypes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from unity_psf.contracts.bundle import sha256_file
from unity_psf.contracts.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointMetadata, load_checkpoint
from unity_psf.contracts.modality import ExpertInstanceSpec, MeasurementChannelSpec, PSFModality

from .experts.astigmatism import AstigmatismExpert


def _validate_astigmatism_condition_schema(schema: Mapping[str, Any]) -> None:
    if schema.get("name") != "astigmatism_film_v1":
        raise ValueError("prototype condition_schema must use astigmatism_film_v1")
    fields = schema.get("fields")
    dimension = schema.get("dimension")
    if isinstance(fields, (str, bytes)) or not isinstance(fields, (list, tuple)):
        raise ValueError("prototype condition_schema fields must be a sequence")
    if any(not isinstance(field, str) or not field.strip() for field in fields):
        raise ValueError("prototype condition_schema fields must be non-empty strings")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension != len(fields):
        raise ValueError("prototype condition_schema dimension must match fields")


class AstigmatismExpertInstance(AstigmatismExpert):
    """A complete astigmatism expert bound to one physical measurement channel."""

    def __init__(
        self,
        *,
        instance_spec: ExpertInstanceSpec,
        parent_checkpoint_hash: str,
        prototype_metadata: CheckpointMetadata,
        **model_config: Any,
    ) -> None:
        if not isinstance(instance_spec, ExpertInstanceSpec):
            raise TypeError("instance_spec must be an ExpertInstanceSpec")
        if not isinstance(prototype_metadata, CheckpointMetadata):
            raise TypeError("prototype_metadata must be CheckpointMetadata")
        super().__init__(**dict(model_config))
        self._instance_spec = instance_spec
        self._parent_checkpoint_hash = str(parent_checkpoint_hash).strip().lower()
        self._prototype_metadata = prototype_metadata
        self._instance_metadata = super().checkpoint_metadata(
            checkpoint_role="instance",
            instance_id=instance_spec.instance_id,
            channel_spec=MeasurementChannelSpec(instance_spec.channel_id),
            parent_checkpoint_hash=self._parent_checkpoint_hash,
            code_version=prototype_metadata.code_version,
        )

    @property
    def expert_type(self) -> PSFModality:
        return self._instance_spec.expert_type

    @property
    def instance_id(self) -> str:
        return self._instance_spec.instance_id

    @property
    def channel_id(self) -> str:
        return self._instance_spec.channel_id

    @property
    def prototype_ref(self) -> str | None:
        return self._instance_spec.prototype_ref

    @property
    def parent_checkpoint_hash(self) -> str:
        return self._parent_checkpoint_hash

    @property
    def metadata(self) -> CheckpointMetadata:
        return self._instance_metadata

    @property
    def prototype_metadata(self) -> CheckpointMetadata:
        return self._prototype_metadata

    def checkpoint_metadata(self) -> CheckpointMetadata:
        """Return immutable instance metadata rather than prototype metadata."""

        return self._instance_metadata


def create_expert_instance_from_prototype(
    prototype_checkpoint: str | Path,
    instance_spec: ExpertInstanceSpec | Mapping[str, object],
    *,
    device: str | torch.device = "cpu",
) -> AstigmatismExpertInstance:
    """Create one independent astigmatism instance from a v2 prototype file."""

    checkpoint_path = Path(prototype_checkpoint)
    if not checkpoint_path.is_file():
        raise ValueError(f"prototype checkpoint does not exist: {checkpoint_path}")
    if not isinstance(instance_spec, ExpertInstanceSpec):
        instance_spec = ExpertInstanceSpec.from_value(instance_spec)  # type: ignore[arg-type]

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("expert instance factory requires a v2 prototype checkpoint")
    metadata_value = payload.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise ValueError("prototype checkpoint metadata must be a mapping")
    metadata = CheckpointMetadata.from_dict(metadata_value)
    if metadata.checkpoint_role != "prototype":
        raise ValueError("prototype checkpoint metadata must have checkpoint_role='prototype'")
    if metadata.expert_type is None:
        raise ValueError("prototype checkpoint metadata requires expert_type")
    if metadata.expert_type is not PSFModality.ASTIGMATISM:
        raise ValueError(
            f"prototype expert_type {metadata.expert_type.value!r} is not supported by the astigmatism instance factory"
        )
    if instance_spec.expert_type is not PSFModality.ASTIGMATISM:
        raise ValueError(
            f"instance expert_type {instance_spec.expert_type.value!r} does not match astigmatism factory"
        )
    _validate_astigmatism_condition_schema(metadata.condition_schema)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("prototype checkpoint model_state_dict must be a mapping")

    parent_hash = sha256_file(checkpoint_path)
    model = AstigmatismExpertInstance(
        instance_spec=instance_spec,
        parent_checkpoint_hash=parent_hash,
        prototype_metadata=metadata,
        **dict(metadata.model_config),
    )
    if dict(model.condition_schema) != dict(metadata.condition_schema):
        raise ValueError("prototype condition_schema does not match model_config")
    model.load_state_dict(model_state, strict=True)
    model.to(torch.device(device))
    return model


def build_instance_optimizer(
    instance: AstigmatismExpertInstance,
    *,
    optimizer_name: str = "adamw",
    optimizer_params: Mapping[str, Any] | None = None,
) -> torch.optim.Optimizer:
    """Build an optimizer from one instance's own parameters."""

    if not isinstance(instance, AstigmatismExpertInstance):
        raise TypeError("instance must be an AstigmatismExpertInstance")
    params = dict(optimizer_params or {})
    normalized_name = str(optimizer_name).strip().lower()
    if normalized_name == "adamw":
        return torch.optim.AdamW(instance.parameters(), **params)
    if normalized_name == "adam":
        return torch.optim.Adam(instance.parameters(), **params)
    if normalized_name == "sgd":
        return torch.optim.SGD(instance.parameters(), **params)
    raise ValueError(f"unsupported instance optimizer {optimizer_name!r}")


def parameter_state_hash(model: nn.Module) -> str:
    """Hash parameter names, dtypes, shapes, and values in deterministic order."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "AstigmatismExpertInstance",
    "build_instance_optimizer",
    "create_expert_instance_from_prototype",
    "parameter_state_hash",
]
