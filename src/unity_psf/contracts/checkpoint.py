"""Versioned checkpoint payload IO and compatibility loaders for UnityPSF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint_metadata import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_EXPERTS,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    LEGACY_STATE_DICT_SCHEMA_VERSION,
    LEGACY_TRAINING_SCHEMA_VERSION,
    CheckpointMetadata,
    _validate_sha256,  # noqa: F401 - preserved for bundle compatibility
)


def detect_checkpoint_format(payload: Mapping[str, Any]) -> str:
    """Identify v2, legacy training, v1, or raw state-dict payloads."""

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    schema = payload.get("checkpoint_schema")
    if schema == CHECKPOINT_SCHEMA_VERSION:
        return "v2"
    if schema == LEGACY_CHECKPOINT_SCHEMA_VERSION:
        return "legacy.v1"
    if "model_state_dict" in payload:
        return "legacy.training"
    if payload and all(isinstance(key, str) for key in payload):
        return "legacy.state_dict"
    raise ValueError("checkpoint payload has an unknown format")


def build_checkpoint(
    model_state: Mapping[str, Any] | None = None,
    *,
    model_state_dict: Mapping[str, Any] | None = None,
    metadata: CheckpointMetadata | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a v2 checkpoint payload using the training loop field names."""

    if model_state is not None and model_state_dict is not None:
        raise ValueError("provide only one of model_state and model_state_dict")
    selected_state = model_state if model_state is not None else model_state_dict
    if not isinstance(selected_state, Mapping):
        raise TypeError("model_state must be a mapping returned by state_dict()")
    payload: dict[str, Any] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "metadata": (metadata or CheckpointMetadata()).to_dict(),
        "model_state_dict": dict(selected_state),
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = dict(optimizer_state)
    if scheduler_state is not None:
        payload["scheduler_state_dict"] = dict(scheduler_state)
    if extra:
        payload.update(dict(extra))
    return payload


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> CheckpointMetadata:
    """Validate a checkpoint and return metadata without upgrading legacy data."""

    checkpoint_format = detect_checkpoint_format(payload)
    if checkpoint_format == "v2":
        if not isinstance(payload.get("metadata"), Mapping):
            raise ValueError("v2 checkpoint metadata must be a mapping")
        if not isinstance(payload.get("model_state_dict"), Mapping):
            raise ValueError("v2 checkpoint model_state_dict must be a mapping")
        return CheckpointMetadata.from_dict(payload["metadata"])
    if checkpoint_format == "legacy.v1":
        if not isinstance(payload.get("metadata"), Mapping):
            raise ValueError("legacy v1 checkpoint metadata must be a mapping")
        if not isinstance(payload.get("model_state"), Mapping):
            raise ValueError("legacy v1 checkpoint model_state must be a mapping")
        return CheckpointMetadata.from_dict(payload["metadata"])
    if checkpoint_format == "legacy.training":
        if not isinstance(payload.get("model_state_dict"), Mapping):
            raise ValueError("legacy training model_state_dict must be a mapping")
        return CheckpointMetadata(
            schema_version=LEGACY_TRAINING_SCHEMA_VERSION,
            model_name="legacy_training",
            checkpoint_role="legacy",
        )
    return CheckpointMetadata(
        schema_version=LEGACY_STATE_DICT_SCHEMA_VERSION,
        model_name="legacy_state_dict",
        checkpoint_role="legacy",
    )


def save_checkpoint(
    path: str | Path,
    model_state: Mapping[str, Any] | None = None,
    *,
    model_state_dict: Mapping[str, Any] | None = None,
    metadata: CheckpointMetadata | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Persist one validated v2 checkpoint payload."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint(
        model_state,
        model_state_dict=model_state_dict,
        metadata=metadata,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        extra=extra,
    )
    torch.save(payload, destination)
    return destination


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and classify a UnityPSF checkpoint without rewriting legacy fields."""

    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint file must contain a mapping payload")
    validate_checkpoint_payload(payload)
    return dict(payload)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointMetadata",
    "DEFAULT_EXPERTS",
    "LEGACY_CHECKPOINT_SCHEMA_VERSION",
    "_validate_sha256",
    "build_checkpoint",
    "detect_checkpoint_format",
    "load_checkpoint",
    "save_checkpoint",
    "validate_checkpoint_payload",
]
