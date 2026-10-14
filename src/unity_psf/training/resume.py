from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from unity_psf.contracts.checkpoint import CheckpointMetadata, detect_checkpoint_format, load_checkpoint
from unity_psf.contracts.modality import PSFModality
from unity_psf.training.checkpoint_state import restore_rng_state
from unity_psf.training.loop_types import ModelInitializationState, TrainingResumeState


def resume_training_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    amp_scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    expected_metadata: CheckpointMetadata | None = None,
    expected_expert_type: PSFModality | str | None = None,
    expected_instance_id: str | None = None,
    expected_channel_id: str | None = None,
    expected_parent_checkpoint_hash: str | None = None,
    restore_rng: bool = True,
    _require_identity: bool = True,
) -> TrainingResumeState:
    """Resume a complete training state with explicit instance lineage checks.

    A v2 instance checkpoint must be matched against the caller's identity
    contract. Legacy payloads remain readable through the compatibility branch,
    but they cannot satisfy an explicit v2 identity expectation.
    """

    checkpoint_path = Path(path)
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    checkpoint_format = detect_checkpoint_format(checkpoint)
    metadata = _checkpoint_metadata(checkpoint, checkpoint_format)
    if checkpoint_format == "v2":
        if metadata is None:
            raise ValueError("v2 checkpoint metadata is required for full resume")
        if metadata.checkpoint_role == "instance" and _require_identity:
            if expected_metadata is None and any(
                value is None
                for value in (
                    expected_expert_type,
                    expected_instance_id,
                    expected_channel_id,
                    expected_parent_checkpoint_hash,
                )
            ):
                raise ValueError(
                    "full instance resume requires expert_type, instance_id, channel_id, "
                    "and parent_checkpoint_hash"
                )
        _validate_resume_identity(
            metadata,
            expected_metadata=expected_metadata,
            expected_expert_type=expected_expert_type,
            expected_instance_id=expected_instance_id,
            expected_channel_id=expected_channel_id,
            expected_parent_checkpoint_hash=expected_parent_checkpoint_hash,
        )
        if metadata.checkpoint_role == "instance" and optimizer is None:
            raise ValueError("full instance resume requires an optimizer")
        if metadata.checkpoint_role == "instance" and "scheduler_state_dict" in checkpoint and scheduler is None:
            raise ValueError("full instance resume requires a scheduler")

    model.load_state_dict(_model_state_dict(checkpoint, checkpoint_format), strict=True)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None:
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint is missing optimizer_state_dict")
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None:
        if not isinstance(scheduler_state, Mapping):
            raise ValueError("checkpoint is missing scheduler_state_dict")
        scheduler.load_state_dict(scheduler_state)
    scaler_state = checkpoint.get("scaler_state_dict")
    if amp_scaler is not None and isinstance(scaler_state, Mapping):
        amp_scaler.load_state_dict(dict(scaler_state))
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))

    return TrainingResumeState(
        epoch=int(checkpoint["epoch"]),
        step_count=int(checkpoint["step_count"]),
        global_step=int(checkpoint["global_step"]),
        path=checkpoint_path,
        physical_state=checkpoint.get("physical_state") if isinstance(checkpoint.get("physical_state"), dict) else None,
        physical_coeff_maps=_checkpoint_physical_coeff_maps(checkpoint),
        physical_state_hash=(
            None if checkpoint.get("physical_state_hash") is None else str(checkpoint["physical_state_hash"])
        ),
        initial_physical_state_hash=(
            None
            if checkpoint.get("initial_physical_state_hash") is None
            else str(checkpoint["initial_physical_state_hash"])
        ),
        checkpoint_format=checkpoint_format,
        metadata=metadata,
        scaler_state_dict=None if not isinstance(scaler_state, Mapping) else dict(scaler_state),
    )


def load_training_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    amp_scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    expected_metadata: CheckpointMetadata | None = None,
    expected_expert_type: PSFModality | str | None = None,
    expected_instance_id: str | None = None,
    expected_channel_id: str | None = None,
    expected_parent_checkpoint_hash: str | None = None,
) -> TrainingResumeState:
    """Backward-compatible name for :func:`resume_training_checkpoint`."""

    return resume_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=amp_scaler,
        map_location=map_location,
        expected_metadata=expected_metadata,
        expected_expert_type=expected_expert_type,
        expected_instance_id=expected_instance_id,
        expected_channel_id=expected_channel_id,
        expected_parent_checkpoint_hash=expected_parent_checkpoint_hash,
        _require_identity=any(
            value is not None
            for value in (
                expected_metadata,
                expected_expert_type,
                expected_instance_id,
                expected_channel_id,
                expected_parent_checkpoint_hash,
            )
        ),
    )


def initialize_model_from_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    expected_expert_type: PSFModality | str | None = None,
) -> ModelInitializationState:
    """Load only model weights, normally from a prototype checkpoint.

    Optimizer, scheduler, scaler, counters, physical state and RNG are
    intentionally ignored. This is the only supported path for starting a new
    instance from an existing prototype without inheriting its training state.
    """

    checkpoint_path = Path(path)
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    checkpoint_format = detect_checkpoint_format(checkpoint)
    metadata = _checkpoint_metadata(checkpoint, checkpoint_format)
    if checkpoint_format == "v2":
        if metadata is None:
            raise ValueError("v2 checkpoint metadata is required for weight initialization")
        if metadata.checkpoint_role != "prototype":
            raise ValueError("weight initialization requires a prototype checkpoint")
        if expected_expert_type is not None and metadata.expert_type != PSFModality.parse(expected_expert_type):
            raise ValueError(
                f"checkpoint expert_type {getattr(metadata.expert_type, 'value', metadata.expert_type)!r} "
                f"does not match expected {PSFModality.parse(expected_expert_type).value!r}"
            )
    model.load_state_dict(_model_state_dict(checkpoint, checkpoint_format), strict=True)
    return ModelInitializationState(
        path=checkpoint_path,
        checkpoint_format=checkpoint_format,
        metadata=metadata,
    )


def _checkpoint_metadata(
    checkpoint: Mapping[str, object], checkpoint_format: str,
) -> CheckpointMetadata | None:
    if checkpoint_format != "v2":
        return None
    value = checkpoint.get("metadata")
    if not isinstance(value, Mapping):
        raise ValueError("v2 checkpoint metadata must be a mapping")
    return CheckpointMetadata.from_dict(value)


def _model_state_dict(checkpoint: Mapping[str, object], checkpoint_format: str) -> Mapping[str, object]:
    key = "model_state_dict" if checkpoint_format != "legacy.v1" else "model_state"
    value = checkpoint.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is missing {key}")
    return value


def _validate_resume_identity(
    metadata: CheckpointMetadata,
    *,
    expected_metadata: CheckpointMetadata | None,
    expected_expert_type: PSFModality | str | None,
    expected_instance_id: str | None,
    expected_channel_id: str | None,
    expected_parent_checkpoint_hash: str | None,
) -> None:
    expected_expert = expected_expert_type
    expected_instance = expected_instance_id
    expected_channel = expected_channel_id
    expected_parent = expected_parent_checkpoint_hash
    if expected_metadata is not None:
        if not isinstance(expected_metadata, CheckpointMetadata):
            raise TypeError("expected_metadata must be CheckpointMetadata")
        if metadata.checkpoint_role != expected_metadata.checkpoint_role:
            raise ValueError(
                f"checkpoint role {metadata.checkpoint_role!r} does not match expected "
                f"{expected_metadata.checkpoint_role!r}"
            )
        expected_expert = expected_metadata.expert_type
        expected_instance = expected_metadata.instance_id
        expected_channel = None if expected_metadata.channel_spec is None else expected_metadata.channel_spec.channel_id
        expected_parent = expected_metadata.parent_checkpoint_hash
    if expected_expert is not None:
        normalized_expert = PSFModality.parse(expected_expert)
        if metadata.expert_type != normalized_expert:
            raise ValueError(
                f"checkpoint expert_type {getattr(metadata.expert_type, 'value', metadata.expert_type)!r} "
                f"does not match expected {normalized_expert.value!r}"
            )
    if expected_instance is not None and metadata.instance_id != str(expected_instance):
        raise ValueError(
            f"checkpoint instance_id {metadata.instance_id!r} does not match expected {expected_instance!r}"
        )
    actual_channel = None if metadata.channel_spec is None else metadata.channel_spec.channel_id
    if expected_channel is not None and actual_channel != str(expected_channel):
        raise ValueError(f"checkpoint channel_id {actual_channel!r} does not match expected {expected_channel!r}")
    if expected_parent is not None:
        normalized_parent = str(expected_parent).strip().lower()
        if metadata.parent_checkpoint_hash != normalized_parent:
            raise ValueError("checkpoint parent_checkpoint_hash does not match expected prototype hash")


def _checkpoint_physical_coeff_maps(checkpoint: Mapping[str, object]) -> tuple[dict[str, str], ...]:
    entries = checkpoint.get("physical_coeff_maps")
    if not isinstance(entries, (tuple, list)):
        return ()
    output = []
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        path = item.get("coeff_maps_npz")
        name = item.get("name")
        if path is None or name is None:
            continue
        output.append({"name": str(name), "coeff_maps_npz": str(path)})
    return tuple(output)
