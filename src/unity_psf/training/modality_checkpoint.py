"""Modality resume shards and joint release checkpoint publication."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from unity_psf.contracts import (
    ModalityChannelState,
    ModalityExpertState,
    ModalityJointCheckpointMetadata,
    PSFModality,
    load_modality_joint_checkpoint,
    save_modality_joint_checkpoint,
)
from unity_psf.training.checkpoint_state import build_training_state_snapshot, restore_rng_state
from unity_psf.training.modality_physical_state import (
    physical_provenance_registry,
    snapshot_channel_physical_state,
)
from unity_psf.training.modality_types import ModalityTrainingRuntime, ModalityTrainingShardState


def commit_modality_joint_checkpoint(
    path: str | Path,
    *,
    runtimes: Mapping[str, ModalityTrainingRuntime],
    completed_modalities: Sequence[PSFModality | str],
    role: str = "release",
    provenance: Mapping[str, Any] | None = None,
    training_progress: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Atomically publish one network per completed modality with nested channels."""

    normalized = {PSFModality.parse(key).value: runtime for key, runtime in runtimes.items()}
    if any(key != runtime.modality.value for key, runtime in normalized.items()):
        raise ValueError("runtime registry keys must match runtime modalities")
    completed = {PSFModality.parse(item).value for item in completed_modalities}
    if completed != set(normalized):
        raise ValueError("all configured modality runtimes must be completed before commit")
    selected_role = str(role).strip().lower()
    snapshots = {}
    training_state = None
    if selected_role == "resume":
        progress = dict(training_progress or {})
        if progress and set(progress) != set(normalized):
            raise ValueError("resume training_progress must cover every modality")
        snapshots = {
            modality: build_training_state_snapshot(
                model=runtime.model, optimizer=runtime.optimizer, scheduler=runtime.scheduler,
                amp_scaler=runtime.amp_scaler, progress=progress.get(modality),
            )
            for modality, runtime in normalized.items()
        }
        training_state = {
            modality: {key: value for key, value in snapshot.items() if key != "model_state_dict"}
            for modality, snapshot in snapshots.items()
        }
    elif training_progress is not None:
        raise ValueError("release checkpoint cannot carry training_progress")
    experts: list[ModalityExpertState] = []
    for modality, runtime in normalized.items():
        metadata_fn = getattr(runtime.model, "checkpoint_metadata", None)
        if not callable(metadata_fn):
            raise TypeError(f"modality expert {modality!r} must expose checkpoint_metadata()")
        model_metadata = metadata_fn()
        experts.append(
            ModalityExpertState(
                modality=modality,
                model_class=f"{type(runtime.model).__module__}.{type(runtime.model).__name__}",
                model_config=model_metadata.model_config,
                input_frame_spec=model_metadata.input_frame_spec,
                condition_schema=model_metadata.condition_schema,
                model_state_dict=(
                    snapshots[modality]["model_state_dict"] if selected_role == "resume" else runtime.model.state_dict()
                ),
                channel_states=tuple(
                    ModalityChannelState(
                        channel_id=channel_id,
                        physical_state=snapshot_channel_physical_state(stream),
                        calibration=stream.calibration,
                        provenance=stream.provenance,
                    )
                    for channel_id, stream in runtime.channels.items()
                ),
                provenance={"training_unit": modality},
            )
        )
    checkpoint_provenance = copy.deepcopy(dict(provenance or {}))
    checkpoint_provenance.setdefault(
        "physical_provenance",
        physical_provenance_registry(normalized),
    )
    return save_modality_joint_checkpoint(
        path,
        metadata=ModalityJointCheckpointMetadata(
            checkpoint_role=selected_role,
            supported_modalities=tuple(runtime.modality for runtime in normalized.values()),
            supported_channels_per_modality={
                modality: tuple(runtime.channels) for modality, runtime in normalized.items()
            },
        ),
        experts=experts,
        provenance=checkpoint_provenance,
        training_state=training_state,
    )


def save_modality_training_shard(
    path: str | Path,
    *,
    runtime: ModalityTrainingRuntime,
    epoch: int,
    optimizer_steps: int,
    channel_progress: Mapping[str, Mapping[str, Any]],
    status: str,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save one atomically replaceable resume checkpoint for a modality rank."""

    normalized_status = str(status).strip().lower()
    if normalized_status not in {"running", "complete"}:
        raise ValueError("modality shard status must be running or complete")
    if int(epoch) < 0 or int(optimizer_steps) < 0:
        raise ValueError("epoch and optimizer_steps must be non-negative")
    progress = {
        str(channel_id): copy.deepcopy(dict(value))
        for channel_id, value in channel_progress.items()
    }
    if set(progress) != set(runtime.channels):
        raise ValueError("channel_progress must exactly match the runtime channels")
    modality = runtime.modality.value
    return commit_modality_joint_checkpoint(
        path,
        runtimes={modality: runtime},
        completed_modalities=(modality,),
        role="resume",
        provenance=provenance,
        training_progress={
            modality: {
                "epoch": int(epoch),
                "optimizer_steps": int(optimizer_steps),
                "channel_progress": progress,
                "status": normalized_status,
            }
        },
    )


def restore_modality_training_shard(
    path: str | Path,
    *,
    runtime: ModalityTrainingRuntime,
    restore_rng: bool = True,
    expected_provenance: Mapping[str, Any] | None = None,
) -> ModalityTrainingShardState:
    """Restore one modality model, optimizer, scheduler, channels, and progress."""

    checkpoint_path = Path(path)
    payload = load_modality_joint_checkpoint(checkpoint_path)
    metadata = ModalityJointCheckpointMetadata.from_dict(payload["metadata"])
    if metadata.checkpoint_role != "resume":
        raise ValueError("modality training restore requires a resume checkpoint")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("modality resume checkpoint provenance must be a mapping")
    for key, expected in dict(expected_provenance or {}).items():
        if provenance.get(key) != expected:
            raise ValueError(f"resume checkpoint provenance mismatch for {key!r}")
    modality = runtime.modality.value
    if set(payload["experts"]) != {modality}:
        raise ValueError(f"resume checkpoint modality does not match {modality!r}")
    saved_channels = payload["channel_states"][modality]
    if set(saved_channels) != set(runtime.channels):
        raise ValueError("resume checkpoint channel inventory does not match the runtime")
    expert = ModalityExpertState.from_payload(
        modality,
        payload["experts"][modality],
        saved_channels,
    )
    runtime.model.load_state_dict(expert.model_state_dict, strict=True)
    state = payload["training_state"][modality]
    optimizer_state = state.get("optimizer")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("modality resume checkpoint is missing optimizer state")
    runtime.optimizer.load_state_dict(optimizer_state)
    scheduler_state = state.get("scheduler")
    if runtime.scheduler is None and scheduler_state is not None:
        raise ValueError("checkpoint has scheduler state but runtime has no scheduler")
    if runtime.scheduler is not None:
        if not isinstance(scheduler_state, Mapping):
            raise ValueError("runtime scheduler requires scheduler state in the checkpoint")
        runtime.scheduler.load_state_dict(scheduler_state)
    scaler_state = state.get("amp_scaler")
    if scaler_state is not None:
        if not isinstance(scaler_state, Mapping):
            raise ValueError("checkpoint AMP scaler state must be a mapping")
        scaler = runtime.amp_scaler
        if scaler is None:
            scaler_device = next(runtime.model.parameters()).device.type
            scaler = torch.amp.GradScaler(
                device=scaler_device,
                enabled=runtime.amp_enabled,
            )
            object.__setattr__(runtime, "amp_scaler", scaler)
        scaler.load_state_dict(dict(scaler_state))
    for channel_id, stream in runtime.channels.items():
        saved = saved_channels[channel_id]
        saved_physical_state = dict(saved.get("physical_state", {}))
        if stream.restore_physical_state is not None:
            stream.restore_physical_state(saved_physical_state)
        restored_physical_state = (
            snapshot_channel_physical_state(stream)
            if stream.snapshot_physical_state is not None
            else saved_physical_state
        )
        saved_version = saved_physical_state.get("condition_store_version")
        restored_version = restored_physical_state.get("condition_store_version")
        if saved_version is not None and restored_version != saved_version:
            raise ValueError(
                f"runtime channel {channel_id!r} condition_store_version "
                f"{restored_version!r} does not match checkpoint {saved_version!r}"
            )
        restored_fields = {
            "physical_state": restored_physical_state,
            "calibration": saved.get("calibration", {}),
            "provenance": saved.get("provenance", {}),
        }
        for field_name, restored_value in restored_fields.items():
            target = getattr(stream, field_name)
            if not isinstance(target, dict):
                raise TypeError(f"runtime channel {field_name} must be mutable for resume")
            target.clear()
            target.update(copy.deepcopy(dict(restored_value)))
    progress = state.get("channel_progress")
    if not isinstance(progress, Mapping) or set(progress) != set(runtime.channels):
        raise ValueError("resume checkpoint channel progress does not match the runtime")
    status = str(state.get("status", "")).strip().lower()
    if status not in {"running", "complete"}:
        raise ValueError("resume checkpoint has invalid modality status")
    if restore_rng:
        restore_rng_state(state.get("rng_state"))
    return ModalityTrainingShardState(
        path=checkpoint_path,
        modality=runtime.modality,
        epoch=int(state["epoch"]),
        optimizer_steps=int(state["optimizer_steps"]),
        channel_progress={
            str(channel_id): copy.deepcopy(dict(value))
            for channel_id, value in progress.items()
        },
        status=status,
    )


def assemble_modality_joint_checkpoint_from_shards(
    path: str | Path,
    *,
    shard_paths: Sequence[str | Path],
    required_modalities: Sequence[PSFModality | str],
    provenance: Mapping[str, Any] | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Publish a release only after every required modality shard is complete."""

    required = tuple(PSFModality.parse(item) for item in required_modalities)
    if not required or len(required) != len(set(required)):
        raise ValueError("required_modalities must be non-empty and unique")
    experts: dict[str, ModalityExpertState] = {}
    for raw_path in shard_paths:
        payload = load_modality_joint_checkpoint(raw_path)
        saved_provenance = payload.get("provenance", {})
        if not isinstance(saved_provenance, Mapping):
            raise ValueError("modality shard provenance must be a mapping")
        for key, expected_value in dict(expected_provenance or {}).items():
            if saved_provenance.get(key) != expected_value:
                raise ValueError(f"modality shard provenance mismatch for {key!r}")
        metadata = ModalityJointCheckpointMetadata.from_dict(payload["metadata"])
        if metadata.checkpoint_role != "resume" or len(payload["experts"]) != 1:
            raise ValueError("each modality shard must be a single-modality resume checkpoint")
        modality = next(iter(payload["experts"]))
        state = payload["training_state"][modality]
        if state.get("status") != "complete":
            raise ValueError(f"modality shard {modality!r} is not complete")
        if modality in experts:
            raise ValueError(f"duplicate modality shard {modality!r}")
        experts[modality] = ModalityExpertState.from_payload(
            modality,
            payload["experts"][modality],
            payload["channel_states"][modality],
        )
    expected = {item.value for item in required}
    if set(experts) != expected:
        raise ValueError("completed modality shards do not match required_modalities")
    checkpoint_provenance = copy.deepcopy(dict(provenance or {}))
    checkpoint_provenance.setdefault(
        "physical_provenance",
        {
            modality: {
                state.channel_id: copy.deepcopy(
                    dict(state.calibration).get("physical_provenance")
                )
                for state in expert.channel_states
                if dict(state.calibration).get("physical_provenance") is not None
            }
            for modality, expert in experts.items()
        },
    )
    return save_modality_joint_checkpoint(
        path,
        metadata=ModalityJointCheckpointMetadata(
            checkpoint_role="release",
            supported_modalities=required,
            supported_channels_per_modality={
                modality: tuple(channel.channel_id for channel in experts[modality].channel_states)
                for modality in experts
            },
        ),
        experts=(experts[item.value] for item in required),
        provenance=checkpoint_provenance,
    )
