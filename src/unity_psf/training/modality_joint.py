"""Shared-expert, multichannel training for one or more PSF modalities."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
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
from unity_psf.training.loop import _capture_rng_state, _restore_rng_state


@dataclass(frozen=True)
class ModalityTrainingBatch:
    """One homogeneous channel batch routed inside a selected modality expert."""

    images: torch.Tensor
    target: Any
    channel_id: str
    conditions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.images, torch.Tensor) or self.images.ndim != 4:
            raise ValueError("images must have shape (N,C,H,W)")
        channel_id = str(self.channel_id).strip()
        if not channel_id or ":" in channel_id:
            raise ValueError("channel_id must be a non-empty identifier without ':'")
        if self.conditions is not None and (
            self.conditions.ndim != 2 or self.conditions.shape[0] != self.images.shape[0]
        ):
            raise ValueError("condition batch size must match image batch size")
        object.__setattr__(self, "channel_id", channel_id)


ModalityBatchProvider = Callable[[int], Iterable[ModalityTrainingBatch]]
ModalityLossFn = Callable[[torch.Tensor, ModalityTrainingBatch], torch.Tensor]
PhysicalStateSnapshot = Callable[[], Mapping[str, Any]]
PhysicalStateRestore = Callable[[Mapping[str, Any]], None]
HeldoutEvalFn = Callable[[torch.nn.Module], Mapping[str, Any]]


@dataclass(frozen=True)
class ModalityChannelStream:
    """Data, loss, and physical state owned by one measurement channel."""

    channel_id: str
    batches: ModalityBatchProvider
    loss_fn: ModalityLossFn
    physical_state: Mapping[str, Any]
    calibration: Mapping[str, Any]
    provenance: Mapping[str, Any]
    snapshot_physical_state: PhysicalStateSnapshot | None = None
    restore_physical_state: PhysicalStateRestore | None = None
    heldout_eval: HeldoutEvalFn | None = None

    def __post_init__(self) -> None:
        channel_id = str(self.channel_id).strip()
        if not channel_id or ":" in channel_id:
            raise ValueError("channel_id must be a non-empty identifier without ':'")
        if not callable(self.batches) or not callable(self.loss_fn):
            raise TypeError("batches and loss_fn must be callable")
        if self.snapshot_physical_state is not None and not callable(
            self.snapshot_physical_state
        ):
            raise TypeError("snapshot_physical_state must be callable")
        if self.restore_physical_state is not None and not callable(
            self.restore_physical_state
        ):
            raise TypeError("restore_physical_state must be callable")
        if self.heldout_eval is not None and not callable(self.heldout_eval):
            raise TypeError("heldout_eval must be callable")
        object.__setattr__(self, "channel_id", channel_id)
        for field_name in ("physical_state", "calibration", "provenance"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            object.__setattr__(self, field_name, copy.deepcopy(dict(value)))


@dataclass(frozen=True)
class ModalityTrainingRuntime:
    """Exactly one trainable network and optimizer for all channels of a modality."""

    modality: PSFModality | str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    channels: Mapping[str, ModalityChannelStream]
    step_budgets: Mapping[str, int]
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    scheduler_step_unit: str = "optimizer_step"
    grad_clip_norm: float | None = None
    amp_enabled: bool = False
    amp_dtype: torch.dtype = torch.float16
    amp_scaler: torch.amp.GradScaler | None = None

    def __post_init__(self) -> None:
        modality = PSFModality.parse(self.modality)
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        channels = dict(self.channels)
        if not channels or any(key != stream.channel_id for key, stream in channels.items()):
            raise ValueError("channel registry keys must match non-empty channel streams")
        budgets = {str(key): int(value) for key, value in self.step_budgets.items()}
        if set(budgets) != set(channels) or any(value <= 0 for value in budgets.values()):
            raise ValueError("every enabled channel requires a positive step budget")
        if self.scheduler_step_unit not in {"optimizer_step", "epoch"}:
            raise ValueError("scheduler_step_unit must be optimizer_step or epoch")
        model_parameters = {id(parameter) for parameter in self.model.parameters()}
        optimizer_parameters = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_parameters != model_parameters:
            raise ValueError("optimizer must own exactly the modality model parameters")
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "step_budgets", budgets)

    def balanced_schedule(self) -> tuple[str, ...]:
        remaining = dict(self.step_budgets)
        schedule: list[str] = []
        while any(value > 0 for value in remaining.values()):
            for channel_id in self.channels:
                if remaining[channel_id] > 0:
                    schedule.append(channel_id)
                    remaining[channel_id] -= 1
        return tuple(schedule)


@dataclass(frozen=True)
class ModalityEpochResult:
    epoch: int
    modality: PSFModality
    schedule: tuple[str, ...]
    step_counts: Mapping[str, int]
    sample_counts: Mapping[str, int]
    losses_by_channel: Mapping[str, tuple[float, ...]]
    optimizer_steps_by_channel: Mapping[str, int]
    skipped_optimizer_steps_by_channel: Mapping[str, int]
    attempted_optimizer_steps: int
    optimizer_steps: int
    skipped_optimizer_steps: int
    mean_loss: float


@dataclass(frozen=True)
class ModalityTrainingShardState:
    path: Path
    modality: PSFModality
    epoch: int
    optimizer_steps: int
    channel_progress: Mapping[str, Mapping[str, Any]]
    status: str


@dataclass(frozen=True)
class ModalityHeldoutEvalResult:
    modality: Mapping[str, Any]
    channels: Mapping[str, Mapping[str, Any]]
    artifacts: Mapping[str, Mapping[str, Any]]


def train_modality_epoch(*, runtime: ModalityTrainingRuntime, epoch: int) -> ModalityEpochResult:
    """Round-robin channel batches through one shared modality expert."""

    schedule = runtime.balanced_schedule()
    iterators = {
        channel_id: iter(stream.batches(int(epoch)))
        for channel_id, stream in runtime.channels.items()
    }
    step_counts = {channel_id: 0 for channel_id in runtime.channels}
    sample_counts = {channel_id: 0 for channel_id in runtime.channels}
    losses: dict[str, list[float]] = {channel_id: [] for channel_id in runtime.channels}
    optimizer_steps_by_channel = {channel_id: 0 for channel_id in runtime.channels}
    device_type = next(runtime.model.parameters()).device.type
    autocast_enabled = bool(runtime.amp_enabled and device_type == "cuda")
    scaler = runtime.amp_scaler
    if scaler is None:
        scaler = torch.amp.GradScaler(device="cuda", enabled=autocast_enabled)
        object.__setattr__(runtime, "amp_scaler", scaler)
    optimizer_steps = 0
    runtime.model.train()
    for channel_id in schedule:
        stream = runtime.channels[channel_id]
        try:
            batch = next(iterators[channel_id])
        except StopIteration as exc:
            raise ValueError(
                f"training data for channel {channel_id!r} ended before its step budget"
            ) from exc
        if not isinstance(batch, ModalityTrainingBatch):
            raise TypeError("channel batches must be ModalityTrainingBatch values")
        if batch.channel_id != channel_id:
            raise ValueError(
                f"scheduled channel {channel_id!r} received batch for {batch.channel_id!r}"
            )
        runtime.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=runtime.amp_dtype,
            enabled=autocast_enabled,
        ):
            output = (
                runtime.model(batch.images, batch.conditions)
                if batch.conditions is not None
                else runtime.model(batch.images)
            )
            loss = stream.loss_fn(output, batch)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError("channel loss function must return one scalar tensor")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError(
                f"non-finite loss for {runtime.modality.value}:{channel_id} at epoch {epoch}"
            )
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(runtime.optimizer)
        else:
            loss.backward()
        if runtime.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), runtime.grad_clip_norm)
        if scaler.is_enabled():
            scale_before_step = scaler.get_scale()
            scaler.step(runtime.optimizer)
            scaler.update()
            optimizer_step_performed = scaler.get_scale() >= scale_before_step
        else:
            runtime.optimizer.step()
            optimizer_step_performed = True
        if optimizer_step_performed:
            optimizer_steps += 1
            optimizer_steps_by_channel[channel_id] += 1
        if (
            optimizer_step_performed
            and runtime.scheduler is not None
            and runtime.scheduler_step_unit == "optimizer_step"
        ):
            runtime.scheduler.step()
        loss_value = float(loss.detach().cpu().item())
        step_counts[channel_id] += 1
        sample_counts[channel_id] += int(batch.images.shape[0])
        losses[channel_id].append(loss_value)
    if (
        optimizer_steps > 0
        and runtime.scheduler is not None
        and runtime.scheduler_step_unit == "epoch"
    ):
        runtime.scheduler.step()
    flat_losses = [value for values in losses.values() for value in values]
    return ModalityEpochResult(
        epoch=int(epoch),
        modality=runtime.modality,
        schedule=schedule,
        step_counts=step_counts,
        sample_counts=sample_counts,
        losses_by_channel={key: tuple(values) for key, values in losses.items()},
        optimizer_steps_by_channel=optimizer_steps_by_channel,
        skipped_optimizer_steps_by_channel={
            channel_id: step_counts[channel_id] - optimizer_steps_by_channel[channel_id]
            for channel_id in runtime.channels
        },
        attempted_optimizer_steps=len(schedule),
        optimizer_steps=optimizer_steps,
        skipped_optimizer_steps=len(schedule) - optimizer_steps,
        mean_loss=sum(flat_losses) / len(flat_losses),
    )


def evaluate_modality_heldout(
    runtime: ModalityTrainingRuntime,
) -> ModalityHeldoutEvalResult:
    """Evaluate every channel and recompute modality metrics from sufficient statistics."""

    missing = [
        channel_id
        for channel_id, stream in runtime.channels.items()
        if stream.heldout_eval is None
    ]
    if missing:
        raise ValueError(f"held-out eval is not configured for channels: {missing}")
    was_training = runtime.model.training
    channel_metrics: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Mapping[str, Any]] = {}
    try:
        runtime.model.eval()
        for channel_id, stream in runtime.channels.items():
            assert stream.heldout_eval is not None
            result = stream.heldout_eval(runtime.model)
            if not isinstance(result, Mapping):
                raise TypeError("held-out eval callback must return a mapping")
            metrics = result.get("metrics")
            artifact = result.get("artifacts", {})
            if not isinstance(metrics, Mapping) or not isinstance(artifact, Mapping):
                raise TypeError("held-out eval result requires metrics and artifacts mappings")
            channel_metrics[channel_id] = _normalize_heldout_metrics(
                metrics,
                modality=runtime.modality,
            )
            artifacts[channel_id] = artifact
    finally:
        if was_training:
            runtime.model.train()
    return ModalityHeldoutEvalResult(
        modality=_aggregate_heldout_metrics(
            channel_metrics.values(),
            modality=runtime.modality,
        ),
        channels=channel_metrics,
        artifacts=artifacts,
    )


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
                model_state_dict=runtime.model.state_dict(),
                channel_states=tuple(
                    ModalityChannelState(
                        channel_id=channel_id,
                        physical_state=_snapshot_channel_physical_state(stream),
                        calibration=stream.calibration,
                        provenance=stream.provenance,
                    )
                    for channel_id, stream in runtime.channels.items()
                ),
                provenance={"training_unit": modality},
            )
        )
    selected_role = str(role).strip().lower()
    training_state = None
    if selected_role == "resume":
        progress = dict(training_progress or {})
        if progress and set(progress) != set(normalized):
            raise ValueError("resume training_progress must cover every modality")
        training_state = {
            modality: {
                **copy.deepcopy(dict(progress.get(modality, {}))),
                "optimizer": runtime.optimizer.state_dict(),
                "scheduler": (
                    None if runtime.scheduler is None else runtime.scheduler.state_dict()
                ),
                "amp_scaler": (
                    None if runtime.amp_scaler is None else runtime.amp_scaler.state_dict()
                ),
            }
            for modality, runtime in normalized.items()
        }
    elif training_progress is not None:
        raise ValueError("release checkpoint cannot carry training_progress")
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
        provenance=provenance,
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
                "rng_state": _capture_rng_state(),
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
            _snapshot_channel_physical_state(stream)
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
        _restore_rng_state(state.get("rng_state"))
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
) -> Path:
    """Publish a release only after every required modality shard is complete."""

    required = tuple(PSFModality.parse(item) for item in required_modalities)
    if not required or len(required) != len(set(required)):
        raise ValueError("required_modalities must be non-empty and unique")
    experts: dict[str, ModalityExpertState] = {}
    for raw_path in shard_paths:
        payload = load_modality_joint_checkpoint(raw_path)
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
        provenance=provenance,
    )


def _snapshot_channel_physical_state(
    stream: ModalityChannelStream,
) -> dict[str, Any]:
    state = (
        stream.physical_state
        if stream.snapshot_physical_state is None
        else stream.snapshot_physical_state()
    )
    if not isinstance(state, Mapping):
        raise TypeError("channel physical state snapshot must be a mapping")
    snapshot = copy.deepcopy(dict(state))
    if isinstance(stream.physical_state, dict):
        stream.physical_state.clear()
        stream.physical_state.update(copy.deepcopy(snapshot))
    return snapshot


def _normalize_heldout_metrics(
    metrics: Mapping[str, Any],
    *,
    modality: PSFModality,
) -> dict[str, Any]:
    required = (
        "eval_loss",
        "true_positive",
        "false_positive",
        "false_negative",
        "lateral_sq_error_nm2_sum",
        "axial_sq_error_nm2_sum",
        "photon_relative_error_sum",
        "matched_photons",
        "sample_count",
        "route_count",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"held-out eval metrics are missing {missing}")
    result = dict(metrics)
    true_positive = int(result["true_positive"])
    false_positive = int(result["false_positive"])
    false_negative = int(result["false_negative"])
    matched_photons = int(result["matched_photons"])
    result.update(
        {
            "eval_loss": float(result["eval_loss"]),
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
            "Jaccard": true_positive
            / max(true_positive + false_positive + false_negative, 1),
            "RMSE_XY_nm": None
            if true_positive == 0
            else (
                float(result["lateral_sq_error_nm2_sum"])
                / true_positive
            )
            ** 0.5,
            "RMSE_Z_nm": None
            if modality is PSFModality.EMITTER_2D or true_positive == 0
            else (
                float(result["axial_sq_error_nm2_sum"])
                / true_positive
            )
            ** 0.5,
            "photon_relative_error": None
            if matched_photons == 0
            else float(result["photon_relative_error_sum"]) / matched_photons,
            "sample_count": int(result["sample_count"]),
            "route_count": int(result["route_count"]),
        }
    )
    return result


def _aggregate_heldout_metrics(
    channels: Iterable[Mapping[str, Any]],
    *,
    modality: PSFModality,
) -> dict[str, Any]:
    values = list(channels)
    sample_count = sum(int(item["sample_count"]) for item in values)
    totals = {
        key: sum(float(item[key]) for item in values)
        for key in (
            "true_positive",
            "false_positive",
            "false_negative",
            "lateral_sq_error_nm2_sum",
            "axial_sq_error_nm2_sum",
            "photon_relative_error_sum",
            "matched_photons",
            "route_count",
        )
    }
    totals.update(
        {
            "eval_loss": sum(
                float(item["eval_loss"]) * int(item["sample_count"])
                for item in values
            )
            / max(sample_count, 1),
            "sample_count": sample_count,
        }
    )
    return _normalize_heldout_metrics(totals, modality=modality)


__all__ = [
    "ModalityChannelStream",
    "ModalityEpochResult",
    "ModalityHeldoutEvalResult",
    "ModalityTrainingBatch",
    "ModalityTrainingRuntime",
    "ModalityTrainingShardState",
    "assemble_modality_joint_checkpoint_from_shards",
    "commit_modality_joint_checkpoint",
    "evaluate_modality_heldout",
    "restore_modality_training_shard",
    "save_modality_training_shard",
    "train_modality_epoch",
]
