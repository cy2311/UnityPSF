"""Data contracts and runtime ownership for modality training."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from unity_psf.contracts import PSFModality
from unity_psf.training.step_progress import validate_scheduler_step_unit


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
        validate_scheduler_step_unit(self.scheduler_step_unit)
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
    optimizer_skip_reasons_by_channel: Mapping[str, Mapping[str, int]] = field(default_factory=dict)


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
