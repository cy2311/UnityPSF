from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from unity_psf.contracts.checkpoint import CheckpointMetadata
from unity_psf.training.step_progress import validate_scheduler_step_unit


@dataclass(frozen=True)
class TrainingBatch:
    inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class TrainingConfig:
    epoch: int
    checkpoint_name: str = "checkpoint_latest.pt"
    metrics_name: str = "training_metrics.jsonl"
    global_step_start: int = 0
    grad_clip_norm: float | None = None
    amp_enabled: bool = False
    amp_dtype: torch.dtype = torch.float16
    amp_scaler: torch.amp.GradScaler | None = None
    scheduler_step_unit: str = "optimizer_step"
    checkpoint_extra_fn: CheckpointExtraFn | None = None
    checkpoint_metadata: CheckpointMetadata | None = None

    def __post_init__(self) -> None:
        validate_scheduler_step_unit(self.scheduler_step_unit)


@dataclass(frozen=True)
class TrainingEpochResult:
    epoch: int
    step_count: int
    mean_loss: float
    metrics_path: Path
    checkpoint_path: Path
    global_step: int
    optimizer_steps: int = 0
    skipped_optimizer_steps: int = 0

    @property
    def attempted_optimizer_steps(self) -> int:
        return self.step_count


@dataclass(frozen=True)
class TrainingStepResult:
    epoch: int
    step_count: int
    global_step: int
    loss: float
    metrics_path: Path
    checkpoint_path: Path | None = None
    optimizer_step_performed: bool = True


@dataclass(frozen=True)
class EpochTrainingConfig:
    start_epoch: int
    stop_epoch: int
    metrics_name: str = "training_metrics.jsonl"
    global_step_start: int = 0
    max_batches: int | None = None
    eval_interval_epochs: int = 1
    grad_clip_norm: float | None = None
    amp_enabled: bool = False
    amp_dtype: torch.dtype = torch.float16
    amp_scaler: torch.amp.GradScaler | None = None
    scaler_state_dict: dict[str, object] | None = None
    scheduler_step_unit: str = "optimizer_step"
    checkpoint_metadata: CheckpointMetadata | None = None

    def __post_init__(self) -> None:
        validate_scheduler_step_unit(self.scheduler_step_unit)


@dataclass(frozen=True)
class TrainingRunEpochResult:
    epoch: int
    step_count: int
    mean_loss: float
    metrics_path: Path
    checkpoint_path: Path
    global_step: int
    eval_loss: float | None = None
    best_checkpoint_path: Path | None = None
    optimizer_steps: int = 0
    skipped_optimizer_steps: int = 0

    @property
    def attempted_optimizer_steps(self) -> int:
        return self.step_count


@dataclass(frozen=True)
class TrainingResumeState:
    epoch: int
    step_count: int
    global_step: int
    path: Path
    physical_state: dict[str, object] | None = None
    physical_coeff_maps: tuple[dict[str, str], ...] = ()
    physical_state_hash: str | None = None
    initial_physical_state_hash: str | None = None
    checkpoint_format: str = "legacy.training"
    metadata: CheckpointMetadata | None = None
    scaler_state_dict: dict[str, object] | None = None


@dataclass(frozen=True)
class ModelInitializationState:
    """Result of loading model weights without training-state restoration."""

    path: Path
    checkpoint_format: str
    metadata: CheckpointMetadata | None = None


BatchProvider = Callable[[int], Iterable[TrainingBatch]]
EvalProvider = Callable[[], Iterable[TrainingBatch]]
EpochEndHook = Callable[[TrainingRunEpochResult], None]
BatchEndHook = Callable[[TrainingStepResult], None]
LossFn = Callable[[torch.nn.Module, TrainingBatch], torch.Tensor]
CheckpointExtraFn = Callable[[], dict[str, object] | None]


__all__ = [
    "BatchEndHook",
    "BatchProvider",
    "CheckpointExtraFn",
    "EpochEndHook",
    "EpochTrainingConfig",
    "EvalProvider",
    "LossFn",
    "ModelInitializationState",
    "TrainingBatch",
    "TrainingConfig",
    "TrainingEpochResult",
    "TrainingResumeState",
    "TrainingRunEpochResult",
    "TrainingStepResult",
]
