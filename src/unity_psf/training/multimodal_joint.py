"""Joint dual-modality training control plane for UnityPSF."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from unity_psf.contracts.joint_checkpoint import (
    JointCheckpointMetadata,
    JointExpertKey,
    JointExpertState,
    save_joint_checkpoint,
)
from unity_psf.contracts.modality import PSFModality

if TYPE_CHECKING:
    from unity_psf.models import UnityPSF


DUAL_MODALITY_INSTANCE_KEYS = (
    "emitter_2d:main",
    "astigmatism:left",
    "astigmatism:right",
)

DUAL_MODALITY_DUAL_CHANNEL_INSTANCE_KEYS = (
    "emitter_2d:left",
    "emitter_2d:right",
    "astigmatism:left",
    "astigmatism:right",
)


@dataclass(frozen=True)
class RoutedTrainingBatch:
    images: torch.Tensor
    target: Any
    conditions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.images, torch.Tensor) or self.images.ndim != 4:
            raise ValueError("images must have shape (N,C,H,W)")
        if self.conditions is not None:
            if self.conditions.ndim != 2 or self.conditions.shape[0] != self.images.shape[0]:
                raise ValueError("condition batch size must match image batch size")


JointLossFn = Callable[[torch.Tensor, RoutedTrainingBatch], torch.Tensor]


@dataclass(frozen=True)
class ExpertTrainingUnit:
    optimizer: torch.optim.Optimizer
    batches: Iterable[RoutedTrainingBatch]
    loss_fn: JointLossFn
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None


@dataclass(frozen=True)
class MultimodalTrainingPlan:
    """Fixed per-instance step budgets sharing one parent epoch definition."""

    instance_keys: tuple[str, ...]
    step_budgets: Mapping[str, int]

    def __post_init__(self) -> None:
        parsed = tuple(JointExpertKey.parse(key).storage_key for key in self.instance_keys)
        if not parsed or len(parsed) != len(set(parsed)):
            raise ValueError("instance_keys must be non-empty and unique")
        budgets = {str(key): int(value) for key, value in self.step_budgets.items()}
        if set(budgets) != set(parsed):
            raise ValueError("step_budgets must exactly match instance_keys")
        if any(value < 0 for value in budgets.values()) or not any(value > 0 for value in budgets.values()):
            raise ValueError("step budgets must be non-negative and at least one must be positive")
        object.__setattr__(self, "instance_keys", parsed)
        object.__setattr__(self, "step_budgets", budgets)

    @classmethod
    def dual_modality(cls, *, step_budgets: Mapping[str, int]) -> "MultimodalTrainingPlan":
        return cls(instance_keys=DUAL_MODALITY_INSTANCE_KEYS, step_budgets=step_budgets)

    @classmethod
    def dual_modality_dual_channel(cls, *, step_budgets: Mapping[str, int]) -> "MultimodalTrainingPlan":
        return cls(instance_keys=DUAL_MODALITY_DUAL_CHANNEL_INSTANCE_KEYS, step_budgets=step_budgets)

    @classmethod
    def from_step_budgets(cls, step_budgets: Mapping[str, int]) -> "MultimodalTrainingPlan":
        return cls(instance_keys=tuple(step_budgets), step_budgets=step_budgets)

    def round_robin_schedule(self) -> tuple[str, ...]:
        remaining = dict(self.step_budgets)
        schedule: list[str] = []
        while any(value > 0 for value in remaining.values()):
            for key in self.instance_keys:
                if remaining[key] <= 0:
                    continue
                schedule.append(key)
                remaining[key] -= 1
        return tuple(schedule)

    @property
    def modalities(self) -> tuple[PSFModality, ...]:
        present = {JointExpertKey.parse(key).modality for key in self.instance_keys}
        return tuple(item for item in PSFModality if item in present)


@dataclass(frozen=True)
class ExpertParallelPlan:
    """Rank ownership contract; execution strategy does not alter model identity."""

    assignments: Mapping[int, tuple[str, ...]]
    required_instances: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = {
            int(rank): tuple(JointExpertKey.parse(key).storage_key for key in keys)
            for rank, keys in self.assignments.items()
        }
        required = tuple(JointExpertKey.parse(key).storage_key for key in self.required_instances)
        if any(rank < 0 for rank in normalized):
            raise ValueError("expert-parallel ranks must be non-negative")
        flat = tuple(key for keys in normalized.values() for key in keys)
        if len(flat) != len(set(flat)) or set(flat) != set(required):
            raise ValueError("expert-parallel assignments must own every required instance exactly once")
        object.__setattr__(self, "assignments", normalized)
        object.__setattr__(self, "required_instances", required)

    @classmethod
    def one_instance_per_rank(cls, plan: MultimodalTrainingPlan) -> "ExpertParallelPlan":
        return cls(
            assignments={rank: (key,) for rank, key in enumerate(plan.instance_keys)},
            required_instances=plan.instance_keys,
        )

    @classmethod
    def one_modality_per_rank(cls, plan: MultimodalTrainingPlan) -> "ExpertParallelPlan":
        assignments: dict[int, tuple[str, ...]] = {}
        rank = 0
        for modality in PSFModality:
            owned = tuple(
                key for key in plan.instance_keys
                if JointExpertKey.parse(key).modality is modality
            )
            if owned:
                assignments[rank] = owned
                rank += 1
        return cls(assignments=assignments, required_instances=plan.instance_keys)

    def rank_for(self, instance_key: str) -> int:
        key = JointExpertKey.parse(instance_key).storage_key
        for rank, owned in self.assignments.items():
            if key in owned:
                return rank
        raise ValueError(f"instance {key!r} has no expert-parallel owner")

    def validate_completed_ranks(self, statuses: Mapping[int, str]) -> None:
        completed = {int(rank) for rank, status in statuses.items() if str(status) == "complete"}
        missing = set(self.assignments).difference(completed)
        if missing:
            raise ValueError(f"missing completed ranks: {sorted(missing)}")


@dataclass(frozen=True)
class JointEpochResult:
    epoch: int
    step_counts: Mapping[str, int]
    losses_by_instance: Mapping[str, tuple[float, ...]]
    schedule: tuple[str, ...]

    @property
    def total_steps(self) -> int:
        return sum(self.step_counts.values())


def train_round_robin_epoch(
    *,
    model: "UnityPSF",
    plan: MultimodalTrainingPlan,
    units: Mapping[str, ExpertTrainingUnit],
    epoch: int,
) -> JointEpochResult:
    """Train homogeneous microbatches while preserving optimizer ownership."""

    from unity_psf.models import UnityPSF

    if not isinstance(model, UnityPSF):
        raise TypeError("model must be UnityPSF")
    if set(units) != set(plan.instance_keys):
        raise ValueError("training units must exactly match the multimodal plan")
    if not set(plan.instance_keys).issubset(model.experts.keys()):
        raise ValueError("UnityPSF expert registry does not satisfy the multimodal plan")
    iterators = {key: iter(units[key].batches) for key in plan.instance_keys}
    counts = {key: 0 for key in plan.instance_keys}
    losses: dict[str, list[float]] = {key: [] for key in plan.instance_keys}
    schedule = plan.round_robin_schedule()
    model.train()
    for key in schedule:
        try:
            batch = next(iterators[key])
        except StopIteration as exc:
            raise ValueError(f"training data for {key!r} ended before its step budget") from exc
        if not isinstance(batch, RoutedTrainingBatch):
            raise TypeError("expert training batches must be RoutedTrainingBatch")
        selected = JointExpertKey.parse(key)
        unit = units[key]
        model.zero_grad(set_to_none=True)
        unit.optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.images,
            modality=selected.modality,
            channel_id=selected.channel_id,
            conditions=batch.conditions,
        )
        loss = unit.loss_fn(output, batch)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError("joint loss function must return one scalar tensor")
        loss.backward()
        unit.optimizer.step()
        if unit.scheduler is not None:
            unit.scheduler.step()
        counts[key] += 1
        losses[key].append(float(loss.detach().cpu().item()))
    return JointEpochResult(
        epoch=int(epoch),
        step_counts=counts,
        losses_by_instance={key: tuple(values) for key, values in losses.items()},
        schedule=schedule,
    )


def commit_joint_training_checkpoint(
    path: str | Path,
    *,
    model: "UnityPSF",
    plan: MultimodalTrainingPlan,
    completed_instances: Sequence[str],
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler | None] | None = None,
    role: str = "release",
    training_progress: Mapping[str, Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Commit only after every required expert instance reaches the barrier."""

    completed = {JointExpertKey.parse(key).storage_key for key in completed_instances}
    required = set(plan.instance_keys)
    if not required.issubset(completed):
        missing = sorted(required.difference(completed))
        raise ValueError(f"required instance training is incomplete: {missing}")
    experts = []
    for key in plan.instance_keys:
        expert = model.experts[key]
        checkpoint_metadata_fn = getattr(expert, "checkpoint_metadata", None)
        if not callable(checkpoint_metadata_fn):
            raise TypeError(f"expert {key!r} does not expose checkpoint_metadata()")
        expert_metadata = checkpoint_metadata_fn()
        experts.append(
            JointExpertState(
                key=JointExpertKey.parse(key),
                instance_id=key.replace(":", "-"),
                model_class=f"{type(expert).__module__}.{type(expert).__name__}",
                model_config=expert_metadata.model_config,
                input_frame_spec=expert_metadata.input_frame_spec,
                condition_schema=expert_metadata.condition_schema,
                model_state_dict=expert.state_dict(),
                physical_state=model.physical_states.get(key, {}),
                calibration=model.calibrations.get(key, {}),
                provenance={"training_instance": key},
            )
        )
    selected_role = str(role).strip().lower()
    training_state = None
    if selected_role == "resume":
        if set(optimizers) != required:
            raise ValueError("resume commit requires one optimizer for every expert instance")
        schedulers = dict(schedulers or {})
        progress = dict(training_progress or {})
        training_state = {
            key: {
                "optimizer": optimizers[key].state_dict(),
                "scheduler": None if schedulers.get(key) is None else schedulers[key].state_dict(),
                **dict(progress.get(key, {})),
            }
            for key in plan.instance_keys
        }
    return save_joint_checkpoint(
        path,
        metadata=JointCheckpointMetadata(
            checkpoint_role=selected_role,
            supported_modalities=plan.modalities,
        ),
        experts=experts,
        provenance=provenance,
        training_state=training_state,
    )


__all__ = [
    "DUAL_MODALITY_DUAL_CHANNEL_INSTANCE_KEYS",
    "DUAL_MODALITY_INSTANCE_KEYS",
    "ExpertParallelPlan",
    "ExpertTrainingUnit",
    "JointEpochResult",
    "MultimodalTrainingPlan",
    "RoutedTrainingBatch",
    "commit_joint_training_checkpoint",
    "train_round_robin_epoch",
]
