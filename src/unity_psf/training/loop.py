from __future__ import annotations

import json
import itertools
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from unity_psf.contracts.checkpoint import (
    CheckpointMetadata,
    build_checkpoint,
    detect_checkpoint_format,
    load_checkpoint,
)
from unity_psf.contracts.modality import PSFModality
from unity_psf.runtime import profiling
from unity_psf.runtime.layout import RunLayout


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


@dataclass(frozen=True)
class TrainingEpochResult:
    epoch: int
    step_count: int
    mean_loss: float
    metrics_path: Path
    checkpoint_path: Path
    global_step: int


@dataclass(frozen=True)
class TrainingStepResult:
    epoch: int
    step_count: int
    global_step: int
    loss: float
    metrics_path: Path
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class EpochTrainingConfig:
    start_epoch: int
    stop_epoch: int
    metrics_name: str = "training_metrics.jsonl"
    global_step_start: int = 0
    max_batches: int | None = None
    grad_clip_norm: float | None = None
    amp_enabled: bool = False
    amp_dtype: torch.dtype = torch.float16
    scheduler_step_unit: str = "optimizer_step"
    checkpoint_metadata: CheckpointMetadata | None = None


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


def train_one_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    batches: Iterable[TrainingBatch],
    layout: RunLayout,
    config: TrainingConfig,
    on_batch_end: BatchEndHook | None = None,
    loss_fn: LossFn | None = None,
) -> TrainingEpochResult:
    model.train()
    compute_loss = loss_fn or _default_mse_loss
    metrics_path = layout.metrics_dir / config.metrics_name
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    device_type = _model_device_type(model)
    autocast_enabled = bool(config.amp_enabled and device_type == "cuda")
    scaler = config.amp_scaler
    if scaler is None and autocast_enabled:
        scaler = torch.amp.GradScaler(device="cuda", enabled=True)
    scaler_enabled = bool(scaler is not None and scaler.is_enabled())

    loss_sum = 0.0
    step_count = 0
    batch_iter = iter(batches)
    step = 0
    while True:
        with profiling.time_block("online_simulation_total"):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
        step += 1
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=config.amp_dtype, enabled=autocast_enabled):
            loss = compute_loss(model, batch)
        if scaler_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        if scaler_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None and config.scheduler_step_unit == "optimizer_step":
            scheduler.step()

        loss_value = float(loss.detach().cpu().item())
        loss_sum += loss_value
        step_count = int(step)
        metric_row: dict[str, object] = {
            "epoch": config.epoch,
            "step": step,
            "global_step": config.global_step_start + step,
            "loss": loss_value,
        }
        metric_row.update(profiling.drain())
        _append_jsonl(
            metrics_path,
            _training_core_metrics_row(metric_row),
        )
        if on_batch_end is not None:
            on_batch_end(
                TrainingStepResult(
                    epoch=int(config.epoch),
                    step_count=int(step),
                    global_step=int(config.global_step_start + step),
                    loss=loss_value,
                    metrics_path=metrics_path,
                )
            )

    checkpoint_path = layout.checkpoints_dir / config.checkpoint_name
    if step_count <= 0:
        raise ValueError("train_one_epoch requires at least one batch")
    _save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=scaler if scaler_enabled else None,
        epoch=config.epoch,
        step_count=step_count,
        global_step=config.global_step_start + step_count,
        extra=_checkpoint_extra(config.checkpoint_extra_fn),
        checkpoint_metadata=config.checkpoint_metadata,
    )

    mean_loss = loss_sum / step_count if step_count else 0.0
    return TrainingEpochResult(
        epoch=config.epoch,
        step_count=step_count,
        mean_loss=mean_loss,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        global_step=config.global_step_start + step_count,
    )


def train_epochs(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    batch_provider: BatchProvider,
    layout: RunLayout,
    config: EpochTrainingConfig,
    on_epoch_end: EpochEndHook | None = None,
    on_batch_end: BatchEndHook | None = None,
    loss_fn: LossFn | None = None,
    eval_provider: EvalProvider | None = None,
    eval_loss_fn: LossFn | None = None,
    checkpoint_extra_fn: CheckpointExtraFn | None = None,
) -> list[TrainingRunEpochResult]:
    if config.stop_epoch < config.start_epoch:
        raise ValueError("stop_epoch must be greater than or equal to start_epoch")
    if config.max_batches is not None and int(config.max_batches) <= 0:
        raise ValueError("max_batches must be positive")

    results: list[TrainingRunEpochResult] = []
    global_step = config.global_step_start
    trained_batches = 0
    best_eval_loss: float | None = None
    best_checkpoint_path = layout.checkpoints_dir / "checkpoint_best.pt"
    if eval_provider is not None:
        best_eval_loss = _existing_best_eval_loss(best_checkpoint_path)
    amp_scaler = (
        torch.amp.GradScaler(device="cuda", enabled=True)
        if config.amp_enabled and _model_device_type(model) == "cuda"
        else None
    )
    for epoch in range(config.start_epoch, config.stop_epoch + 1):
        if config.max_batches is not None:
            remaining = int(config.max_batches) - int(trained_batches)
            if remaining <= 0:
                break
        epoch_batches: Iterable[TrainingBatch] = batch_provider(epoch)
        if config.max_batches is not None:
            epoch_batches = itertools.islice(epoch_batches, int(remaining))
        epoch_result = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batches=epoch_batches,
            layout=layout,
            config=TrainingConfig(
                epoch=epoch,
                checkpoint_name="checkpoint_latest.pt",
                metrics_name=config.metrics_name,
                global_step_start=global_step,
                grad_clip_norm=config.grad_clip_norm,
                amp_enabled=config.amp_enabled,
                amp_dtype=config.amp_dtype,
                amp_scaler=amp_scaler,
                scheduler_step_unit=config.scheduler_step_unit,
                checkpoint_extra_fn=checkpoint_extra_fn,
                checkpoint_metadata=config.checkpoint_metadata,
            ),
            on_batch_end=on_batch_end,
            loss_fn=loss_fn,
        )
        global_step = epoch_result.global_step
        trained_batches += int(epoch_result.step_count)
        if scheduler is not None and config.scheduler_step_unit == "epoch":
            scheduler.step()
            _save_checkpoint(
                path=epoch_result.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                amp_scaler=amp_scaler if amp_scaler is not None and amp_scaler.is_enabled() else None,
                epoch=epoch_result.epoch,
                step_count=epoch_result.step_count,
                global_step=global_step,
                extra=_checkpoint_extra(checkpoint_extra_fn),
                checkpoint_metadata=config.checkpoint_metadata,
            )
        eval_loss = None
        current_best_path = None
        if eval_provider is not None:
            _reset_eval_metrics(eval_loss_fn or loss_fn)
            eval_loss = evaluate(
                model=model,
                batches=eval_provider(),
                loss_fn=eval_loss_fn or loss_fn,
            )
            eval_row: dict[str, object] = {
                "epoch": epoch,
                "global_step": global_step,
                "eval_loss": eval_loss,
            }
            eval_metrics = _aggregate_eval_metrics(eval_loss_fn or loss_fn)
            if isinstance(eval_metrics, Mapping):
                eval_row.update(dict(eval_metrics))
            eval_core_row = _eval_core_metrics_row(eval_row)
            _append_jsonl(
                layout.metrics_dir / "eval_metrics.jsonl",
                eval_core_row,
            )
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                _save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    amp_scaler=amp_scaler if amp_scaler is not None and amp_scaler.is_enabled() else None,
                    epoch=epoch,
                    step_count=epoch_result.step_count,
                    global_step=global_step,
                    extra={**_checkpoint_extra(checkpoint_extra_fn), "eval_loss": eval_loss},
                    checkpoint_metadata=config.checkpoint_metadata,
                )
            current_best_path = best_checkpoint_path

        result = TrainingRunEpochResult(
            epoch=epoch_result.epoch,
            step_count=epoch_result.step_count,
            mean_loss=epoch_result.mean_loss,
            metrics_path=epoch_result.metrics_path,
            checkpoint_path=epoch_result.checkpoint_path,
            global_step=epoch_result.global_step,
            eval_loss=eval_loss,
            best_checkpoint_path=current_best_path,
        )
        results.append(result)
        if on_epoch_end is not None:
            on_epoch_end(result)
    return results


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
        _restore_rng_state(checkpoint.get("rng_state"))

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


def evaluate(
    *,
    model: torch.nn.Module,
    batches: Iterable[TrainingBatch],
    loss_fn: LossFn | None = None,
) -> float:
    batch_list = list(batches)
    if not batch_list:
        raise ValueError("evaluate requires at least one batch")

    was_training = model.training
    model.eval()
    compute_loss = loss_fn or _default_mse_loss
    losses: list[float] = []
    with torch.no_grad():
        for batch in batch_list:
            loss = compute_loss(model, batch)
            losses.append(float(loss.detach().cpu().item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _reset_eval_metrics(loss_fn: LossFn | None) -> None:
    reset = getattr(loss_fn, "reset_eval_metrics", None)
    if callable(reset):
        reset()


def _aggregate_eval_metrics(loss_fn: LossFn | None):
    aggregate = getattr(loss_fn, "aggregate_eval_metrics", None)
    if callable(aggregate):
        return aggregate()
    return getattr(loss_fn, "last_metrics", None)


def _training_core_metrics_row(metric_row: Mapping[str, object]) -> dict[str, object]:
    core_keys = ("global_step", "loss")
    row = {key: metric_row[key] for key in core_keys if key in metric_row}
    row.update({key: value for key, value in metric_row.items() if str(key).startswith("profile_")})
    return row


def _eval_core_metrics_row(eval_row: Mapping[str, object]) -> dict[str, object]:
    core_keys = (
        "global_step",
        "eval_loss",
        "jaccard",
        "rmse_lat",
        "rmse_ax",
        "recall",
        "precision",
        "predicted_emitters",
        "target_emitters",
    )
    return {key: eval_row[key] for key in core_keys if key in eval_row}


def _checkpoint_extra(checkpoint_extra_fn: CheckpointExtraFn | None) -> dict[str, object]:
    if checkpoint_extra_fn is None:
        return {}
    payload = checkpoint_extra_fn()
    if payload is None:
        return {}
    return dict(payload)


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


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": _capture_numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(value: object) -> None:
    if not isinstance(value, Mapping):
        return
    python_state = value.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = value.get("numpy")
    if isinstance(numpy_state, Mapping):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    elif isinstance(numpy_state, (tuple, list)) and len(numpy_state) == 5:
        np.random.set_state(
            (
                str(numpy_state[0]),
                np.asarray(numpy_state[1], dtype=np.uint32),
                int(numpy_state[2]),
                int(numpy_state[3]),
                float(numpy_state[4]),
            )
        )
    torch_cpu_state = value.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(torch.as_tensor(torch_cpu_state, dtype=torch.uint8, device="cpu"))
    torch_cuda_state = value.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(torch_cuda_state, (tuple, list)):
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in torch_cuda_state]
        )


def _capture_numpy_rng_state() -> dict[str, object]:
    bit_generator, state, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": str(bit_generator),
        "state": np.asarray(state, dtype=np.uint32).tolist(),
        "pos": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


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


def _default_mse_loss(model: torch.nn.Module, batch: TrainingBatch) -> torch.Tensor:
    prediction = model(batch.inputs)
    return torch.nn.functional.mse_loss(prediction, batch.targets)


def _model_device_type(model: torch.nn.Module) -> str:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device.type
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device.type
    return "cpu"


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    amp_scaler: torch.amp.GradScaler | None = None,
    epoch: int,
    step_count: int,
    global_step: int,
    extra: dict[str, object] | None = None,
    checkpoint_metadata: CheckpointMetadata | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    training_extra: dict[str, object] = {
        "epoch": epoch,
        "step_count": step_count,
        "global_step": global_step,
    }
    if checkpoint_metadata is not None:
        training_extra["rng_state"] = _capture_rng_state()
    if extra:
        training_extra.update(extra)
    if amp_scaler is not None and amp_scaler.is_enabled():
        training_extra["scaler_state_dict"] = amp_scaler.state_dict()
    if checkpoint_metadata is None:
        payload: dict[str, object] = {
            **training_extra,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
    else:
        payload = build_checkpoint(
            model_state_dict=model.state_dict(),
            metadata=checkpoint_metadata,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None if scheduler is None else scheduler.state_dict(),
            extra=training_extra,
        )
    _atomic_torch_save(payload, path)


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    """Write a checkpoint beside its destination, then replace it atomically."""

    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temporary_path = Path(temporary_name)
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _existing_best_eval_loss(path: Path) -> float | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    eval_loss = checkpoint.get("eval_loss")
    return None if eval_loss is None else float(eval_loss)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
