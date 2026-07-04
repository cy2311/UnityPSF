from __future__ import annotations

import json
import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from neptune_v03.runtime import profiling
from neptune_v03.runtime.layout import RunLayout


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


def load_training_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> TrainingResumeState:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if scheduler is not None and "scheduler_state_dict" not in checkpoint:
        raise ValueError("checkpoint is missing scheduler_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return TrainingResumeState(
        epoch=int(checkpoint["epoch"]),
        step_count=int(checkpoint["step_count"]),
        global_step=int(checkpoint["global_step"]),
        path=checkpoint_path,
        physical_state=checkpoint.get("physical_state") if isinstance(checkpoint.get("physical_state"), dict) else None,
        physical_coeff_maps=_checkpoint_physical_coeff_maps(checkpoint),
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "epoch": epoch,
        "step_count": step_count,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if amp_scaler is not None and amp_scaler.is_enabled():
        payload["scaler_state_dict"] = amp_scaler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


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
