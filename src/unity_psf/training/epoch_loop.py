from __future__ import annotations

from collections.abc import Iterable

import torch

from unity_psf.runtime import profiling
from unity_psf.runtime.layout import RunLayout
from unity_psf.training.checkpoint_io import append_jsonl, checkpoint_extra_with_rng, save_checkpoint
from unity_psf.training.evaluation import default_mse_loss, training_core_metrics_row
from unity_psf.training.loop_types import (
    BatchEndHook,
    LossFn,
    TrainingBatch,
    TrainingConfig,
    TrainingEpochResult,
    TrainingStepResult,
)
from unity_psf.training.optimizer_step import run_optimizer_step
from unity_psf.training.step_progress import OptimizerStepProgress, advance_scheduler


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
    compute_loss = loss_fn or default_mse_loss
    metrics_path = layout.metrics_dir / config.metrics_name
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    device_type = model_device_type(model)
    autocast_enabled = bool(config.amp_enabled and device_type == "cuda")
    scaler = config.amp_scaler
    if scaler is None and autocast_enabled:
        scaler = torch.amp.GradScaler(device="cuda", enabled=True)
    scaler_enabled = bool(scaler is not None and scaler.is_enabled())

    loss_sum = 0.0
    progress = OptimizerStepProgress()
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
        step_result = run_optimizer_step(
            model=model,
            optimizer=optimizer,
            compute_loss=lambda: compute_loss(model, batch),
            scaler=scaler,
            autocast_enabled=autocast_enabled,
            amp_dtype=config.amp_dtype,
            grad_clip_norm=config.grad_clip_norm,
            loss_context=f"epoch {config.epoch}",
            profile=True,
        )
        progress.record(performed=step_result.optimizer_step_performed)
        advance_scheduler(
            scheduler, step_unit=config.scheduler_step_unit, event="optimizer_step",
            optimizer_steps=int(step_result.optimizer_step_performed),
        )

        loss_value = step_result.loss
        loss_sum += loss_value
        step_count = int(step)
        metric_row: dict[str, object] = {
            "epoch": config.epoch,
            "step": step,
            "global_step": config.global_step_start + step,
            "loss": loss_value,
        }
        metric_row.update(profiling.drain())
        append_jsonl(
            metrics_path,
            training_core_metrics_row(metric_row),
        )
        if on_batch_end is not None:
            on_batch_end(
                TrainingStepResult(
                    epoch=int(config.epoch),
                    step_count=int(step),
                    global_step=int(config.global_step_start + step),
                    loss=loss_value,
                    metrics_path=metrics_path,
                    optimizer_step_performed=step_result.optimizer_step_performed,
                )
            )

    checkpoint_path = layout.checkpoints_dir / config.checkpoint_name
    if step_count <= 0:
        raise ValueError("train_one_epoch requires at least one batch")
    advance_scheduler(
        scheduler, step_unit=config.scheduler_step_unit, event="epoch",
        optimizer_steps=progress.optimizer_steps,
    )
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=scaler if scaler_enabled else None,
        epoch=config.epoch,
        step_count=step_count,
        global_step=config.global_step_start + step_count,
        extra=checkpoint_extra_with_rng(config.checkpoint_extra_fn),
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
        optimizer_steps=progress.optimizer_steps,
        skipped_optimizer_steps=progress.skipped_optimizer_steps,
    )


def model_device_type(model: torch.nn.Module) -> str:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device.type
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device.type
    return "cpu"


def model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")
