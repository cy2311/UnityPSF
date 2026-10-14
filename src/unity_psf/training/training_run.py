from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping

import torch

from unity_psf.runtime import profiling
from unity_psf.runtime.layout import RunLayout
from unity_psf.training.checkpoint_io import append_jsonl, checkpoint_extra_with_rng, existing_best_eval_loss, save_checkpoint
from unity_psf.training.epoch_loop import train_one_epoch, model_device, model_device_type
from unity_psf.training.evaluation import aggregate_eval_metrics, eval_core_metrics_row, reset_eval_metrics, evaluate
from unity_psf.training.loop_types import (
    BatchEndHook,
    BatchProvider,
    CheckpointExtraFn,
    EpochEndHook,
    EpochTrainingConfig,
    EvalProvider,
    LossFn,
    TrainingBatch,
    TrainingConfig,
    TrainingRunEpochResult,
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
    if int(config.eval_interval_epochs) <= 0:
        raise ValueError("eval_interval_epochs must be positive")

    results: list[TrainingRunEpochResult] = []
    global_step = config.global_step_start
    trained_batches = int(config.global_step_start)
    best_eval_loss: float | None = None
    best_checkpoint_path = layout.checkpoints_dir / "checkpoint_best.pt"
    if eval_provider is not None:
        best_eval_loss = existing_best_eval_loss(best_checkpoint_path)
    amp_scaler = config.amp_scaler
    if amp_scaler is None and config.amp_enabled and model_device_type(model) == "cuda":
        amp_scaler = torch.amp.GradScaler(device="cuda", enabled=True)
    if amp_scaler is not None and config.scaler_state_dict is not None:
        amp_scaler.load_state_dict(config.scaler_state_dict)
    for epoch in range(config.start_epoch, config.stop_epoch + 1):
        if config.max_batches is not None:
            remaining = int(config.max_batches) - int(trained_batches)
            if remaining <= 0:
                break
        epoch_batches: Iterable[TrainingBatch] = batch_provider(epoch)
        prefetch_depth = int(getattr(epoch_batches, "prefetch_batches", 0) or 0)
        if config.max_batches is not None:
            epoch_batches = itertools.islice(epoch_batches, int(remaining))
        if prefetch_depth > 0:
            from unity_psf.localization.training_adapter import prefetch_training_batches

            epoch_batches = prefetch_training_batches(
                epoch_batches,
                model_device(model),
                prefetch_batches=prefetch_depth,
            )
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
        eval_loss = None
        current_best_path = None
        should_eval = (
            eval_provider is not None
            and int(epoch) % int(config.eval_interval_epochs) == 0
        )
        if should_eval:
            reset_eval_metrics(eval_loss_fn or loss_fn)
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
            # Keep evaluation timing separate from training-step timing so
            # epoch wall-clock regressions can be attributed to the correct
            # path (simulation, model, or held-out validation).
            eval_row.update(profiling.drain())
            eval_metrics = aggregate_eval_metrics(eval_loss_fn or loss_fn)
            if isinstance(eval_metrics, Mapping):
                eval_row.update(dict(eval_metrics))
            eval_core_row = eval_core_metrics_row(eval_row)
            append_jsonl(
                layout.metrics_dir / "eval_metrics.jsonl",
                eval_core_row,
            )
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    amp_scaler=amp_scaler if amp_scaler is not None and amp_scaler.is_enabled() else None,
                    epoch=epoch,
                    step_count=epoch_result.step_count,
                    global_step=global_step,
                    extra={**checkpoint_extra_with_rng(checkpoint_extra_fn), "eval_loss": eval_loss},
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
            optimizer_steps=epoch_result.optimizer_steps,
            skipped_optimizer_steps=epoch_result.skipped_optimizer_steps,
        )
        results.append(result)
        if on_epoch_end is not None:
            on_epoch_end(result)
    return results
