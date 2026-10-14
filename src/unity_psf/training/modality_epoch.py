"""Round-robin training for a shared modality expert."""

from __future__ import annotations

import torch

from unity_psf.training.modality_types import (
    ModalityEpochResult,
    ModalityTrainingBatch,
    ModalityTrainingRuntime,
)
from unity_psf.training.optimizer_step import run_optimizer_step
from unity_psf.training.step_progress import OptimizerStepProgress, advance_scheduler


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
    channel_progress = {channel_id: OptimizerStepProgress() for channel_id in runtime.channels}
    skip_reasons = {channel_id: {} for channel_id in runtime.channels}
    device_type = next(runtime.model.parameters()).device.type
    autocast_enabled = bool(runtime.amp_enabled and device_type == "cuda")
    scaler = runtime.amp_scaler
    if scaler is None:
        scaler = torch.amp.GradScaler(device="cuda", enabled=autocast_enabled, init_scale=128.0)
        object.__setattr__(runtime, "amp_scaler", scaler)
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
        model_device = next(runtime.model.parameters()).device
        images = batch.images.to(device=model_device, non_blocking=True)
        conditions = None if batch.conditions is None else batch.conditions.to(device=model_device, non_blocking=True)

        def compute_loss():
            output = (
                runtime.model(images, conditions)
                if conditions is not None
                else runtime.model(images)
            )
            return stream.loss_fn(output, batch)

        step_result = run_optimizer_step(
            model=runtime.model,
            optimizer=runtime.optimizer,
            compute_loss=compute_loss,
            scaler=scaler,
            autocast_enabled=autocast_enabled,
            amp_dtype=runtime.amp_dtype,
            grad_clip_norm=runtime.grad_clip_norm,
            loss_context=f"{runtime.modality.value}:{channel_id} at epoch {epoch}",
        )
        optimizer_step_performed = step_result.optimizer_step_performed
        if step_result.skip_reason is not None:
            reasons = skip_reasons[channel_id]
            reasons[step_result.skip_reason] = reasons.get(step_result.skip_reason, 0) + 1
        channel_progress[channel_id].record(performed=optimizer_step_performed)
        advance_scheduler(
            runtime.scheduler, step_unit=runtime.scheduler_step_unit, event="optimizer_step",
            optimizer_steps=int(optimizer_step_performed),
        )
        loss_value = step_result.loss
        step_counts[channel_id] += 1
        sample_counts[channel_id] += int(batch.images.shape[0])
        losses[channel_id].append(loss_value)
    optimizer_steps = sum(progress.optimizer_steps for progress in channel_progress.values())
    advance_scheduler(
        runtime.scheduler, step_unit=runtime.scheduler_step_unit, event="epoch",
        optimizer_steps=optimizer_steps,
    )
    flat_losses = [value for values in losses.values() for value in values]
    return ModalityEpochResult(
        epoch=int(epoch),
        modality=runtime.modality,
        schedule=schedule,
        step_counts=step_counts,
        sample_counts=sample_counts,
        losses_by_channel={key: tuple(values) for key, values in losses.items()},
        optimizer_steps_by_channel={
            channel_id: progress.optimizer_steps for channel_id, progress in channel_progress.items()
        },
        skipped_optimizer_steps_by_channel={
            channel_id: progress.skipped_optimizer_steps for channel_id, progress in channel_progress.items()
        },
        optimizer_skip_reasons_by_channel={
            channel_id: dict(reasons) for channel_id, reasons in skip_reasons.items()
        },
        attempted_optimizer_steps=sum(progress.attempted_optimizer_steps for progress in channel_progress.values()),
        optimizer_steps=optimizer_steps,
        skipped_optimizer_steps=sum(progress.skipped_optimizer_steps for progress in channel_progress.values()),
        mean_loss=sum(flat_losses) / len(flat_losses),
    )
