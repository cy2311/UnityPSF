from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from unity_psf.runtime.profiling import time_block
from unity_psf.training.loop_types import TrainingBatch


@dataclass(frozen=True)
class LocalizationTrainBatch:
    model_input: torch.Tensor | tuple[torch.Tensor, ...]
    detect_tar: torch.Tensor
    bkg_tar: torch.Tensor
    pxyz_tar: torch.Tensor
    mask_tar: torch.Tensor
    metadata: dict[str, Any]


def localization_batch_to_device(
    batch: LocalizationTrainBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> LocalizationTrainBatch:
    return LocalizationTrainBatch(
        model_input=_model_input_to_device(batch.model_input, device, non_blocking=non_blocking),
        detect_tar=batch.detect_tar.to(device=device, non_blocking=non_blocking),
        bkg_tar=batch.bkg_tar.to(device=device, non_blocking=non_blocking),
        pxyz_tar=batch.pxyz_tar.to(device=device, non_blocking=non_blocking),
        mask_tar=batch.mask_tar.to(device=device, non_blocking=non_blocking),
        metadata=batch.metadata,
    )


def prefetch_training_batches(
    batches: Iterable[TrainingBatch],
    device: torch.device | str,
    *,
    prefetch_batches: int = 1,
) -> Iterator[TrainingBatch]:
    """Move localization batches on a side CUDA stream with a one-batch lead.

    The iterator remains unchanged on CPU or for non-localization batches. A
    CUDA event is waited on by the caller's current stream immediately before
    each yielded batch, allowing the next H2D copy to overlap model compute.
    """
    depth = int(prefetch_batches)
    device_obj = torch.device(device)
    if depth <= 0 or device_obj.type != "cuda" or not torch.cuda.is_available():
        yield from batches
        return

    stream = torch.cuda.Stream(device=device_obj)
    iterator = iter(batches)
    queue: deque[tuple[TrainingBatch, torch.cuda.Event | None]] = deque()

    def enqueue(batch: TrainingBatch) -> None:
        if not isinstance(batch.inputs, LocalizationTrainBatch):
            queue.append((batch, None))
            return
        pinned = _pin_localization_batch(batch.inputs)
        with torch.cuda.stream(stream):
            moved = localization_batch_to_device(
                pinned,
                device_obj,
                non_blocking=True,
            )
            event = torch.cuda.Event()
            event.record(stream)
        queue.append((TrainingBatch(inputs=moved, targets=moved.detect_tar), event))

    for _ in range(depth + 1):
        try:
            enqueue(next(iterator))
        except StopIteration:
            break

    current_stream = torch.cuda.current_stream(device_obj)
    while queue:
        batch, event = queue.popleft()
        try:
            enqueue(next(iterator))
        except StopIteration:
            pass
        if event is not None:
            current_stream.wait_event(event)
            for tensor in _localization_batch_tensors(batch.inputs):
                tensor.record_stream(current_stream)
        yield batch


def _pin_localization_batch(batch: LocalizationTrainBatch) -> LocalizationTrainBatch:
    def pin(value: torch.Tensor | tuple[torch.Tensor, ...]):
        if isinstance(value, tuple):
            return tuple(pin(item) for item in value)
        if value.device.type != "cpu" or value.is_pinned():
            return value
        return value.pin_memory()

    return LocalizationTrainBatch(
        model_input=pin(batch.model_input),
        detect_tar=pin(batch.detect_tar),
        bkg_tar=pin(batch.bkg_tar),
        pxyz_tar=pin(batch.pxyz_tar),
        mask_tar=pin(batch.mask_tar),
        metadata=batch.metadata,
    )


def _localization_batch_tensors(batch: LocalizationTrainBatch):
    values = (batch.model_input, batch.detect_tar, batch.bkg_tar, batch.pxyz_tar, batch.mask_tar)
    for value in values:
        if isinstance(value, tuple):
            yield from value
        else:
            yield value


def to_training_batch(batch: LocalizationTrainBatch) -> TrainingBatch:
    return TrainingBatch(inputs=batch, targets=batch.detect_tar)


def make_localization_loss(criterion):
    def loss_from_output(y_out: torch.Tensor, batch: TrainingBatch) -> torch.Tensor:
        loc_batch = batch.inputs
        if not isinstance(loc_batch, LocalizationTrainBatch):
            raise TypeError("localization loss expects TrainingBatch.inputs to be LocalizationTrainBatch")
        loc_batch = localization_batch_to_device(loc_batch, y_out.device)
        with time_block("gmm_posterior_loss"):
            loss = criterion.forward(
                y_out,
                loc_batch.detect_tar,
                loc_batch.pxyz_tar,
                loc_batch.mask_tar,
                loc_batch.bkg_tar,
            )
        return loss.mean()

    def loss_fn(model: torch.nn.Module, batch: TrainingBatch) -> torch.Tensor:
        loc_batch = batch.inputs
        if not isinstance(loc_batch, LocalizationTrainBatch):
            raise TypeError("localization loss expects TrainingBatch.inputs to be LocalizationTrainBatch")
        loc_batch = localization_batch_to_device(loc_batch, _model_device(model))
        return loss_from_output(model(loc_batch.model_input), TrainingBatch(inputs=loc_batch, targets=loc_batch.detect_tar))

    loss_fn.from_output = loss_from_output
    return loss_fn


def _model_input_to_device(
    model_input: torch.Tensor | tuple[torch.Tensor, ...],
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(model_input, tuple):
        return tuple(item.to(device=device, non_blocking=non_blocking) for item in model_input)
    return model_input.to(device=device, non_blocking=non_blocking)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
