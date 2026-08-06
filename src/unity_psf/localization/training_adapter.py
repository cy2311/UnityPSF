from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from unity_psf.runtime.profiling import time_block
from unity_psf.training.loop import TrainingBatch


@dataclass(frozen=True)
class LocalizationTrainBatch:
    model_input: torch.Tensor | tuple[torch.Tensor, ...]
    detect_tar: torch.Tensor
    bkg_tar: torch.Tensor
    pxyz_tar: torch.Tensor
    mask_tar: torch.Tensor
    metadata: dict[str, Any]


def localization_batch_to_device(batch: LocalizationTrainBatch, device: torch.device | str) -> LocalizationTrainBatch:
    return LocalizationTrainBatch(
        model_input=_model_input_to_device(batch.model_input, device),
        detect_tar=batch.detect_tar.to(device=device),
        bkg_tar=batch.bkg_tar.to(device=device),
        pxyz_tar=batch.pxyz_tar.to(device=device),
        mask_tar=batch.mask_tar.to(device=device),
        metadata=batch.metadata,
    )


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
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(model_input, tuple):
        return tuple(item.to(device=device) for item in model_input)
    return model_input.to(device=device)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
