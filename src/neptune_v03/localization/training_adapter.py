from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neptune_v03.localization.smlm_output import SMLMOutputChannels
from neptune_v03.training.loop import TrainingBatch


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
    def loss_fn(model: torch.nn.Module, batch: TrainingBatch) -> torch.Tensor:
        loss_fn.last_metrics = {}
        loc_batch = batch.inputs
        if not isinstance(loc_batch, LocalizationTrainBatch):
            raise TypeError("localization loss expects TrainingBatch.inputs to be LocalizationTrainBatch")
        loc_batch = localization_batch_to_device(loc_batch, _model_device(model))
        y_out = model(loc_batch.model_input)
        loss = criterion.forward(
            y_out,
            loc_batch.detect_tar,
            loc_batch.pxyz_tar,
            loc_batch.mask_tar,
            loc_batch.bkg_tar,
        )
        components = getattr(criterion, "last_components", None)
        if isinstance(components, dict):
            loss_fn.last_metrics = {"loss_components": dict(components)}
        detection_metrics = _detection_metrics(y_out, loc_batch.detect_tar)
        if detection_metrics:
            loss_fn.last_metrics.update(detection_metrics)
        return loss.mean()

    return loss_fn


def _detection_metrics(y_out: torch.Tensor, detect_tar: torch.Tensor) -> dict[str, float]:
    if not isinstance(y_out, torch.Tensor) or y_out.ndim != 4 or int(y_out.shape[1]) != 10:
        return {}
    prob = y_out[:, SMLMOutputChannels.p]
    target = detect_tar.to(device=prob.device, dtype=torch.bool)
    pred = prob >= 0.5
    true_positive = torch.logical_and(pred, target).sum().to(dtype=torch.float32)
    target_count = target.sum().to(dtype=torch.float32)
    pred_count = pred.sum().to(dtype=torch.float32)
    union = torch.logical_or(pred, target).sum().to(dtype=torch.float32)
    recall = true_positive / target_count.clamp_min(1.0)
    jaccard = true_positive / union.clamp_min(1.0)
    batch_size = max(1, int(prob.shape[0]))
    return {
        "pixel_recall": float(recall.detach().cpu().item()),
        "pixel_jaccard": float(jaccard.detach().cpu().item()),
        "pixel_predicted_emitters_per_sample": float((pred_count / batch_size).detach().cpu().item()),
        "pixel_target_emitters_per_sample": float((target_count / batch_size).detach().cpu().item()),
        "pixel_predicted_emitters_total": float(pred_count.detach().cpu().item()),
        "pixel_target_emitters_total": float(target_count.detach().cpu().item()),
    }


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
