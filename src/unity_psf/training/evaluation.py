from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from unity_psf.training.evaluation_mode import preserve_module_training_modes
from unity_psf.training.loop_types import LossFn, TrainingBatch


def evaluate(
    *,
    model: torch.nn.Module,
    batches: Iterable[TrainingBatch],
    loss_fn: LossFn | None = None,
) -> float:
    batch_list = list(batches)
    if not batch_list:
        raise ValueError("evaluate requires at least one batch")

    compute_loss = loss_fn or default_mse_loss
    losses: list[float] = []
    with preserve_module_training_modes(model), torch.no_grad():
        for batch in batch_list:
            loss = compute_loss(model, batch)
            losses.append(float(loss.detach().cpu().item()))
    return sum(losses) / len(losses)


def reset_eval_metrics(loss_fn: LossFn | None) -> None:
    reset = getattr(loss_fn, "reset_eval_metrics", None)
    if callable(reset):
        reset()


def aggregate_eval_metrics(loss_fn: LossFn | None):
    aggregate = getattr(loss_fn, "aggregate_eval_metrics", None)
    if callable(aggregate):
        return aggregate()
    return getattr(loss_fn, "last_metrics", None)


def training_core_metrics_row(metric_row: Mapping[str, object]) -> dict[str, object]:
    core_keys = ("global_step", "loss")
    row = {key: metric_row[key] for key in core_keys if key in metric_row}
    row.update({key: value for key, value in metric_row.items() if str(key).startswith("profile_")})
    return row


def eval_core_metrics_row(eval_row: Mapping[str, object]) -> dict[str, object]:
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


def default_mse_loss(model: torch.nn.Module, batch: TrainingBatch) -> torch.Tensor:
    prediction = model(batch.inputs)
    return torch.nn.functional.mse_loss(prediction, batch.targets)


__all__ = [
    "aggregate_eval_metrics",
    "eval_core_metrics_row",
    "reset_eval_metrics",
    "training_core_metrics_row",
    "evaluate",
]
