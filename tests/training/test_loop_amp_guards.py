from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unity_psf.runtime.layout import ensure_run_layout
from unity_psf.training.loop import (
    TrainingBatch,
    TrainingConfig,
    _existing_best_eval_loss,
    train_one_epoch,
)


def _run(tmp_path: Path, *, init_scale: float, loss_fn=None):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler(device="cpu", init_scale=init_scale)
    weights_before = model.weight.detach().clone()
    train_one_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batches=[TrainingBatch(inputs=torch.ones(4, 2), targets=torch.zeros(4, 1))],
        layout=ensure_run_layout(tmp_path, "amp"),
        config=TrainingConfig(epoch=1, amp_scaler=scaler),
        loss_fn=loss_fn,
    )
    return model, weights_before, scheduler


def test_scheduler_advances_when_optimizer_steps(tmp_path: Path) -> None:
    model, before, scheduler = _run(tmp_path, init_scale=1.0)
    assert not torch.equal(model.weight, before)
    assert scheduler.last_epoch == 1


def test_scheduler_holds_when_scaler_skips_step(tmp_path: Path) -> None:
    def huge_loss(model, batch):
        return model(batch.inputs).mean() * 1e30

    model, before, scheduler = _run(tmp_path, init_scale=2.0**127, loss_fn=huge_loss)
    assert torch.equal(model.weight, before)
    assert scheduler.last_epoch == 0


def test_non_finite_loss_raises(tmp_path: Path) -> None:
    def nan_loss(model, batch):
        return model(batch.inputs).sum() * float("nan")

    with pytest.raises(FloatingPointError):
        _run(tmp_path, init_scale=1.0, loss_fn=nan_loss)


def test_existing_best_eval_loss_reads_checkpoint_with_metadata_objects(tmp_path: Path) -> None:
    from unity_psf.contracts.modality import PSFModality

    path = tmp_path / "best.pt"
    torch.save({"eval_loss": 0.25, "expert_type": PSFModality.ASTIGMATISM}, path)
    assert _existing_best_eval_loss(path) == 0.25
