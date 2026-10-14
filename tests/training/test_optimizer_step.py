from pathlib import Path

import pytest
import torch

from unity_psf.runtime.layout import ensure_run_layout
from unity_psf.training.loop import TrainingBatch, TrainingConfig, train_one_epoch
from unity_psf.training.modality_joint import (
    ModalityChannelStream,
    ModalityTrainingBatch,
    ModalityTrainingRuntime,
    train_modality_epoch,
)


@pytest.mark.parametrize("route", ["generic", "modality"])
@pytest.mark.parametrize("failure", ["vector", "nan", "not_tensor"])
def test_training_rejects_invalid_loss_before_backward(tmp_path: Path, route: str, failure: str) -> None:
    model = torch.nn.Conv2d(1, 1, 1, bias=False)
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    images = torch.ones(1, 1, 2, 2)

    def invalid_loss(output):
        if failure == "vector":
            return output.flatten()
        if failure == "not_tensor":
            return 1.0
        return output.sum() * float("nan")

    expected = FloatingPointError if failure == "nan" else ValueError
    with pytest.raises(expected):
        if route == "generic":
            train_one_epoch(
                model=model,
                optimizer=optimizer,
                batches=[TrainingBatch(images, torch.zeros_like(images))],
                layout=ensure_run_layout(tmp_path, "invalid"),
                config=TrainingConfig(epoch=1),
                loss_fn=lambda network, batch: invalid_loss(network(batch.inputs)),
            )
        else:
            stream = ModalityChannelStream(
                channel_id="left",
                batches=lambda epoch: [ModalityTrainingBatch(images, None, "left")],
                loss_fn=lambda output, batch: invalid_loss(output),
                physical_state={},
                calibration={},
                provenance={},
            )
            train_modality_epoch(
                runtime=ModalityTrainingRuntime(
                    modality="astigmatism", model=model, optimizer=optimizer,
                    channels={"left": stream}, step_budgets={"left": 1},
                ),
                epoch=1,
            )
    assert torch.equal(model.weight, before)
    assert model.weight.grad is None


@pytest.mark.parametrize("scale, multiplier, performed", [(1.0, 1.0, True), (2.0**127, 1e30, False)])
@pytest.mark.parametrize("growth_interval", [1, 2000])
def test_optimizer_step_reports_amp_update_and_skip(
    scale: float, multiplier: float, performed: bool, growth_interval: int,
) -> None:
    from unity_psf.training.optimizer_step import run_optimizer_step

    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(device="cpu", init_scale=scale, growth_interval=growth_interval)
    before = model.weight.detach().clone()
    result = run_optimizer_step(
        model=model, optimizer=optimizer,
        compute_loss=lambda: model(torch.ones(1, 2)).sum() * multiplier,
        scaler=scaler, autocast_enabled=False, amp_dtype=torch.float16,
        grad_clip_norm=1.0, loss_context="test",
    )
    assert result.optimizer_step_performed is performed
    assert result.skip_reason == (None if performed else "amp_scale_reduced")
    expected_scale = scale * (2.0 if growth_interval == 1 else 1.0) if performed else scale / 2.0
    assert scaler.get_scale() == expected_scale
    assert isinstance(result.loss, float)
    assert torch.equal(model.weight, before) is (not performed)


@pytest.mark.parametrize("scaler_mode", ["absent", "disabled"])
def test_optimizer_step_without_amp_has_no_skip_reason(scaler_mode: str) -> None:
    from unity_psf.training.optimizer_step import run_optimizer_step

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    scaler = None if scaler_mode == "absent" else torch.amp.GradScaler(device="cpu", enabled=False)
    result = run_optimizer_step(
        model=model, optimizer=optimizer,
        compute_loss=lambda: model(torch.ones(1, 1)).sum(),
        scaler=scaler, autocast_enabled=False, amp_dtype=torch.float16,
        grad_clip_norm=None, loss_context="test",
    )
    assert result.optimizer_step_performed is True
    assert result.skip_reason is None
    assert not torch.equal(model.weight, before)


def test_optimizer_step_clips_unscaled_gradient() -> None:
    from unity_psf.training.optimizer_step import run_optimizer_step

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = model.weight.detach().clone()
    run_optimizer_step(
        model=model, optimizer=optimizer,
        compute_loss=lambda: model(torch.ones(1, 1)).sum() * 100,
        scaler=torch.amp.GradScaler(device="cpu", init_scale=128),
        autocast_enabled=False, amp_dtype=torch.float16, grad_clip_norm=0.5,
        loss_context="test",
    )
    assert torch.allclose(before - model.weight, torch.full_like(before, 0.5))
