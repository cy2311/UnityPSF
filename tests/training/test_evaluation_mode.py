from __future__ import annotations

import pytest
import numpy as np
import torch
from torch import nn

from unity_psf.training.evaluation_mode import preserve_module_training_modes
from unity_psf.training.loop_types import TrainingBatch
from unity_psf.training.evaluation import evaluate
from unity_psf.training.modality_evaluation import evaluate_modality_heldout
from unity_psf.training.modality_types import ModalityChannelStream, ModalityTrainingRuntime
from unity_psf.training.high_fidelity.roi_inference import model_raw_tiff_infer_fn
from unity_psf.roi_library import ROIBankDomain


class _NestedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.dropout = nn.Dropout()
        self.normalizer = nn.BatchNorm1d(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalizer(self.dropout(self.linear(inputs)))


def _mixed_mode_model() -> _NestedModel:
    model = _NestedModel()
    model.train()
    model.normalizer.eval()
    return model


def test_evaluate_restores_every_nested_module_training_mode() -> None:
    model = _mixed_mode_model()
    before = {name: module.training for name, module in model.named_modules()}

    evaluate(
        model=model,
        batches=[TrainingBatch(inputs=torch.ones(2, 2), targets=torch.zeros(2, 2))],
    )

    assert {name: module.training for name, module in model.named_modules()} == before


def test_evaluate_restores_nested_modes_when_loss_raises() -> None:
    model = _mixed_mode_model()
    before = {name: module.training for name, module in model.named_modules()}

    def fail(_model: nn.Module, _batch: TrainingBatch) -> torch.Tensor:
        raise RuntimeError("evaluation failure")

    with pytest.raises(RuntimeError, match="evaluation failure"):
        evaluate(
            model=model,
            batches=[TrainingBatch(inputs=torch.ones(2, 2), targets=torch.zeros(2, 2))],
            loss_fn=fail,
        )

    assert {name: module.training for name, module in model.named_modules()} == before


def test_preserve_module_training_modes_restores_nested_modes_after_exception() -> None:
    model = _mixed_mode_model()
    before = {name: module.training for name, module in model.named_modules()}

    with pytest.raises(RuntimeError, match="mode failure"):
        with preserve_module_training_modes(model):
            assert all(not module.training for module in model.modules())
            raise RuntimeError("mode failure")

    assert {name: module.training for name, module in model.named_modules()} == before


@pytest.mark.parametrize("raise_error", [False, True])
def test_preserve_module_training_modes_restores_shared_submodules(raise_error: bool) -> None:
    shared = nn.Dropout()
    left = nn.Sequential(shared)
    right = nn.Sequential(shared)
    left.train()
    right.eval()
    shared.train()
    model = nn.ModuleDict({"left": left, "right": right})
    before = [module.training for module in model.modules()]

    def run_session():
        with preserve_module_training_modes(model):
            assert all(not module.training for module in model.modules())
            if raise_error:
                raise RuntimeError("shared failure")

    if raise_error:
        with pytest.raises(RuntimeError, match="shared failure"):
            run_session()
    else:
        run_session()

    assert [module.training for module in model.modules()] == before


@pytest.mark.parametrize("root_training", [False, True])
def test_modality_callback_failure_restores_mixed_modes(root_training: bool) -> None:
    model = _mixed_mode_model()
    model.training = root_training
    before = [module.training for module in model.modules()]

    def fail(current_model):
        assert all(not module.training for module in current_model.modules())
        raise RuntimeError("heldout failure")

    stream = ModalityChannelStream(
        channel_id="left", batches=lambda epoch: (), loss_fn=lambda output, batch: output.sum(),
        physical_state={}, calibration={}, provenance={}, heldout_eval=fail,
    )
    runtime = ModalityTrainingRuntime(
        modality="astigmatism", model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        channels={"left": stream}, step_budgets={"left": 1},
    )
    with pytest.raises(RuntimeError, match="heldout failure"):
        evaluate_modality_heldout(runtime)
    assert [module.training for module in model.modules()] == before


def test_roi_model_failure_restores_modes_and_grad_context(monkeypatch) -> None:
    model = _mixed_mode_model()
    before = [module.training for module in model.modules()]

    def fail(inputs):
        assert all(not module.training for module in model.modules())
        assert not torch.is_grad_enabled()
        raise RuntimeError("roi failure")

    monkeypatch.setattr(model, "forward", fail)
    infer = model_raw_tiff_infer_fn(
        model=model, threshold=0.5, max_emitters=1, seed=1,
        expected_channels=1, photon_scale=1000.0, z_scale_nm=700.0,
    )
    with pytest.raises(RuntimeError, match="roi failure"):
        infer(
            domain=ROIBankDomain("main", 0, 0, 8, 8), frame_window=(0, 1),
            raw_domain_frames_photon=np.ones((1, 8, 8), dtype=np.float32),
        )
    assert [module.training for module in model.modules()] == before
    assert torch.is_grad_enabled()
