from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unity_psf.localization.model import LocalizationModelOutput
from unity_psf.roi_library import EmitterPosterior, ROIBank, ROIRecord
from unity_psf.training.high_fidelity import gamma_workflow, heldout_monitor, posterior_workflow, roi_projection


class _ROIModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> LocalizationModelOutput:
        self.seen_inputs.append(inputs.detach().clone())
        shape = (inputs.shape[0], inputs.shape[-2], inputs.shape[-1])
        logits = inputs.new_full(shape, -20.0)
        logits[:, 2, 2] = 20.0
        return LocalizationModelOutput(
            detection_logits=logits, xy_offset=inputs.new_zeros((shape[0], 2, *shape[1:])),
            z=inputs.new_full(shape, 0.25), photons=inputs.new_full(shape, 0.5),
        )


class _ProjectionRenderer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        return kwargs["gamma"].square().sum()


@pytest.fixture
def roi_bank() -> ROIBank:
    emitter = EmitterPosterior(
        probability=0.99, cell_xy_px=(2.0, 2.0), mu_xy_px=(0.0, 0.0),
        sigma_xy_px=(0.01, 0.01), mu_z_nm=20.0, sigma_z_nm=0.01,
        mu_photons=100.0, sigma_photons=0.01, local_xy_px=(2.0, 2.0),
        full_xy_px=(12.0, 22.0), frame_index=11,
    )
    raw = np.stack([np.full((5, 5), value, dtype=np.float32) for value in (3.0, 7.0, 11.0)])
    background = np.full((5, 5), 2.0, dtype=np.float32)
    record = ROIRecord(
        roi_id=0, domain_name="left", frame_window=(10, 13),
        roi_origin_xy_px=(10.0, 20.0), raw_frames_photon=raw,
        background_mu=background, background_smoothed=background, grid_cell_id=0,
        emitters=(emitter,), summary={"valid_core_size_px": 3, "valid_core_offset_xy_px": (1, 1)},
    )
    return ROIBank(records=(record, replace(record, roi_id=1, grid_cell_id=1)), metadata={"source": "test"})


def test_record_projection_tensors_preserves_emitter_frames_and_valid_core(roi_bank: ROIBank) -> None:
    raw, background, samples, origins, domains, loss_mask = roi_projection.record_projection_tensors(roi_bank)

    torch.testing.assert_close(raw, torch.full((2, 5, 5), 7.0))
    torch.testing.assert_close(background, torch.full((2, 5, 5), 2.0))
    torch.testing.assert_close(samples.xyzph[:, 0], torch.tensor([[2.0, 2.0, 20.0, 100.0]] * 2))
    assert samples.mask.tolist() == [[True], [True]]
    assert samples.metadata["frame_index"] == [11, 11]
    torch.testing.assert_close(origins, torch.tensor([[10.0, 20.0]] * 2))
    assert domains == ["left", "left"]
    assert loss_mask is not None
    assert loss_mask.sum().item() == 18
    assert loss_mask[:, 1:4, 1:4].all()
    assert not loss_mask[:, 0].any()


def test_heldout_plain_posterior_uses_physical_scales(roi_bank: ROIBank) -> None:
    model = _ROIModel()
    context = heldout_monitor.build_heldout_monitor_context(
        {}, model=model, bank=roi_bank, objective=_ProjectionRenderer(), max_emitters=1,
        mode="fixed_samples", split_source="test", train_cfg={"scaling": {"photon_max": 200.0, "z_max": 0.8}},
    )

    assert context is not None
    assert context.samples_mask_count == 2
    torch.testing.assert_close(context.samples.xyzph[:, 0], torch.tensor([[2.0, 2.0, 200.0, 100.0]] * 2))
    assert context.loss_mask.sum().item() == 18
    assert model.training


def test_current_model_posterior_applies_real_fd_normalizer(roi_bank: ROIBank) -> None:
    model = _ROIModel()
    context, metrics = posterior_workflow.sample_roi_bank_posterior_update_from_current_model(
        roi_bank, {"roi_size_px": 5, "target_projected_emitters": 1, "sample_continuous": False},
        model=model, step_index=1,
        train_cfg={"normalization": {"mode": "fd_deeploc_style", "train_background_adu": 40.0},
                   "scaling": {"photon_max": 200.0, "z_max": 0.8}},
    )

    assert context.samples.mask.any()
    assert metrics["selected_sampled_emitter_count"] >= 1
    assert model.seen_inputs
    assert model.seen_inputs[0].mean().item() == pytest.approx(40.0, abs=1.0)
    assert torch.isfinite(context.samples.xyzph).all()


@pytest.mark.parametrize("entry", ["prepare", "lazy_active"])
def test_importance_objective_reaches_real_cached_posterior(entry: str, roi_bank: ROIBank, tmp_path: Path) -> None:
    renderer = _ProjectionRenderer()
    objective_fn, _metrics_fn, prepare_fn = gamma_workflow._build_roi_projection_objective(
        {"roi_bank_objective": "importance_wake", "posterior_source": "cached_emitters",
         "roi_size_px": 5, "target_projected_emitters": 1, "sample_continuous": False},
        model=_ROIModel(), bank=roi_bank, objective_source="test", roi_library_path=None,
        heldout_bank=None, heldout_mode="not_configured", heldout_split_source=None,
        layout=SimpleNamespace(run_dir=tmp_path), objective=renderer, train_cfg={},
    )
    if entry == "prepare":
        metrics = prepare_fn(SimpleNamespace(global_step=3, epoch=1))
        assert metrics["selected_sampled_emitter_count"] >= 1
    gamma = torch.tensor([1.0], requires_grad=True)
    loss = objective_fn(gamma)
    loss.backward()

    assert gamma.grad.item() == pytest.approx(2.0)
    assert renderer.calls[0]["samples"].mask.any()
    assert renderer.calls[0]["raw_frames"].shape[-2:] == (5, 5)
    assert renderer.calls[0]["loss_mask"].sum().item() >= 9


@pytest.mark.parametrize("source", ["persisted", "harvest"])
def test_gamma_routing_splits_real_roi_bank_before_renderer_setup(
    source: str, roi_bank: ROIBank, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class RendererBoundaryReached(Exception):
        pass

    class RoutingRenderer(_ProjectionRenderer):
        def initial_gamma(self):
            raise RendererBoundaryReached

    model = _ROIModel()
    renderer = RoutingRenderer()
    monkeypatch.setattr(gamma_workflow, "_build_vector_roi_gamma_objective", lambda *args, **kwargs: renderer)
    gamma_cfg = {"enabled": True, "auto_heldout_min_rois": 1, "auto_heldout_max_rois": 1}
    if source == "persisted":
        gamma_cfg["roi_library_path"] = str(tmp_path / "roi.h5")
        monkeypatch.setattr(gamma_workflow, "load_roi_bank", lambda path: roi_bank)
    else:
        gamma_cfg["roi_bank_source"] = {"mode": "auto_build", "raw_path": str(tmp_path / "raw.tif")}
        monkeypatch.setattr(gamma_workflow, "_auto_build_roi_bank", lambda *args, **kwargs: roi_bank)

    def reach_renderer() -> None:
        hook = gamma_workflow.build_roi_bank_gamma_hook(
            {"roi_bank_gamma": gamma_cfg}, config={}, config_base_dir=tmp_path,
            model=model, layout=SimpleNamespace(run_dir=tmp_path),
        )
        if source == "harvest":
            assert callable(hook)
            hook(SimpleNamespace(epoch=1, global_step=1))

    with pytest.raises(RendererBoundaryReached):
        reach_renderer()
    assert len(model.seen_inputs) == 1
    assert model.seen_inputs[0].shape == (1, 3, 5, 5)
