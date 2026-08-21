from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unity_psf.roi_library import EmitterPosterior, ROIBank, ROIRecord
from unity_psf.training.high_fidelity.roi_posterior import (
    record_observed_center_frame,
    record_projection_tensors,
    sample_roi_bank_posterior_update_from_cached_emitters,
    select_roi_bank_update_subset,
    sample_roi_bank_posterior_update_from_current_model,
)


def _record(
    roi_id: int,
    *,
    domain: str = "left",
    frame_window: tuple[int, int] = (10, 13),
    valid_core: bool = False,
) -> ROIRecord:
    raw = np.stack(
        [np.full((4, 4), float(frame), dtype=np.float32) for frame in range(frame_window[0], frame_window[1])],
        axis=0,
    )
    emitters = (
        EmitterPosterior(
            probability=0.95,
            cell_xy_px=(1.0, 1.0),
            mu_xy_px=(1.25, 1.5),
            sigma_xy_px=(0.0, 0.0),
            mu_z_nm=25.0,
            sigma_z_nm=0.0,
            mu_photons=120.0,
            sigma_photons=0.0,
            local_xy_px=(1.25, 1.5),
            full_xy_px=(1.25, 1.5),
            frame_index=10,
        ),
        EmitterPosterior(
            probability=0.9,
            cell_xy_px=(2.0, 2.0),
            mu_xy_px=(2.25, 2.5),
            sigma_xy_px=(0.0, 0.0),
            mu_z_nm=-30.0,
            sigma_z_nm=0.0,
            mu_photons=80.0,
            sigma_photons=0.0,
            local_xy_px=(2.25, 2.5),
            full_xy_px=(2.25, 2.5),
            frame_index=12,
        ),
    )
    summary = {}
    if valid_core:
        summary = {"valid_core_size_px": 2, "valid_core_offset_xy_px": (1, 1)}
    return ROIRecord(
        roi_id=roi_id,
        domain_name=domain,
        frame_window=frame_window,
        roi_origin_xy_px=(10.0 + roi_id, 20.0 + roi_id),
        raw_frames_photon=raw,
        background_mu=np.full((4, 4), 2.0, dtype=np.float32),
        background_smoothed=np.full((4, 4), 3.0, dtype=np.float32),
        grid_cell_id=roi_id,
        emitters=emitters,
        summary=summary,
    )


def test_cached_posterior_is_deterministic_and_preserves_frame_alignment() -> None:
    bank = ROIBank(records=(_record(2), _record(1)))
    cfg = {
        "target_projected_emitters": 1,
        "max_sampling_rounds": 1,
        "seed": 17,
        "num_posterior_samples": 1,
        "sample_continuous": False,
        "posterior_probability_threshold": 0.5,
        "roi_size_px": 4,
    }

    first, first_metrics = sample_roi_bank_posterior_update_from_cached_emitters(bank, cfg, epoch=3)
    second, second_metrics = sample_roi_bank_posterior_update_from_cached_emitters(bank, cfg, epoch=3)

    assert first_metrics == second_metrics
    assert torch.equal(first.raw_frames, second.raw_frames)
    assert torch.equal(first.background, second.background)
    assert torch.equal(first.samples.xyzph, second.samples.xyzph)
    assert torch.equal(first.samples.mask, second.samples.mask)
    assert torch.equal(first.roi_origin_xy_px, second.roi_origin_xy_px)
    assert first.domain_names == second.domain_names
    assert first.samples.metadata["frame_index"] == [10, 12]
    assert torch.equal(first.raw_frames[:, 0, 0], torch.tensor([10.0, 12.0]))


def test_update_subset_keeps_seeded_roi_order_and_target_metrics() -> None:
    bank = ROIBank(records=(_record(1), _record(2), _record(3)))
    subset, metrics = select_roi_bank_update_subset(
        bank,
        {"target_projected_emitters": 2, "max_sampling_rounds": 1, "seed": 5},
        epoch=4,
    )

    assert metrics["selected_roi_ids"] == [record.roi_id for record in subset.records]
    assert metrics["sampled_emitter_count_total"] == sum(len(record.emitters) for record in subset.records)
    assert metrics["target_emitters_reached"] is True


def test_record_projection_uses_emitter_frames_and_stacks_loss_masks() -> None:
    bank = ROIBank(records=(_record(1, valid_core=True), _record(2)))

    raw_frames, background, samples, origins, domains, loss_mask = record_projection_tensors(bank)

    assert raw_frames.shape == (4, 4, 4)
    assert torch.equal(raw_frames[:, 0, 0], torch.tensor([10.0, 12.0, 10.0, 12.0]))
    assert torch.equal(background, torch.full((4, 4, 4), 3.0))
    assert samples.xyzph.shape == (4, 1, 4)
    assert samples.mask.shape == (4, 1)
    assert samples.metadata["frame_index"] == [10, 12, 10, 12]
    assert torch.equal(origins, torch.tensor([[11.0, 21.0], [11.0, 21.0], [12.0, 22.0], [12.0, 22.0]]))
    assert domains == ["left", "left", "left", "left"]
    assert loss_mask is not None
    assert bool(loss_mask[0, 0, 0]) is False
    assert bool(loss_mask[0, 1, 1]) is True
    assert bool(loss_mask[1, 1, 1]) is True
    assert bool(loss_mask[2].all()) is True


def test_observed_center_frame_uses_the_middle_raw_window_frame() -> None:
    assert torch.equal(record_observed_center_frame(_record(1)), torch.full((4, 4), 11.0))


def test_record_projection_rejects_an_empty_bank() -> None:
    with pytest.raises(ValueError, match="at least one ROI record"):
        record_projection_tensors(ROIBank(records=()))


def test_current_model_empty_posterior_keeps_error_contract() -> None:
    class EmptyModel(torch.nn.Module):
        def forward(self, value):
            if isinstance(value, tuple):
                value = value[0]
            batch, _, height, width = value.shape
            output = torch.zeros((batch, 10, height, width), dtype=value.dtype)
            return output

    with pytest.raises(ValueError, match="selected no samples"):
        sample_roi_bank_posterior_update_from_current_model(
            ROIBank(records=(_record(1),)),
            {"target_projected_emitters": 1, "max_sampling_rounds": 1, "seed": 3, "roi_size_px": 4},
            model=EmptyModel(),
            train_cfg={},
            step_index=0,
        )
