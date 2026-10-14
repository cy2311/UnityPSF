from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg.physical_update import (
    _poisson_nll_from_lunar_normalized,
    _project_patch_to_roi,
    _offset_roi_bank_frame_indices,
    _split_roi_samples_by_frame,
    physical_update_quality_reasons,
    normalized_field_xy,
    resolve_roi_update_budget,
    samples_from_roi_bank,
    should_update_ppg,
)
from unity_psf.data.normalization import LunarInputNormalization
from unity_psf.roi_library import EmitterPosterior, ROIBank, ROIRecord
from ppg.train_dh_physical import (
    _json_default,
    _prepare_working_map,
    _select_update_records,
)


def _write_ppg_map(path: Path, *, support: int = 21, marker: int = 0) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray("ppg.pca_map.v1"),
        gamma=np.zeros((9, 4), dtype=np.float32),
        roi_shape_yx=np.asarray([support, support]),
        coefficient_field=np.zeros((4, 40, 40), dtype=np.float32),
        marker=np.asarray(marker),
    )


def test_physical_update_metrics_serialize_numpy_scalars() -> None:
    encoded = json.dumps(
        {"accepted": np.bool_(True), "score": np.float32(1.25), "count": np.int64(3)},
        default=_json_default,
    )

    assert json.loads(encoded) == {"accepted": True, "score": 1.25, "count": 3}


def test_resume_preserves_existing_working_map(tmp_path: Path) -> None:
    initial_map = tmp_path / "initial.npz"
    working_map = tmp_path / "working.npz"
    _write_ppg_map(initial_map, marker=1)
    _write_ppg_map(working_map, marker=2)

    _prepare_working_map(initial_map, working_map, resume=True, psf_support_px=21)

    with np.load(working_map, allow_pickle=False) as payload:
        assert int(payload["marker"]) == 2


def test_fresh_working_map_copies_the_validated_artifact(tmp_path: Path) -> None:
    initial_map = tmp_path / "initial.npz"
    working_map = tmp_path / "working.npz"
    _write_ppg_map(initial_map, marker=3)

    _prepare_working_map(initial_map, working_map, resume=False, psf_support_px=21)

    assert working_map.read_bytes() == initial_map.read_bytes()


def test_fresh_working_map_rejects_nonstandard_support(tmp_path: Path) -> None:
    initial_map = tmp_path / "initial.npz"
    _write_ppg_map(initial_map, support=25)

    with pytest.raises(ValueError, match="ROI shape"):
        _prepare_working_map(
            initial_map,
            tmp_path / "working.npz",
            resume=False,
            psf_support_px=21,
        )


def test_physical_update_quality_gate_requires_ncc_and_nll() -> None:
    assert (
        physical_update_quality_reasons(
            ncc_after=0.80,
            poisson_nll_before=10.0,
            poisson_nll_after=9.0,
            ncc_threshold=0.75,
        )
        == []
    )
    assert physical_update_quality_reasons(
        ncc_after=0.70,
        poisson_nll_before=10.0,
        poisson_nll_after=9.0,
        ncc_threshold=0.75,
    ) == ["ncc_below_threshold"]
    assert physical_update_quality_reasons(
        ncc_after=0.80,
        poisson_nll_before=10.0,
        poisson_nll_after=10.0,
        ncc_threshold=0.75,
    ) == ["poisson_nll_not_improved"]


def test_physical_nll_uses_lunar_normalized_inputs_without_changing_likelihood() -> None:
    observed_counts = torch.tensor([110.0, 730.0, 1350.0])
    expected_counts = torch.tensor([120.0, 700.0, 1300.0])
    normalization = LunarInputNormalization(
        background_range=(110.0, 110.0),
        photon_range=(0.0, 31000.0),
    )

    actual = _poisson_nll_from_lunar_normalized(
        normalization.normalize(expected_counts),
        normalization.normalize(observed_counts),
        normalization=normalization,
    )
    expected = (
        expected_counts - observed_counts * torch.log(expected_counts)
    ).mean() / normalization.scale_ph_factor

    torch.testing.assert_close(actual, expected)


def test_ppg_update_schedule_starts_at_30_then_runs_every_10_epochs() -> None:
    assert not should_update_ppg(29, start_epoch=30, interval=10)
    assert should_update_ppg(30, start_epoch=30, interval=10)
    assert not should_update_ppg(31, start_epoch=30, interval=10)
    assert should_update_ppg(40, start_epoch=30, interval=10)
    assert should_update_ppg(300, start_epoch=30, interval=10)


def test_unitypsf_roi_update_budget_is_256_pool_and_256_per_update() -> None:
    assert resolve_roi_update_budget({"max_rois": 256}) == (256, 256)
    assert resolve_roi_update_budget(
        {"roi_library_max_rois": 256, "roi_update_max_rois": 256}
    ) == (256, 256)


def test_roi_update_budget_rejects_update_larger_than_pool() -> None:
    try:
        resolve_roi_update_budget(
            {"roi_library_max_rois": 128, "roi_update_max_rois": 256}
        )
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("an update budget larger than the ROI pool must fail")


def test_roi_update_subset_rotates_without_shrinking_library() -> None:
    records = tuple(range(8))

    first = _select_update_records(records, max_rois=3, update_index=0)
    second = _select_update_records(records, max_rois=3, update_index=1)
    wrapped = _select_update_records(records, max_rois=3, update_index=2)

    assert first == (0, 1, 2)
    assert second == (3, 4, 5)
    assert wrapped == (6, 7, 0)
    assert records == tuple(range(8))


def test_normalized_coordinates_align_different_field_sizes() -> None:
    sample = normalized_field_xy(255.5, 255.5, field_shape_yx=(512, 512))
    beads = normalized_field_xy(319.5, 319.5, field_shape_yx=(640, 640))
    np.testing.assert_allclose(sample, beads, atol=2e-3)
    np.testing.assert_allclose(
        normalized_field_xy(0.0, 0.0, field_shape_yx=(512, 512)),
        (0.0, 0.0),
    )
    np.testing.assert_allclose(
        normalized_field_xy(511.0, 511.0, field_shape_yx=(512, 512)),
        (1.0, 1.0),
    )


def test_sample_physical_update_preserves_large_multi_emitter_roi() -> None:
    emitters = tuple(
        EmitterPosterior(
            probability=0.9,
            cell_xy_px=xy,
            mu_xy_px=xy,
            sigma_xy_px=(0.0, 0.0),
            mu_z_nm=z_nm,
            sigma_z_nm=0.0,
            mu_photons=1000.0,
            sigma_photons=0.0,
            local_xy_px=xy,
            full_xy_px=(xy[0] + 96.0, xy[1] + 192.0),
            frame_index=101,
        )
        for xy, z_nm in (((24.5, 30.5), -200.0), ((70.5, 65.5), 300.0))
    )
    record = ROIRecord(
        roi_id=7,
        domain_name="sample",
        frame_window=(100, 103),
        roi_origin_xy_px=(96.0, 192.0),
        raw_frames_photon=np.ones((3, 96, 96), dtype=np.float32),
        background_mu=np.zeros((96, 96), dtype=np.float32),
        background_smoothed=np.zeros((96, 96), dtype=np.float32),
        grid_cell_id=3,
        emitters=emitters,
    )

    samples = samples_from_roi_bank(
        ROIBank(records=(record,)), field_shape_yx=(512, 512)
    )

    assert len(samples) == 1
    assert samples[0].observed.shape == (96, 96)
    assert len(samples[0].emitters) == 2
    assert samples[0].emitters[0].xy_normalized == normalized_field_xy(
        120.5, 222.5, field_shape_yx=(512, 512)
    )


def test_psf_projection_clips_support_at_large_roi_boundary() -> None:
    projected = _project_patch_to_roi(
        torch.ones((25, 25), dtype=torch.float32),
        local_xy_px=(3.5, 4.5),
        roi_shape_yx=(96, 96),
    )

    assert projected.shape == (96, 96)
    assert 0.0 < float(projected.sum()) < 625.0


def test_train_heldout_split_never_shares_raw_frames() -> None:
    class Sample:
        def __init__(self, roi_id: int, frame_index: int) -> None:
            self.roi_id = roi_id
            self.frame_index = frame_index

    samples = [
        Sample(index, frame)
        for index, frame in enumerate((100, 100, 101, 102, 102, 103, 104))
    ]

    training, heldout = _split_roi_samples_by_frame(samples, heldout_fraction=0.2)

    assert {item.frame_index for item in training}.isdisjoint(
        {item.frame_index for item in heldout}
    )
    assert {item.frame_index for item in training + heldout} == {
        100,
        101,
        102,
        103,
        104,
    }


def test_lazy_tiff_roi_bank_frame_indices_are_restored_to_global_range() -> None:
    emitter = EmitterPosterior(
        probability=0.9,
        cell_xy_px=(48.0, 48.0),
        mu_xy_px=(48.0, 48.0),
        sigma_xy_px=(0.0, 0.0),
        mu_z_nm=0.0,
        sigma_z_nm=0.0,
        mu_photons=1000.0,
        sigma_photons=0.0,
        local_xy_px=(48.0, 48.0),
        full_xy_px=(48.0, 48.0),
        frame_index=7,
    )
    record = ROIRecord(
        roi_id=0,
        domain_name="sample",
        frame_window=(6, 9),
        roi_origin_xy_px=(0.0, 0.0),
        raw_frames_photon=np.ones((3, 96, 96), dtype=np.float32),
        background_mu=np.zeros((96, 96), dtype=np.float32),
        background_smoothed=np.zeros((96, 96), dtype=np.float32),
        grid_cell_id=0,
        emitters=(emitter,),
    )

    shifted = _offset_roi_bank_frame_indices(
        ROIBank(records=(record,)), frame_offset=277
    )

    assert shifted.records[0].frame_window == (283, 286)
    assert shifted.records[0].emitters[0].frame_index == 284
