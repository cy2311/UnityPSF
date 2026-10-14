from __future__ import annotations

from unity_psf.roi_library import ROIBank, ROIRecord
from unity_psf.training.high_fidelity.roi_split import resolve_roi_gamma_banks


def _bank(*domains: str) -> ROIBank:
    records = tuple(
        ROIRecord(
            roi_id=index,
            domain_name=domain,
            frame_window=(0, 1),
            roi_origin_xy_px=(0.0, 0.0),
            raw_frames_photon=[],
            background_mu=[],
            background_smoothed=[],
            grid_cell_id=index,
        )
        for index, domain in enumerate(domains)
    )
    return ROIBank(records=records, metadata={"source": "test"})


def test_resolve_roi_gamma_banks_splits_each_domain() -> None:
    selected, heldout, mode, source = resolve_roi_gamma_banks(
        _bank("left", "left", "right", "right"),
        {"auto_heldout_min_rois": 1, "auto_heldout_max_rois": 1},
    )

    assert [record.roi_id for record in selected.records] == [0, 2]
    assert [record.roi_id for record in heldout.records] == [1, 3]
    assert mode == "auto_split_fixed_samples"
    assert source == "auto_from_roi_library"
