from __future__ import annotations

from pathlib import Path

from unity_psf.training.high_fidelity.roi_bank_source import (
    auto_build_domains,
    camera_backward_from_training_config,
    resolve_roi_bank_source,
    roi_bank_source_metrics,
)


def test_roi_source_resolves_relative_path_and_metadata(tmp_path: Path) -> None:
    source = resolve_roi_bank_source(
        {
            "roi_bank_source": {
                "mode": "loc_infer_raw_tiff",
                "raw_path": "frames.tif",
                "candidate_mode": "dense_tile_temporal",
                "frame_range": [5, 8],
                "domains": [{"name": "left", "crop_width": 32, "crop_height": 24}],
            }
        },
        train_cfg={},
        config={},
        config_base_dir=tmp_path,
    )

    assert source is not None
    assert source.raw_path == str((tmp_path / "frames.tif").resolve())
    assert source.alias == "loc_infer_raw_tiff"
    assert source.frame_range == (5, 8)
    assert roi_bank_source_metrics(source) == {
        "roi_bank_source_mode": "auto_build",
        "roi_bank_raw_path": str((tmp_path / "frames.tif").resolve()),
        "roi_bank_candidate_mode": "dense_tile_temporal",
        "roi_bank_source_alias": "loc_infer_raw_tiff",
        "roi_bank_frame_range": [5, 8],
    }
    assert auto_build_domains({}, roi_size=8, roi_source=source)[0].name == "left"


def test_roi_source_defaults_and_camera_metadata_contract() -> None:
    source = resolve_roi_bank_source(
        {"auto_build_roi_bank": True, "auto_build_source_path": "raw.tif"},
        train_cfg={},
        config={},
        config_base_dir=Path("/tmp"),
    )

    assert source is not None
    assert auto_build_domains({}, roi_size=8, roi_source=source)[0].crop_width == 12
    assert camera_backward_from_training_config(
        {
            "camera": {"qe": 0.8, "e_per_adu": 2.0, "baseline_adu": 100.0},
            "normalization": {"em_gain": 3.0, "spurious_charge": 0.1},
        }
    ) == {
        "baseline": 100.0,
        "e_per_adu": 2.0,
        "em_gain": 3.0,
        "qe": 0.8,
        "spurious_charge": 0.1,
    }
