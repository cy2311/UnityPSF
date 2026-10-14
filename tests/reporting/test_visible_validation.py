from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from unity_psf.reporting.visible_validation import (
    InstanceVisualRecord,
    generate_visible_validation_report,
)


def _record(key: str, offset: float, *, physical: bool) -> InstanceVisualRecord:
    image = np.zeros((24, 24), dtype=np.float32)
    image[5:19, 8:16] = np.linspace(0.1, 1.0, 14, dtype=np.float32)[:, None] + offset
    return InstanceVisualRecord(
        instance_key=key,
        input_image=image,
        patches=(image[4:12, 6:14], image[10:18, 10:18]),
        loss_history=(2.0 + offset, 1.2 + offset, 0.7 + offset),
        route_count=12,
        step_count=3,
        sample_count=24,
        prediction_xy=np.asarray([[10.0, 8.0], [13.0, 16.0]], dtype=np.float32),
        reconstruction=np.flipud(image).copy(),
        z_values=np.asarray([-500.0, 0.0, 500.0], dtype=np.float32) if physical else None,
        z_errors=np.asarray([80.0, -20.0, 45.0], dtype=np.float32) if physical else None,
        physical_initial=np.eye(12, dtype=np.float32) if physical else None,
        physical_current=np.fliplr(np.eye(12, dtype=np.float32)) if physical else None,
        status="trained",
        checkpoint_hash=("a" if key.endswith("main") else "b") * 64,
    )


def _heldout_record(key: str, offset: float) -> InstanceVisualRecord:
    record = _record(key, offset, physical=key.startswith("astigmatism"))
    return replace(
        record,
        status="heldout-evaluated",
        heldout_metrics={
            "precision": 0.9,
            "recall": 0.8,
            "Jaccard": 0.75,
            "RMSE_XY_nm": 18.5,
            "RMSE_Z_nm": 42.0 if key.startswith("astigmatism") else None,
            "photon_relative_error": 0.08,
        },
    )


def test_visible_report_writes_nonblank_fixed_figure_pack_and_static_html(tmp_path: Path) -> None:
    records = (
        _record("emitter_2d:main", 0.0, physical=False),
        _record("astigmatism:left", 0.1, physical=True),
        _record("astigmatism:right", 0.2, physical=True),
    )

    result = generate_visible_validation_report(tmp_path, records, run_id="dual-smoke")

    assert result.report_path == tmp_path / "report" / "report.html"
    assert result.report_path.is_file()
    assert "emitter_2d:main" in result.report_path.read_text(encoding="utf-8")
    required = {
        "00_input_audit.png",
        "01_psf_patch_montage.png",
        "02_route_and_step_balance.png",
        "03_training_curves.png",
        "07_physical_state_emitter_2d_main.png",
        "08_cross_modality_scorecard.png",
    }
    assert required.issubset({path.name for path in result.figure_paths})
    assert any(path.name == "04_prediction_overlay_astigmatism_left.png" for path in result.figure_paths)
    assert any(path.name == "07_physical_state_astigmatism_right.png" for path in result.figure_paths)
    for path in result.figure_paths:
        pixels = mpimg.imread(path)
        assert pixels.size > 0
        assert float(np.std(pixels)) > 0.01

    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "dual-smoke"
    assert set(summary["instances"]) == {
        "emitter_2d:main",
        "astigmatism:left",
        "astigmatism:right",
    }


def test_scorecard_reserves_space_for_four_long_instance_statuses(tmp_path: Path) -> None:
    records = (
        _heldout_record("emitter_2d:left", 0.0),
        _heldout_record("emitter_2d:right", 0.1),
        _heldout_record("astigmatism:left", 0.2),
        _heldout_record("astigmatism:right", 0.3),
    )

    modality_metrics = {
        "emitter_2d": records[0].heldout_metrics,
        "astigmatism": records[2].heldout_metrics,
    }
    generate_visible_validation_report(
        tmp_path,
        records,
        run_id="four-channel",
        modality_metrics=modality_metrics,
    )

    scorecard = mpimg.imread(tmp_path / "figures" / "08_cross_modality_scorecard.png")
    assert scorecard.shape[1] / scorecard.shape[0] >= 2.6
    assert (tmp_path / "figures" / "09_heldout_metrics_scorecard.png").is_file()
    summary = json.loads((tmp_path / "metrics" / "summary.json").read_text())
    assert summary["instances"]["emitter_2d:left"]["heldout_metrics"]["RMSE_Z_nm"] is None
    assert summary["modality_metrics"]["astigmatism"]["RMSE_Z_nm"] == 42.0
