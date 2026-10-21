from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile


def test_first3_overlay_uses_middle_raw_frame_and_frame0_predictions(tmp_path: Path):
    case_dir = tmp_path / "left_o0_s200_first3"
    case_dir.mkdir(parents=True)
    sample_path = tmp_path / "sample.ome.tif"

    frames = np.zeros((3, 4, 6), dtype=np.uint16)
    frames[1, :, :3] = 200
    frames[1, :, 3:] = 50
    tifffile.imwrite(sample_path, frames)

    summary = {
        "sample_tiffs": [str(sample_path)],
        "half": "left",
        "crop_left": 0,
        "crop_width": 3,
        "crop_height": 4,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with (case_dir / "predictions_merged.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "x_px", "y_px", "x_px_full", "y_px_full", "z", "photon", "prob", "sample_tiff"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame": 0,
                "x_px": 1.0,
                "y_px": 2.0,
                "x_px_full": 1.0,
                "y_px_full": 2.0,
                "z": 0.0,
                "photon": 1000.0,
                "prob": 0.9,
                "sample_tiff": sample_path.name,
            }
        )
        writer.writerow(
            {
                "frame": 1,
                "x_px": 2.0,
                "y_px": 1.0,
                "x_px_full": 2.0,
                "y_px_full": 1.0,
                "z": 0.0,
                "photon": 1000.0,
                "prob": 0.9,
                "sample_tiff": sample_path.name,
            }
        )

    from scripts.analysis.render_first3_norm_overlays import load_case_overlay_inputs, render_overlay_image

    base_frame, xy_points = load_case_overlay_inputs(case_dir)

    assert base_frame.shape == (4, 3)
    assert np.all(base_frame == 200)
    assert xy_points == [(1.0, 2.0)]

    overlay = render_overlay_image(base_frame, xy_points, marker_radius=0)

    assert overlay.shape == (4, 3, 3)
    assert tuple(overlay[2, 1]) == (255, 0, 0)
    assert tuple(overlay[0, 0]) == (255, 255, 255)


def test_first3_overlay_accepts_single_sample_tiff_key(tmp_path: Path):
    case_dir = tmp_path / "single_sample_key_case"
    case_dir.mkdir(parents=True)
    sample_path = tmp_path / "sample.ome.tif"

    frames = np.zeros((3, 4, 6), dtype=np.uint16)
    frames[1, :, :3] = 123
    tifffile.imwrite(sample_path, frames)

    summary = {
        "sample_tiff": str(sample_path),
        "crop_left": 0,
        "crop_width": 3,
        "crop_height": 4,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with (case_dir / "predictions_merged.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "x_px", "y_px", "x_px_full", "y_px_full", "z", "photon", "prob", "sample_tiff"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame": 0,
                "x_px": 1.0,
                "y_px": 1.0,
                "x_px_full": 1.0,
                "y_px_full": 1.0,
                "z": 0.0,
                "photon": 1000.0,
                "prob": 0.9,
                "sample_tiff": sample_path.name,
            }
        )

    from scripts.analysis.render_first3_norm_overlays import load_case_overlay_inputs

    base_frame, xy_points = load_case_overlay_inputs(case_dir)

    assert base_frame.shape == (4, 3)
    assert np.all(base_frame == 123)
    assert xy_points == [(1.0, 1.0)]
