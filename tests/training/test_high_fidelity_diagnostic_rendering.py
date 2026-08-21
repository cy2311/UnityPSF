from __future__ import annotations

from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tifffile

from unity_psf.training.high_fidelity.diagnostic_rendering import (
    background_anchored_display_scale,
    ncc_value,
    poisson_nll_value,
    render_gamma_monitor_markdown,
    tile_frames_uint8,
    to_uint8,
    to_uint8_background_anchored,
    write_grayscale_png,
)
from unity_psf.training.high_fidelity.diagnostics import (
    _artifact_group_metrics,
    _metadata_item,
    _path_token,
    _raw_tiff_adu_frame_for_diagnostic,
)


def test_uint8_rendering_preserves_existing_scaling_contract() -> None:
    frame = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    background = torch.full((2, 2), 2.0)

    assert np.array_equal(to_uint8(frame), np.asarray([[0, 85], [170, 255]], dtype=np.uint8))

    scale = background_anchored_display_scale([frame], [background])
    assert scale == {
        "mode": "background_anchored",
        "background_gray_uint8": 24,
        "background_reference_photon": 2.0,
        "signal_high_photon": pytest.approx(5.98),
    }
    anchored = to_uint8_background_anchored(frame, background, scale)
    assert anchored.dtype == np.uint8
    assert anchored[0, 0] == 24
    assert anchored[-1, -1] == 255

    tiled = tile_frames_uint8([frame, frame.flip(1)])
    assert tiled.shape == (2, 4)
    assert np.array_equal(tiled[:, :2], to_uint8(frame))


def test_diagnostic_metrics_preserve_existing_numerical_contract() -> None:
    raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstruction = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

    expected_poisson = (reconstruction - raw * torch.log(reconstruction)).mean().item()
    assert poisson_nll_value(raw, reconstruction) == pytest.approx(expected_poisson)
    assert ncc_value(raw, reconstruction) == pytest.approx(1.0)


def test_grayscale_png_writer_preserves_dimensions_and_png_signature(tmp_path: Path) -> None:
    path = tmp_path / "diagnostic.png"
    image = np.asarray([[0, 64, 255], [255, 64, 0]], dtype=np.uint8)

    write_grayscale_png(path, image)

    payload = path.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", payload[16:24])
    assert (height, width) == image.shape


def test_gamma_monitor_markdown_preserves_field_order_and_terminal_newline() -> None:
    payload = {
        "epoch": 3,
        "best_step": 17,
        "selected_poisson_nll": 1.25,
        "diagnostic_png_path": "artifacts/example.png",
    }

    report = render_gamma_monitor_markdown(payload)

    assert report.startswith("# Gamma Update Monitor\n\n- epoch: 3\n- best_step: 17\n")
    assert "- selected_poisson_nll: 1.25\n" in report
    assert report.endswith("- diagnostic_png_path: artifacts/example.png\n")


def test_diagnostic_grouping_and_tokens_preserve_artifact_paths() -> None:
    bank = SimpleNamespace(
        records=(SimpleNamespace(domain_name="right"), SimpleNamespace(domain_name="left"))
    )

    assert _artifact_group_metrics(
        bank,
        roi_library_source="Raw TIFF",
        objective_source="fallback",
    ) == {
        "artifact_source_group": "Raw TIFF",
        "artifact_domain_group": "multi",
        "selected_domain_names": ["left", "right"],
    }
    assert _path_token(" Raw TIFF / Right ") == "raw_tiff_right"
    assert _path_token("---") == "unknown"


def test_raw_tiff_diagnostic_crop_preserves_right_domain_offset(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.tif"
    frame = np.arange(6 * 12, dtype=np.float32).reshape(6, 12)
    tifffile.imwrite(raw_path, frame)
    samples = SimpleNamespace(metadata={"frame_index": [0]})

    crop = _raw_tiff_adu_frame_for_diagnostic(
        {"roi_bank_raw_path": str(raw_path)},
        samples=samples,
        roi_origin_xy_px=torch.tensor([[1.0, 2.0]]),
        domain_names=["right"],
        fallback_shape=(2, 3),
    )

    assert crop is not None
    assert torch.equal(crop, torch.from_numpy(frame[2:4, 7:10].copy()))


def test_metadata_item_normalizes_numpy_scalars() -> None:
    assert _metadata_item(np.asarray([np.int64(7), np.int64(8)]), 1) == 8
    assert _metadata_item([np.float32(1.25)], 0) == pytest.approx(1.25)
    assert _metadata_item([True], 0) is True
    assert _metadata_item([], 0) is None
