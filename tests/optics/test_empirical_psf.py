from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tifffile

from unity_psf.optics.empirical_psf import load_empirical_focal_psf


def _write_dual_channel_stack(path: Path) -> None:
    stack = np.full((3, 7, 14), 10, dtype=np.uint16)
    stack[1, 2, 3] = 110
    stack[1, 4, 7 + 2] = 210
    tifffile.imwrite(path, stack, photometric="minisblack")


def test_empirical_focal_psf_selects_and_normalizes_each_measurement_channel(tmp_path: Path) -> None:
    path = tmp_path / "PSF_aligned.tif"
    _write_dual_channel_stack(path)

    left = load_empirical_focal_psf(path, channel_id="left", focus_index=1)
    right = load_empirical_focal_psf(path, channel_id="right", focus_index=1)

    assert left.kernel.shape == (7, 7)
    assert right.kernel.shape == (7, 7)
    assert np.isclose(float(left.kernel.sum()), 1.0)
    assert np.isclose(float(right.kernel.sum()), 1.0)
    assert np.unravel_index(int(left.kernel.argmax()), left.kernel.shape) == (3, 3)
    assert np.unravel_index(int(right.kernel.argmax()), right.kernel.shape) == (3, 3)
    assert left.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert left.kernel_sha256 != right.kernel_sha256
    assert left.focus_index == right.focus_index == 1
    assert left.source_channel_slice == (0, 7)
    assert right.source_channel_slice == (7, 14)


def test_empirical_focal_psf_rejects_non_dual_or_out_of_range_calibration(tmp_path: Path) -> None:
    path = tmp_path / "invalid.tif"
    tifffile.imwrite(path, np.ones((2, 7, 13), dtype=np.uint16), photometric="minisblack")

    try:
        load_empirical_focal_psf(path, channel_id="left", focus_index=1)
    except ValueError as exc:
        assert "two equal square channel patches" in str(exc)
    else:
        raise AssertionError("invalid dual-channel calibration was accepted")
