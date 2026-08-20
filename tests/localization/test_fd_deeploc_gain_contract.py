from __future__ import annotations

import numpy as np
from pathlib import Path
import sys
import torch

from unity_psf.roi_library.loc_harvest import _FDDeeplocTileNormalizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "neptune_iwae"))
from Normalization import FDDeeplocStyleNormalizer  # noqa: E402


def _raw_window() -> np.ndarray:
    yy, xx = np.mgrid[:3, :8]
    return (500.0 + 4.0 * yy + 2.0 * xx + (xx == 7) * 900.0).astype(np.float32)[None, ...]


def test_fixed_gain_is_applied_after_anchor_recenter() -> None:
    raw = _raw_window()
    normalizer = _FDDeeplocTileNormalizer(
        train_background_adu=495.58422534346505,
        background_percentile=50.0,
        signal_gain=0.03,
    )

    background = normalizer.estimate_background_numpy(raw)
    result = normalizer.forward(torch.from_numpy(raw)).numpy()
    expected = 495.58422534346505 + (raw - background) * 0.03

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-5)
    assert normalizer.to_dict()["signal_gain"] == 0.03
    assert normalizer.to_dict()["gain_scope"] == "global_run"


def test_gain_contract_rejects_non_global_scope() -> None:
    try:
        _FDDeeplocTileNormalizer(signal_gain=0.03, gain_scope="per_window")
    except ValueError as exc:
        assert "global_run" in str(exc)
    else:
        raise AssertionError("per-window gain must be rejected")


def test_neptune_reference_and_unitypsf_gain_inputs_are_pixel_equivalent() -> None:
    raw = _raw_window()
    anchor = 495.58422534346505
    gain = 0.03
    neptune = FDDeeplocStyleNormalizer()
    unity = _FDDeeplocTileNormalizer(train_background_adu=anchor, signal_gain=gain)

    neptune_background = neptune.estimate_background_numpy(raw)
    unity_background = unity.estimate_background_numpy(raw)
    expected = anchor + (neptune.recenter_numpy(raw) - anchor) * gain
    actual = unity.forward(torch.from_numpy(raw)).numpy()

    assert unity_background == neptune_background
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
