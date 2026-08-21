from __future__ import annotations

import numpy as np
import torch

from unity_psf.roi_library.loc_harvest import _FDDeeplocTileNormalizer


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
