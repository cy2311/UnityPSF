from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from ppg.lunar_background import (
    estimate_lunar_background_range,
    estimate_lunar_tiff_background,
)


def test_lunar_background_matches_darkest_temporal_mean_gaussian_fit() -> None:
    frames = np.arange(300, dtype=np.float32).reshape(3, 10, 10)
    temporal_mean = frames.mean(axis=0)
    mask = temporal_mean < np.percentile(temporal_mean, 10.0)
    selected = frames[:, mask].reshape(-1)
    expected_mu = float(selected.mean())
    expected_sigma = float(selected.std(ddof=0))

    estimate = estimate_lunar_background_range(frames, percentile=10.0)

    np.testing.assert_allclose(
        estimate,
        (max(expected_mu - 2.0 * expected_sigma, 0.0), expected_mu + 2.0 * expected_sigma),
    )


def test_lunar_tiff_background_is_one_global_estimate(tmp_path: Path) -> None:
    frames = np.stack(
        [
            np.arange(100, dtype=np.float32).reshape(10, 10) + offset
            for offset in (0.0, 100.0, 200.0, 300.0)
        ]
    )
    source = tmp_path / "sample.tif"
    tifffile.imwrite(source, frames, photometric="minisblack")

    estimate = estimate_lunar_tiff_background(
        source,
        percentile=10.0,
        max_size_gb=2.0,
        chunk_frames=2,
    )

    np.testing.assert_allclose(
        estimate.background_range,
        estimate_lunar_background_range(frames, percentile=10.0),
    )
    assert estimate.frame_range == (0, 4)
    assert estimate.camera_backward == "identity"
