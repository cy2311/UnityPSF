from __future__ import annotations

import pytest
import torch

from unity_psf.localization.simulator import LocalizationSimulatorConfig, _sample_counts


def test_density_sampling_preserves_target_mean_without_upper_tail_clipping() -> None:
    config = LocalizationSimulatorConfig(
        batch_size=4096,
        height=96,
        width=96,
        pixel_size_nm_x=101.11,
        pixel_size_nm_y=98.83,
        emitter_density_um2=0.5,
    )
    counts = _sample_counts(
        config,
        batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(17),
    )
    expected = 0.5 * (96 * 101.11 / 1000.0) * (96 * 98.83 / 1000.0)

    assert expected == pytest.approx(46.0463675904)
    assert float(counts.float().mean()) == pytest.approx(expected, rel=0.02)
    assert int(counts.max()) > int(expected)


def test_density_sampling_keeps_valid_zero_emitter_frames() -> None:
    config = LocalizationSimulatorConfig(
        batch_size=4096,
        height=8,
        width=8,
        pixel_size_nm_x=100.0,
        pixel_size_nm_y=100.0,
        emitter_density_um2=0.5,
    )

    counts = _sample_counts(
        config,
        batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(19),
    )

    assert int(counts.min()) == 0
