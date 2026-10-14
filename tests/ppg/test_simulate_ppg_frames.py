from __future__ import annotations

import numpy as np

from ppg.simulate_ppg_frames import (
    edge_emitter_mask,
    sliding_field_origins,
)


def test_sliding_origins_cover_640_field_with_100_nonoverlapping_64_rois() -> None:
    origins = sliding_field_origins(
        field_shape_yx=(640, 640),
        roi_shape_yx=(64, 64),
        stride_px=64,
    )

    assert origins.shape == (100, 2)
    assert tuple(origins[0]) == (0, 0)
    assert tuple(origins[-1]) == (576, 576)
    assert len({tuple(origin) for origin in origins}) == 100


def test_sliding_origins_include_last_valid_edge_when_stride_is_inexact() -> None:
    origins = sliding_field_origins(
        field_shape_yx=(130, 130),
        roi_shape_yx=(64, 64),
        stride_px=64,
    )

    assert {int(value) for value in origins[:, 0]} == {0, 64, 66}
    assert {int(value) for value in origins[:, 1]} == {0, 64, 66}


def test_edge_emitter_mask_uses_half_psf_support() -> None:
    xy = np.asarray(
        [
            [0.0, 32.0],
            [9.9, 32.0],
            [10.0, 32.0],
            [32.0, 53.0],
            [32.0, 54.0],
            [63.0, 63.0],
        ]
    )

    np.testing.assert_array_equal(
        edge_emitter_mask(xy, roi_shape_yx=(64, 64), psf_support_px=21),
        [True, True, False, False, True, True],
    )
