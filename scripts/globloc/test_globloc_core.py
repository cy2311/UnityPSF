import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from globloc_core import build_dT_all, match_frame_candidates, roi_start_xy
from globloc_ctypes import GlobLocFit


def test_build_dT_all_uses_channel_one_minus_channel_zero_for_local_roi_shift():
    left = np.asarray([[10.25, 11.5]], dtype=np.float32)
    right = np.asarray([[11.0, 11.0]], dtype=np.float32)

    dts = build_dT_all(left, right)

    np.testing.assert_allclose(dts.shape, (1, 4, 5))
    np.testing.assert_allclose(dts[0, 0, :], 0.0)
    np.testing.assert_allclose(dts[0, 1, :2], [-0.75, 0.5])
    np.testing.assert_allclose(dts[0, 2:, :], 1.0)


def test_match_frame_candidates_is_one_to_one_and_distance_limited():
    left = np.asarray([[10.0, 10.0], [30.0, 30.0]], dtype=np.float32)
    right = np.asarray([[10.5, 10.0], [30.4, 30.3], [10.2, 10.1]], dtype=np.float32)

    pairs, distances = match_frame_candidates(left, right, max_distance=1.0)

    np.testing.assert_array_equal(pairs, [[0, 2], [1, 1]])
    np.testing.assert_allclose(distances, [0.2236068, 0.5], rtol=1e-5)


def test_roi_start_is_stable_for_subpixel_candidate():
    assert roi_start_xy(20.25, 20.5, 20) == (10, 10)
    assert roi_start_xy(10.0, 10.0, 20) == (0, 0)


def test_parameter_count_is_shared_xyz_plus_per_channel_photons_and_background():
    shared = np.asarray([[1, 1, 1, 0, 0]], dtype=np.int32)

    assert GlobLocFit.parameter_count(shared, 2) == 7
