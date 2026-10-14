from __future__ import annotations

import inspect

import numpy as np
import pytest

from ppg.beads_initialization import (
    BASIS_TERMS,
    fit_beads_initialization,
    legendre_design,
    normalize_psf,
)


def test_degree2_legendre_design_has_nine_terms() -> None:
    positions = np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    design = legendre_design(positions)

    assert BASIS_TERMS.tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2],
    ]
    assert design.shape == (3, 9)
    np.testing.assert_allclose(
        design[1], [1.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.25]
    )


def test_psf_normalization_is_nonnegative_unit_flux_per_plane() -> None:
    stack = np.asarray([[[-1.0, 1.0], [2.0, 1.0]], [[3.0, 1.0], [0.0, 0.0]]])

    normalized = normalize_psf(stack)

    assert np.all(normalized >= 0.0)
    np.testing.assert_allclose(normalized.sum(axis=(-2, -1)), 1.0)


def test_default_boundary_anchor_strength_is_one() -> None:
    parameter = inspect.signature(fit_beads_initialization).parameters[
        "boundary_anchor_strength"
    ]

    assert parameter.default == 1.0


def test_negative_boundary_anchor_strength_is_rejected_before_fitting() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        fit_beads_initialization(
            np.empty((0, 0, 0)),
            bead_ids=np.empty(0),
            centers_yx=np.empty((0, 2), dtype=int),
            positions_xy_normalized=np.empty((0, 2)),
            field_shape_yx=np.asarray([640, 640]),
            holdout_ids=np.empty(0),
            boundary_anchor_strength=-0.1,
        )
