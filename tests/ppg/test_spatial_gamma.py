from __future__ import annotations

import numpy as np

from ppg.spatial_gamma import (
    apply_gamma_updates,
    boundary_anchors,
    coefficient_field_from_gamma,
    degree2_legendre_design,
    fit_center_shrunk_gamma,
)


def _legendre_design(positions: np.ndarray) -> np.ndarray:
    x = 2.0 * positions[:, 0] - 1.0
    y = 2.0 * positions[:, 1] - 1.0
    px = np.polynomial.legendre.legvander(x, 2)
    py = np.polynomial.legendre.legvander(y, 2)
    return np.stack([px[:, i] * py[:, j] for i in range(3) for j in range(3)], axis=1)


def test_sparse_boundary_is_shrunk_toward_zero_score() -> None:
    rng = np.random.default_rng(7)
    positions = rng.uniform((0.20, 0.05), (1.0, 0.82), size=(32, 2))
    design = _legendre_design(positions)
    scores = rng.normal(0.0, 0.025, size=(len(positions), 2))
    anchors, density_weights = boundary_anchors(positions, samples_per_axis=5)
    anchor_design = _legendre_design(anchors)

    ordinary = np.linalg.lstsq(design, scores, rcond=None)[0]
    shrunk = fit_center_shrunk_gamma(
        design,
        scores,
        anchor_design,
        density_weights,
        anchor_strength=0.1,
        ridge=0.01,
    )
    sparse_corner = _legendre_design(np.asarray([[0.0, 1.0]]))

    assert (
        density_weights[np.argmax(np.sum((anchors - (0.0, 1.0)) ** 2, axis=1) == 0)]
        == 1.0
    )
    assert np.linalg.norm(sparse_corner @ shrunk) < np.linalg.norm(
        sparse_corner @ ordinary
    )
    assert np.mean((design @ shrunk - scores) ** 2) < 0.002


def test_sparse_boundary_can_shrink_toward_center_score() -> None:
    rng = np.random.default_rng(11)
    positions = rng.uniform((0.20, 0.05), (1.0, 0.82), size=(32, 2))
    design = _legendre_design(positions)
    scores = rng.normal(0.0, 0.03, size=(len(positions), 2))
    target = np.asarray([0.08, -0.04])
    anchors, density_weights = boundary_anchors(positions, samples_per_axis=5)
    corner_design = _legendre_design(np.asarray([[0.0, 1.0]]))
    ordinary = np.linalg.lstsq(design, scores, rcond=None)[0]
    shrunk = fit_center_shrunk_gamma(
        design,
        scores,
        _legendre_design(anchors),
        density_weights,
        anchor_target=target,
        anchor_strength=0.1,
        ridge=0.01,
    )

    assert np.linalg.norm(corner_design @ shrunk - target) < np.linalg.norm(
        corner_design @ ordinary - target
    )


def test_physical_update_changes_only_36_gamma_parameters() -> None:
    positions = np.asarray([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.5, 0.5]])
    expected_delta_gamma = np.zeros((9, 4))
    expected_delta_gamma[0] = [0.01, -0.02, 0.03, -0.01]
    expected_delta_gamma[1] = [0.02, 0.01, 0.0, -0.02]
    deltas = degree2_legendre_design(positions) @ expected_delta_gamma
    initial = np.zeros((9, 4))

    updated = apply_gamma_updates(
        initial,
        positions_xy_normalized=positions,
        deltas=deltas,
        weights=np.ones(len(positions)),
        ridge=1e-8,
        boundary_anchor_strength=0.0,
        max_abs_delta=1.0,
    )
    field = coefficient_field_from_gamma(updated, np.linspace(0.0, 1.0, 40))

    assert updated.shape == (9, 4)
    assert field.shape == (4, 40, 40)
    np.testing.assert_allclose(
        degree2_legendre_design(positions) @ updated,
        deltas,
        atol=1e-6,
    )


def test_gamma_update_rejects_too_few_emitters() -> None:
    try:
        apply_gamma_updates(
            np.zeros((9, 4)),
            positions_xy_normalized=np.zeros((3, 2)),
            deltas=np.zeros((3, 4)),
            weights=np.ones(3),
        )
    except ValueError as exc:
        assert "at least four" in str(exc)
    else:
        raise AssertionError("an underdetermined gamma update must be rejected")
