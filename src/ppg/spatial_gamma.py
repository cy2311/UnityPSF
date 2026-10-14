from __future__ import annotations

import numpy as np


DEGREE2_TERMS = np.asarray([(i, j) for i in range(3) for j in range(3)])


def degree2_legendre_design(positions_xy_normalized: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_xy_normalized, dtype=np.float64)
    x = 2.0 * positions[:, 0] - 1.0
    y = 2.0 * positions[:, 1] - 1.0
    x_basis = np.polynomial.legendre.legvander(x, 2)
    y_basis = np.polynomial.legendre.legvander(y, 2)
    return np.stack(
        [x_basis[:, i] * y_basis[:, j] for i, j in DEGREE2_TERMS],
        axis=1,
    )


def coefficient_field_from_gamma(gamma: np.ndarray, grid: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(grid, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    values = degree2_legendre_design(np.c_[xx.ravel(), yy.ravel()]) @ gamma
    return values.reshape(len(coordinates), len(coordinates), gamma.shape[1]).transpose(
        2, 0, 1
    )


def boundary_anchors(
    training_positions: np.ndarray,
    *,
    samples_per_axis: int = 5,
    density_scale: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.linspace(0.0, 1.0, samples_per_axis)
    anchors = np.asarray(
        [
            (x, y)
            for x in coordinates
            for y in coordinates
            if x in (0.0, 1.0) or y in (0.0, 1.0)
        ]
    )
    nearest_distance = np.min(
        np.linalg.norm(anchors[:, None, :] - training_positions[None, :, :], axis=2),
        axis=1,
    )
    density_weights = np.clip((nearest_distance / density_scale) ** 2, 0.0, 1.0)
    return anchors, density_weights


def fit_center_shrunk_gamma(
    design: np.ndarray,
    scores: np.ndarray,
    anchor_design: np.ndarray,
    density_weights: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    anchor_target: np.ndarray | None = None,
    anchor_strength: float = 1.0,
    ridge: float = 0.01,
) -> np.ndarray:
    if sample_weights is not None:
        root_weights = np.sqrt(np.asarray(sample_weights, dtype=np.float64))[:, None]
        design = root_weights * design
        scores = root_weights * scores
    weighted_anchors = (
        np.sqrt(anchor_strength * density_weights)[:, None] * anchor_design
    )
    target = (
        np.zeros(scores.shape[1], dtype=scores.dtype)
        if anchor_target is None
        else np.asarray(anchor_target, dtype=scores.dtype)
    )
    augmented_design = np.vstack((design, weighted_anchors))
    augmented_scores = np.vstack(
        (scores, np.sqrt(anchor_strength * density_weights)[:, None] * target[None, :])
    )
    penalty = np.diag(np.r_[0.0, np.ones(design.shape[1] - 1)])
    return np.linalg.solve(
        augmented_design.T @ augmented_design + ridge * penalty,
        augmented_design.T @ augmented_scores,
    )


def apply_gamma_updates(
    gamma: np.ndarray,
    *,
    positions_xy_normalized: np.ndarray,
    deltas: np.ndarray,
    weights: np.ndarray,
    ridge: float = 0.01,
    boundary_anchor_strength: float = 1.0,
    max_abs_delta: float = 0.05,
) -> np.ndarray:
    positions = np.asarray(positions_xy_normalized, dtype=np.float64)
    if len(positions) < 4:
        raise ValueError("gamma update requires at least four accepted emitters")
    anchors, density_weights = boundary_anchors(positions)
    delta_gamma = fit_center_shrunk_gamma(
        degree2_legendre_design(positions),
        np.asarray(deltas, dtype=np.float64),
        degree2_legendre_design(anchors),
        density_weights,
        sample_weights=np.asarray(weights, dtype=np.float64),
        anchor_strength=boundary_anchor_strength,
        ridge=ridge,
    )
    step = np.clip(delta_gamma, -max_abs_delta, max_abs_delta)
    return (np.asarray(gamma, dtype=np.float64) + step).astype(np.asarray(gamma).dtype)
