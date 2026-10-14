from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from ppg.beads_alignment import apply_affine_z_shift, fit_affine_z_shift, stack_ncc
from ppg.spatial_gamma import (
    DEGREE2_TERMS,
    boundary_anchors,
    coefficient_field_from_gamma,
    degree2_legendre_design,
    fit_center_shrunk_gamma,
)


PLANES = np.arange(140, 261, 20)
Z_UM = (PLANES - 200) * 0.01
Z_NORMALIZED = Z_UM / np.max(np.abs(Z_UM))
N_COMPONENTS = 4
NCC_THRESHOLD = 0.85
ALIGNMENT_ROUNDS = 3
MAX_CONSTANT_SHIFT_NORM_PX = 2.5
MAX_SLOPE_SHIFT_NORM_PX = 1.0
BASIS_TERMS = DEGREE2_TERMS


@dataclass(frozen=True)
class BeadsInitialization:
    arrays: dict[str, np.ndarray]
    summary: dict[str, object]
    raw_calibration: np.ndarray
    predicted_calibration: np.ndarray
    registered_predictions: np.ndarray
    registered_ncc: np.ndarray


def normalize_psf(stack: np.ndarray) -> np.ndarray:
    nonnegative = np.maximum(stack, 0.0)
    return nonnegative / np.maximum(
        nonnegative.sum(axis=(-2, -1), keepdims=True), 1e-12
    )


def legendre_design(positions_xy_normalized: np.ndarray) -> np.ndarray:
    return degree2_legendre_design(positions_xy_normalized)


def extract_bead_signal(stack: np.ndarray, centers_yx: np.ndarray) -> np.ndarray:
    raw = np.stack([stack[:, y - 12 : y + 13, x - 12 : x + 13] for y, x in centers_yx])
    if raw.shape[-2:] != (25, 25):
        raise ValueError("all bead centers must support a complete 25 x 25 ROI")
    border = np.concatenate(
        (
            raw[:, :, :2].reshape(len(raw), len(stack), -1),
            raw[:, :, -2:].reshape(len(raw), len(stack), -1),
            raw[:, :, 2:-2, :2].reshape(len(raw), len(stack), -1),
            raw[:, :, 2:-2, -2:].reshape(len(raw), len(stack), -1),
        ),
        axis=2,
    )
    background = np.median(border, axis=2)
    return np.maximum(raw - background[:, :, None, None], 0.0)[:, PLANES, 2:-2, 2:-2]


def poisson_deviance_per_photon(
    observed: np.ndarray, expected_unit_flux: np.ndarray
) -> float:
    total = float(observed.sum())
    expected = np.maximum(expected_unit_flux * total, 1e-6)
    positive = observed > 0
    deviance = expected - observed
    deviance[positive] += observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return float(2.0 * deviance.sum() / max(total, 1e-12))


def _predict_one(
    calibration: np.ndarray,
    positions: np.ndarray,
    train_mask: np.ndarray,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=N_COMPONENTS, svd_solver="full")
    scores = pca.fit_transform(
        calibration[train_mask].reshape(int(train_mask.sum()), -1)
    )
    gamma = np.linalg.lstsq(legendre_design(positions[train_mask]), scores, rcond=None)[
        0
    ]
    target_scores = legendre_design(positions[target_index : target_index + 1]) @ gamma
    prediction = normalize_psf(
        pca.inverse_transform(target_scores).reshape(len(PLANES), 21, 21)
    )
    baseline = normalize_psf(pca.mean_.reshape(len(PLANES), 21, 21))
    return prediction, baseline


def evaluate_leave_one_out(
    calibration: np.ndarray,
    signal: np.ndarray,
    positions: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    ncc_values = np.full((len(calibration), len(PLANES)), np.nan)
    nll_after = np.full_like(ncc_values, np.nan)
    predictions: dict[int, np.ndarray] = {}
    for target_index in range(len(calibration)):
        train_mask = eligible.copy()
        if eligible[target_index]:
            train_mask[target_index] = False
        prediction, _ = _predict_one(calibration, positions, train_mask, target_index)
        predictions[target_index] = prediction
        ncc_values[target_index] = stack_ncc(calibration[target_index], prediction)
        for z_index in range(len(PLANES)):
            nll_after[target_index, z_index] = poisson_deviance_per_photon(
                signal[target_index, z_index], prediction[z_index]
            )
    return ncc_values, nll_after, predictions


def align_bead_stacks(
    raw_signal: np.ndarray,
    positions: np.ndarray,
    eligible: np.ndarray,
    rounds: int,
    initial_parameters: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    parameters = (
        np.zeros((len(raw_signal), 4), dtype=np.float64)
        if initial_parameters is None
        else initial_parameters.copy()
    )
    aligned = np.stack(
        [
            np.maximum(
                apply_affine_z_shift(
                    raw_signal[index], -parameters[index], Z_NORMALIZED
                ),
                0.0,
            )
            for index in range(len(raw_signal))
        ]
    )
    for _ in range(rounds):
        calibration = normalize_psf(aligned)
        _, _, predictions = evaluate_leave_one_out(
            calibration, aligned, positions, eligible
        )
        increments = np.stack(
            [
                fit_affine_z_shift(calibration[index], predictions[index], Z_NORMALIZED)
                for index in range(len(raw_signal))
            ]
        )
        increments -= np.median(increments[eligible], axis=0, keepdims=True)
        parameters += increments
        aligned = np.stack(
            [
                np.maximum(
                    apply_affine_z_shift(
                        raw_signal[index], -parameters[index], Z_NORMALIZED
                    ),
                    0.0,
                )
                for index in range(len(raw_signal))
            ]
        )
    return aligned, parameters


def fit_beads_initialization(
    stack: np.ndarray,
    *,
    bead_ids: np.ndarray,
    centers_yx: np.ndarray,
    positions_xy_normalized: np.ndarray,
    field_shape_yx: np.ndarray,
    holdout_ids: np.ndarray,
    grid_size: int = 40,
    boundary_anchor_strength: float = 1.0,
) -> BeadsInitialization:
    if boundary_anchor_strength < 0.0:
        raise ValueError("boundary_anchor_strength must be nonnegative")
    signal = extract_bead_signal(np.asarray(stack, dtype=np.float64), centers_yx)
    raw_calibration = normalize_psf(signal)
    holdout = np.isin(bead_ids, holdout_ids)
    if int(holdout.sum()) != len(holdout_ids):
        missing = sorted(set(holdout_ids.tolist()) - set(bead_ids.tolist()))
        raise ValueError(f"holdout bead IDs missing from catalog: {missing}")
    train_candidate = ~holdout

    aligned_signal, alignment_parameters = align_bead_stacks(
        signal, positions_xy_normalized, train_candidate, ALIGNMENT_ROUNDS
    )
    aligned_calibration = normalize_psf(aligned_signal)
    train_ncc, train_after, _ = evaluate_leave_one_out(
        aligned_calibration, aligned_signal, positions_xy_normalized, train_candidate
    )
    train_deviance = np.nanmedian(train_after, axis=1)
    reference_deviance = train_deviance[train_candidate]
    deviance_median = float(np.median(reference_deviance))
    deviance_mad = float(
        1.4826 * np.median(np.abs(reference_deviance - deviance_median))
    )
    deviance_threshold = deviance_median + 3.0 * max(deviance_mad, 1e-12)
    median_ncc = np.nanmedian(train_ncc, axis=1)
    constant_shift_norm = np.linalg.norm(alignment_parameters[:, :2], axis=1)
    slope_shift_norm = np.linalg.norm(alignment_parameters[:, 2:], axis=1)
    train_failed = (
        (median_ncc < NCC_THRESHOLD)
        | ((train_deviance > deviance_threshold) & (median_ncc < 0.90))
        | (constant_shift_norm > MAX_CONSTANT_SHIFT_NORM_PX)
        | (slope_shift_norm > MAX_SLOPE_SHIFT_NORM_PX)
    )
    accepted = train_candidate & ~train_failed

    aligned_signal, alignment_parameters = align_bead_stacks(
        signal,
        positions_xy_normalized,
        accepted,
        1,
        initial_parameters=alignment_parameters,
    )
    aligned_calibration = normalize_psf(aligned_signal)
    pca = PCA(n_components=N_COMPONENTS, svd_solver="full")
    scores = pca.fit_transform(
        aligned_calibration[accepted].reshape(int(accepted.sum()), -1)
    )

    anchors, anchor_density_weights = boundary_anchors(
        positions_xy_normalized[accepted]
    )
    center_neighbors = np.argsort(
        np.linalg.norm(positions_xy_normalized[accepted] - 0.5, axis=1)
    )[:8]
    center_score_target = np.median(scores[center_neighbors], axis=0)
    gamma = fit_center_shrunk_gamma(
        legendre_design(positions_xy_normalized[accepted]),
        scores,
        legendre_design(anchors),
        anchor_density_weights,
        anchor_target=center_score_target,
        anchor_strength=float(boundary_anchor_strength),
        ridge=0.01,
    )

    predicted_scores = legendre_design(positions_xy_normalized) @ gamma
    predicted = normalize_psf(
        pca.inverse_transform(predicted_scores).reshape(
            len(bead_ids), len(PLANES), 21, 21
        )
    )
    for index in np.flatnonzero(holdout):
        alignment_parameters[index] = fit_affine_z_shift(
            raw_calibration[index],
            predicted[index],
            Z_NORMALIZED,
            constant_bound_px=3.0,
            slope_bound_px=1.0,
        )
    registered = np.stack(
        [
            normalize_psf(
                apply_affine_z_shift(
                    predicted[index], alignment_parameters[index], Z_NORMALIZED
                )
            )
            for index in range(len(bead_ids))
        ]
    )
    registered_ncc = np.stack(
        [
            stack_ncc(raw_calibration[index], registered[index])
            for index in range(len(bead_ids))
        ]
    )

    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    field = coefficient_field_from_gamma(gamma, grid)
    holdout_medians = np.median(registered_ncc[holdout], axis=1)
    arrays = {
        "schema": np.asarray("ppg.pca_map.v1"),
        "initialization_schema": np.asarray("ppg.beads_gamma36.v1"),
        "pca_mean": pca.mean_,
        "pca_components": pca.components_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "coefficient_field": field,
        "coefficient_positions_xy_normalized": positions_xy_normalized[accepted],
        "coefficient_values": scores,
        "xy_grid_normalized": grid,
        "train_z_um": Z_UM,
        "roi_shape_yx": np.asarray([21, 21]),
        "field_shape_yx": np.asarray(field_shape_yx),
        "accepted_ids": bead_ids[accepted],
        "rejected_ids": bead_ids[train_candidate & ~accepted],
        "heldout_ids": bead_ids[holdout],
        "gamma": gamma,
        "basis_terms": BASIS_TERMS,
        "boundary_anchor_positions_xy_normalized": anchors,
        "boundary_anchor_density_weights": anchor_density_weights,
        "boundary_anchor_center_score_target": center_score_target,
        "boundary_anchor_strength": np.asarray(boundary_anchor_strength),
        "gamma_ridge": np.asarray(0.01),
        "alignment_parameters_dx0_dy0_dxz_dyz": alignment_parameters,
    }
    summary: dict[str, object] = {
        "schema": "ppg.beads_gamma36_manifest.v1",
        "method": "4 PCA modes x 9 degree-2 2D Legendre basis functions",
        "gamma_parameter_count": 36,
        "psf_support_px": 21,
        "calibration_only_nuisance_parameters_per_bead": 4,
        "holdout_protocol": (
            "five fixed spatial beads excluded from alignment reference, PCA, gamma, and gate; "
            "only evaluation-time xy nuisance fitted"
        ),
        "gate": (
            "reject median leave-one-out NCC < 0.85, excessive affine-z shift, or robust "
            "Poisson deviance outlier when NCC < 0.90"
        ),
        "boundary_regularization": (
            "16 density-weighted boundary anchors shrink sparse edges toward the median PCA "
            "score of the 8 nearest center-FOV accepted beads"
        ),
        "boundary_anchor_strength": float(boundary_anchor_strength),
        "gamma_ridge": 0.01,
        "input_bead_count": int(len(bead_ids)),
        "accepted_training_count": int(accepted.sum()),
        "accepted_training_bead_ids": bead_ids[accepted].tolist(),
        "rejected_training_bead_ids": bead_ids[train_candidate & ~accepted].tolist(),
        "heldout_bead_ids": bead_ids[holdout].tolist(),
        "heldout_positions_xy_normalized": positions_xy_normalized[holdout].tolist(),
        "heldout_ncc_per_bead": {
            str(bead_id): float(value)
            for bead_id, value in zip(bead_ids[holdout], holdout_medians, strict=True)
        },
        "heldout_ncc_median": float(np.median(holdout_medians)),
        "heldout_ncc_min": float(np.min(holdout_medians)),
        "poisson_deviance_per_photon_threshold": deviance_threshold,
        "grid_size": int(grid_size),
    }
    return BeadsInitialization(
        arrays=arrays,
        summary=summary,
        raw_calibration=raw_calibration,
        predicted_calibration=predicted,
        registered_predictions=registered,
        registered_ncc=registered_ncc,
    )
