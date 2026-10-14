from __future__ import annotations

import numpy as np
from scipy.ndimage import shift
from scipy.optimize import minimize


def stack_ncc(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_flat = left.reshape(len(left), -1) - left.reshape(len(left), -1).mean(
        axis=1, keepdims=True
    )
    right_flat = right.reshape(len(right), -1) - right.reshape(len(right), -1).mean(
        axis=1, keepdims=True
    )
    numerator = np.sum(left_flat * right_flat, axis=1)
    denominator = np.sqrt(
        np.sum(left_flat * left_flat, axis=1) * np.sum(right_flat * right_flat, axis=1)
    )
    return numerator / np.maximum(denominator, 1e-12)


def apply_affine_z_shift(
    stack: np.ndarray,
    parameters: np.ndarray,
    z_normalized: np.ndarray,
) -> np.ndarray:
    dx = parameters[0] + parameters[2] * z_normalized
    dy = parameters[1] + parameters[3] * z_normalized
    return np.stack(
        [
            shift(
                image,
                (shift_y, shift_x),
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
            for image, shift_x, shift_y in zip(stack, dx, dy, strict=True)
        ]
    )


def fit_affine_z_shift(
    observed: np.ndarray,
    prediction: np.ndarray,
    z_normalized: np.ndarray,
    *,
    constant_bound_px: float = 2.0,
    slope_bound_px: float = 1.0,
) -> np.ndarray:
    def objective(parameters: np.ndarray) -> float:
        shifted = apply_affine_z_shift(prediction, parameters, z_normalized)
        return 1.0 - float(np.mean(stack_ncc(observed, shifted)))

    result = minimize(
        objective,
        np.zeros(4, dtype=np.float64),
        method="Powell",
        bounds=[
            (-constant_bound_px, constant_bound_px),
            (-constant_bound_px, constant_bound_px),
            (-slope_bound_px, slope_bound_px),
            (-slope_bound_px, slope_bound_px),
        ],
        options={"maxiter": 120, "xtol": 1e-3, "ftol": 1e-5},
    )
    return result.x
