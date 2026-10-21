from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldFitConfig:
    image_shape_hw: tuple[int, int] = (150, 150)
    active_padding_px: int = 15
    spatial_degree: int = 2
    global_z_degree: int = 3
    z_range_nm: tuple[float, float] = (33.3, 3962.7)
    ridge: float = 1e-3
    split_modulo: int = 5
    spatial_block_grid: int = 5
    bootstrap_iterations: int = 200
    random_seed: int = 20260723


@dataclass(frozen=True)
class FieldFitResult:
    fd_z_offset_nm: np.ndarray
    candidate_map_nm: np.ndarray
    uncertainty_std_nm: np.ndarray
    active_fov_mask: np.ndarray
    field_accepted: bool
    spatial_terms: tuple[tuple[int, int], ...]
    spatial_coefficients_nm: np.ndarray
    global_z_coefficients_nm: np.ndarray
    metrics: dict[str, dict[str, float | int | list[float]]]


def fit_field_dependent_z(
    x_px: np.ndarray,
    y_px: np.ndarray,
    z_gt_nm: np.ndarray,
    z_residual_nm: np.ndarray,
    frame_index: np.ndarray,
    *,
    config: FieldFitConfig | None = None,
) -> FieldFitResult:
    cfg = FieldFitConfig() if config is None else config
    x, y, z, residual, frames = _validated_inputs(x_px, y_px, z_gt_nm, z_residual_nm, frame_index)
    active_mask, grid_x, grid_y = _active_grid(cfg)
    terms = _spatial_terms(cfg.spatial_degree)
    grid_spatial_raw = _spatial_design(grid_x.reshape(-1), grid_y.reshape(-1), terms, cfg.image_shape_hw)
    active_flat = active_mask.reshape(-1)
    gauge_mean = grid_spatial_raw[active_flat].mean(axis=0)
    grid_spatial = grid_spatial_raw - gauge_mean[None, :]
    observed_spatial = _spatial_design(x, y, terms, cfg.image_shape_hw) - gauge_mean[None, :]
    global_design = _global_z_design(z, cfg)

    block_ids = _spatial_block_ids(x, y, cfg)
    frame_holdout = frames % cfg.split_modulo == 0
    spatial_holdout = _spatial_holdout_mask(block_ids, cfg)
    train = ~frame_holdout & ~spatial_holdout
    frame_validation = frame_holdout & ~spatial_holdout
    spatial_validation = ~frame_holdout & spatial_holdout
    if min(int(train.sum()), int(frame_validation.sum()), int(spatial_validation.sum())) == 0:
        raise ValueError("Frame/spatial split produced an empty partition.")

    global_coeff_cv = _ridge_solve(global_design[train], residual[train], ridge=0.0)
    combined_design = np.concatenate([global_design, observed_spatial], axis=1)
    combined_coeff_cv = _ridge_solve(
        combined_design[train],
        residual[train],
        ridge=cfg.ridge,
        unregularized_columns=global_design.shape[1],
    )
    global_prediction = global_design @ global_coeff_cv
    fd_prediction = combined_design @ combined_coeff_cv

    rng = np.random.default_rng(cfg.random_seed)
    metrics: dict[str, dict[str, float | int | list[float]]] = {}
    for name, mask in (
        ("train", train),
        ("frame_holdout", frame_validation),
        ("spatial_holdout", spatial_validation),
    ):
        metrics[name] = _comparison_metrics(
            residual[mask],
            global_prediction[mask],
            fd_prediction[mask],
            block_ids[mask],
            bootstrap_iterations=cfg.bootstrap_iterations,
            rng=rng,
        )

    field_accepted = all(
        float(metrics[name]["bootstrap_ci95_nm"][0]) > 0.0
        for name in ("frame_holdout", "spatial_holdout")
    )

    combined_coeff = _ridge_solve(
        combined_design,
        residual,
        ridge=cfg.ridge,
        unregularized_columns=global_design.shape[1],
    )
    global_coeff = combined_coeff[: global_design.shape[1]]
    spatial_coeff = combined_coeff[global_design.shape[1] :]
    candidate = (grid_spatial @ spatial_coeff).reshape(cfg.image_shape_hw)
    candidate -= float(candidate[active_mask].mean())
    uncertainty = _bootstrap_map_uncertainty(
        combined_design,
        residual,
        block_ids,
        grid_spatial,
        global_columns=global_design.shape[1],
        config=cfg,
        rng=rng,
    ).reshape(cfg.image_shape_hw)
    fd_map = candidate if field_accepted else np.zeros_like(candidate)

    return FieldFitResult(
        fd_z_offset_nm=fd_map.astype(np.float32),
        candidate_map_nm=candidate.astype(np.float32),
        uncertainty_std_nm=uncertainty.astype(np.float32),
        active_fov_mask=active_mask,
        field_accepted=field_accepted,
        spatial_terms=terms,
        spatial_coefficients_nm=spatial_coeff.astype(np.float64),
        global_z_coefficients_nm=global_coeff.astype(np.float64),
        metrics=metrics,
    )


def _validated_inputs(
    x_px: np.ndarray,
    y_px: np.ndarray,
    z_gt_nm: np.ndarray,
    z_residual_nm: np.ndarray,
    frame_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = [np.asarray(value).reshape(-1) for value in (x_px, y_px, z_gt_nm, z_residual_nm, frame_index)]
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError("All field-fit inputs must have the same length.")
    x, y, z, residual = (array.astype(np.float64) for array in arrays[:4])
    frames = arrays[4].astype(np.int64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(residual)
    if not np.all(finite):
        raise ValueError("Field-fit inputs must be finite.")
    return x, y, z, residual, frames


def _active_grid(config: FieldFitConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = config.image_shape_hw
    padding = int(config.active_padding_px)
    active = np.zeros((height, width), dtype=bool)
    active[padding : height - padding, padding : width - padding] = True
    yy, xx = np.mgrid[:height, :width]
    return active, xx.astype(np.float64), yy.astype(np.float64)


def _spatial_terms(degree: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (px, py)
        for total in range(1, int(degree) + 1)
        for px in range(total, -1, -1)
        for py in (total - px,)
    )


def _spatial_design(
    x_px: np.ndarray,
    y_px: np.ndarray,
    terms: tuple[tuple[int, int], ...],
    image_shape_hw: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape_hw
    xn = -1.0 + 2.0 * np.asarray(x_px, dtype=np.float64) / float(width)
    yn = -1.0 + 2.0 * np.asarray(y_px, dtype=np.float64) / float(height)
    return np.stack([_legendre(px, xn) * _legendre(py, yn) for px, py in terms], axis=1)


def _global_z_design(z_nm: np.ndarray, config: FieldFitConfig) -> np.ndarray:
    z_min, z_max = config.z_range_nm
    normalized = 2.0 * (np.asarray(z_nm, dtype=np.float64) - z_min) / (z_max - z_min) - 1.0
    return np.stack([_legendre(degree, normalized) for degree in range(config.global_z_degree + 1)], axis=1)


def _legendre(degree: int, values: np.ndarray) -> np.ndarray:
    if degree == 0:
        return np.ones_like(values)
    if degree == 1:
        return values
    previous_previous = np.ones_like(values)
    previous = values
    for current_degree in range(2, int(degree) + 1):
        current = (
            (2.0 * current_degree - 1.0) * values * previous
            - (current_degree - 1.0) * previous_previous
        ) / current_degree
        previous_previous, previous = previous, current
    return previous


def _ridge_solve(
    design: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
    unregularized_columns: int = 0,
) -> np.ndarray:
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    penalty[: int(unregularized_columns), : int(unregularized_columns)] = 0.0
    return np.linalg.solve(gram + penalty, design.T @ target)


def _spatial_block_ids(x_px: np.ndarray, y_px: np.ndarray, config: FieldFitConfig) -> np.ndarray:
    height, width = config.image_shape_hw
    padding = float(config.active_padding_px)
    grid = int(config.spatial_block_grid)
    active_width = width - 2.0 * padding
    active_height = height - 2.0 * padding
    cell_x = np.clip(((x_px - padding) / active_width * grid).astype(np.int64), 0, grid - 1)
    cell_y = np.clip(((y_px - padding) / active_height * grid).astype(np.int64), 0, grid - 1)
    return cell_y * grid + cell_x


def _spatial_holdout_mask(block_ids: np.ndarray, config: FieldFitConfig) -> np.ndarray:
    grid = int(config.spatial_block_grid)
    cell_x = block_ids % grid
    cell_y = block_ids // grid
    return (cell_x + 2 * cell_y) % int(config.split_modulo) == 0


def _comparison_metrics(
    target: np.ndarray,
    global_prediction: np.ndarray,
    fd_prediction: np.ndarray,
    block_ids: np.ndarray,
    *,
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> dict[str, float | int | list[float]]:
    global_error = target - global_prediction
    fd_error = target - fd_prediction
    global_rmse = float(np.sqrt(np.mean(global_error**2)))
    fd_rmse = float(np.sqrt(np.mean(fd_error**2)))
    improvements = _block_bootstrap_rmse_improvement(
        global_error,
        fd_error,
        block_ids,
        iterations=bootstrap_iterations,
        rng=rng,
    )
    ci = np.percentile(improvements, [2.5, 97.5]) if improvements.size else np.array([np.nan, np.nan])
    return {
        "n": int(target.shape[0]),
        "global_rmse_nm": global_rmse,
        "fd_rmse_nm": fd_rmse,
        "rmse_improvement_nm": global_rmse - fd_rmse,
        "bootstrap_ci95_nm": [float(ci[0]), float(ci[1])],
    }


def _block_bootstrap_rmse_improvement(
    global_error: np.ndarray,
    fd_error: np.ndarray,
    block_ids: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_blocks = np.unique(block_ids)
    if iterations <= 0 or unique_blocks.size == 0:
        return np.empty((0,), dtype=np.float64)
    rows_by_block = {block: np.flatnonzero(block_ids == block) for block in unique_blocks}
    results = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        sampled_blocks = rng.choice(unique_blocks, size=unique_blocks.size, replace=True)
        rows = np.concatenate([rows_by_block[block] for block in sampled_blocks])
        results[iteration] = np.sqrt(np.mean(global_error[rows] ** 2)) - np.sqrt(np.mean(fd_error[rows] ** 2))
    return results


def _bootstrap_map_uncertainty(
    combined_design: np.ndarray,
    target: np.ndarray,
    block_ids: np.ndarray,
    grid_spatial: np.ndarray,
    *,
    global_columns: int,
    config: FieldFitConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    iterations = int(config.bootstrap_iterations)
    if iterations <= 1:
        return np.zeros(grid_spatial.shape[0], dtype=np.float64)
    unique_blocks = np.unique(block_ids)
    rows_by_block = {block: np.flatnonzero(block_ids == block) for block in unique_blocks}
    maps = np.empty((iterations, grid_spatial.shape[0]), dtype=np.float32)
    for iteration in range(iterations):
        sampled_blocks = rng.choice(unique_blocks, size=unique_blocks.size, replace=True)
        rows = np.concatenate([rows_by_block[block] for block in sampled_blocks])
        coefficients = _ridge_solve(
            combined_design[rows],
            target[rows],
            ridge=config.ridge,
            unregularized_columns=global_columns,
        )
        maps[iteration] = grid_spatial @ coefficients[global_columns:]
    return maps.std(axis=0, ddof=1, dtype=np.float64)


__all__ = ["FieldFitConfig", "FieldFitResult", "fit_field_dependent_z"]
