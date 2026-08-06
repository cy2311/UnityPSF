from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .calibration import profile_photometry
from .local_fit import OracleObservations
from .vector_model import DoubleHelixVectorPSF


@dataclass(frozen=True)
class FieldGammaFitConfig:
    image_shape_hw: tuple[int, int] = (150, 150)
    active_padding_px: int = 15
    spatial_degree: int = 2
    spatial_block_grid: int = 5
    split_modulo: int = 5
    max_training_emitters: int = 3000
    max_validation_emitters: int = 1200
    batch_size: int = 64
    adam_steps: int = 600
    learning_rate_nm: float = 0.5
    spatial_regularization: float = 1e-5
    noise_variance_adu2: float = 2504.757
    bootstrap_iterations: int = 500
    na: float = 1.27
    wavelength_nm: float = 660.0
    refractive_index: float = 1.33
    pixel_size_nm: float = 200.0
    npupil: int = 128
    psf_size: int = 31
    seed: int = 20260723
    device: str = "cuda"


@dataclass(frozen=True)
class FieldPartition:
    train: np.ndarray
    frame_holdout: np.ndarray
    spatial_holdout: np.ndarray
    block_ids: np.ndarray


@dataclass(frozen=True)
class FieldGammaFitResult:
    gamma_nm: np.ndarray
    candidate_gamma_nm: np.ndarray
    mode_order: tuple[tuple[int, int], ...]
    field_accepted: bool
    spatial_terms: tuple[tuple[int, int], ...]
    metrics: dict[str, Any]
    partition: FieldPartition


def spatial_gamma_terms(degree: int) -> tuple[tuple[int, int], ...]:
    if int(degree) < 1:
        raise ValueError("degree must be positive.")
    return tuple(
        (px, total - px)
        for total in range(1, int(degree) + 1)
        for px in range(total, -1, -1)
    )


def assemble_direct_gamma(
    global_gamma_nm: np.ndarray,
    spatial_coefficients_nm: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    global_values = np.asarray(global_gamma_nm, dtype=np.float32).reshape(-1)
    spatial_values = np.asarray(spatial_coefficients_nm, dtype=np.float32)
    terms = spatial_gamma_terms(degree)
    if spatial_values.shape != (global_values.size, len(terms)):
        raise ValueError("spatial_coefficients_nm must have shape (C, spatial_term_count).")
    gamma = np.zeros((global_values.size, int(degree) + 1, int(degree) + 1), dtype=np.float32)
    gamma[:, 0, 0] = global_values
    for column, (px, py) in enumerate(terms):
        gamma[:, px, py] = spatial_values[:, column]
    return gamma


def partition_field_observations(
    x_px: np.ndarray,
    y_px: np.ndarray,
    frame_index: np.ndarray,
    *,
    config: FieldGammaFitConfig,
) -> FieldPartition:
    x = np.asarray(x_px, dtype=np.float64).reshape(-1)
    y = np.asarray(y_px, dtype=np.float64).reshape(-1)
    frames = np.asarray(frame_index, dtype=np.int64).reshape(-1)
    if not (x.shape == y.shape == frames.shape):
        raise ValueError("x_px, y_px, and frame_index must have matching shapes.")
    block_ids = _spatial_block_ids(x, y, config)
    grid = int(config.spatial_block_grid)
    cell_x = block_ids % grid
    cell_y = block_ids // grid
    spatial_flag = (cell_x + 2 * cell_y) % int(config.split_modulo) == 0
    frame_flag = frames % int(config.split_modulo) == 0
    return FieldPartition(
        train=~frame_flag & ~spatial_flag,
        frame_holdout=frame_flag & ~spatial_flag,
        spatial_holdout=~frame_flag & spatial_flag,
        block_ids=block_ids,
    )


def select_field_gamma(
    global_gamma_nm: np.ndarray,
    candidate_gamma_nm: np.ndarray,
    *,
    spatial_loss_improvement_ci95: tuple[float, float],
    baseline_median_ncc: float,
    candidate_median_ncc: float,
) -> tuple[np.ndarray, bool]:
    candidate = np.asarray(candidate_gamma_nm, dtype=np.float32)
    global_values = np.asarray(global_gamma_nm, dtype=np.float32).reshape(-1)
    if candidate.ndim != 3 or candidate.shape[0] != global_values.size:
        raise ValueError("candidate_gamma_nm must have shape (C,Px,Py).")
    accepted = bool(
        float(spatial_loss_improvement_ci95[0]) > 0.0
        and float(candidate_median_ncc) >= float(baseline_median_ncc)
    )
    if accepted:
        return candidate.copy(), True
    selected = np.zeros_like(candidate)
    selected[:, 0, 0] = global_values
    return selected, False


def fit_field_gamma(
    observations: OracleObservations,
    *,
    global_gamma_nm: np.ndarray,
    mode_order: tuple[tuple[int, int], ...],
    config: FieldGammaFitConfig,
    carrier_complex: np.ndarray | torch.Tensor | None = None,
) -> FieldGammaFitResult:
    global_values = np.asarray(global_gamma_nm, dtype=np.float32).reshape(-1)
    if global_values.size != len(mode_order):
        raise ValueError("global_gamma_nm length must match mode_order.")
    if observations.patches.shape[0] != observations.x_px.shape[0]:
        raise ValueError("Observation arrays must share the same leading dimension.")
    partition = partition_field_observations(
        observations.x_px,
        observations.y_px,
        observations.frame_index,
        config=config,
    )
    rng = np.random.default_rng(config.seed)
    train_rows = _sample_rows(partition.train, config.max_training_emitters, rng)
    if train_rows.size == 0:
        raise ValueError("Field partition produced no training observations.")

    device = torch.device(config.device)
    patches = torch.as_tensor(observations.patches, dtype=torch.float32, device=device)
    x_values = torch.as_tensor(observations.x_px, dtype=torch.float32, device=device)
    y_values = torch.as_tensor(observations.y_px, dtype=torch.float32, device=device)
    z_values = torch.as_tensor(observations.z_gt_nm, dtype=torch.float32, device=device)
    global_t = torch.as_tensor(global_values, dtype=torch.float32, device=device)
    terms = spatial_gamma_terms(config.spatial_degree)
    spatial = torch.zeros(
        (len(mode_order), len(terms)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    model = DoubleHelixVectorPSF(
        mode_order=mode_order,
        na=config.na,
        wavelength_nm=config.wavelength_nm,
        pixel_size_nm=config.pixel_size_nm,
        refractive_index=config.refractive_index,
        npupil=config.npupil,
        psf_size=config.psf_size,
        device=device,
    )
    carrier = None
    if carrier_complex is not None:
        carrier = torch.as_tensor(carrier_complex, device=device)
    optimizer = torch.optim.Adam((spatial,), lr=config.learning_rate_nm)
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(config.seed)
    train_t = torch.as_tensor(train_rows, dtype=torch.long, device=device)
    for _ in range(int(config.adam_steps)):
        choice = torch.randint(
            train_t.numel(),
            (min(config.batch_size, train_t.numel()),),
            generator=torch_generator,
            device=device,
        )
        rows = train_t[choice]
        optimizer.zero_grad(set_to_none=True)
        design = _spatial_design_torch(x_values[rows], y_values[rows], terms, config.image_shape_hw)
        coefficients = global_t[None] + design @ spatial.T
        unit_flux = model.render(
            coefficients_nm=coefficients,
            z_nm=z_values[rows],
            carrier_complex=carrier,
        )
        reconstruction, _, _ = profile_photometry(patches[rows], unit_flux)
        data_loss = (patches[rows] - reconstruction).square().mean() / config.noise_variance_adu2
        regularization = config.spatial_regularization * (spatial / config.wavelength_nm).square().sum()
        (data_loss + regularization).backward()
        optimizer.step()

    candidate_gamma = assemble_direct_gamma(
        global_values,
        spatial.detach().cpu().numpy(),
        degree=config.spatial_degree,
    )
    evaluation = _evaluate_field_models(
        observations,
        global_values=global_values,
        candidate_gamma=candidate_gamma,
        mode_order=mode_order,
        partition=partition,
        model=model,
        config=config,
        rng=rng,
        carrier_complex=carrier,
    )
    spatial_metrics = evaluation["spatial_holdout"]
    selected, accepted = select_field_gamma(
        global_values,
        candidate_gamma,
        spatial_loss_improvement_ci95=tuple(spatial_metrics["relative_loss_improvement_ci95"]),
        baseline_median_ncc=float(spatial_metrics["baseline_median_ncc"]),
        candidate_median_ncc=float(spatial_metrics["candidate_median_ncc"]),
    )
    return FieldGammaFitResult(
        gamma_nm=selected,
        candidate_gamma_nm=candidate_gamma,
        mode_order=mode_order,
        field_accepted=accepted,
        spatial_terms=terms,
        metrics=evaluation,
        partition=partition,
    )


def _evaluate_field_models(
    observations: OracleObservations,
    *,
    global_values: np.ndarray,
    candidate_gamma: np.ndarray,
    mode_order: tuple[tuple[int, int], ...],
    partition: FieldPartition,
    model: DoubleHelixVectorPSF,
    config: FieldGammaFitConfig,
    rng: np.random.Generator,
    carrier_complex: torch.Tensor | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, mask in (
        ("train", partition.train),
        ("frame_holdout", partition.frame_holdout),
        ("spatial_holdout", partition.spatial_holdout),
    ):
        rows = _sample_rows(mask, config.max_validation_emitters, rng)
        baseline_loss, baseline_ncc = _evaluate_rows(
            observations,
            rows,
            gamma_nm=_constant_gamma(global_values, candidate_gamma.shape[1:]),
            model=model,
            config=config,
            carrier_complex=carrier_complex,
        )
        candidate_loss, candidate_ncc = _evaluate_rows(
            observations,
            rows,
            gamma_nm=candidate_gamma,
            model=model,
            config=config,
            carrier_complex=carrier_complex,
        )
        relative = (baseline_loss - candidate_loss) / np.maximum(baseline_loss, 1e-12)
        ci = _block_bootstrap_ci(
            relative,
            partition.block_ids[rows],
            iterations=config.bootstrap_iterations,
            rng=rng,
        )
        output[name] = {
            "n": int(rows.size),
            "baseline_mean_gaussian_loss": float(baseline_loss.mean()),
            "candidate_mean_gaussian_loss": float(candidate_loss.mean()),
            "mean_relative_loss_improvement": float(relative.mean()),
            "relative_loss_improvement_ci95": [float(ci[0]), float(ci[1])],
            "baseline_median_ncc": float(np.median(baseline_ncc)),
            "candidate_median_ncc": float(np.median(candidate_ncc)),
        }
    return output


def _evaluate_rows(
    observations: OracleObservations,
    rows: np.ndarray,
    *,
    gamma_nm: np.ndarray,
    model: DoubleHelixVectorPSF,
    config: FieldGammaFitConfig,
    carrier_complex: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray]:
    device = model.device
    losses = []
    correlations = []
    for start in range(0, rows.size, config.batch_size):
        batch = rows[start : start + config.batch_size]
        x = torch.as_tensor(observations.x_px[batch], dtype=torch.float32, device=device)
        y = torch.as_tensor(observations.y_px[batch], dtype=torch.float32, device=device)
        z = torch.as_tensor(observations.z_gt_nm[batch], dtype=torch.float32, device=device)
        observed = torch.as_tensor(observations.patches[batch], dtype=torch.float32, device=device)
        coefficients = _evaluate_gamma_torch(gamma_nm, x, y, config.image_shape_hw)
        with torch.no_grad():
            unit_flux = model.render(
                coefficients_nm=coefficients,
                z_nm=z,
                carrier_complex=carrier_complex,
            )
            reconstruction, photons, background = profile_photometry(observed, unit_flux)
            loss = (observed - reconstruction).square().mean(dim=(1, 2)) / config.noise_variance_adu2
            observed_unit = ((observed - background[:, None, None]) / photons[:, None, None]).clamp_min(0.0)
            observed_unit = observed_unit / observed_unit.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            ncc = _ncc(observed_unit, unit_flux)
        losses.append(loss.cpu().numpy())
        correlations.append(ncc.cpu().numpy())
    return np.concatenate(losses), np.concatenate(correlations)


def _evaluate_gamma_torch(
    gamma_nm: np.ndarray,
    x_px: torch.Tensor,
    y_px: torch.Tensor,
    image_shape_hw: tuple[int, int],
) -> torch.Tensor:
    gamma = torch.as_tensor(gamma_nm, dtype=x_px.dtype, device=x_px.device)
    terms = tuple(
        (px, py)
        for px in range(gamma.shape[1])
        for py in range(gamma.shape[2])
        if bool(torch.count_nonzero(gamma[:, px, py]).item())
    )
    if not terms:
        return torch.zeros(
            (x_px.numel(), gamma.shape[0]),
            dtype=x_px.dtype,
            device=x_px.device,
        )
    design = _spatial_design_torch(x_px, y_px, terms, image_shape_hw)
    values = torch.stack([gamma[:, px, py] for px, py in terms], dim=1)
    return design @ values.T


def _spatial_design_torch(
    x_px: torch.Tensor,
    y_px: torch.Tensor,
    terms: tuple[tuple[int, int], ...],
    image_shape_hw: tuple[int, int],
) -> torch.Tensor:
    height, width = image_shape_hw
    x_normalized = -1.0 + 2.0 * x_px / float(width)
    y_normalized = -1.0 + 2.0 * y_px / float(height)
    return torch.stack(
        [_legendre(px, x_normalized) * _legendre(py, y_normalized) for px, py in terms],
        dim=1,
    )


def _legendre(degree: int, values: torch.Tensor) -> torch.Tensor:
    if degree == 0:
        return torch.ones_like(values)
    if degree == 1:
        return values
    previous_previous = torch.ones_like(values)
    previous = values
    for current_degree in range(2, degree + 1):
        current = (
            (2.0 * current_degree - 1.0) * values * previous
            - (current_degree - 1.0) * previous_previous
        ) / float(current_degree)
        previous_previous, previous = previous, current
    return previous


def _constant_gamma(global_values: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
    gamma = np.zeros((global_values.size, *spatial_shape), dtype=np.float32)
    gamma[:, 0, 0] = global_values
    return gamma


def _spatial_block_ids(x: np.ndarray, y: np.ndarray, config: FieldGammaFitConfig) -> np.ndarray:
    height, width = config.image_shape_hw
    padding = float(config.active_padding_px)
    grid = int(config.spatial_block_grid)
    cell_x = np.clip(((x - padding) / (width - 2.0 * padding) * grid).astype(np.int64), 0, grid - 1)
    cell_y = np.clip(((y - padding) / (height - 2.0 * padding) * grid).astype(np.int64), 0, grid - 1)
    return cell_y * grid + cell_x


def _sample_rows(mask: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    rows = np.flatnonzero(mask)
    if rows.size > int(maximum):
        rows = np.sort(rng.choice(rows, size=int(maximum), replace=False))
    return rows


def _block_bootstrap_ci(
    values: np.ndarray,
    block_ids: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    unique = np.unique(block_ids)
    rows_by_block = {block: np.flatnonzero(block_ids == block) for block in unique}
    means = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([rows_by_block[block] for block in sampled])
        means[iteration] = float(np.mean(values[rows]))
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def _ncc(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_flat = first.flatten(1)
    second_flat = second.flatten(1)
    first_centered = first_flat - first_flat.mean(dim=1, keepdim=True)
    second_centered = second_flat - second_flat.mean(dim=1, keepdim=True)
    return (first_centered * second_centered).sum(dim=1) / torch.sqrt(
        first_centered.square().sum(dim=1) * second_centered.square().sum(dim=1)
    ).clamp_min(1e-12)


__all__ = [
    "FieldGammaFitConfig",
    "FieldGammaFitResult",
    "FieldPartition",
    "assemble_direct_gamma",
    "fit_field_gamma",
    "partition_field_observations",
    "select_field_gamma",
    "spatial_gamma_terms",
]
