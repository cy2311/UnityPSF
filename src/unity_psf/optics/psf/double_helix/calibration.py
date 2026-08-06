from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch

from .dataset import Microscope1Dataset
from .vector_model import DoubleHelixVectorPSF


SUPPORTED_CALIBRATION_MODE_COUNTS = (13, 21, 30, 42, 64, 72, 84, 92)


def calibration_mode_order(mode_count: int) -> tuple[tuple[int, int], ...]:
    if mode_count not in SUPPORTED_CALIBRATION_MODE_COUNTS:
        raise ValueError(
            f"Supported mode counts are {SUPPORTED_CALIBRATION_MODE_COUNTS}; got {mode_count}."
        )
    modes = []
    radial_order = 2
    while len(modes) < mode_count:
        for azimuthal_order in range(-radial_order, radial_order + 1, 2):
            if (radial_order, azimuthal_order) != (2, 0):
                modes.append((radial_order, azimuthal_order))
        radial_order += 1
    return tuple(modes[:mode_count])


CALIBRATION_MODE_ORDER = calibration_mode_order(13)


@dataclass(frozen=True)
class CalibrationFitConfig:
    mode_count: int = 13
    wavelength_nm: float = 660.0
    wavelength_source: str = "assumption; Zenodo record does not specify emission wavelength"
    na: float = 1.27
    refractive_index: float = 1.33
    pixel_size_nm: float = 200.0
    z_step_nm: float = 33.3
    z_index_origin: float = 1.0
    z_sign: int = 1
    npupil: int = 128
    psf_size: int = 31
    restart_count: int = 4
    seed_scale_nm: float = 220.0
    adam_steps: int = 300
    adam_learning_rate_nm: float = 3.0
    shift_learning_rate_px: float = 0.01
    fit_z_range_nm: float | None = None
    learning_rate_schedule: str = "constant"
    minimum_learning_rate_ratio: float = 0.1
    high_order_regularization: float = 1e-6
    warm_start_mode_count: int = 0
    warm_start_noise_nm: float = 10.0
    optimize_z_calibration: bool = False
    z_calibration_learning_rate: float = 0.01
    max_z_offset_nm: float = 250.0
    max_z_scale_delta: float = 0.25
    deep_z_loss: bool = False
    poisson_loss_weight: float = 1.0
    ncc_loss_weight: float = 2.0
    lobe_geometry_loss_weight: float = 3.0
    paired_helix_loss_weight: float = 0.0
    paired_angle_weight: float = 1.0
    paired_separation_weight: float = 1.0
    paired_center_weight: float = 1.0
    z_bin_edges_nm: tuple[float, ...] = (400.0, 800.0)
    z_bin_weights: tuple[float, ...] = (1.0, 1.5, 2.5)
    seed: int = 20260723
    device: str = "cuda"


@dataclass(frozen=True)
class CalibrationFitResult:
    gamma_nm: np.ndarray
    mode_order: tuple[tuple[int, int], ...]
    reconstruction_adu: np.ndarray
    reconstruction_unit_flux: np.ndarray
    photons_adu: np.ndarray
    background_adu: np.ndarray
    dx_affine_px: np.ndarray
    dy_affine_px: np.ndarray
    source_plane_indices: np.ndarray
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    z_nm: np.ndarray
    metrics: dict[str, Any]
    stage_z_nm: np.ndarray | None = None


def calibration_plane_z_nm(
    plane_count: int,
    *,
    config: CalibrationFitConfig,
) -> np.ndarray:
    indices = np.arange(int(plane_count), dtype=np.float64)
    return config.z_sign * (indices + config.z_index_origin) * config.z_step_nm


def calibration_fit_plane_indices(
    plane_count: int,
    *,
    config: CalibrationFitConfig,
) -> np.ndarray:
    all_indices = np.arange(int(plane_count), dtype=np.int64)
    if config.fit_z_range_nm is None:
        return all_indices
    if config.fit_z_range_nm <= 0:
        raise ValueError("fit_z_range_nm must be positive when specified.")
    full_z_nm = calibration_plane_z_nm(plane_count, config=config)
    relative_z_nm = full_z_nm - 0.5 * (full_z_nm[0] + full_z_nm[-1])
    return all_indices[np.abs(relative_z_nm) <= config.fit_z_range_nm]


def calibration_fit_z_nm(
    plane_count: int,
    *,
    config: CalibrationFitConfig,
) -> np.ndarray:
    full_z_nm = calibration_plane_z_nm(plane_count, config=config)
    source_indices = calibration_fit_plane_indices(plane_count, config=config)
    if config.fit_z_range_nm is None:
        return full_z_nm
    return full_z_nm[source_indices] - 0.5 * (full_z_nm[0] + full_z_nm[-1])


def calibration_learning_rate_multiplier(
    step: int,
    *,
    total_steps: int,
    config: CalibrationFitConfig,
) -> float:
    if config.learning_rate_schedule == "constant":
        return 1.0
    if config.learning_rate_schedule != "cosine":
        raise ValueError(
            "learning_rate_schedule must be either 'constant' or 'cosine'."
        )
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError("step must be between zero and total_steps, inclusive.")
    if not 0.0 <= config.minimum_learning_rate_ratio <= 1.0:
        raise ValueError("minimum_learning_rate_ratio must be between zero and one.")
    cosine = 0.5 * (1.0 + np.cos(np.pi * step / total_steps))
    return float(
        config.minimum_learning_rate_ratio
        + (1.0 - config.minimum_learning_rate_ratio) * cosine
    )


def interleaved_calibration_split(plane_count: int) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(int(plane_count), dtype=np.int64)
    heldout = np.arange(4, int(plane_count), 5, dtype=np.int64)
    train = np.setdiff1d(all_indices, heldout, assume_unique=True)
    return train, heldout


def profile_photometry(
    observed: torch.Tensor,
    unit_flux_psf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if observed.shape != unit_flux_psf.shape or observed.ndim != 3:
        raise ValueError("observed and unit_flux_psf must have matching (N,H,W) shapes.")
    observed_flat = observed.flatten(1)
    psf_flat = unit_flux_psf.flatten(1)
    observed_mean = observed_flat.mean(dim=1)
    psf_mean = psf_flat.mean(dim=1)
    centered_psf = psf_flat - psf_mean[:, None]
    numerator = (centered_psf * (observed_flat - observed_mean[:, None])).sum(dim=1)
    denominator = centered_psf.square().sum(dim=1).clamp_min(torch.finfo(observed.dtype).eps)
    photons = (numerator / denominator).clamp_min(torch.finfo(observed.dtype).eps)
    background = (observed_mean - photons * psf_mean).clamp_min(0.0)
    reconstruction = photons[:, None, None] * unit_flux_psf + background[:, None, None]
    return reconstruction, photons, background


def expand_warm_start_gamma(
    initial_gamma_nm: np.ndarray,
    *,
    target_mode_count: int,
) -> np.ndarray:
    values = np.asarray(initial_gamma_nm, dtype=np.float32)
    if values.ndim > 1:
        if int(np.prod(values.shape[1:])) != 1:
            raise ValueError("Warm-start gamma must contain one coefficient per mode.")
        values = values.reshape(values.shape[0])
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Warm-start gamma must be a non-empty coefficient vector.")
    if values.size > int(target_mode_count):
        raise ValueError("Warm-start gamma has more modes than the target calibration.")
    expanded = np.zeros(int(target_mode_count), dtype=np.float32)
    expanded[: values.size] = values
    return expanded


def calibrated_z_values(
    stage_z_nm: torch.Tensor,
    *,
    z_offset_nm: torch.Tensor,
    z_scale: torch.Tensor,
) -> torch.Tensor:
    return z_scale * stage_z_nm + z_offset_nm


def z_bin_balanced_mean(
    per_plane_loss: torch.Tensor,
    z_nm: torch.Tensor,
    *,
    bin_edges_nm: tuple[float, ...],
    bin_weights: tuple[float, ...],
) -> torch.Tensor:
    if per_plane_loss.ndim != 1 or z_nm.shape != per_plane_loss.shape:
        raise ValueError("per_plane_loss and z_nm must be matching one-dimensional tensors.")
    if len(bin_weights) != len(bin_edges_nm) + 1:
        raise ValueError("bin_weights must contain one value more than bin_edges_nm.")
    if any(right <= left for left, right in zip(bin_edges_nm, bin_edges_nm[1:])):
        raise ValueError("bin_edges_nm must be strictly increasing.")
    edges = torch.as_tensor(bin_edges_nm, dtype=z_nm.dtype, device=z_nm.device)
    bins = torch.bucketize(z_nm.abs(), edges)
    weighted_means = []
    present_weights = []
    for index, weight in enumerate(bin_weights):
        selected = bins == index
        if selected.any():
            weighted_means.append(float(weight) * per_plane_loss[selected].mean())
            present_weights.append(float(weight))
    if not weighted_means:
        raise ValueError("At least one z bin must contain a plane.")
    return torch.stack(weighted_means).sum() / sum(present_weights)


def _lobe_geometry_features(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 3:
        raise ValueError("images must have shape (N,H,W).")
    normalized = images.clamp_min(0.0)
    normalized = normalized / normalized.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    height, width = images.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, dtype=images.dtype, device=images.device),
        torch.linspace(-1.0, 1.0, width, dtype=images.dtype, device=images.device),
        indexing="ij",
    )
    center_x = (normalized * xx).sum(dim=(1, 2))
    center_y = (normalized * yy).sum(dim=(1, 2))
    delta_x = xx[None] - center_x[:, None, None]
    delta_y = yy[None] - center_y[:, None, None]
    moment_xx = (normalized * delta_x.square()).sum(dim=(1, 2))
    moment_yy = (normalized * delta_y.square()).sum(dim=(1, 2))
    moment_xy = (normalized * delta_x * delta_y).sum(dim=(1, 2))
    trace = (moment_xx + moment_yy).clamp_min(1e-12)
    anisotropy_x = (moment_xx - moment_yy) / trace
    anisotropy_y = 2.0 * moment_xy / trace
    determinant = (moment_xx * moment_yy - moment_xy.square()).clamp_min(0.0)
    return torch.stack((anisotropy_x, anisotropy_y, trace, determinant.sqrt()), dim=1)


def lobe_geometry_loss(observed_unit: torch.Tensor, reconstructed_unit: torch.Tensor) -> torch.Tensor:
    if observed_unit.shape != reconstructed_unit.shape:
        raise ValueError("observed_unit and reconstructed_unit must have matching shapes.")
    difference = _lobe_geometry_features(observed_unit) - _lobe_geometry_features(reconstructed_unit)
    return difference.square().mean(dim=1)


def symmetric_z_pair_indices(
    z_nm: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(z_nm, torch.Tensor):
        values = z_nm.detach().cpu().numpy()
    else:
        values = np.asarray(z_nm)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("z_nm must be one-dimensional.")
    focus_candidates = np.flatnonzero(np.isclose(values, 0.0, rtol=0.0, atol=1e-6))
    if focus_candidates.size != 1:
        raise ValueError("Paired double-helix loss requires exactly one z=0 plane.")
    negative_indices = []
    positive_indices = []
    for negative_index in np.flatnonzero(values < 0.0):
        matches = np.flatnonzero(np.isclose(values, -values[negative_index], rtol=0.0, atol=1e-6))
        if matches.size == 1:
            negative_indices.append(int(negative_index))
            positive_indices.append(int(matches[0]))
    if not negative_indices:
        raise ValueError("Paired double-helix loss requires at least one complete (-z,+z) pair.")
    return (
        np.asarray(negative_indices, dtype=np.int64),
        np.asarray(positive_indices, dtype=np.int64),
        int(focus_candidates[0]),
    )


def _double_helix_pair_features(
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if images.ndim != 3:
        raise ValueError("images must have shape (N,H,W).")
    normalized = images.clamp_min(0.0)
    normalized = normalized / normalized.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    height, width = images.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, dtype=images.dtype, device=images.device),
        torch.linspace(-1.0, 1.0, width, dtype=images.dtype, device=images.device),
        indexing="ij",
    )
    center_x = (normalized * xx).sum(dim=(1, 2))
    center_y = (normalized * yy).sum(dim=(1, 2))
    delta_x = xx[None] - center_x[:, None, None]
    delta_y = yy[None] - center_y[:, None, None]
    moment_xx = (normalized * delta_x.square()).sum(dim=(1, 2))
    moment_yy = (normalized * delta_y.square()).sum(dim=(1, 2))
    moment_xy = (normalized * delta_x * delta_y).sum(dim=(1, 2))
    orientation = torch.stack((moment_xx - moment_yy, 2.0 * moment_xy), dim=1)
    anisotropy = torch.linalg.vector_norm(orientation, dim=1).clamp_min(1e-12)
    orientation = orientation / anisotropy[:, None]
    separation = 2.0 * torch.sqrt(anisotropy)
    center = torch.stack((center_x, center_y), dim=1)
    return center, orientation, separation


def double_helix_pair_loss_components(
    images: torch.Tensor,
    *,
    negative_indices: np.ndarray | torch.Tensor,
    positive_indices: np.ndarray | torch.Tensor,
    focus_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center, orientation, separation = _double_helix_pair_features(images)
    negative = torch.as_tensor(negative_indices, dtype=torch.long, device=images.device)
    positive = torch.as_tensor(positive_indices, dtype=torch.long, device=images.device)
    if negative.ndim != 1 or positive.shape != negative.shape or negative.numel() == 0:
        raise ValueError("negative_indices and positive_indices must identify matching non-empty pairs.")

    focus_orientation = orientation[int(focus_index)]
    focus_squared = torch.stack(
        (
            focus_orientation[0].square() - focus_orientation[1].square(),
            2.0 * focus_orientation[0] * focus_orientation[1],
        )
    )
    positive_conjugate = torch.stack(
        (orientation[positive, 0], -orientation[positive, 1]),
        dim=1,
    )
    reflected_positive = torch.stack(
        (
            focus_squared[0] * positive_conjugate[:, 0]
            - focus_squared[1] * positive_conjugate[:, 1],
            focus_squared[1] * positive_conjugate[:, 0]
            + focus_squared[0] * positive_conjugate[:, 1],
        ),
        dim=1,
    )
    angle_loss = (orientation[negative] - reflected_positive).square().mean()

    mean_separation = (0.5 * (separation[negative] + separation[positive])).clamp_min(1e-6)
    separation_loss = ((separation[negative] - separation[positive]) / mean_separation).square().mean()

    pair_center_scale = mean_separation[:, None]
    center_residual = center[negative] + center[positive] - 2.0 * center[int(focus_index)]
    center_loss = (center_residual / pair_center_scale).square().mean()
    return angle_loss, separation_loss, center_loss


def deep_z_per_plane_losses(
    observed: torch.Tensor,
    unit_flux_psf: torch.Tensor,
    *,
    config: CalibrationFitConfig,
) -> torch.Tensor:
    reconstruction, photons, background = profile_photometry(observed, unit_flux_psf)
    observed_unit = ((observed - background[:, None, None]) / photons[:, None, None]).clamp_min(0.0)
    observed_unit = observed_unit / observed_unit.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    mean = reconstruction.clamp_min(torch.finfo(reconstruction.dtype).eps)
    values = observed.clamp_min(0.0)
    log_ratio = torch.where(values > 0.0, values * (values.clamp_min(1e-12).log() - mean.log()), 0.0)
    poisson_deviance = 2.0 * (mean - values + log_ratio)
    poisson_loss = poisson_deviance.mean(dim=(1, 2)) / values.mean(dim=(1, 2)).clamp_min(1.0)
    ncc_loss = 1.0 - _ncc(observed_unit, unit_flux_psf)
    geometry_loss = lobe_geometry_loss(observed_unit, unit_flux_psf)
    return (
        config.poisson_loss_weight * poisson_loss
        + config.ncc_loss_weight * ncc_loss
        + config.lobe_geometry_loss_weight * geometry_loss
    )


def fit_microscope1_calibration(
    dataset: Microscope1Dataset,
    *,
    config: CalibrationFitConfig,
) -> CalibrationFitResult:
    return fit_calibration_stack(dataset.read_calibration(), config=config)


def fit_calibration_stack(
    stack_adu: np.ndarray,
    *,
    config: CalibrationFitConfig,
    initial_gamma_nm: np.ndarray | None = None,
) -> CalibrationFitResult:
    observed_np = np.asarray(stack_adu, dtype=np.float32)
    if observed_np.ndim != 3 or observed_np.shape[1:] != (config.psf_size, config.psf_size):
        raise ValueError("stack_adu must have shape (N, psf_size, psf_size).")
    source_plane_indices = calibration_fit_plane_indices(observed_np.shape[0], config=config)
    z_np = calibration_fit_z_nm(observed_np.shape[0], config=config)
    observed_np = observed_np[source_plane_indices]
    device = torch.device(config.device)
    observed = torch.as_tensor(observed_np, dtype=torch.float32, device=device)
    stage_z_values = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    z_normalized = 2.0 * (stage_z_values - stage_z_values.min()) / (
        stage_z_values.max() - stage_z_values.min()
    ) - 1.0
    train_indices, heldout_indices = interleaved_calibration_split(observed_np.shape[0])
    train_t = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    train_pair_indices = None
    if config.paired_helix_loss_weight > 0.0:
        train_pair_indices = symmetric_z_pair_indices(stage_z_values[train_t])
    noise_sigma = _border_noise_sigma(observed)

    mode_order = calibration_mode_order(config.mode_count)
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
    radial_weight = torch.as_tensor(
        [(n * (n + 1)) ** 2 for n, _ in mode_order],
        dtype=torch.float32,
        device=device,
    )
    warm_start = None
    if initial_gamma_nm is not None:
        warm_start = torch.as_tensor(
            expand_warm_start_gamma(initial_gamma_nm, target_mode_count=len(mode_order)),
            dtype=torch.float32,
            device=device,
        )
    candidates = []
    for restart in range(int(config.restart_count)):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config.seed + restart))
        coefficients = torch.zeros(len(mode_order), dtype=torch.float32, device=device)
        if warm_start is not None:
            coefficients.copy_(warm_start)
            if restart > 0:
                coefficients.add_(
                    torch.randn(len(mode_order), dtype=torch.float32, device=device, generator=generator)
                    * config.warm_start_noise_nm
                )
        elif restart > 0:
            coefficients.normal_(0.0, config.seed_scale_nm, generator=generator)
        coefficients.requires_grad_(True)
        dx_affine = torch.tensor(
            _initial_affine_shift(observed_np, axis="x"),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        dy_affine = torch.tensor(
            _initial_affine_shift(observed_np, axis="y"),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        z_offset_parameter = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        z_scale_parameter = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        parameter_groups = [
            {"params": [coefficients], "lr": config.adam_learning_rate_nm},
            {"params": [dx_affine, dy_affine], "lr": config.shift_learning_rate_px},
        ]
        base_learning_rates = [
            config.adam_learning_rate_nm,
            config.shift_learning_rate_px,
        ]
        if config.optimize_z_calibration:
            parameter_groups.append(
                {
                    "params": [z_offset_parameter, z_scale_parameter],
                    "lr": config.z_calibration_learning_rate,
                }
            )
            base_learning_rates.append(config.z_calibration_learning_rate)
        optimizer = torch.optim.Adam(parameter_groups)
        for step in range(int(config.adam_steps)):
            learning_rate_multiplier = calibration_learning_rate_multiplier(
                step,
                total_steps=int(config.adam_steps),
                config=config,
            )
            for parameter_group, base_learning_rate in zip(
                optimizer.param_groups,
                base_learning_rates,
                strict=True,
            ):
                parameter_group["lr"] = base_learning_rate * learning_rate_multiplier
            optimizer.zero_grad(set_to_none=True)
            if config.optimize_z_calibration:
                z_offset_nm = config.max_z_offset_nm * torch.tanh(z_offset_parameter)
                z_scale = 1.0 + config.max_z_scale_delta * torch.tanh(z_scale_parameter)
            else:
                z_offset_nm = stage_z_values.new_zeros(())
                z_scale = stage_z_values.new_ones(())
            fitted_z_values = calibrated_z_values(
                stage_z_values,
                z_offset_nm=z_offset_nm,
                z_scale=z_scale,
            )
            unit_flux = model.render(
                coefficients_nm=coefficients[None].expand(train_t.numel(), -1),
                z_nm=fitted_z_values[train_t],
                dx_px=dx_affine[0] + dx_affine[1] * z_normalized[train_t],
                dy_px=dy_affine[0] + dy_affine[1] * z_normalized[train_t],
            )
            if config.deep_z_loss:
                per_plane_loss = deep_z_per_plane_losses(observed[train_t], unit_flux, config=config)
                data_loss = z_bin_balanced_mean(
                    per_plane_loss,
                    stage_z_values[train_t],
                    bin_edges_nm=config.z_bin_edges_nm,
                    bin_weights=config.z_bin_weights,
                )
            else:
                reconstruction, _, _ = profile_photometry(observed[train_t], unit_flux)
                data_loss = (
                    (observed[train_t] - reconstruction).square()
                    / noise_sigma[train_t, None, None].square()
                ).mean()
            regularization = config.high_order_regularization * (
                radial_weight * (coefficients / config.wavelength_nm).square()
            ).sum()
            paired_loss = unit_flux.new_zeros(())
            if train_pair_indices is not None:
                angle_loss, separation_loss, center_loss = double_helix_pair_loss_components(
                    unit_flux,
                    negative_indices=train_pair_indices[0],
                    positive_indices=train_pair_indices[1],
                    focus_index=train_pair_indices[2],
                )
                paired_loss = (
                    config.paired_angle_weight * angle_loss
                    + config.paired_separation_weight * separation_loss
                    + config.paired_center_weight * center_loss
                )
            (data_loss + config.paired_helix_loss_weight * paired_loss + regularization).backward()
            optimizer.step()
        candidates.append(
            _evaluate_candidate(
                model=model,
                coefficients=coefficients,
                dx_affine=dx_affine,
                dy_affine=dy_affine,
                observed=observed,
                stage_z_values=stage_z_values,
                z_values=fitted_z_values,
                z_normalized=z_normalized,
                train_indices=train_indices,
                heldout_indices=heldout_indices,
                noise_sigma=noise_sigma,
                mode_order=mode_order,
                config=config,
                z_offset_nm=z_offset_nm,
                z_scale=z_scale,
            )
        )

    selection_metric = "heldout_deep_z_loss" if config.deep_z_loss else "heldout_gaussian_loss"
    best = min(candidates, key=lambda item: item["metrics"][selection_metric])
    metrics = dict(best["metrics"])
    metrics["accepted"] = bool(
        metrics["heldout_median_ncc"] >= 0.95
        and metrics["heldout_p10_ncc"] >= 0.90
        and metrics["heldout_median_nrmse"] <= 0.10
        and metrics["edge_flux_fraction"] <= 0.05
        and metrics["heldout_median_lobe_angle_error_deg"] <= 10.0
    )
    return CalibrationFitResult(
        gamma_nm=best["coefficients"].detach().cpu().numpy().astype(np.float32)[:, None, None],
        mode_order=best["mode_order"],
        reconstruction_adu=best["reconstruction"].detach().cpu().numpy().astype(np.float32),
        reconstruction_unit_flux=best["unit_flux"].detach().cpu().numpy().astype(np.float32),
        photons_adu=best["photons"].detach().cpu().numpy().astype(np.float32),
        background_adu=best["background"].detach().cpu().numpy().astype(np.float32),
        dx_affine_px=best["dx_affine"].detach().cpu().numpy().astype(np.float32),
        dy_affine_px=best["dy_affine"].detach().cpu().numpy().astype(np.float32),
        source_plane_indices=source_plane_indices,
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        z_nm=best["z_values"].detach().cpu().numpy().astype(np.float64),
        metrics=metrics,
        stage_z_nm=z_np,
    )


def _border_noise_sigma(observed: torch.Tensor) -> torch.Tensor:
    border = torch.cat(
        (
            observed[:, :3].flatten(1),
            observed[:, -3:].flatten(1),
            observed[:, 3:-3, :3].flatten(1),
            observed[:, 3:-3, -3:].flatten(1),
        ),
        dim=1,
    )
    median = border.median(dim=1).values
    mad = (border - median[:, None]).abs().median(dim=1).values
    return (1.4826 * mad).clamp_min(1.0)


def _initial_affine_shift(stack: np.ndarray, *, axis: str) -> np.ndarray:
    yy, xx = np.indices(stack.shape[1:], dtype=np.float64)
    border = np.concatenate(
        (stack[:, :3].reshape(stack.shape[0], -1), stack[:, -3:].reshape(stack.shape[0], -1)),
        axis=1,
    )
    background = np.median(border, axis=1)
    signal = np.maximum(stack - background[:, None, None], 0.0)
    coordinates = xx if axis == "x" else yy
    centroid = (signal * coordinates[None]).sum(axis=(1, 2)) / signal.sum(axis=(1, 2))
    normalized = np.linspace(-1.0, 1.0, stack.shape[0])
    slope, intercept = np.polyfit(normalized, centroid - (stack.shape[-1] - 1) / 2.0, 1)
    return np.asarray((intercept, slope), dtype=np.float32)


def _evaluate_candidate(
    *,
    model: DoubleHelixVectorPSF,
    coefficients: torch.Tensor,
    dx_affine: torch.Tensor,
    dy_affine: torch.Tensor,
    observed: torch.Tensor,
    stage_z_values: torch.Tensor,
    z_values: torch.Tensor,
    z_normalized: torch.Tensor,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    noise_sigma: torch.Tensor,
    mode_order: tuple[tuple[int, int], ...],
    config: CalibrationFitConfig,
    z_offset_nm: torch.Tensor,
    z_scale: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        unit_flux = model.render(
            coefficients_nm=coefficients[None].expand(observed.shape[0], -1),
            z_nm=z_values,
            dx_px=dx_affine[0] + dx_affine[1] * z_normalized,
            dy_px=dy_affine[0] + dy_affine[1] * z_normalized,
        )
        reconstruction, photons, background = profile_photometry(observed, unit_flux)
        per_plane_loss = (
            (observed - reconstruction).square() / noise_sigma[:, None, None].square()
        ).mean(dim=(1, 2))
        observed_unit = ((observed - background[:, None, None]) / photons[:, None, None]).clamp_min(0.0)
        observed_unit = observed_unit / observed_unit.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        ncc = _ncc(observed_unit, unit_flux)
        nrmse = torch.sqrt((observed_unit - unit_flux).square().mean(dim=(1, 2))) / torch.sqrt(
            observed_unit.square().mean(dim=(1, 2))
        ).clamp_min(1e-12)
        lobe_error = _lobe_angle_errors(observed_unit, unit_flux)
        edge_flux = torch.cat(
            (
                unit_flux[:, 0],
                unit_flux[:, -1],
                unit_flux[:, 1:-1, 0],
                unit_flux[:, 1:-1, -1],
            ),
            dim=1,
        ).sum(dim=1)
        train = torch.as_tensor(train_indices, device=observed.device)
        heldout = torch.as_tensor(heldout_indices, device=observed.device)
        deep_z_loss = deep_z_per_plane_losses(observed, unit_flux, config=config)
        metrics = {
            "train_gaussian_loss": float(per_plane_loss[train].mean().item()),
            "heldout_gaussian_loss": float(per_plane_loss[heldout].mean().item()),
            "heldout_median_ncc": float(ncc[heldout].median().item()),
            "heldout_p10_ncc": float(torch.quantile(ncc[heldout], 0.1).item()),
            "heldout_median_nrmse": float(nrmse[heldout].median().item()),
            "heldout_median_lobe_angle_error_deg": float(torch.nanmedian(lobe_error[heldout]).item()),
            "edge_flux_fraction": float(edge_flux.max().item()),
            "train_deep_z_loss": float(
                z_bin_balanced_mean(
                    deep_z_loss[train],
                    stage_z_values[train],
                    bin_edges_nm=config.z_bin_edges_nm,
                    bin_weights=config.z_bin_weights,
                ).item()
            ),
            "heldout_deep_z_loss": float(
                z_bin_balanced_mean(
                    deep_z_loss[heldout],
                    stage_z_values[heldout],
                    bin_edges_nm=config.z_bin_edges_nm,
                    bin_weights=config.z_bin_weights,
                ).item()
            ),
            "fitted_z_offset_nm": float(z_offset_nm.item()),
            "fitted_z_scale": float(z_scale.item()),
        }
        if config.paired_helix_loss_weight > 0.0:
            negative, positive, focus = symmetric_z_pair_indices(stage_z_values)
            angle_loss, separation_loss, center_loss = double_helix_pair_loss_components(
                unit_flux,
                negative_indices=negative,
                positive_indices=positive,
                focus_index=focus,
            )
            metrics.update(
                {
                    "full_paired_helix_angle_loss": float(angle_loss.item()),
                    "full_paired_helix_separation_loss": float(separation_loss.item()),
                    "full_paired_helix_center_loss": float(center_loss.item()),
                    "full_paired_helix_pair_count": int(len(negative)),
                }
            )
    return {
        "coefficients": coefficients.detach().clone(),
        "mode_order": mode_order,
        "dx_affine": dx_affine.detach().clone(),
        "dy_affine": dy_affine.detach().clone(),
        "unit_flux": unit_flux,
        "reconstruction": reconstruction,
        "photons": photons,
        "background": background,
        "z_values": z_values.detach().clone(),
        "metrics": metrics,
    }


def _ncc(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_centered = first.flatten(1) - first.flatten(1).mean(dim=1, keepdim=True)
    second_centered = second.flatten(1) - second.flatten(1).mean(dim=1, keepdim=True)
    numerator = (first_centered * second_centered).sum(dim=1)
    denominator = torch.sqrt(first_centered.square().sum(dim=1) * second_centered.square().sum(dim=1))
    return numerator / denominator.clamp_min(1e-12)


def _lobe_angle_errors(observed: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    observed_angles = _two_peak_angles(observed)
    reconstructed_angles = _two_peak_angles(reconstructed)
    difference = torch.remainder(observed_angles - reconstructed_angles + torch.pi / 2.0, torch.pi) - torch.pi / 2.0
    return difference.abs() * (180.0 / torch.pi)


def _two_peak_angles(images: torch.Tensor) -> torch.Tensor:
    height, width = images.shape[-2:]
    flattened = images.flatten(1)
    first = flattened.argmax(dim=1)
    first_y = torch.div(first, width, rounding_mode="floor")
    first_x = first % width
    yy, xx = torch.meshgrid(
        torch.arange(height, device=images.device),
        torch.arange(width, device=images.device),
        indexing="ij",
    )
    distance_squared = (xx[None] - first_x[:, None, None]).square() + (yy[None] - first_y[:, None, None]).square()
    masked = images.masked_fill(distance_squared < 9, -torch.inf)
    second = masked.flatten(1).argmax(dim=1)
    second_y = torch.div(second, width, rounding_mode="floor")
    second_x = second % width
    return torch.atan2((second_y - first_y).to(images.dtype), (second_x - first_x).to(images.dtype))


def calibration_config_dict(config: CalibrationFitConfig) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


__all__ = [
    "CALIBRATION_MODE_ORDER",
    "SUPPORTED_CALIBRATION_MODE_COUNTS",
    "CalibrationFitConfig",
    "CalibrationFitResult",
    "calibration_fit_plane_indices",
    "calibration_fit_z_nm",
    "calibration_config_dict",
    "calibration_learning_rate_multiplier",
    "calibration_mode_order",
    "calibration_plane_z_nm",
    "calibrated_z_values",
    "deep_z_per_plane_losses",
    "double_helix_pair_loss_components",
    "expand_warm_start_gamma",
    "fit_calibration_stack",
    "fit_microscope1_calibration",
    "interleaved_calibration_split",
    "lobe_geometry_loss",
    "profile_photometry",
    "symmetric_z_pair_indices",
    "z_bin_balanced_mean",
]
