from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter

from .calibration import (
    calibration_mode_order,
    deep_z_per_plane_losses,
    double_helix_pair_loss_components,
    interleaved_calibration_split,
    profile_photometry,
    symmetric_z_pair_indices,
    z_bin_balanced_mean,
)
from .lg_carrier import CANONICAL_DH_LG_MODES, laguerre_gaussian_basis, lg_dh_carrier
from .numerics import normalized_cross_correlation
from .vector_model import DoubleHelixVectorPSF


@dataclass(frozen=True)
class LGResidualFitConfig:
    mode_count: int = 13
    wavelength_nm: float = 660.0
    na: float = 1.27
    refractive_index: float = 1.33
    pixel_size_nm: float = 207.0
    z_step_nm: float = 40.0
    fit_z_range_nm: float = 2000.0
    roi_size: int = 17
    npupil: int = 128
    lg_waist: float = 0.72
    alternating_rounds: int = 4
    local_steps: int = 100
    global_steps: int = 500
    shift_learning_rate_px: float = 0.01
    gamma_learning_rate_nm: float = 1.5
    carrier_learning_rate: float = 0.02
    poisson_loss_weight: float = 1.0
    ncc_loss_weight: float = 2.0
    lobe_geometry_loss_weight: float = 3.0
    paired_helix_loss_weight: float = 2.0
    paired_angle_weight: float = 5.0
    paired_separation_weight: float = 4.0
    paired_center_weight: float = 3.0
    z_bin_edges_nm: tuple[float, ...] = (400.0, 800.0, 1400.0)
    z_bin_weights: tuple[float, ...] = (5.0, 4.0, 3.0, 4.0)
    residual_regularization: float = 2e-5
    residual_mean_regularization: float = 1e-4
    field_ridge: float = 1e-3
    seed: int = 20260724
    device: str = "cuda"


@dataclass(frozen=True)
class LGResidualFitResult:
    centers_yx: np.ndarray
    z_nm: np.ndarray
    source_plane_indices: np.ndarray
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    mode_order: tuple[tuple[int, int], ...]
    lg_mode_order: tuple[tuple[int, int], ...]
    lg_weights: np.ndarray
    lg_weight_logits: np.ndarray
    lg_phase_offsets_rad: np.ndarray
    lg_rotation_rad: float
    carrier_phase_rad: np.ndarray
    residual_gamma_nm: np.ndarray
    residual_field_coefficients_nm: np.ndarray
    zernike_maps_nm: np.ndarray
    reconstruction_unit_flux: np.ndarray
    reconstruction_adu: np.ndarray
    photons_adu: np.ndarray
    background_adu: np.ndarray
    dx_affine_px: np.ndarray
    dy_affine_px: np.ndarray
    metrics: dict[str, object]


def detect_stable_dh_centers(
    stack: np.ndarray,
    *,
    roi_size: int,
    maximum_count: int,
    minimum_distance_px: int,
) -> np.ndarray:
    values = np.asarray(stack)
    if values.ndim != 3:
        raise ValueError("stack must have shape (z, y, x).")
    if roi_size <= 0 or roi_size % 2 == 0:
        raise ValueError("roi_size must be a positive odd integer.")
    if maximum_count <= 0 or minimum_distance_px <= 0:
        raise ValueError("maximum_count and minimum_distance_px must be positive.")

    float_values = values.astype(np.float32, copy=False)
    border_width = min(5, values.shape[1] // 4, values.shape[2] // 4)
    border = np.concatenate(
        (
            float_values[:, :border_width].reshape(len(values), -1),
            float_values[:, -border_width:].reshape(len(values), -1),
            float_values[:, border_width:-border_width, :border_width].reshape(len(values), -1),
            float_values[:, border_width:-border_width, -border_width:].reshape(len(values), -1),
        ),
        axis=1,
    )
    background = np.median(border, axis=1)
    signal = np.maximum(float_values - background[:, None, None], 0.0)
    aggregate = gaussian_filter(signal.mean(axis=0), sigma=2.0)
    local_maximum = maximum_filter(aggregate, size=int(minimum_distance_px), mode="nearest")
    candidate_y, candidate_x = np.nonzero(
        (aggregate == local_maximum) & (aggregate >= np.quantile(aggregate, 0.85))
    )
    strengths = aggregate[candidate_y, candidate_x]
    half_width = roi_size // 2
    inside = (
        (candidate_y >= half_width)
        & (candidate_y < values.shape[1] - half_width)
        & (candidate_x >= half_width)
        & (candidate_x < values.shape[2] - half_width)
    )
    candidates = np.stack((candidate_y[inside], candidate_x[inside]), axis=1)
    strengths = strengths[inside]
    order = np.argsort(strengths)[::-1]
    selected: list[np.ndarray] = []
    for index in order:
        candidate = candidates[index]
        if all(np.linalg.norm(candidate - previous) >= minimum_distance_px for previous in selected):
            selected.append(candidate)
        if len(selected) == maximum_count:
            break
    if len(selected) != maximum_count:
        raise ValueError(
            f"Detected only {len(selected)} complete DH centers; requested {maximum_count}."
        )
    return np.asarray(selected, dtype=np.int64)


def extract_centered_roi_stacks(
    stack: np.ndarray,
    centers_yx: np.ndarray,
    *,
    roi_size: int,
) -> np.ndarray:
    values = np.asarray(stack)
    centers = np.asarray(centers_yx)
    if values.ndim != 3 or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("stack must be (z,y,x) and centers_yx must be (bead,2).")
    if roi_size <= 0 or roi_size % 2 == 0:
        raise ValueError("roi_size must be a positive odd integer.")
    half_width = roi_size // 2
    rois = []
    for center_y, center_x in np.rint(centers).astype(np.int64):
        y0, y1 = center_y - half_width, center_y + half_width + 1
        x0, x1 = center_x - half_width, center_x + half_width + 1
        if y0 < 0 or x0 < 0 or y1 > values.shape[1] or x1 > values.shape[2]:
            raise ValueError("Every center must define a complete ROI inside the stack.")
        rois.append(values[:, y0:y1, x0:x1])
    return np.stack(rois)


def fit_affine_residual_gamma_maps(
    centers_yx: np.ndarray,
    gamma_nm: np.ndarray,
    *,
    field_shape_yx: tuple[int, int],
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers_yx, dtype=np.float64)
    gamma = np.asarray(gamma_nm, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_yx must have shape (bead,2).")
    if gamma.ndim != 2 or gamma.shape[0] != centers.shape[0]:
        raise ValueError("gamma_nm must have shape (bead,mode).")
    height, width = map(int, field_shape_yx)
    if height <= 1 or width <= 1 or ridge < 0.0:
        raise ValueError("field dimensions must exceed one and ridge must be non-negative.")
    y = -1.0 + 2.0 * centers[:, 0] / float(height - 1)
    x = -1.0 + 2.0 * centers[:, 1] / float(width - 1)
    design = np.stack((np.ones_like(x), x, y), axis=1)
    penalty = np.diag((0.0, float(ridge), float(ridge)))
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ gamma).T

    field_y = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    field_x = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    yy, xx = np.meshgrid(field_y, field_x, indexing="ij")
    field_design = np.stack((np.ones_like(xx), xx, yy), axis=0)
    maps = np.einsum("ct,tyx->cyx", coefficients, field_design)
    return maps.astype(np.float32), coefficients.astype(np.float32)


def fit_lg_residual_calibration(
    roi_stacks_adu: np.ndarray,
    *,
    centers_yx: np.ndarray,
    field_shape_yx: tuple[int, int],
    config: LGResidualFitConfig,
) -> LGResidualFitResult:
    stacks = np.asarray(roi_stacks_adu, dtype=np.float32)
    centers = np.asarray(centers_yx, dtype=np.float32)
    if stacks.ndim != 4 or stacks.shape[2:] != (config.roi_size, config.roi_size):
        raise ValueError("roi_stacks_adu must have shape (bead,z,roi_size,roi_size).")
    if centers.shape != (stacks.shape[0], 2):
        raise ValueError("centers_yx must contain one field position per bead stack.")
    if config.mode_count not in (13, 21, 30, 42, 64, 72, 84, 92):
        raise ValueError("mode_count must use the established DH calibration mode counts.")

    full_z_nm = (
        np.arange(stacks.shape[1], dtype=np.float64) - 0.5 * (stacks.shape[1] - 1)
    ) * config.z_step_nm
    source_plane_indices = np.flatnonzero(np.abs(full_z_nm) <= config.fit_z_range_nm + 1e-6)
    selected = stacks[:, source_plane_indices]
    z_np = full_z_nm[source_plane_indices]
    train_indices, heldout_indices = interleaved_calibration_split(len(z_np))
    device = torch.device(config.device)
    observed = torch.as_tensor(selected, dtype=torch.float32, device=device)
    z_values = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    z_normalized = z_values / max(float(config.fit_z_range_nm), 1.0)

    mode_order = calibration_mode_order(config.mode_count)
    model = DoubleHelixVectorPSF(
        mode_order=mode_order,
        na=config.na,
        wavelength_nm=config.wavelength_nm,
        pixel_size_nm=config.pixel_size_nm,
        refractive_index=config.refractive_index,
        npupil=config.npupil,
        psf_size=config.roi_size,
        device=device,
    )
    pupil_step = 2.0 / float(config.npupil)
    pupil_coordinates = torch.arange(
        -1.0 + pupil_step / 2.0,
        1.0,
        pupil_step,
        dtype=torch.float32,
        device=device,
    )
    x_pupil, y_pupil = torch.meshgrid(pupil_coordinates, pupil_coordinates, indexing="ij")
    lg_basis = laguerre_gaussian_basis(
        CANONICAL_DH_LG_MODES,
        x_pupil,
        y_pupil,
        waist=config.lg_waist,
    )

    torch.manual_seed(int(config.seed))
    bead_count = stacks.shape[0]
    residual_gamma = torch.zeros(
        (bead_count, len(mode_order)), dtype=torch.float32, device=device, requires_grad=True
    )
    lg_logits = torch.zeros(
        len(CANONICAL_DH_LG_MODES), dtype=torch.float32, device=device, requires_grad=True
    )
    lg_phases = torch.zeros(
        len(CANONICAL_DH_LG_MODES), dtype=torch.float32, device=device, requires_grad=True
    )
    lg_rotation = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    dx_affine = torch.zeros((bead_count, 2), dtype=torch.float32, device=device, requires_grad=True)
    dy_affine = torch.zeros((bead_count, 2), dtype=torch.float32, device=device, requires_grad=True)
    train_t = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    train_pair_indices = symmetric_z_pair_indices(z_values[train_t])
    history: list[dict[str, float | int]] = []

    for round_index in range(int(config.alternating_rounds)):
        local_optimizer = torch.optim.Adam(
            (dx_affine, dy_affine), lr=config.shift_learning_rate_px
        )
        local_loss = observed.new_zeros(())
        for _ in range(int(config.local_steps)):
            local_optimizer.zero_grad(set_to_none=True)
            carrier = lg_dh_carrier(
                lg_basis,
                mode_order=CANONICAL_DH_LG_MODES,
                weight_logits=lg_logits.detach(),
                phase_offsets_rad=lg_phases.detach(),
                rotation_rad=lg_rotation.detach(),
            )
            rendered = _render_bead_planes(
                model,
                residual_gamma=residual_gamma.detach(),
                carrier=carrier,
                z_values=z_values,
                z_normalized=z_normalized,
                plane_indices=train_t,
                dx_affine=dx_affine,
                dy_affine=dy_affine,
            )
            local_loss = _balanced_multibead_loss(
                observed[:, train_t], rendered, z_values[train_t], config=config
            )
            local_loss.backward()
            local_optimizer.step()

        global_optimizer = torch.optim.Adam(
            (
                {"params": [residual_gamma], "lr": config.gamma_learning_rate_nm},
                {"params": [lg_logits, lg_phases, lg_rotation], "lr": config.carrier_learning_rate},
            )
        )
        global_loss = observed.new_zeros(())
        for step in range(int(config.global_steps)):
            global_optimizer.zero_grad(set_to_none=True)
            carrier = lg_dh_carrier(
                lg_basis,
                mode_order=CANONICAL_DH_LG_MODES,
                weight_logits=lg_logits,
                phase_offsets_rad=lg_phases,
                rotation_rad=lg_rotation,
            )
            rendered = _render_bead_planes(
                model,
                residual_gamma=residual_gamma,
                carrier=carrier,
                z_values=z_values,
                z_normalized=z_normalized,
                plane_indices=train_t,
                dx_affine=dx_affine.detach(),
                dy_affine=dy_affine.detach(),
            )
            data_loss = _balanced_multibead_loss(
                observed[:, train_t], rendered, z_values[train_t], config=config
            )
            paired_loss = _multibead_paired_loss(
                rendered,
                pair_indices=train_pair_indices,
                config=config,
            )
            radial_weight = residual_gamma.new_tensor(
                [(n * (n + 1)) ** 2 for n, _ in mode_order]
            )
            residual_penalty = config.residual_regularization * (
                radial_weight[None]
                * (residual_gamma / config.wavelength_nm).square()
            ).mean()
            mean_penalty = config.residual_mean_regularization * (
                residual_gamma.mean(dim=0) / config.wavelength_nm
            ).square().mean()
            global_loss = (
                data_loss
                + config.paired_helix_loss_weight * paired_loss
                + residual_penalty
                + mean_penalty
            )
            global_loss.backward()
            global_optimizer.step()
            progress = (round_index * config.global_steps + step + 1) / max(
                config.alternating_rounds * config.global_steps, 1
            )
            multiplier = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))
            global_optimizer.param_groups[0]["lr"] = config.gamma_learning_rate_nm * multiplier
            global_optimizer.param_groups[1]["lr"] = config.carrier_learning_rate * multiplier
        history.append(
            {
                "round": round_index + 1,
                "local_loss": float(local_loss.detach().item()),
                "global_loss": float(global_loss.detach().item()),
            }
        )

    with torch.no_grad():
        carrier = lg_dh_carrier(
            lg_basis,
            mode_order=CANONICAL_DH_LG_MODES,
            weight_logits=lg_logits,
            phase_offsets_rad=lg_phases,
            rotation_rad=lg_rotation,
        )
        all_planes = torch.arange(len(z_values), dtype=torch.long, device=device)
        unit_flux = _render_bead_planes(
            model,
            residual_gamma=residual_gamma,
            carrier=carrier,
            z_values=z_values,
            z_normalized=z_normalized,
            plane_indices=all_planes,
            dx_affine=dx_affine,
            dy_affine=dy_affine,
        )
        flat_observed = observed.reshape(-1, config.roi_size, config.roi_size)
        flat_unit = unit_flux.reshape_as(flat_observed)
        reconstruction, photons, background = profile_photometry(flat_observed, flat_unit)
        ncc = normalized_cross_correlation(flat_observed, reconstruction).reshape(bead_count, len(z_values))

    gamma_np = residual_gamma.detach().cpu().numpy().astype(np.float32)
    zernike_maps, field_coefficients = fit_affine_residual_gamma_maps(
        centers,
        gamma_np,
        field_shape_yx=field_shape_yx,
        ridge=config.field_ridge,
    )
    weights_np = torch.softmax(lg_logits.detach(), dim=0).cpu().numpy().astype(np.float32)
    per_bead_metrics = []
    for bead_index in range(bead_count):
        per_bead_metrics.append(
            {
                "bead_index": bead_index,
                "center_yx": centers[bead_index].astype(float).tolist(),
                "median_ncc": float(torch.median(ncc[bead_index]).item()),
                "heldout_median_ncc": float(
                    torch.median(ncc[bead_index, heldout_indices]).item()
                ),
                "negative_edge_ncc": float(ncc[bead_index, 0].item()),
                "positive_edge_ncc": float(ncc[bead_index, -1].item()),
            }
        )
    metrics: dict[str, object] = {
        "device": str(device),
        "bead_count": bead_count,
        "heldout_median_ncc": float(torch.median(ncc[:, heldout_indices]).item()),
        "full_median_ncc": float(torch.median(ncc).item()),
        "per_bead": per_bead_metrics,
        "alternation_history": history,
    }
    return LGResidualFitResult(
        centers_yx=centers.astype(np.float32),
        z_nm=z_np,
        source_plane_indices=source_plane_indices.astype(np.int64),
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        mode_order=mode_order,
        lg_mode_order=CANONICAL_DH_LG_MODES,
        lg_weights=weights_np,
        lg_weight_logits=lg_logits.detach().cpu().numpy().astype(np.float32),
        lg_phase_offsets_rad=(
            lg_phases.detach() - lg_phases.detach()[0]
        ).cpu().numpy().astype(np.float32),
        lg_rotation_rad=float(lg_rotation.detach().item()),
        carrier_phase_rad=torch.angle(carrier).cpu().numpy().astype(np.float32),
        residual_gamma_nm=gamma_np,
        residual_field_coefficients_nm=field_coefficients,
        zernike_maps_nm=zernike_maps,
        reconstruction_unit_flux=unit_flux.cpu().numpy().astype(np.float32),
        reconstruction_adu=reconstruction.reshape_as(observed).cpu().numpy().astype(np.float32),
        photons_adu=photons.reshape(bead_count, len(z_values)).cpu().numpy().astype(np.float32),
        background_adu=background.reshape(bead_count, len(z_values)).cpu().numpy().astype(np.float32),
        dx_affine_px=dx_affine.detach().cpu().numpy().astype(np.float32),
        dy_affine_px=dy_affine.detach().cpu().numpy().astype(np.float32),
        metrics=metrics,
    )


def _render_bead_planes(
    model: DoubleHelixVectorPSF,
    *,
    residual_gamma: torch.Tensor,
    carrier: torch.Tensor,
    z_values: torch.Tensor,
    z_normalized: torch.Tensor,
    plane_indices: torch.Tensor,
    dx_affine: torch.Tensor,
    dy_affine: torch.Tensor,
) -> torch.Tensor:
    bead_count = residual_gamma.shape[0]
    plane_count = plane_indices.numel()
    coefficients = residual_gamma[:, None].expand(-1, plane_count, -1).reshape(
        bead_count * plane_count, -1
    )
    selected_z = z_values[plane_indices]
    selected_z_normalized = z_normalized[plane_indices]
    dx = (
        dx_affine[:, 0, None] + dx_affine[:, 1, None] * selected_z_normalized[None]
    ).reshape(-1)
    dy = (
        dy_affine[:, 0, None] + dy_affine[:, 1, None] * selected_z_normalized[None]
    ).reshape(-1)
    rendered = model.render(
        coefficients_nm=coefficients,
        z_nm=selected_z.repeat(bead_count),
        carrier_complex=carrier,
        dx_px=dx,
        dy_px=dy,
    )
    return rendered.reshape(bead_count, plane_count, model.psf_size, model.psf_size)


def _balanced_multibead_loss(
    observed: torch.Tensor,
    rendered: torch.Tensor,
    z_values: torch.Tensor,
    *,
    config: LGResidualFitConfig,
) -> torch.Tensor:
    bead_count, plane_count = observed.shape[:2]
    per_plane = deep_z_per_plane_losses(
        observed.reshape(-1, config.roi_size, config.roi_size),
        rendered.reshape(-1, config.roi_size, config.roi_size),
        config=config,
    ).reshape(bead_count, plane_count)
    return torch.stack(
        [
            z_bin_balanced_mean(
                per_plane[bead],
                z_values,
                bin_edges_nm=config.z_bin_edges_nm,
                bin_weights=config.z_bin_weights,
            )
            for bead in range(bead_count)
        ]
    ).mean()


def _multibead_paired_loss(
    rendered: torch.Tensor,
    *,
    pair_indices: tuple[np.ndarray, np.ndarray, int],
    config: LGResidualFitConfig,
) -> torch.Tensor:
    losses = []
    for bead in range(rendered.shape[0]):
        angle, separation, center = double_helix_pair_loss_components(
            rendered[bead],
            negative_indices=pair_indices[0],
            positive_indices=pair_indices[1],
            focus_index=pair_indices[2],
        )
        losses.append(
            config.paired_angle_weight * angle
            + config.paired_separation_weight * separation
            + config.paired_center_weight * center
        )
    return torch.stack(losses).mean()


__all__ = [
    "LGResidualFitConfig",
    "LGResidualFitResult",
    "detect_stable_dh_centers",
    "extract_centered_roi_stacks",
    "fit_affine_residual_gamma_maps",
    "fit_lg_residual_calibration",
]
