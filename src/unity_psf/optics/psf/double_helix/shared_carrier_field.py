from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .calibration import calibration_mode_order, interleaved_calibration_split, profile_photometry
from .lg_calibration import fit_affine_residual_gamma_maps
from .pixel_pupil_calibration import phase_only_complex_pupil, pupil_grid
from .vector_model import DoubleHelixVectorPSF, evaluate_normalized_zernike


@dataclass(frozen=True)
class SharedCarrierFieldConfig:
    mode_count: int = 21
    wavelength_nm: float = 660.0
    na: float = 1.27
    refractive_index: float = 1.33
    pixel_size_nm: float = 207.0
    roi_size: int = 19
    npupil: int = 128
    alternating_rounds: int = 6
    gamma_steps: int = 300
    gamma_learning_rate_nm: float = 1.0
    huber_delta_rad: float = 0.5
    gamma_regularization: float = 1e-6
    field_ridge: float = 1e-3
    device: str = "cuda"


@dataclass(frozen=True)
class SharedCarrierFieldResult:
    mode_order: tuple[tuple[int, int], ...]
    centers_yx: np.ndarray
    shared_carrier_phase_rad: np.ndarray
    shared_carrier_complex: np.ndarray
    residual_gamma_nm: np.ndarray
    field_coefficients_nm: np.ndarray
    zernike_maps_nm: np.ndarray
    field_gamma_at_beads_nm: np.ndarray
    reconstructed_pupil_phase_rad: np.ndarray
    reconstructed_complex_pupils: np.ndarray
    loss_history: np.ndarray
    metrics: dict[str, object]


@dataclass(frozen=True)
class SharedFieldCalibrationResult:
    z_nm: np.ndarray
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    reconstruction_unit_flux: np.ndarray
    reconstruction_adu: np.ndarray
    photons_adu: np.ndarray
    background_adu: np.ndarray
    per_plane_ncc: np.ndarray
    metrics: dict[str, object]


def fit_shared_carrier_field(
    pupil_phases_rad: np.ndarray,
    *,
    centers_yx: np.ndarray,
    field_shape_yx: tuple[int, int],
    config: SharedCarrierFieldConfig,
) -> SharedCarrierFieldResult:
    phases_np = np.asarray(pupil_phases_rad, dtype=np.float32)
    centers = np.asarray(centers_yx, dtype=np.float32)
    if phases_np.ndim != 3 or phases_np.shape[1:] != (config.npupil, config.npupil):
        raise ValueError("pupil_phases_rad must have shape (bead,npupil,npupil).")
    if centers.shape != (phases_np.shape[0], 2):
        raise ValueError("centers_yx must contain one field position per pupil.")
    if config.alternating_rounds <= 0 or config.gamma_steps <= 0:
        raise ValueError("alternating_rounds and gamma_steps must be positive.")

    device = torch.device(config.device)
    phases = torch.as_tensor(phases_np, dtype=torch.float32, device=device)
    x_pupil, y_pupil, pupil_mask = pupil_grid(
        config.npupil,
        device=device,
        dtype=torch.float32,
    )
    observed_complex = phase_only_complex_pupil(phases, pupil_mask)
    mode_order = calibration_mode_order(config.mode_count)
    basis = evaluate_normalized_zernike(mode_order, x_pupil, y_pupil)
    shared_complex = _robust_circular_mean(
        observed_complex,
        pupil_mask,
        delta=config.huber_delta_rad,
    ).detach()
    gamma_parameter = torch.zeros(
        (len(phases), len(mode_order)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    radial_weight = gamma_parameter.new_tensor(
        [(n * (n + 1)) ** 2 for n, _ in mode_order]
    )
    loss_history: list[float] = []

    def centered_gamma() -> torch.Tensor:
        return gamma_parameter - gamma_parameter.mean(dim=0, keepdim=True)

    def residual_phase(gamma_nm: torch.Tensor) -> torch.Tensor:
        phase = (
            2.0
            * math.pi
            * torch.einsum("bc,chw->bhw", gamma_nm, basis)
            / config.wavelength_nm
        )
        return phase * pupil_mask

    def complex_loss(candidate_shared: torch.Tensor, gamma_nm: torch.Tensor) -> torch.Tensor:
        residual_complex = phase_only_complex_pupil(residual_phase(gamma_nm), pupil_mask)
        prediction = candidate_shared[None] * residual_complex
        distance = (observed_complex - prediction).abs()[..., pupil_mask]
        delta = float(config.huber_delta_rad)
        robust = torch.where(
            distance <= delta,
            0.5 * distance.square(),
            delta * (distance - 0.5 * delta),
        ).mean()
        regularization = config.gamma_regularization * (
            radial_weight[None] * (gamma_nm / config.wavelength_nm).square()
        ).mean()
        return robust + regularization

    with torch.no_grad():
        initial_gamma = centered_gamma()
        initial_prediction = shared_complex[None] * phase_only_complex_pupil(
            residual_phase(initial_gamma), pupil_mask
        )
        initial_complex_nrmse = _complex_nrmse(
            observed_complex,
            initial_prediction,
            pupil_mask,
        )

    for _ in range(int(config.alternating_rounds)):
        optimizer = torch.optim.Adam(
            (gamma_parameter,),
            lr=config.gamma_learning_rate_nm,
        )
        for step in range(int(config.gamma_steps)):
            progress = step / max(int(config.gamma_steps) - 1, 1)
            optimizer.param_groups[0]["lr"] = config.gamma_learning_rate_nm * (
                0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
            )
            optimizer.zero_grad(set_to_none=True)
            loss = complex_loss(shared_complex, centered_gamma())
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().item()))
        with torch.no_grad():
            residual_complex = phase_only_complex_pupil(
                residual_phase(centered_gamma()), pupil_mask
            )
            corrected = observed_complex * residual_complex.conj()
            shared_complex = _robust_circular_mean(
                corrected,
                pupil_mask,
                delta=config.huber_delta_rad,
            ).detach()

    with torch.no_grad():
        gamma = centered_gamma()
        residual_complex = phase_only_complex_pupil(residual_phase(gamma), pupil_mask)
        reconstructed_complex = shared_complex[None] * residual_complex
        reconstructed_phase = torch.angle(reconstructed_complex) * pupil_mask
        shared_phase = torch.angle(shared_complex) * pupil_mask
        final_complex_nrmse = _complex_nrmse(
            observed_complex,
            reconstructed_complex,
            pupil_mask,
        )
        per_bead_complex_nrmse = [
            _complex_nrmse(
                observed_complex[index],
                reconstructed_complex[index],
                pupil_mask,
            )
            for index in range(len(observed_complex))
        ]

    gamma_np = gamma.cpu().numpy().astype(np.float32)
    zernike_maps, field_coefficients = fit_affine_residual_gamma_maps(
        centers,
        gamma_np,
        field_shape_yx=field_shape_yx,
        ridge=config.field_ridge,
    )
    center_indices = np.rint(centers).astype(np.int64)
    field_gamma_at_beads = np.stack(
        [zernike_maps[:, y, x] for y, x in center_indices],
        axis=0,
    )
    metrics: dict[str, object] = {
        "bead_count": int(len(phases)),
        "mode_count": int(len(mode_order)),
        "device": str(device),
        "initial_complex_nrmse": float(initial_complex_nrmse),
        "final_complex_nrmse": float(final_complex_nrmse),
        "per_bead_complex_nrmse": [float(value) for value in per_bead_complex_nrmse],
        "residual_gamma_mean_abs_nm": float(np.max(np.abs(gamma_np.mean(axis=0)))),
        "field_model_gamma_rmse_nm": float(
            np.sqrt(np.mean((field_gamma_at_beads - gamma_np) ** 2))
        ),
        "gamma_semantics": "field-dependent residual OPD above shared DH carrier",
        "gauge": "residual excludes piston, tip, tilt, and Z(2,0); mean residual gamma is zero",
    }
    return SharedCarrierFieldResult(
        mode_order=mode_order,
        centers_yx=centers,
        shared_carrier_phase_rad=shared_phase.cpu().numpy().astype(np.float32),
        shared_carrier_complex=shared_complex.cpu().numpy().astype(np.complex64),
        residual_gamma_nm=gamma_np,
        field_coefficients_nm=field_coefficients,
        zernike_maps_nm=zernike_maps,
        field_gamma_at_beads_nm=field_gamma_at_beads.astype(np.float32),
        reconstructed_pupil_phase_rad=reconstructed_phase.cpu().numpy().astype(np.float32),
        reconstructed_complex_pupils=reconstructed_complex.cpu().numpy().astype(np.complex64),
        loss_history=np.asarray(loss_history, dtype=np.float32),
        metrics=metrics,
    )


def render_shared_field_calibration(
    observed_stacks_adu: np.ndarray,
    *,
    z_nm: np.ndarray,
    dx_affine_px: np.ndarray,
    dy_affine_px: np.ndarray,
    decomposition: SharedCarrierFieldResult,
    config: SharedCarrierFieldConfig,
) -> SharedFieldCalibrationResult:
    observed_np = np.asarray(observed_stacks_adu, dtype=np.float32)
    z_np = np.asarray(z_nm, dtype=np.float32)
    dx_np = np.asarray(dx_affine_px, dtype=np.float32)
    dy_np = np.asarray(dy_affine_px, dtype=np.float32)
    bead_count = len(decomposition.centers_yx)
    if observed_np.shape != (bead_count, len(z_np), config.roi_size, config.roi_size):
        raise ValueError("observed_stacks_adu must match bead, z, and configured ROI dimensions.")
    if dx_np.shape != (bead_count, 2) or dy_np.shape != (bead_count, 2):
        raise ValueError("dx_affine_px and dy_affine_px must have shape (bead,2).")

    device = torch.device(config.device)
    observed = torch.as_tensor(observed_np, dtype=torch.float32, device=device)
    z_values = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    z_normalized = z_values / max(float(np.max(np.abs(z_np))), 1.0)
    carrier = torch.as_tensor(decomposition.shared_carrier_complex, device=device)
    model = DoubleHelixVectorPSF(
        mode_order=decomposition.mode_order,
        na=config.na,
        wavelength_nm=config.wavelength_nm,
        pixel_size_nm=config.pixel_size_nm,
        refractive_index=config.refractive_index,
        npupil=config.npupil,
        psf_size=config.roi_size,
        device=device,
    )
    unit_flux = []
    with torch.no_grad():
        for bead in range(bead_count):
            gamma = torch.as_tensor(
                decomposition.field_gamma_at_beads_nm[bead],
                dtype=torch.float32,
                device=device,
            )
            unit_flux.append(
                model.render(
                    coefficients_nm=gamma[None].expand(len(z_values), -1),
                    z_nm=z_values,
                    carrier_complex=carrier,
                    dx_px=dx_np[bead, 0] + dx_np[bead, 1] * z_normalized,
                    dy_px=dy_np[bead, 0] + dy_np[bead, 1] * z_normalized,
                )
            )
        rendered = torch.stack(unit_flux)
        flat_observed = observed.flatten(0, 1)
        flat_rendered = rendered.flatten(0, 1)
        reconstruction, photons, background = profile_photometry(flat_observed, flat_rendered)
        reconstruction = reconstruction.reshape_as(observed)
        photons = photons.reshape(bead_count, len(z_values))
        background = background.reshape(bead_count, len(z_values))
        ncc = _ncc(observed.flatten(0, 1), reconstruction.flatten(0, 1)).reshape(
            bead_count, len(z_values)
        )

    train_indices, heldout_indices = interleaved_calibration_split(len(z_values))
    per_bead = []
    for bead in range(bead_count):
        heldout_ncc = ncc[bead, heldout_indices]
        per_bead.append(
            {
                "bead_number": bead + 1,
                "center_yx": decomposition.centers_yx[bead].astype(float).tolist(),
                "train_median_ncc": float(torch.median(ncc[bead, train_indices]).item()),
                "heldout_median_ncc": float(torch.median(heldout_ncc).item()),
                "heldout_p10_ncc": float(torch.quantile(heldout_ncc, 0.1).item()),
                "negative_edge_ncc": float(ncc[bead, 0].item()),
                "positive_edge_ncc": float(ncc[bead, -1].item()),
            }
        )
    heldout_values = ncc[:, heldout_indices]
    metrics: dict[str, object] = {
        "heldout_median_ncc_across_beads": float(torch.median(heldout_values).item()),
        "heldout_p10_ncc_across_beads": float(torch.quantile(heldout_values, 0.1).item()),
        "minimum_bead_heldout_median_ncc": float(
            min(item["heldout_median_ncc"] for item in per_bead)
        ),
        "accepted": bool(
            min(item["heldout_median_ncc"] for item in per_bead) >= 0.90
            and min(
                min(item["negative_edge_ncc"], item["positive_edge_ncc"])
                for item in per_bead
            )
            >= 0.85
        ),
        "per_bead": per_bead,
    }
    return SharedFieldCalibrationResult(
        z_nm=z_np.astype(np.float64),
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        reconstruction_unit_flux=rendered.cpu().numpy().astype(np.float32),
        reconstruction_adu=reconstruction.cpu().numpy().astype(np.float32),
        photons_adu=photons.cpu().numpy().astype(np.float32),
        background_adu=background.cpu().numpy().astype(np.float32),
        per_plane_ncc=ncc.cpu().numpy().astype(np.float32),
        metrics=metrics,
    )


def _complex_nrmse(
    observed: torch.Tensor,
    predicted: torch.Tensor,
    pupil_mask: torch.Tensor,
) -> float:
    observed = observed[..., pupil_mask]
    predicted = predicted[..., pupil_mask]
    return float(
        torch.sqrt((observed - predicted).abs().square().mean())
        / torch.sqrt(observed.abs().square().mean()).clamp_min(1e-12)
    )


def _robust_circular_mean(
    values: torch.Tensor,
    pupil_mask: torch.Tensor,
    *,
    delta: float,
    iterations: int = 4,
) -> torch.Tensor:
    eps = torch.finfo(values.real.dtype).eps
    vector = values.sum(dim=0)
    mean = vector / vector.abs().clamp_min(eps)
    mean = mean * pupil_mask
    for _ in range(int(iterations)):
        distance = (values - mean[None]).abs()
        weights = torch.where(
            distance <= float(delta),
            torch.ones_like(distance),
            float(delta) / distance.clamp_min(eps),
        )
        vector = (weights * values).sum(dim=0)
        mean = vector / vector.abs().clamp_min(eps)
        mean = mean * pupil_mask
    return mean


def _ncc(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_centered = first - first.mean(dim=(-2, -1), keepdim=True)
    second_centered = second - second.mean(dim=(-2, -1), keepdim=True)
    numerator = (first_centered * second_centered).sum(dim=(-2, -1))
    denominator = torch.sqrt(
        first_centered.square().sum(dim=(-2, -1))
        * second_centered.square().sum(dim=(-2, -1))
    ).clamp_min(1e-12)
    return numerator / denominator


__all__ = [
    "SharedCarrierFieldConfig",
    "SharedCarrierFieldResult",
    "SharedFieldCalibrationResult",
    "fit_shared_carrier_field",
    "render_shared_field_calibration",
]
