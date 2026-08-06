from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from .calibration import interleaved_calibration_split, profile_photometry, z_bin_balanced_mean
from .vector_model import DoubleHelixVectorPSF, evaluate_normalized_zernike


@dataclass(frozen=True)
class PixelPupilFitConfig:
    wavelength_nm: float = 660.0
    na: float = 1.27
    refractive_index: float = 1.33
    pixel_size_nm: float = 207.0
    roi_size: int = 19
    npupil: int = 128
    alternating_rounds: int = 2
    local_steps: int = 100
    phase_adam_steps: int = 500
    lbfgs_steps: int = 80
    phase_learning_rate: float = 0.03
    shift_learning_rate_px: float = 0.01
    poisson_loss_weight: float = 1.0
    ncc_loss_weight: float = 2.0
    z_bin_edges_nm: tuple[float, ...] = (400.0, 800.0, 1400.0)
    z_bin_weights: tuple[float, ...] = (5.0, 4.0, 3.0, 4.0)
    phase_anchor_weight: float = 1e-4
    seed: int = 20260725
    device: str = "cuda"


@dataclass(frozen=True)
class PixelPupilFitResult:
    z_nm: np.ndarray
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    pupil_phase_rad: np.ndarray
    complex_pupil: np.ndarray
    reconstruction_unit_flux: np.ndarray
    reconstruction_adu: np.ndarray
    photons_adu: np.ndarray
    background_adu: np.ndarray
    dx_affine_px: np.ndarray
    dy_affine_px: np.ndarray
    per_plane_ncc: np.ndarray
    loss_history: np.ndarray
    metrics: dict[str, float | int | bool | str]


def pupil_grid(
    npupil: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(npupil) <= 0:
        raise ValueError("npupil must be positive.")
    step = 2.0 / float(npupil)
    coordinates = torch.arange(
        -1.0 + step / 2.0,
        1.0,
        step,
        device=device,
        dtype=dtype,
    )
    x_pupil, y_pupil = torch.meshgrid(coordinates, coordinates, indexing="ij")
    pupil_mask = x_pupil.square() + y_pupil.square() < 1.0
    return x_pupil, y_pupil, pupil_mask


def gauge_fixed_phase(
    raw_phase_rad: torch.Tensor,
    x_pupil: torch.Tensor,
    y_pupil: torch.Tensor,
    pupil_mask: torch.Tensor,
) -> torch.Tensor:
    if raw_phase_rad.shape[-2:] != x_pupil.shape or x_pupil.shape != y_pupil.shape:
        raise ValueError("raw phase and pupil coordinates must have matching spatial shapes.")
    if pupil_mask.shape != x_pupil.shape or pupil_mask.dtype != torch.bool:
        raise ValueError("pupil_mask must be a boolean tensor matching the pupil coordinates.")
    gauge_basis = torch.stack(
        (
            torch.ones_like(x_pupil),
            x_pupil,
            y_pupil,
            2.0 * (x_pupil.square() + y_pupil.square()) - 1.0,
        )
    ) * pupil_mask[None]
    gram = torch.einsum("khw,lhw->kl", gauge_basis, gauge_basis)
    projection = torch.einsum("...hw,khw->...k", raw_phase_rad, gauge_basis)
    flat_projection = projection.reshape(-1, gauge_basis.shape[0])
    coefficients = torch.linalg.solve(gram, flat_projection.T).T.reshape(projection.shape)
    correction = torch.einsum("...k,khw->...hw", coefficients, gauge_basis)
    return (raw_phase_rad - correction) * pupil_mask


def phase_only_complex_pupil(
    phase_rad: torch.Tensor,
    pupil_mask: torch.Tensor,
) -> torch.Tensor:
    if phase_rad.shape[-2:] != pupil_mask.shape or pupil_mask.dtype != torch.bool:
        raise ValueError("phase_rad and boolean pupil_mask must have matching spatial shapes.")
    return torch.exp(1j * phase_rad) * pupil_mask


def load_zernike_phase_initialization(
    path: str | Path,
    *,
    npupil: int,
    wavelength_nm: float,
) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "gamma_nm" not in payload or "mode_order" not in payload:
            raise ValueError("Initialization NPZ must contain gamma_nm and mode_order.")
        gamma = np.asarray(payload["gamma_nm"], dtype=np.float32).reshape(-1)
        mode_values = np.asarray(payload["mode_order"], dtype=np.int64)
    if mode_values.shape != (len(gamma), 2):
        raise ValueError("mode_order must contain one (n,m) pair per gamma coefficient.")
    mode_order = tuple((int(n), int(m)) for n, m in mode_values)
    x_pupil, y_pupil, pupil_mask = pupil_grid(
        npupil,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    basis = evaluate_normalized_zernike(mode_order, x_pupil, y_pupil)
    phase = 2.0 * math.pi * torch.einsum(
        "c,chw->hw",
        torch.from_numpy(gamma),
        basis,
    ) / float(wavelength_nm)
    fixed = gauge_fixed_phase(phase, x_pupil, y_pupil, pupil_mask)
    return fixed.numpy().astype(np.float32)


def fit_single_pixel_pupil(
    stack_adu: np.ndarray,
    *,
    z_nm: np.ndarray,
    initial_phase_rad: np.ndarray,
    config: PixelPupilFitConfig,
) -> PixelPupilFitResult:
    stack = np.asarray(stack_adu, dtype=np.float32)
    z_values_np = np.asarray(z_nm, dtype=np.float32)
    initial_phase_np = np.asarray(initial_phase_rad, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[1:] != (config.roi_size, config.roi_size):
        raise ValueError("stack_adu must have shape (z, roi_size, roi_size).")
    if z_values_np.shape != (len(stack),):
        raise ValueError("z_nm must contain one coordinate per stack plane.")
    if initial_phase_np.shape != (config.npupil, config.npupil):
        raise ValueError("initial_phase_rad must match the configured pupil size.")
    if len(config.z_bin_weights) != len(config.z_bin_edges_nm) + 1:
        raise ValueError("z_bin_weights must contain one more entry than z_bin_edges_nm.")

    device = torch.device(config.device)
    torch.manual_seed(int(config.seed))
    observed = torch.as_tensor(stack, dtype=torch.float32, device=device)
    z_values = torch.as_tensor(z_values_np, dtype=torch.float32, device=device)
    z_scale = max(float(np.max(np.abs(z_values_np))), 1.0)
    z_normalized = z_values / z_scale
    train_indices, heldout_indices = interleaved_calibration_split(len(stack))
    train = torch.as_tensor(train_indices, dtype=torch.long, device=device)

    x_pupil, y_pupil, pupil_mask = pupil_grid(
        config.npupil,
        device=device,
        dtype=torch.float32,
    )
    initial_phase = gauge_fixed_phase(
        torch.as_tensor(initial_phase_np, device=device),
        x_pupil,
        y_pupil,
        pupil_mask,
    ).detach()
    raw_phase = initial_phase.clone().requires_grad_(True)
    dx_affine = torch.zeros(2, dtype=torch.float32, device=device, requires_grad=True)
    dy_affine = torch.zeros(2, dtype=torch.float32, device=device, requires_grad=True)
    model = DoubleHelixVectorPSF(
        mode_order=((0, 0),),
        na=config.na,
        wavelength_nm=config.wavelength_nm,
        pixel_size_nm=config.pixel_size_nm,
        refractive_index=config.refractive_index,
        npupil=config.npupil,
        psf_size=config.roi_size,
        device=device,
    )
    loss_history: list[float] = []

    def render(
        phase_parameter: torch.Tensor,
        shift_x: torch.Tensor,
        shift_y: torch.Tensor,
        plane_indices: torch.Tensor,
    ) -> torch.Tensor:
        phase = gauge_fixed_phase(
            phase_parameter,
            x_pupil,
            y_pupil,
            pupil_mask,
        )
        pupil = phase_only_complex_pupil(phase, pupil_mask)
        selected_z = z_values[plane_indices]
        return model.render(
            coefficients_nm=observed.new_zeros((plane_indices.numel(), 1)),
            z_nm=selected_z,
            carrier_complex=pupil,
            dx_px=shift_x[0] + shift_x[1] * z_normalized[plane_indices],
            dy_px=shift_y[0] + shift_y[1] * z_normalized[plane_indices],
        )

    def objective(
        phase_parameter: torch.Tensor,
        shift_x: torch.Tensor,
        shift_y: torch.Tensor,
        plane_indices: torch.Tensor,
    ) -> torch.Tensor:
        unit_flux = render(phase_parameter, shift_x, shift_y, plane_indices)
        per_plane = _profiled_per_plane_loss(
            observed[plane_indices],
            unit_flux,
            poisson_weight=config.poisson_loss_weight,
            ncc_weight=config.ncc_loss_weight,
        )
        data_loss = z_bin_balanced_mean(
            per_plane,
            z_values[plane_indices],
            bin_edges_nm=config.z_bin_edges_nm,
            bin_weights=config.z_bin_weights,
        )
        phase = gauge_fixed_phase(
            phase_parameter,
            x_pupil,
            y_pupil,
            pupil_mask,
        )
        anchor = (
            phase_only_complex_pupil(phase, pupil_mask)
            - phase_only_complex_pupil(initial_phase, pupil_mask)
        ).abs().square()[pupil_mask].mean()
        return data_loss + config.phase_anchor_weight * anchor

    with torch.no_grad():
        initial_objective = float(objective(raw_phase, dx_affine, dy_affine, train).item())

    for _ in range(int(config.alternating_rounds)):
        if config.local_steps > 0:
            local_optimizer = torch.optim.Adam(
                (dx_affine, dy_affine),
                lr=config.shift_learning_rate_px,
            )
            for _ in range(int(config.local_steps)):
                local_optimizer.zero_grad(set_to_none=True)
                loss = objective(raw_phase.detach(), dx_affine, dy_affine, train)
                loss.backward()
                local_optimizer.step()
                loss_history.append(float(loss.detach().item()))

        if config.phase_adam_steps > 0:
            phase_optimizer = torch.optim.Adam((raw_phase,), lr=config.phase_learning_rate)
            for step in range(int(config.phase_adam_steps)):
                progress = step / max(int(config.phase_adam_steps) - 1, 1)
                phase_optimizer.param_groups[0]["lr"] = config.phase_learning_rate * (
                    0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))
                )
                phase_optimizer.zero_grad(set_to_none=True)
                loss = objective(raw_phase, dx_affine.detach(), dy_affine.detach(), train)
                loss.backward()
                phase_optimizer.step()
                loss_history.append(float(loss.detach().item()))

    if config.lbfgs_steps > 0:
        optimizer = torch.optim.LBFGS(
            (raw_phase, dx_affine, dy_affine),
            lr=0.5,
            max_iter=int(config.lbfgs_steps),
            tolerance_change=1e-9,
            tolerance_grad=1e-7,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(raw_phase, dx_affine, dy_affine, train)
            loss.backward()
            loss_history.append(float(loss.detach().item()))
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        all_planes = torch.arange(len(stack), dtype=torch.long, device=device)
        pupil_phase = gauge_fixed_phase(raw_phase, x_pupil, y_pupil, pupil_mask)
        complex_pupil = phase_only_complex_pupil(pupil_phase, pupil_mask)
        unit_flux = render(raw_phase, dx_affine, dy_affine, all_planes)
        reconstruction, photons, background = profile_photometry(observed, unit_flux)
        ncc = _ncc(observed, reconstruction)
        final_objective = float(objective(raw_phase, dx_affine, dy_affine, train).item())
        heldout = torch.as_tensor(heldout_indices, dtype=torch.long, device=device)
        metrics: dict[str, float | int | bool | str] = {
            "device": str(device),
            "initial_train_objective": initial_objective,
            "final_train_objective": final_objective,
            "train_median_ncc": float(torch.median(ncc[train]).item()),
            "heldout_median_ncc": float(torch.median(ncc[heldout]).item()),
            "heldout_p10_ncc": float(torch.quantile(ncc[heldout], 0.1).item()),
            "negative_edge_ncc": float(ncc[0].item()),
            "positive_edge_ncc": float(ncc[-1].item()),
            "fixed_z_offset_nm": 0.0,
            "fixed_z_scale": 1.0,
            "defocus_gauge_fixed": True,
        }

    return PixelPupilFitResult(
        z_nm=z_values_np.astype(np.float64),
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        pupil_phase_rad=pupil_phase.cpu().numpy().astype(np.float32),
        complex_pupil=complex_pupil.cpu().numpy().astype(np.complex64),
        reconstruction_unit_flux=unit_flux.cpu().numpy().astype(np.float32),
        reconstruction_adu=reconstruction.cpu().numpy().astype(np.float32),
        photons_adu=photons.cpu().numpy().astype(np.float32),
        background_adu=background.cpu().numpy().astype(np.float32),
        dx_affine_px=dx_affine.detach().cpu().numpy().astype(np.float32),
        dy_affine_px=dy_affine.detach().cpu().numpy().astype(np.float32),
        per_plane_ncc=ncc.cpu().numpy().astype(np.float32),
        loss_history=np.asarray(loss_history, dtype=np.float32),
        metrics=metrics,
    )


def _profiled_per_plane_loss(
    observed: torch.Tensor,
    unit_flux: torch.Tensor,
    *,
    poisson_weight: float,
    ncc_weight: float,
) -> torch.Tensor:
    reconstruction, _, _ = profile_photometry(observed, unit_flux)
    eps = torch.finfo(observed.dtype).eps
    expected = reconstruction.clamp_min(eps)
    observed_positive = observed.clamp_min(eps)
    deviance = 2.0 * (
        expected
        - observed
        + observed * (torch.log(observed_positive) - torch.log(expected))
    )
    poisson = deviance.mean(dim=(-2, -1)) / observed.mean(dim=(-2, -1)).clamp_min(1.0)
    return poisson_weight * poisson + ncc_weight * (1.0 - _ncc(observed, expected))


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
    "PixelPupilFitConfig",
    "PixelPupilFitResult",
    "fit_single_pixel_pupil",
    "gauge_fixed_phase",
    "load_zernike_phase_initialization",
    "phase_only_complex_pupil",
    "pupil_grid",
]
