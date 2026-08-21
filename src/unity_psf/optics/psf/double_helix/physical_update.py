from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .localization import IndependentLocalizations
from .numerics import legendre_polynomial
from .vector_model import DoubleHelixVectorPSF


@dataclass(frozen=True)
class FullFOVPhysicalUpdateConfig:
    image_shape_hw: tuple[int, int] = (150, 150)
    spatial_degree: int = 2
    gamma_steps: int = 100
    gamma_lr: float = 0.025
    frame_batch_size: int = 8
    spatial_regularization: float = 1e-6
    na: float = 1.27
    wavelength_nm: float = 660.0
    refractive_index: float = 1.33
    pixel_size_nm: float = 200.0
    npupil: int = 128
    psf_size: int = 31
    device: str = "cuda"


@dataclass(frozen=True)
class FullFOVPhysicalUpdateResult:
    gamma_terms_nm: np.ndarray
    gamma_nm: np.ndarray
    spatial_terms: tuple[tuple[int, int], ...]
    mode_order: tuple[tuple[int, int], ...]
    loss_history: np.ndarray
    metrics: dict[str, Any]


def spatial_gamma_terms(degree: int) -> tuple[tuple[int, int], ...]:
    if int(degree) < 0:
        raise ValueError("degree must be non-negative.")
    return tuple(
        (px, total - px)
        for total in range(int(degree) + 1)
        for px in range(total, -1, -1)
    )


def evaluate_residual_coefficients(
    gamma_terms_nm: torch.Tensor,
    *,
    x_px: torch.Tensor,
    y_px: torch.Tensor,
    image_shape_hw: tuple[int, int],
    terms: Sequence[tuple[int, int]],
) -> torch.Tensor:
    parameters = torch.as_tensor(gamma_terms_nm)
    if parameters.ndim != 2 or parameters.shape[1] != len(terms):
        raise ValueError("gamma_terms_nm must have shape (mode_count, spatial_term_count).")
    x = torch.as_tensor(x_px, dtype=parameters.dtype, device=parameters.device).reshape(-1)
    y = torch.as_tensor(y_px, dtype=parameters.dtype, device=parameters.device).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x_px and y_px must have matching shapes.")
    height, width = (int(value) for value in image_shape_hw)
    x_normalized = -1.0 + 2.0 * x / float(width)
    y_normalized = -1.0 + 2.0 * y / float(height)
    design = torch.stack(
        [legendre_polynomial(px, x_normalized) * legendre_polynomial(py, y_normalized) for px, py in terms],
        dim=1,
    )
    return design @ parameters.T


def gamma_terms_to_tensor(
    gamma_terms_nm: torch.Tensor,
    *,
    terms: Sequence[tuple[int, int]],
    degree: int,
) -> torch.Tensor:
    parameters = torch.as_tensor(gamma_terms_nm)
    if parameters.ndim != 2 or parameters.shape[1] != len(terms):
        raise ValueError("gamma_terms_nm must have shape (mode_count, spatial_term_count).")
    gamma = parameters.new_zeros((parameters.shape[0], int(degree) + 1, int(degree) + 1))
    for column, (px, py) in enumerate(terms):
        gamma[:, int(px), int(py)] = parameters[:, column]
    return gamma


def project_patches_to_frames(
    patches: torch.Tensor,
    *,
    batch_index: torch.Tensor,
    center_x_px: torch.Tensor,
    center_y_px: torch.Tensor,
    output_shape: tuple[int, int, int],
) -> torch.Tensor:
    batch, height, width = (int(value) for value in output_shape)
    projected = torch.zeros((batch, height, width), device=patches.device, dtype=patches.dtype)
    if patches.numel() == 0:
        return projected
    if patches.ndim != 3 or patches.shape[-1] != patches.shape[-2]:
        raise ValueError("patches must have shape (N,S,S).")
    patch_size = int(patches.shape[-1])
    radius = patch_size // 2
    yy, xx = torch.meshgrid(
        torch.arange(patch_size, device=patches.device),
        torch.arange(patch_size, device=patches.device),
        indexing="ij",
    )
    image_x = center_x_px.to(device=patches.device, dtype=torch.long)[:, None, None] - radius + xx
    image_y = center_y_px.to(device=patches.device, dtype=torch.long)[:, None, None] - radius + yy
    batch_ix = batch_index.to(device=patches.device, dtype=torch.long)[:, None, None].expand_as(image_x)
    valid = (
        (batch_ix >= 0)
        & (batch_ix < batch)
        & (image_x >= 0)
        & (image_x < width)
        & (image_y >= 0)
        & (image_y < height)
    )
    if bool(valid.any()):
        flat_index = ((batch_ix * height + image_y) * width + image_x)[valid]
        projected.reshape(-1).scatter_add_(0, flat_index.reshape(-1), patches[valid].reshape(-1))
    return projected


def fit_full_fov_physical_update(
    frames_adu: np.ndarray,
    *,
    localizations: IndependentLocalizations,
    initial_gamma_terms_nm: np.ndarray,
    mode_order: tuple[tuple[int, int], ...],
    carrier_complex: np.ndarray | torch.Tensor,
    config: FullFOVPhysicalUpdateConfig,
    model: Any | None = None,
) -> FullFOVPhysicalUpdateResult:
    frames = np.asarray(frames_adu, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != config.image_shape_hw:
        raise ValueError("frames_adu must have shape (frame_count,H,W) matching image_shape_hw.")
    terms = spatial_gamma_terms(config.spatial_degree)
    initial = np.asarray(initial_gamma_terms_nm, dtype=np.float32)
    if initial.shape != (len(mode_order), len(terms)):
        raise ValueError("initial_gamma_terms_nm shape must match mode_order and spatial degree.")
    if localizations.frame_index.size == 0:
        raise ValueError("The full-FOV ROI bank must contain at least one localized emitter.")
    if np.any(localizations.frame_index < 0) or np.any(localizations.frame_index >= len(frames)):
        raise ValueError("Localization frame indices must address frames_adu.")

    device = torch.device(config.device)
    renderer = model or DoubleHelixVectorPSF(
        mode_order=mode_order,
        na=config.na,
        wavelength_nm=config.wavelength_nm,
        pixel_size_nm=config.pixel_size_nm,
        refractive_index=config.refractive_index,
        npupil=config.npupil,
        psf_size=config.psf_size,
        device=device,
    )
    parameters = torch.nn.Parameter(torch.as_tensor(initial, dtype=torch.float32, device=device))
    carrier = torch.as_tensor(carrier_complex, device=device)
    optimizer = torch.optim.Adam((parameters,), lr=float(config.gamma_lr))
    before = parameters.detach().clone()
    loss_history: list[float] = []

    for _ in range(int(config.gamma_steps)):
        optimizer.zero_grad(set_to_none=True)
        data_loss = _backward_full_fov_loss(
            frames,
            localizations=localizations,
            gamma_terms_nm=parameters,
            terms=terms,
            carrier_complex=carrier,
            model=renderer,
            config=config,
        )
        regularization = float(config.spatial_regularization) * (
            parameters / float(config.wavelength_nm)
        ).square().mean()
        regularization.backward()
        optimizer.step()
        loss_history.append(float((data_loss + regularization.detach()).cpu().item()))

    final_loss = evaluate_full_fov_poisson_loss(
        frames,
        localizations=localizations,
        gamma_terms_nm=parameters.detach(),
        terms=terms,
        carrier_complex=carrier,
        model=renderer,
        config=config,
    )
    loss_history.append(final_loss)
    dense_gamma = gamma_terms_to_tensor(
        parameters.detach(),
        terms=terms,
        degree=config.spatial_degree,
    )
    delta = parameters.detach() - before
    metrics = {
        "roi_bank_count": 1,
        "roi_bank_shape_hw": list(config.image_shape_hw),
        "roi_bank_frame_count": int(len(frames)),
        "selected_sampled_emitter_count": int(localizations.frame_index.size),
        "gamma_update_accepted": True,
        "heldout_accept_policy": "monitor",
        "feedback_applied": True,
        "steps": int(config.gamma_steps),
        "lr": float(config.gamma_lr),
        "optimizer": "adam",
        "gamma_before_norm": float(torch.linalg.norm(before).cpu().item()),
        "gamma_after_norm": float(torch.linalg.norm(parameters.detach()).cpu().item()),
        "gamma_delta_norm": float(torch.linalg.norm(delta).cpu().item()),
        "initial_poisson_nll": float(loss_history[0]),
        "final_poisson_nll": float(final_loss),
        "shared_carrier_fixed": True,
    }
    return FullFOVPhysicalUpdateResult(
        gamma_terms_nm=parameters.detach().cpu().numpy().astype(np.float32),
        gamma_nm=dense_gamma.cpu().numpy().astype(np.float32),
        spatial_terms=terms,
        mode_order=mode_order,
        loss_history=np.asarray(loss_history, dtype=np.float32),
        metrics=metrics,
    )


def evaluate_full_fov_poisson_loss(
    frames_adu: np.ndarray,
    *,
    localizations: IndependentLocalizations,
    gamma_terms_nm: torch.Tensor,
    terms: Sequence[tuple[int, int]],
    carrier_complex: torch.Tensor,
    model: Any,
    config: FullFOVPhysicalUpdateConfig,
) -> float:
    total = 0.0
    total_pixels = int(np.prod(np.asarray(frames_adu).shape))
    with torch.no_grad():
        for start in range(0, len(frames_adu), int(config.frame_batch_size)):
            stop = min(start + int(config.frame_batch_size), len(frames_adu))
            loss_sum = _full_fov_chunk_loss_sum(
                frames_adu[start:stop],
                frame_start=start,
                localizations=localizations,
                gamma_terms_nm=gamma_terms_nm,
                terms=terms,
                carrier_complex=carrier_complex,
                model=model,
                config=config,
            )
            total += float(loss_sum.cpu().item())
    return total / max(total_pixels, 1)


def _backward_full_fov_loss(
    frames_adu: np.ndarray,
    *,
    localizations: IndependentLocalizations,
    gamma_terms_nm: torch.Tensor,
    terms: Sequence[tuple[int, int]],
    carrier_complex: torch.Tensor,
    model: Any,
    config: FullFOVPhysicalUpdateConfig,
) -> torch.Tensor:
    total_pixels = int(np.prod(frames_adu.shape))
    detached_total = gamma_terms_nm.detach().new_tensor(0.0)
    for start in range(0, len(frames_adu), int(config.frame_batch_size)):
        stop = min(start + int(config.frame_batch_size), len(frames_adu))
        loss_sum = _full_fov_chunk_loss_sum(
            frames_adu[start:stop],
            frame_start=start,
            localizations=localizations,
            gamma_terms_nm=gamma_terms_nm,
            terms=terms,
            carrier_complex=carrier_complex,
            model=model,
            config=config,
        )
        (loss_sum / max(total_pixels, 1)).backward()
        detached_total = detached_total + loss_sum.detach()
    return detached_total / max(total_pixels, 1)


def _full_fov_chunk_loss_sum(
    frames_adu: np.ndarray,
    *,
    frame_start: int,
    localizations: IndependentLocalizations,
    gamma_terms_nm: torch.Tensor,
    terms: Sequence[tuple[int, int]],
    carrier_complex: torch.Tensor,
    model: Any,
    config: FullFOVPhysicalUpdateConfig,
) -> torch.Tensor:
    device = gamma_terms_nm.device
    frame_stop = frame_start + len(frames_adu)
    rows = (localizations.frame_index >= frame_start) & (localizations.frame_index < frame_stop)
    local_frame = torch.as_tensor(
        localizations.frame_index[rows] - frame_start,
        dtype=torch.long,
        device=device,
    )
    x = torch.as_tensor(localizations.x_px[rows], dtype=torch.float32, device=device)
    y = torch.as_tensor(localizations.y_px[rows], dtype=torch.float32, device=device)
    z = torch.as_tensor(localizations.z_nm[rows], dtype=torch.float32, device=device)
    photons = torch.as_tensor(localizations.photons_adu[rows], dtype=torch.float32, device=device)
    centers_x = torch.floor(x + 0.5).to(dtype=torch.long)
    centers_y = torch.floor(y + 0.5).to(dtype=torch.long)
    coefficients = evaluate_residual_coefficients(
        gamma_terms_nm,
        x_px=x,
        y_px=y,
        image_shape_hw=config.image_shape_hw,
        terms=terms,
    )
    psfs = model.render(
        coefficients_nm=coefficients,
        z_nm=z,
        carrier_complex=carrier_complex,
        dx_px=x - centers_x.to(dtype=x.dtype),
        dy_px=y - centers_y.to(dtype=y.dtype),
    )
    signal = project_patches_to_frames(
        psfs * photons[:, None, None],
        batch_index=local_frame,
        center_x_px=centers_x,
        center_y_px=centers_y,
        output_shape=(len(frames_adu), *config.image_shape_hw),
    )
    background = _frame_backgrounds(
        frames_adu,
        frame_start=frame_start,
        localizations=localizations,
        device=device,
    )
    observed = torch.as_tensor(frames_adu, dtype=torch.float32, device=device).clamp_min(0.0)
    expected = (signal + background[:, None, None]).clamp_min(1e-6)
    return (expected - observed * torch.log(expected) + torch.lgamma(observed + 1.0)).sum()


def _frame_backgrounds(
    frames_adu: np.ndarray,
    *,
    frame_start: int,
    localizations: IndependentLocalizations,
    device: torch.device,
) -> torch.Tensor:
    backgrounds = np.empty(len(frames_adu), dtype=np.float32)
    for local_index in range(len(frames_adu)):
        rows = localizations.frame_index == frame_start + local_index
        if np.any(rows):
            backgrounds[local_index] = float(np.median(localizations.background_adu[rows]))
        else:
            backgrounds[local_index] = float(np.median(frames_adu[local_index]))
    return torch.as_tensor(backgrounds, dtype=torch.float32, device=device)


__all__ = [
    "FullFOVPhysicalUpdateConfig",
    "FullFOVPhysicalUpdateResult",
    "evaluate_full_fov_poisson_loss",
    "evaluate_residual_coefficients",
    "fit_full_fov_physical_update",
    "gamma_terms_to_tensor",
    "project_patches_to_frames",
    "spatial_gamma_terms",
]
