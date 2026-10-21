"""Torch-based PSF utilities shared by simulator_diff."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from .zernike import noll_to_nm, zernike_polynomial


def pupil_coords_from_optics(
    shape: Tuple[int, int],
    *,
    pixel_size_m: float,
    wavelength_m: float,
    NA: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized pupil coordinates (rho, theta) on the frequency grid."""
    H, W = shape
    real_dtype = torch.float32 if dtype in (torch.complex64, torch.float32) else torch.float64
    fx = torch.fft.fftfreq(W, d=pixel_size_m, device=device, dtype=real_dtype)
    fy = torch.fft.fftfreq(H, d=pixel_size_m, device=device, dtype=real_dtype)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    FX = torch.fft.fftshift(FX)
    FY = torch.fft.fftshift(FY)
    f_max = NA / wavelength_m
    rho = torch.sqrt(FX**2 + FY**2) / f_max
    theta = torch.atan2(FY, FX)
    return rho.to(dtype=torch.float32), theta.to(dtype=torch.float32)


def build_pupil_from_zernike(
    coeffs: torch.Tensor,
    pupil_mask: torch.Tensor,
    noll_indices: Sequence[int],
    *,
    rho: torch.Tensor | None = None,
    theta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build complex pupil given Zernike coefficients (radians) and pupil mask."""
    device = coeffs.device
    if rho is None or theta is None:
        H, W = pupil_mask.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, steps=H, device=device),
            torch.linspace(-1, 1, steps=W, device=device),
            indexing="ij",
        )
        rho = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx)
    else:
        rho = rho.to(device=device)
        theta = theta.to(device=device)
    phase = torch.zeros_like(pupil_mask, dtype=coeffs.dtype, device=device)
    for c, j in zip(coeffs, noll_indices):
        n, m = noll_to_nm(j)
        phase = phase + c * zernike_polynomial(n, m, rho, theta)
    aperture = pupil_mask
    return aperture * torch.exp(1j * phase)


def _make_gaussian_kernel(sigma_px: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(3 * sigma_px))
    real_dtype = torch.float32 if dtype in (torch.complex64, torch.float32) else torch.float64
    x = torch.arange(-radius, radius + 1, device=device, dtype=real_dtype)
    kernel_1d = torch.exp(-0.5 * (x / sigma_px) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d


def render_psf(
    pupil: torch.Tensor,
    defocus_z: torch.Tensor,
    pixel_size_m: float,
    wavelength_m: float,
    NA: float,
    upsample: int = 3,
    extra_blur_sigma_px: float = 0.0,
    crop_hw: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """Render PSF stack for given pupil and defocus planes."""
    device = pupil.device
    dtype = pupil.dtype
    H, W = pupil.shape

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    fx = torch.fft.fftfreq(W, d=pixel_size_m, device=device, dtype=real_dtype)
    fy = torch.fft.fftfreq(H, d=pixel_size_m, device=device, dtype=real_dtype)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    FX = torch.fft.fftshift(FX)
    FY = torch.fft.fftshift(FY)
    rho2 = FX**2 + FY**2
    f_max = NA / wavelength_m
    pupil = pupil * (torch.sqrt(rho2) <= f_max)

    psfs = []
    for z in defocus_z:
        phase_defocus = torch.exp(1j * math.pi * wavelength_m * z * rho2)
        pz = pupil * phase_defocus
        field = torch.fft.ifft2(torch.fft.ifftshift(pz))
        field = torch.fft.fftshift(field)
        intensity = (field.conj() * field).real
        psfs.append(intensity)
    psf = torch.stack(psfs, dim=0)

    if crop_hw is not None:
        ch, cw = crop_hw
        h, w = psf.shape[-2:]
        top = (h - ch) // 2
        left = (w - cw) // 2
        psf = psf[:, top : top + ch, left : left + cw]

    if upsample > 1:
        psf = F.interpolate(psf.unsqueeze(1), scale_factor=upsample, mode="bilinear", align_corners=False)
        psf = F.avg_pool2d(psf, kernel_size=upsample, stride=upsample)
        psf = psf[:, 0]

    if extra_blur_sigma_px > 0:
        kernel = _make_gaussian_kernel(extra_blur_sigma_px, device, dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        psf = F.conv2d(psf.unsqueeze(1), kernel, padding=kernel.shape[-1] // 2)
        psf = psf[:, 0]

    psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    return psf.real


__all__ = ["pupil_coords_from_optics", "build_pupil_from_zernike", "render_psf"]
