from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence

import numpy as np
import torch

from unity_psf.optics.vector_psf import LocalVectorPSFTorchFit


def evaluate_normalized_zernike(
    mode_order: Sequence[tuple[int, int]],
    x_pupil: torch.Tensor,
    y_pupil: torch.Tensor,
) -> torch.Tensor:
    if x_pupil.shape != y_pupil.shape or x_pupil.ndim != 2:
        raise ValueError("x_pupil and y_pupil must be two-dimensional tensors with matching shapes.")
    modes = tuple(_validate_mode(mode) for mode in mode_order)
    rho = torch.sqrt(x_pupil.square() + y_pupil.square())
    phi = torch.atan2(y_pupil, x_pupil)
    aperture = rho < 1.0
    basis = []
    for n, m in modes:
        if n <= 8 and abs(m) <= 4:
            order = torch.tensor(((n, m),), dtype=x_pupil.dtype, device=x_pupil.device)
            unnormalized = LocalVectorPSFTorchFit.get_zernike(
                order,
                x_pupil,
                y_pupil,
                x_pupil.device,
            )[0].to(x_pupil.dtype)
        else:
            radial = torch.zeros_like(rho)
            absolute_m = abs(m)
            for k in range((n - absolute_m) // 2 + 1):
                numerator = (-1) ** k * math.factorial(n - k)
                denominator = (
                    math.factorial(k)
                    * math.factorial((n + absolute_m) // 2 - k)
                    * math.factorial((n - absolute_m) // 2 - k)
                )
                radial = radial + (numerator / denominator) * rho.pow(n - 2 * k)
            if m >= 0:
                angular = torch.cos(float(m) * phi)
            else:
                angular = torch.sin(float(-m) * phi)
            unnormalized = (radial * angular).to(torch.float32).to(x_pupil.dtype)
        normalization = math.sqrt(2.0 * (n + 1.0) / (1.0 + float(m == 0)))
        basis.append(torch.where(aperture, normalization * unnormalized, 0.0))
    if not basis:
        return x_pupil.new_empty((0, *x_pupil.shape))
    return torch.stack(basis)


def fourier_shift(
    images: torch.Tensor,
    *,
    dx_px: torch.Tensor | Sequence[float] | float,
    dy_px: torch.Tensor | Sequence[float] | float,
) -> torch.Tensor:
    if images.ndim < 2 or not torch.is_floating_point(images):
        raise ValueError("images must be a floating-point tensor with at least two dimensions.")
    height, width = images.shape[-2:]
    batch_shape = images.shape[:-2]
    dx = torch.as_tensor(dx_px, dtype=images.dtype, device=images.device)
    dy = torch.as_tensor(dy_px, dtype=images.dtype, device=images.device)
    dx, dy = torch.broadcast_tensors(dx, dy)
    dx = torch.broadcast_to(dx, batch_shape)
    dy = torch.broadcast_to(dy, batch_shape)
    frequency_x = torch.fft.fftfreq(width, device=images.device, dtype=images.dtype)
    frequency_y = torch.fft.fftfreq(height, device=images.device, dtype=images.dtype)
    phase = torch.exp(
        -2j
        * math.pi
        * (
            dy[..., None, None] * frequency_y[:, None]
            + dx[..., None, None] * frequency_x[None, :]
        )
    )
    return torch.fft.ifft2(torch.fft.fft2(images) * phase).real


class DoubleHelixVectorPSF:
    def __init__(
        self,
        *,
        mode_order: Sequence[tuple[int, int]],
        na: float,
        wavelength_nm: float,
        pixel_size_nm: float,
        refractive_index: float,
        npupil: int,
        psf_size: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.mode_order = tuple(_validate_mode(mode) for mode in mode_order)
        if not self.mode_order:
            raise ValueError("mode_order must contain at least one Zernike mode.")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64.")
        self.device = torch.device(device)
        self.dtype = dtype
        self.wavelength_nm = float(wavelength_nm)
        self.psf_size = int(psf_size)
        if self.psf_size <= 0 or int(npupil) <= 0:
            raise ValueError("npupil and psf_size must be positive.")

        parameters = {
            "na": float(na),
            "wavelength": self.wavelength_nm,
            "refmed": float(refractive_index),
            "refcov": float(refractive_index),
            "refimm": float(refractive_index),
            "zernike_mode": np.asarray(((0, 0),), dtype=np.float32),
            "zernike_coef": np.zeros(1, dtype=np.float32),
            "objstage0": 0.0,
            "zemit0": 0.0,
            "pixel_size_xy": (float(pixel_size_nm), float(pixel_size_nm)),
            "otf_rescale_xy": (0.0, 0.0),
            "npupil": int(npupil),
            "psf_size": self.psf_size,
        }
        self._optics = LocalVectorPSFTorchFit(
            parameters,
            req_grad=False,
            data_type=dtype,
            device=str(self.device),
        )
        pupil_step = 2.0 / float(npupil)
        coordinates = torch.arange(
            -1.0 + pupil_step / 2.0,
            1.0,
            pupil_step,
            dtype=dtype,
            device=self.device,
        )
        x_pupil, y_pupil = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.zernike_basis = evaluate_normalized_zernike(self.mode_order, x_pupil, y_pupil)

    def render(
        self,
        *,
        coefficients_nm: torch.Tensor | Sequence[Sequence[float]],
        z_nm: torch.Tensor | Sequence[float] | float,
        carrier_complex: torch.Tensor | None = None,
        dx_px: torch.Tensor | Sequence[float] | float | None = None,
        dy_px: torch.Tensor | Sequence[float] | float | None = None,
    ) -> torch.Tensor:
        coefficients = torch.as_tensor(coefficients_nm, dtype=self.dtype, device=self.device)
        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0)
        if coefficients.ndim != 2 or coefficients.shape[1] != len(self.mode_order):
            raise ValueError("coefficients_nm must have shape (N, C) matching mode_order.")
        z_values = torch.as_tensor(z_nm, dtype=self.dtype, device=self.device).reshape(-1)
        if z_values.numel() == 1 and coefficients.shape[0] != 1:
            z_values = z_values.expand(coefficients.shape[0])
        if z_values.shape[0] != coefficients.shape[0]:
            raise ValueError("z_nm must be scalar or have the same leading dimension as coefficients_nm.")

        optics = self._optics
        pupil_phase = torch.exp(
            2j
            * math.pi
            * torch.einsum("nc,chw->nhw", coefficients, self.zernike_basis)
            / self.wavelength_nm
        )
        if carrier_complex is not None:
            complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128
            carrier = torch.as_tensor(carrier_complex, dtype=complex_dtype, device=self.device)
            if carrier.ndim == 2:
                carrier = carrier.unsqueeze(0)
            if carrier.ndim != 3 or carrier.shape[-2:] != self.zernike_basis.shape[-2:]:
                raise ValueError("carrier_complex must match the sampled pupil shape.")
            if carrier.shape[0] == 1:
                carrier = carrier.expand(coefficients.shape[0], -1, -1)
            if carrier.shape[0] != coefficients.shape[0]:
                raise ValueError("carrier_complex batch dimension must be one or match coefficients_nm.")
            carrier = carrier / carrier.abs().clamp_min(torch.finfo(self.dtype).eps)
            pupil_phase = pupil_phase * carrier
        pupil_matrix = (
            pupil_phase[None, None]
            * optics.polarizationvector[:, :, None]
            * optics.amplitude[None, None, None]
        )
        axial_phase = torch.exp(1j * z_values[:, None, None] * optics.wavevectorzmed[None])
        pupil_matrix = axial_phase[None, None] * pupil_matrix
        intermediate = torch.transpose(
            optics.czt_parallel(pupil_matrix, optics.ay, optics.by, optics.dy),
            -1,
            -2,
        )
        field = torch.transpose(
            optics.czt_parallel(intermediate, optics.ax, optics.bx, optics.dx),
            -1,
            -2,
        )
        intensity = field.abs().square().sum(dim=(0, 1)) / 3.0
        intensity = intensity / optics.norm_intensity
        intensity = intensity.clamp_min(0.0)
        intensity = intensity / intensity.sum(dim=(-2, -1), keepdim=True)

        if dx_px is not None or dy_px is not None:
            shift_x = 0.0 if dx_px is None else dx_px
            shift_y = 0.0 if dy_px is None else dy_px
            intensity = fourier_shift(intensity, dx_px=shift_x, dy_px=shift_y).clamp_min(0.0)
            intensity = intensity / intensity.sum(dim=(-2, -1), keepdim=True)
        return intensity


def _validate_mode(mode: Sequence[int]) -> tuple[int, int]:
    if isinstance(mode, (str, bytes)) or not hasattr(mode, "__len__") or len(mode) != 2:
        raise ValueError("Each mode_order entry must be an (n, m) pair.")
    n, m = mode
    if (
        isinstance(n, bool)
        or isinstance(m, bool)
        or not isinstance(n, Integral)
        or not isinstance(m, Integral)
        or n < 0
        or abs(m) > n
        or (n - abs(m)) % 2 != 0
    ):
        raise ValueError(f"Invalid Zernike mode: {(n, m)}")
    return int(n), int(m)


__all__ = ["DoubleHelixVectorPSF", "evaluate_normalized_zernike", "fourier_shift"]
