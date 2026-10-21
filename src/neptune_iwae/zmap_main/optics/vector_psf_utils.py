"""Shared vector-PSF utilities for simulation and phase retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .local_vector_psf_fit import LocalVectorPSFTorchFit
from .zernike import noll_to_nm


@dataclass(frozen=True)
class VectorPSFParams:
    npupil: int
    psf_size: int
    refmed: float
    refcov: float
    refimm: float
    objstage0: float
    otf_rescale_xy: tuple[float, float]
    batch_size: int = 64


@dataclass(frozen=True)
class VectorPSFContext:
    psf: "VectorPSFTorchFit"
    noll_indices: list[int]
    normfac: torch.Tensor
    wavelength_nm: float
    device: torch.device
    batch_size: int

def _noll_normfac(noll_indices: Sequence[int]) -> np.ndarray:
    zernike_mode = np.array([noll_to_nm(int(j)) for j in noll_indices], dtype=np.float32)
    n = zernike_mode[:, 0]
    m = zernike_mode[:, 1]
    normfac = np.sqrt(2.0 * (n + 1.0) / (1.0 + (m == 0.0).astype(np.float32)))
    return normfac.astype(np.float32)


def build_vector_psf_context(
    *,
    NA: float,
    wavelength_nm: float,
    pixel_size_nm_x: float,
    pixel_size_nm_y: float,
    noll_indices: Sequence[int],
    params: VectorPSFParams,
    device: torch.device,
) -> VectorPSFContext:
    zernike_mode = np.array([noll_to_nm(int(j)) for j in noll_indices], dtype=np.float32)
    psf_params = {
        "na": float(NA),
        "wavelength": float(wavelength_nm),
        "refmed": float(params.refmed),
        "refcov": float(params.refcov),
        "refimm": float(params.refimm),
        "zernike_mode": zernike_mode,
        "zernike_coef": np.zeros(len(noll_indices), dtype=np.float32),
        "objstage0": float(params.objstage0),
        "pixel_size_xy": [float(pixel_size_nm_x), float(pixel_size_nm_y)],
        "otf_rescale_xy": tuple(float(v) for v in params.otf_rescale_xy),
        "npupil": int(params.npupil),
        "psf_size": int(params.psf_size),
    }
    psf = LocalVectorPSFTorchFit(psf_params, req_grad=False, data_type=torch.float32, device=str(device))
    normfac = torch.from_numpy(_noll_normfac(noll_indices)).to(device=device, dtype=torch.float32)
    return VectorPSFContext(
        psf=psf,
        noll_indices=list(noll_indices),
        normfac=normfac,
        wavelength_nm=float(wavelength_nm),
        device=device,
        batch_size=int(params.batch_size),
    )


def _center_crop_or_pad_torch(psf: torch.Tensor, size: int) -> torch.Tensor:
    if psf.shape[-1] == size and psf.shape[-2] == size:
        return psf
    h, w = psf.shape[-2:]
    if h >= size and w >= size:
        top = (h - size) // 2
        left = (w - size) // 2
        return psf[..., top : top + size, left : left + size]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(psf, (pad_left, pad_right, pad_top, pad_bottom))


def _make_gaussian_kernel(sigma_px: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(3 * sigma_px))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * (x / sigma_px) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d


def _render_vector_psf_chunk(
    ctx: VectorPSFContext,
    coeffs_rad: torch.Tensor,
    z_m: torch.Tensor,
    orient_vecs: torch.Tensor | None,
) -> torch.Tensor:
    psf = ctx.psf
    device = ctx.device

    coeffs_rad = coeffs_rad.to(device=device, dtype=torch.float32)
    z_m = z_m.to(device=device, dtype=torch.float32)

    coeffs_nm = coeffs_rad * (ctx.wavelength_nm / (2.0 * math.pi))
    coeffs_nm = coeffs_nm / ctx.normfac[None, :]

    z_nm = z_m * 1e9
    x_nm = torch.zeros_like(z_nm)
    y_nm = torch.zeros_like(z_nm)

    zernike_phase = torch.exp(
        1j
        * 2
        * math.pi
        * torch.sum(coeffs_nm[:, :, None, None] * psf.allzernikes[None], dim=1)
        / psf.wavelength
    )
    pupilmatrix = (
        zernike_phase[None, None] * psf.polarizationvector[:, :, None] * psf.amplitude[None, None, None]
    )

    z_total = z_nm + psf.zemit0
    base_phase_xy = (
        -y_nm[:, None, None] * psf.wavevector[0][None]
        - x_nm[:, None, None] * psf.wavevector[1][None]
    )
    phase_medium = (
        base_phase_xy
        + z_total[:, None, None] * psf.wavevectorzmed[None]
        + (psf.objstage0) * psf.wavevectorzimm[None]
    )
    phase_immersion = (
        base_phase_xy
        + (psf.objstage0 + z_total)[:, None, None] * psf.wavevectorzimm[None]
    )
    position_phase = torch.exp(
        1j * torch.where((z_total >= 0)[:, None, None], phase_medium, phase_immersion)
    )

    pupil_tmp = position_phase[None, None] * pupilmatrix
    inter_image = torch.transpose(psf.czt_parallel(pupil_tmp, psf.ay, psf.by, psf.dy), -1, -2)
    field_matrix = torch.transpose(psf.czt_parallel(inter_image, psf.ax, psf.bx, psf.dx), -1, -2)

    if orient_vecs is None:
        intensity = (torch.abs(field_matrix) ** 2).sum(dim=(0, 1)) / 3.0
    else:
        fm = field_matrix.permute(2, 0, 1, 3, 4)
        orient_slice = orient_vecs[:, None, :, None, None]
        e_pol = (fm * orient_slice).sum(dim=2)
        intensity = (e_pol.abs() ** 2).sum(dim=1)

    intensity = intensity / psf.norm_intensity
    return intensity.to(dtype=torch.float32)


def render_vector_psf_bank(
    ctx: VectorPSFContext,
    coeffs_rad: np.ndarray | torch.Tensor,
    z_m: np.ndarray | torch.Tensor,
    *,
    orient_vecs: np.ndarray | torch.Tensor | None = None,
    out_size: int | None = None,
    batch_size: int | None = None,
    blur_sigma_px: float = 0.0,
    return_torch: bool = False,
) -> np.ndarray | torch.Tensor:
    device = ctx.device
    coeffs_t = torch.as_tensor(coeffs_rad, device=device, dtype=torch.float32)
    z_m_t = torch.as_tensor(z_m, device=device, dtype=torch.float32)

    if coeffs_t.ndim == 1:
        coeffs_t = coeffs_t.unsqueeze(0)
    if z_m_t.ndim == 0:
        z_m_t = z_m_t.unsqueeze(0)

    if coeffs_t.shape[0] != z_m_t.shape[0]:
        raise ValueError("coeffs_rad and z_m must have the same leading dimension.")

    orient_t = None
    if orient_vecs is not None:
        orient_t = torch.as_tensor(orient_vecs, device=device, dtype=torch.float32)
        if orient_t.ndim != 2 or orient_t.shape[1] != 3:
            raise ValueError("orient_vecs must have shape (N, 3).")
        if orient_t.shape[0] != coeffs_t.shape[0]:
            raise ValueError("orient_vecs length must match coeffs_rad.")

    n_emitters = int(coeffs_t.shape[0])
    size_out = int(out_size) if out_size is not None else int(ctx.psf.psf_size)
    chunk = int(batch_size or ctx.batch_size or n_emitters)
    chunk = max(1, min(chunk, n_emitters))

    chunks: list[torch.Tensor] = []
    for start in range(0, n_emitters, chunk):
        end = min(n_emitters, start + chunk)
        coeffs_chunk = coeffs_t[start:end]
        z_chunk = z_m_t[start:end]
        orient_chunk = orient_t[start:end] if orient_t is not None else None
        intensity = _render_vector_psf_chunk(ctx, coeffs_chunk, z_chunk, orient_chunk)
        if intensity.shape[-1] != size_out or intensity.shape[-2] != size_out:
            intensity = _center_crop_or_pad_torch(intensity, size_out)
        if blur_sigma_px > 0:
            kernel = _make_gaussian_kernel(float(blur_sigma_px), device=device)
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            pad = kernel.shape[-1] // 2
            intensity = F.conv2d(intensity.unsqueeze(1), kernel, padding=pad).squeeze(1)
        chunks.append(intensity)

    if chunks:
        psf_bank = torch.cat(chunks, dim=0)
    else:
        psf_bank = torch.empty((0, size_out, size_out), device=device, dtype=torch.float32)

    psf_bank = torch.clamp(psf_bank, min=0.0)
    sum_vals = psf_bank.sum(dim=(1, 2), keepdim=True)
    psf_bank = torch.where(sum_vals > 0, psf_bank / sum_vals, psf_bank)

    if return_torch:
        return psf_bank
    return psf_bank.detach().cpu().numpy()


def render_vector_psf_stack(
    ctx: VectorPSFContext,
    coeffs_rad: torch.Tensor | np.ndarray,
    defocus_z_m: torch.Tensor | np.ndarray,
    *,
    blur_sigma_px: float = 0.0,
    out_size: int | None = None,
    batch_size: int | None = None,
) -> torch.Tensor:
    coeffs_t = torch.as_tensor(coeffs_rad, device=ctx.device, dtype=torch.float32)
    if coeffs_t.ndim != 1:
        raise ValueError("coeffs_rad must be a 1D array for render_vector_psf_stack.")
    defocus_z_t = torch.as_tensor(defocus_z_m, device=ctx.device, dtype=torch.float32)
    if defocus_z_t.ndim != 1:
        raise ValueError("defocus_z_m must be a 1D array for render_vector_psf_stack.")
    coeffs_rep = coeffs_t.unsqueeze(0).expand(defocus_z_t.shape[0], -1)
    return render_vector_psf_bank(
        ctx,
        coeffs_rep,
        defocus_z_t,
        orient_vecs=None,
        out_size=out_size,
        batch_size=batch_size,
        blur_sigma_px=blur_sigma_px,
        return_torch=True,
    )


__all__ = [
    "VectorPSFParams",
    "VectorPSFContext",
    "build_vector_psf_context",
    "render_vector_psf_bank",
    "render_vector_psf_stack",
]
