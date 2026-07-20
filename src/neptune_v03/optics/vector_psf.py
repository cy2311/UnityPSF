from __future__ import annotations

import math

import torch
import torch.nn as nn


def noll_to_nm(j: int) -> tuple[int, int]:
    if int(j) < 1:
        raise ValueError("Noll index must be >= 1")
    n = int(math.ceil((-3.0 + math.sqrt(1.0 + 8.0 * int(j))) / 2.0))
    m = int(j - (n * (n + 1)) // 2 - 1)
    if (n % 2) != (m % 2):
        m += 1
    if (int(j) % 2) == 1:
        m = -m
    return n, m


class LocalVectorPSFTorchFit:
    """Project-local vector PSF torch model, vendored into Neptune."""

    def __init__(self, psf_params, req_grad: bool = False, data_type: torch.dtype = torch.float32, device: str = "cuda"):
        self.device = torch.device(device)
        self.data_type = data_type
        if data_type == torch.float64:
            self.complex_type = torch.complex128
        elif data_type == torch.float32:
            self.complex_type = torch.complex64
        else:
            raise ValueError(f"unsupported data type {data_type}")

        self.na = torch.tensor(psf_params["na"], device=self.device, dtype=self.data_type)
        self.wavelength = torch.tensor(psf_params["wavelength"], device=self.device, dtype=self.data_type)
        self.refmed = torch.tensor(psf_params["refmed"], device=self.device, dtype=self.data_type)
        self.refcov = torch.tensor(psf_params["refcov"], device=self.device, dtype=self.data_type)
        self.refimm = torch.tensor(psf_params["refimm"], device=self.device, dtype=self.data_type)
        self.zernike_mode = torch.tensor(psf_params["zernike_mode"], device=self.device, dtype=self.data_type)
        self.zernike_coef = torch.tensor(
            psf_params["zernike_coef"],
            device=self.device,
            dtype=self.data_type,
            requires_grad=req_grad,
        )
        self.objstage0 = torch.tensor(psf_params["objstage0"], device=self.device, dtype=self.data_type)
        self.zemit0 = torch.tensor(
            psf_params.get("zemit0", -psf_params["objstage0"] / psf_params["refimm"] * psf_params["refmed"]),
            device=self.device,
            dtype=self.data_type,
        )
        self.pixel_size_xy = torch.tensor(psf_params["pixel_size_xy"], device=self.device, dtype=self.data_type)
        self.otf_rescale_xy = torch.tensor(psf_params["otf_rescale_xy"], device=self.device, dtype=self.data_type)
        self.npupil = int(psf_params["npupil"])
        self.psf_size = int(psf_params["psf_size"])

        self._pre_compute()

    @staticmethod
    def _gauss2D_kernel(shape=(3, 3), sigmax=0.5, sigmay=0.5, device="cuda", data_type=torch.float32):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = torch.meshgrid(
            torch.arange(-m, m + 1, 1, device=device, dtype=data_type),
            torch.arange(-n, n + 1, 1, device=device, dtype=data_type),
            indexing="ij",
        )
        h = torch.exp(-(x * x) / (2.0 * sigmax * sigmax + 1e-6) - (y * y) / (2.0 * sigmay * sigmay + 1e-6))
        return h / h.sum().clamp_min(torch.finfo(h.dtype).eps)

    @staticmethod
    def otf_rescale(psfdata, sigma_xy):
        kernel = LocalVectorPSFTorchFit._gauss2D_kernel(
            shape=(5, 5),
            sigmax=float(sigma_xy[0]),
            sigmay=float(sigma_xy[1]),
            device=str(psfdata.device),
            data_type=psfdata.dtype,
        ).reshape((1, 1, 5, 5))
        psf_size = psfdata.shape[1]
        psfdata = psfdata.view(-1, 1, psf_size, psf_size)
        tmp = nn.functional.conv2d(psfdata, kernel, padding=2, stride=1)
        return tmp.view(-1, psf_size, psf_size)

    @staticmethod
    def get_zernike(orders, xpupil, ypupil, device):
        xpupil = torch.real(xpupil)
        ypupil = torch.real(ypupil)
        nzer = orders.shape[0]
        radormax = int(torch.max(orders[:, 0]).item())
        azormax = int(torch.max(torch.abs(orders[:, 1])).item())
        nx, ny = xpupil.shape
        zerpol = torch.zeros([21, 6, nx, ny], device=device)
        rhosq = xpupil**2 + ypupil**2
        rho = torch.sqrt(rhosq)
        zerpol[0, 0, :, :] = torch.ones_like(xpupil)

        for jm in range(1, azormax + 3):
            m = jm - 1
            if m > 0:
                zerpol[jm - 1, jm - 1, :, :] = rho * zerpol[jm - 2, jm - 2, :, :]
            zerpol[jm + 1, jm - 1, :, :] = ((m + 2) * rhosq - m - 1) * zerpol[jm - 1, jm - 1, :, :]
            for p in range(2, radormax - m + 3):
                n = m + 2 * p
                jn = n + 1
                zerpol[jn - 1, jm - 1, :, :] = (
                    2
                    * (n - 1)
                    * (n * (n - 2) * (2 * rhosq - 1) - m**2)
                    * zerpol[jn - 3, jm - 1, :, :]
                    - n * (n + m - 2) * (n - m - 2) * zerpol[jn - 5, jm - 1, :, :]
                ) / ((n - 2) * (n + m) * (n - m))

        phi = torch.atan2(ypupil, xpupil)
        allzernikes = torch.zeros([nzer, nx, ny], device=device)
        for j in range(1, nzer + 1):
            n = int(orders[j - 1, 0].item())
            m = int(orders[j - 1, 1].item())
            if m >= 0:
                allzernikes[j - 1, :, :] = zerpol[n, m, :, :] * torch.cos(m * phi)
            else:
                allzernikes[j - 1, :, :] = zerpol[n, -m, :, :] * torch.sin(-m * phi)
        return allzernikes

    def prechirpz(self, xsize, qsize, n, m):
        l = n + m - 1
        sigma = 2 * math.pi * xsize * qsize / n / m
        afac = torch.exp(torch.tensor(2j * sigma * (1 - m), dtype=self.complex_type, device=self.device))
        bfac = torch.exp(torch.tensor(2j * sigma * (1 - n), dtype=self.complex_type, device=self.device))
        sqw = torch.exp(torch.tensor(2j * sigma, dtype=self.complex_type, device=self.device))
        w = sqw**2
        gfac = (2 * xsize / n) * torch.exp(torch.tensor(1j * sigma * (1 - n) * (1 - m), dtype=self.complex_type, device=self.device))

        utmp = torch.zeros([1, n], dtype=self.complex_type, device=self.device)
        a = torch.zeros([1, n], dtype=self.complex_type, device=self.device)
        utmp[0, 0] = sqw * afac
        a[0, 0] = 1.0
        for i in range(1, n):
            a[0, i] = utmp[0, i - 1] * a[0, i - 1]
            utmp[0, i] = utmp[0, i - 1] * w

        utmp = torch.zeros([1, m], dtype=self.complex_type, device=self.device)
        b = torch.ones([1, m], dtype=self.complex_type, device=self.device)
        utmp[0, 0] = sqw * bfac
        b[0, 0] = gfac
        for i in range(1, m):
            b[0, i] = utmp[0, i - 1] * b[0, i - 1]
            utmp[0, i] = utmp[0, i - 1] * w

        utmp = torch.zeros([1, max(n, m) + 1], dtype=self.complex_type, device=self.device)
        vtmp = torch.zeros([1, max(n, m) + 1], dtype=self.complex_type, device=self.device)
        utmp[0, 0] = sqw
        vtmp[0, 0] = 1.0
        for i in range(1, max(n, m) + 1):
            vtmp[0, i] = utmp[0, i - 1] * vtmp[0, i - 1]
            utmp[0, i] = utmp[0, i - 1] * w

        d = torch.ones([1, l], dtype=self.complex_type, device=self.device)
        for i in range(0, m):
            d[0, i] = torch.conj(vtmp[0, i])
        for i in range(0, n):
            d[0, l - 1 - i] = torch.conj(vtmp[0, i + 1])
        d = torch.fft.fft(d, dim=1)
        return a, b, d

    def czt_parallel(self, datain, a, b, d):
        n = a.shape[1]
        m = b.shape[1]
        l = d.shape[1]
        k = datain.shape[-2]
        n_mol = datain.shape[-3]

        amt = a.expand(k, n)
        bmt = b.expand(k, m)
        dmt = d.expand(k, l)
        cztin = torch.zeros([2, 3, n_mol, k, l], dtype=self.complex_type, device=self.device)
        cztin[:, :, :, :, 0:n] = amt[None, None, None] * datain
        tmp = dmt * torch.fft.fft(cztin, dim=-1)
        cztout = torch.fft.ifft(tmp, dim=-1)
        return bmt[None, None, None] * cztout[:, :, :, :, 0:m]

    def _pre_compute(self):
        pupil_size = 1.0
        dxypupil = 2 * pupil_size / self.npupil
        xypupil = torch.arange(
            -pupil_size + dxypupil / 2,
            pupil_size,
            dxypupil,
            device=self.device,
            dtype=self.data_type,
        )
        xpupil, ypupil = torch.meshgrid(xypupil, xypupil, indexing="ij")
        ypupil = torch.complex(ypupil, torch.zeros_like(ypupil))
        xpupil = torch.complex(xpupil, torch.zeros_like(xpupil))

        costhetamed = torch.sqrt(1.0 - (xpupil**2 + ypupil**2) * (self.na**2) / (self.refmed**2))
        costhetacov = torch.sqrt(1.0 - (xpupil**2 + ypupil**2) * (self.na**2) / (self.refcov**2))
        costhetaimm = torch.sqrt(1.0 - (xpupil**2 + ypupil**2) * (self.na**2) / (self.refimm**2))
        fresnelpmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetacov + self.refcov * costhetamed)
        fresnelsmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetamed + self.refcov * costhetacov)
        fresnelpcovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetaimm + self.refimm * costhetacov)
        fresnelscovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetacov + self.refimm * costhetaimm)
        fresnelp = fresnelpmedcov * fresnelpcovimm
        fresnels = fresnelsmedcov * fresnelscovimm
        apod = torch.sqrt(costhetaimm) / costhetamed
        aperturemask = torch.where((xpupil**2 + ypupil**2).real < 1.0, 1.0, 0.0)
        self.amplitude = aperturemask * apod

        phi = torch.atan2(torch.real(ypupil), torch.real(xpupil))
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        costheta = costhetamed
        sintheta = torch.sqrt(1 - costheta**2)

        pvec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device=self.device)
        pvec[0] = fresnelp * costheta * cosphi
        pvec[1] = fresnelp * costheta * sinphi
        pvec[2] = -fresnelp * sintheta
        svec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device=self.device)
        svec[0] = -fresnels * sinphi
        svec[1] = fresnels * cosphi
        svec[2] = 0 * cosphi

        self.polarizationvector = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device=self.device)
        self.polarizationvector[0] = cosphi * pvec - sinphi * svec
        self.polarizationvector[1] = sinphi * pvec + cosphi * svec

        self.wavevector = torch.empty([2, self.npupil, self.npupil], dtype=self.complex_type, device=self.device)
        self.wavevector[0] = 2 * math.pi * self.na / self.wavelength * xpupil
        self.wavevector[1] = 2 * math.pi * self.na / self.wavelength * ypupil
        self.wavevectorzimm = 2 * math.pi * self.refimm / self.wavelength * costhetaimm
        self.wavevectorzmed = 2 * math.pi * self.refmed / self.wavelength * costhetamed

        normfac = torch.sqrt(2 * (self.zernike_mode[:, 0] + 1) / (1 + torch.where(self.zernike_mode[:, 1] == 0, 1.0, 0.0)))
        self.allzernikes = self.get_zernike(self.zernike_mode, xpupil, ypupil, self.device) * normfac[:, None, None] * aperturemask[None]

        xrange = self.pixel_size_xy[0] * self.psf_size / 2
        yrange = self.pixel_size_xy[1] * self.psf_size / 2
        imagesizex = xrange * self.na / self.wavelength
        imagesizey = yrange * self.na / self.wavelength
        self.ax, self.bx, self.dx = self.prechirpz(pupil_size, imagesizey, self.npupil, self.psf_size)
        self.ay, self.by, self.dy = self.prechirpz(pupil_size, imagesizex, self.npupil, self.psf_size)

        pupilfunction_norm = self.amplitude[None, None, None] * self.polarizationvector[:, :, None]
        inter_image_norm = torch.transpose(self.czt_parallel(pupilfunction_norm, self.ax, self.bx, self.dx), -1, -2)
        fieldmatrix_norm = torch.transpose(self.czt_parallel(inter_image_norm, self.ay, self.by, self.dy), -1, -2)
        int_focus = torch.zeros([self.psf_size, self.psf_size], dtype=self.data_type, device=self.device)
        int_focus += (torch.abs(fieldmatrix_norm) ** 2).sum(dim=(0, 1, 2)) / 3.0
        self.norm_intensity = torch.sum(int_focus)



"""Shared vector-PSF utilities for simulation and phase retrieval."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F



@dataclass(frozen=True)
class VectorPSFParams:
    npupil: int
    psf_size: int
    refmed: float
    refcov: float
    refimm: float
    objstage0: float
    otf_rescale_xy: tuple[float, float]
    zemit0: float | None = None
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
    if params.zemit0 is not None:
        psf_params["zemit0"] = float(params.zemit0)
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
