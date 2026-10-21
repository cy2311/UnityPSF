from __future__ import annotations

import math

import torch
import torch.nn as nn


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

