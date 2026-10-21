from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from optics.vector_psf_utils import (
    VectorPSFParams,
    build_vector_psf_context,
    render_vector_psf_stack,
)
from simulation.core.phase_payload import coerce_phase_payload

from ..generic.localizations import FrameLocSet
from ..utils.param_types import RecursiveNamespace


@dataclass(frozen=True)
class MeanPSFRenderConfig:
    input_scale: float
    qe: float
    e_per_adu: float
    z_min: float
    z_max: float
    z_steps: int
    image_size: int


class MeanPSFCenterRenderer:
    """Render accepted emitters into the normalized center-frame input domain."""

    def __init__(
        self,
        *,
        psf_stack: torch.Tensor,
        cfg: MeanPSFRenderConfig,
        device: torch.device,
    ) -> None:
        self.psf_stack = psf_stack.to(device=device, dtype=torch.float32)
        self.cfg = cfg
        self.device = torch.device(device)
        self.emit_scale = float(cfg.qe) / max(float(cfg.e_per_adu), 1e-6) / max(float(cfg.input_scale), 1e-6)

        size = int(cfg.image_size)
        x_lin = torch.linspace(-1.0, 1.0, size, device=self.device)
        y_lin = torch.linspace(-1.0, 1.0, size, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing="ij")
        self.base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        self.center = float(size // 2)
        self.denom_x = float(size - 1) if size > 1 else 1.0
        self.denom_y = float(size - 1) if size > 1 else 1.0

    @classmethod
    def from_pipeline(
        cls,
        *,
        cfg: dict[str, Any],
        param: RecursiveNamespace,
        phase_payload: Any,
        device: torch.device,
        image_size: int,
    ) -> "MeanPSFCenterRenderer":
        sim_cfg = (cfg.get("simulation") or {}).get("psf", {}) or {}
        vector_cfg = sim_cfg.get("vector") or {}
        mean_cfg = vector_cfg.get("mean_psf") or {}
        if not bool(mean_cfg.get("enabled", False)):
            raise ValueError("Residual refinement renderer requires simulation.psf.vector.mean_psf.enabled=true.")

        payload = coerce_phase_payload(phase_payload, require_coeff_mag=False)
        coeff_phase = payload.get("coeff_phase")
        if coeff_phase is None or np.asarray(coeff_phase).size == 0:
            raise ValueError("Phase payload missing coeff_phase required for mean-PSF residual rendering.")
        coeff_phase = np.asarray(coeff_phase, dtype=np.float32)
        ncc = payload.get("ncc")
        ncc_thresh = float(mean_cfg.get("ncc_thresh", 0.0))
        if ncc is not None and np.asarray(ncc).size == coeff_phase.shape[0]:
            ncc = np.asarray(ncc, dtype=np.float32)
            keep = np.isfinite(ncc) & (ncc >= ncc_thresh)
            if np.any(keep):
                coeff_phase = coeff_phase[keep]

        mean_coeffs = coeff_phase.mean(axis=0, dtype=np.float32)

        optical_cfg = cfg.get("optical") or {}
        pr_cfg = (cfg.get("phase_retrieval") or {}).get("simulator_diff", {}) or {}
        zernike_cfg = pr_cfg.get("zernike") or {}
        noll_start = int(zernike_cfg.get("noll_start", 4))
        noll_end = int(zernike_cfg.get("noll_end", noll_start))
        noll_indices = list(range(noll_start, noll_end))
        if len(noll_indices) != int(mean_coeffs.shape[0]):
            raise ValueError(
                "Mean-PSF coefficient dimension mismatch: "
                f"coeffs={int(mean_coeffs.shape[0])} noll_indices={len(noll_indices)} "
                f"(noll_start={noll_start}, noll_end={noll_end})."
            )

        vector_params = VectorPSFParams(
            npupil=int(vector_cfg.get("npupil", 128)),
            psf_size=int(vector_cfg.get("psf_size", image_size)),
            refmed=float(vector_cfg.get("refmed", optical_cfg.get("n_medium", 1.518))),
            refcov=float(vector_cfg.get("refcov", optical_cfg.get("n_medium", 1.518))),
            refimm=float(vector_cfg.get("refimm", optical_cfg.get("n_medium", 1.518))),
            objstage0=float(vector_cfg.get("objstage0", 0.0)),
            otf_rescale_xy=tuple(float(v) for v in vector_cfg.get("otf_rescale_xy", [0.0, 0.0])),
            batch_size=int(vector_cfg.get("batch_size", 64)),
        )
        vector_ctx = build_vector_psf_context(
            NA=float(optical_cfg.get("NA", 1.4)),
            wavelength_nm=float(optical_cfg.get("wavelength_nm", 660.0)),
            pixel_size_nm_x=float(optical_cfg.get("pixel_size_nm_x", 100.0)),
            pixel_size_nm_y=float(optical_cfg.get("pixel_size_nm_y", 100.0)),
            noll_indices=noll_indices,
            params=vector_params,
            device=torch.device(device),
        )

        z_min = float(param.Simulation.emitter_extent[2][0])
        z_max = float(param.Simulation.emitter_extent[2][1])
        z_steps = int(mean_cfg.get("z_steps", 121))
        defocus_z = torch.linspace(z_min, z_max, z_steps, device=device, dtype=torch.float32) * 1e-6
        psf_stack = render_vector_psf_stack(
            vector_ctx,
            mean_coeffs,
            defocus_z,
            out_size=int(image_size),
            batch_size=vector_ctx.batch_size,
        )
        render_cfg = MeanPSFRenderConfig(
            input_scale=float(param.Scaling.input_scale),
            qe=float((cfg.get("camera") or {}).get("qe", 1.0)),
            e_per_adu=float((cfg.get("camera") or {}).get("e_per_adu", 1.0)),
            z_min=z_min,
            z_max=z_max,
            z_steps=z_steps,
            image_size=int(image_size),
        )
        return cls(psf_stack=psf_stack, cfg=render_cfg, device=torch.device(device))

    def _interp_psf_stack(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(device=self.device, dtype=torch.float32)
        z_steps = int(self.psf_stack.shape[0])
        if z_steps <= 1:
            return self.psf_stack[0].unsqueeze(0).expand(z.shape[0], -1, -1)

        denom = max(float(self.cfg.z_max) - float(self.cfg.z_min), 1e-12)
        z_clamped = z.clamp(float(self.cfg.z_min), float(self.cfg.z_max))
        z_norm = (z_clamped - float(self.cfg.z_min)) / denom * float(z_steps - 1)
        z0 = torch.floor(z_norm).to(dtype=torch.int64)
        z1 = torch.clamp(z0 + 1, max=z_steps - 1)
        t = (z_norm - z0.to(dtype=z_norm.dtype)).view(-1, 1, 1)
        return self.psf_stack[z0] * (1.0 - t) + self.psf_stack[z1] * t

    def render(
        self,
        locs: FrameLocSet,
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        device = torch.device(device)
        if height != width:
            raise ValueError(f"Residual renderer expects square frames, got H={height} W={width}.")
        if height != int(self.cfg.image_size):
            raise ValueError(
                f"Residual renderer was initialized for size={self.cfg.image_size}, got {height}."
            )

        render = torch.zeros((batch_size, height, width), device=device, dtype=torch.float32)
        if len(locs) == 0:
            return render

        xyz = torch.as_tensor(locs.xyz, device=self.device, dtype=torch.float32)
        phot = torch.as_tensor(locs.phot, device=self.device, dtype=torch.float32)
        indf = torch.as_tensor(locs.indf, device=self.device, dtype=torch.int64)

        psf_bank = self._interp_psf_stack(xyz[:, 2])
        dx = xyz[:, 0] - self.center
        dy = xyz[:, 1] - self.center
        dx_norm = (dx / self.denom_x) * 2.0
        dy_norm = (dy / self.denom_y) * 2.0
        shift = torch.stack((dx_norm, dy_norm), dim=-1).view(-1, 1, 1, 2)
        shifted = F.grid_sample(
            psf_bank.unsqueeze(1),
            self.base_grid - shift,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)
        shifted = torch.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)
        shifted = torch.clamp(shifted, min=0.0)
        weighted = shifted * phot[:, None, None] * self.emit_scale

        render_flat = render.view(batch_size, height * width)
        render_flat.index_add_(0, indf, weighted.view(weighted.shape[0], -1).to(device=device))
        return render
