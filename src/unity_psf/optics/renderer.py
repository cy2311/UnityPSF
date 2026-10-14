from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaussianPSFRendererConfig:
    patch_size_px: int = 15
    sigma_base_px: float = 1.2
    coefficient_sigma_scale: float = 0.1


class GaussianPSFRenderer:
    def __init__(self, config: GaussianPSFRendererConfig | None = None) -> None:
        self.config = GaussianPSFRendererConfig() if config is None else config
        if int(self.config.patch_size_px) <= 0 or int(self.config.patch_size_px) % 2 == 0:
            raise ValueError("patch_size_px must be a positive odd integer.")

    def render(
        self,
        *,
        local_x_px: torch.Tensor,
        local_y_px: torch.Tensor,
        photons: torch.Tensor,
        background: torch.Tensor,
        coefficients_nm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_x = torch.as_tensor(local_x_px, dtype=torch.float32).reshape(-1)
        device = local_x.device
        local_y = torch.as_tensor(local_y_px, dtype=torch.float32, device=device).reshape(-1)
        photons_t = torch.as_tensor(photons, dtype=torch.float32, device=device).reshape(-1)
        background_t = torch.as_tensor(background, dtype=torch.float32, device=device).reshape(-1)
        if not (local_x.shape == local_y.shape == photons_t.shape == background_t.shape):
            raise ValueError("local_x_px, local_y_px, photons, and background must have matching lengths.")
        sigma = self._sigma(local_x.shape[0], coefficients_nm=coefficients_nm, device=device)
        size = int(self.config.patch_size_px)
        center = (float(size) - 1.0) / 2.0
        yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32, device=device), torch.arange(size, dtype=torch.float32, device=device), indexing="ij")
        x0 = center + local_x.reshape(-1, 1, 1)
        y0 = center + local_y.reshape(-1, 1, 1)
        sigma_t = sigma.reshape(-1, 1, 1).clamp_min(0.2)
        kernel = torch.exp(-0.5 * (((xx[None] - x0) / sigma_t) ** 2 + ((yy[None] - y0) / sigma_t) ** 2))
        kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return kernel * photons_t.reshape(-1, 1, 1) + background_t.reshape(-1, 1, 1)

    def _sigma(self, count: int, *, coefficients_nm: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        base = torch.full((count,), float(self.config.sigma_base_px), dtype=torch.float32, device=device)
        if coefficients_nm is None:
            return base
        coeffs = torch.as_tensor(coefficients_nm, dtype=torch.float32, device=device)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(-1, 1)
        if coeffs.shape[0] != count:
            raise ValueError("coefficients_nm must have one row per emitter.")
        return (base + float(self.config.coefficient_sigma_scale) * coeffs[:, 0]).clamp_min(0.2)


__all__ = [
    "GaussianPSFRenderer",
    "GaussianPSFRendererConfig",
]
