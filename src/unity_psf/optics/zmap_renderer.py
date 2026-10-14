"""Differentiable empirical zmap renderer for linked raw-pixel fitting."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .vector_psf import VectorPSFContext, render_vector_psf_bank


def add_fixed_astigmatism_anchor(
    coeffs_nm: torch.Tensor,
    *,
    mode_order: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    anchor_nm: float | None,
) -> torch.Tensor:
    """Add a fixed positive astigmatism coefficient without changing the map."""

    if anchor_nm is None or float(anchor_nm) == 0.0:
        return coeffs_nm
    try:
        mode_index = tuple((int(n), int(m)) for n, m in mode_order).index((2, 2))
    except ValueError as exc:
        raise ValueError("fixed astigmatism anchor requires Zernike mode (2, 2)") from exc
    result = coeffs_nm.clone()
    result[..., mode_index] = result[..., mode_index] + float(anchor_nm)
    return result


class ZmapLUTRenderer:
    """Render a normalized ``(z, y, x)`` PSF stack into a full ROI."""

    def __init__(
        self,
        planes_zyx: torch.Tensor,
        *,
        z_min_nm: float,
        z_max_nm: float,
        pixel_size_nm_xy: tuple[float, float],
        roi_shape_yx: tuple[int, int],
        roi_origin_nm_xy: tuple[float, float] = (0.0, 0.0),
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        planes = torch.as_tensor(planes_zyx, dtype=torch.float32)
        if planes.ndim != 3 or planes.shape[1] % 2 != 1 or planes.shape[2] % 2 != 1:
            raise ValueError("planes_zyx must have shape (Z, odd, odd)")
        if float(z_max_nm) <= float(z_min_nm):
            raise ValueError("z_max_nm must exceed z_min_nm")
        if any(float(value) <= 0 for value in pixel_size_nm_xy):
            raise ValueError("pixel sizes must be positive")
        if any(int(value) <= 0 for value in roi_shape_yx):
            raise ValueError("roi shape must be positive")
        self.planes = (planes / planes.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)).contiguous()
        self.z_min_nm = float(z_min_nm)
        self.z_max_nm = float(z_max_nm)
        self.pixel_size_nm_xy = tuple(float(value) for value in pixel_size_nm_xy)
        self.roi_shape_yx = tuple(int(value) for value in roi_shape_yx)
        self.roi_origin_nm_xy = tuple(float(value) for value in roi_origin_nm_xy)
        self.transform = transform

    def __call__(self, xyz_nm: torch.Tensor, photons: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        xyz = torch.as_tensor(xyz_nm, dtype=torch.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz_nm must have shape (N, 3)")
        if self.transform is not None:
            channel_xyz = self.transform(xyz)
        else:
            channel_xyz = xyz
        device = xyz.device
        planes = self.planes.to(device=device)
        photons_t = torch.as_tensor(photons, dtype=xyz.dtype, device=device).reshape(-1)
        background_t = torch.as_tensor(background, dtype=xyz.dtype, device=device).reshape(-1)
        count = int(xyz.shape[0])
        if photons_t.shape != (count,):
            raise ValueError("photons must have one value per emitter")
        if background_t.numel() not in (1, count):
            raise ValueError("background must be scalar or have one value per emitter")
        z_index = ((channel_xyz[:, 2] - self.z_min_nm) / (self.z_max_nm - self.z_min_nm) * (planes.shape[0] - 1)).clamp(0.0, float(planes.shape[0] - 1))
        low = torch.floor(z_index).long()
        high = (low + 1).clamp_max(planes.shape[0] - 1)
        fraction = (z_index - low.to(z_index.dtype)).reshape(-1, 1, 1)
        patch = planes[low] * (1.0 - fraction) + planes[high] * fraction
        px, py = self.pixel_size_nm_xy
        x = (channel_xyz[:, 0] - self.roi_origin_nm_xy[0]) / px
        y = (channel_xyz[:, 1] - self.roi_origin_nm_xy[1]) / py
        center_x = torch.round(x)
        center_y = torch.round(y)
        patch = self._shift_patches(patch, x - center_x, y - center_y) * photons_t[:, None, None]
        height, width = self.roi_shape_yx
        radius_y, radius_x = patch.shape[-2] // 2, patch.shape[-1] // 2
        yy, xx = torch.meshgrid(torch.arange(-radius_y, radius_y + 1, device=device), torch.arange(-radius_x, radius_x + 1, device=device), indexing="ij")
        ix = center_x.long()[:, None, None] + xx[None]
        iy = center_y.long()[:, None, None] + yy[None]
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        indices = (iy.clamp(0, height - 1) * width + ix.clamp(0, width - 1)).reshape(-1)
        values = (patch * valid.to(patch.dtype)).reshape(-1)
        image = torch.zeros(height * width, dtype=patch.dtype, device=device).scatter_add(0, indices, values).reshape(height, width)
        return image + background_t.mean()

    @staticmethod
    def _shift_patches(patches: torch.Tensor, shift_x: torch.Tensor, shift_y: torch.Tensor) -> torch.Tensor:
        _, height, width = patches.shape
        yy, xx = torch.meshgrid(torch.arange(height, device=patches.device, dtype=patches.dtype), torch.arange(width, device=patches.device, dtype=patches.dtype), indexing="ij")
        source_x = xx[None] - shift_x[:, None, None]
        source_y = yy[None] - shift_y[:, None, None]
        grid = torch.stack((2.0 * source_x / max(width - 1, 1) - 1.0, 2.0 * source_y / max(height - 1, 1) - 1.0), dim=-1)
        return F.grid_sample(patches[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]


class ZmapCoefficientRenderer:
    """Render PSFs from field-dependent Zernike coefficient maps."""

    def __init__(
        self,
        coeff_maps_nm: torch.Tensor,
        *,
        mode_order: tuple[tuple[int, int], ...],
        vector_context: VectorPSFContext,
        pixel_size_nm_xy: tuple[float, float],
        roi_shape_yx: tuple[int, int],
        field_origin_px_xy: tuple[float, float] = (0.0, 0.0),
        roi_origin_px_xy: tuple[float, float] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        psf_size: int | None = None,
    ) -> None:
        maps = torch.as_tensor(coeff_maps_nm, dtype=torch.float32)
        if maps.ndim != 3 or maps.shape[0] != len(mode_order):
            raise ValueError("coeff_maps_nm must have shape (modes, height, width) matching mode_order")
        if maps.shape[1] < 2 or maps.shape[2] < 2:
            raise ValueError("coefficient maps must cover at least a 2x2 field")
        if len(vector_context.noll_indices) != len(mode_order):
            raise ValueError("vector context modes must match coefficient map modes")
        self.coeff_maps_nm = maps.contiguous()
        self.mode_order = tuple((int(n), int(m)) for n, m in mode_order)
        self.vector_context = vector_context
        self.pixel_size_nm_xy = tuple(float(value) for value in pixel_size_nm_xy)
        self.roi_shape_yx = tuple(int(value) for value in roi_shape_yx)
        self.field_origin_px_xy = tuple(float(value) for value in field_origin_px_xy)
        self.roi_origin_px_xy = self.field_origin_px_xy if roi_origin_px_xy is None else tuple(float(value) for value in roi_origin_px_xy)
        self.transform = transform
        self.psf_size = int(psf_size or vector_context.psf.psf_size)

    @classmethod
    def from_npz(cls, path: str | Path, **kwargs) -> "ZmapCoefficientRenderer":
        payload = np.load(Path(path), allow_pickle=False)
        required = {"zernike_maps_nm", "mode_order"}
        if not required.issubset(payload.files):
            raise ValueError(f"zmap NPZ must contain {sorted(required)}")
        return cls(payload["zernike_maps_nm"], mode_order=tuple(map(tuple, payload["mode_order"].tolist())), **kwargs)

    def __call__(self, xyz_nm: torch.Tensor, photons: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        xyz = torch.as_tensor(xyz_nm, dtype=torch.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz_nm must have shape (N, 3)")
        channel_xyz = self.transform(xyz) if self.transform is not None else xyz
        device = xyz.device
        maps = self.coeff_maps_nm.to(device=device)
        px, py = self.pixel_size_nm_xy
        field_x = (channel_xyz[:, 0] / px - self.field_origin_px_xy[0])
        field_y = (channel_xyz[:, 1] / py - self.field_origin_px_xy[1])
        height, width = maps.shape[-2:]
        grid = torch.stack((2.0 * field_x / max(width - 1, 1) - 1.0, 2.0 * field_y / max(height - 1, 1) - 1.0), dim=1).reshape(-1, 1, 1, 2)
        coeffs_nm = F.grid_sample(maps[None].expand(xyz.shape[0], -1, -1, -1), grid, mode="bilinear", padding_mode="border", align_corners=True)[:, :, 0, 0]
        coeffs_rad = coeffs_nm * (2.0 * torch.pi / max(float(self.vector_context.wavelength_nm), 1e-6)) * self.vector_context.normfac.to(device=device)[None]
        patches = render_vector_psf_bank(
            self.vector_context,
            coeffs_rad,
            channel_xyz[:, 2] * 1e-9,
            out_size=self.psf_size,
            batch_size=self.vector_context.batch_size,
            return_torch=True,
        )
        x = channel_xyz[:, 0] / px - self.roi_origin_px_xy[0]
        y = channel_xyz[:, 1] / py - self.roi_origin_px_xy[1]
        center_x, center_y = torch.round(x), torch.round(y)
        patches = ZmapLUTRenderer._shift_patches(patches, x - center_x, y - center_y)
        patches = patches * torch.as_tensor(photons, dtype=xyz.dtype, device=device).reshape(-1, 1, 1)
        radius = self.psf_size // 2
        yy, xx = torch.meshgrid(torch.arange(-radius, radius + 1, device=device), torch.arange(-radius, radius + 1, device=device), indexing="ij")
        ix, iy = center_x.long()[:, None, None] + xx[None], center_y.long()[:, None, None] + yy[None]
        valid = (ix >= 0) & (ix < self.roi_shape_yx[1]) & (iy >= 0) & (iy < self.roi_shape_yx[0])
        indices = (iy.clamp(0, self.roi_shape_yx[0] - 1) * self.roi_shape_yx[1] + ix.clamp(0, self.roi_shape_yx[1] - 1)).reshape(-1)
        values = (patches * valid.to(patches.dtype)).reshape(-1)
        image = torch.zeros(self.roi_shape_yx[0] * self.roi_shape_yx[1], dtype=patches.dtype, device=device).scatter_add(0, indices, values).reshape(self.roi_shape_yx)
        return image + torch.as_tensor(background, dtype=xyz.dtype, device=device).mean()


__all__ = ["ZmapCoefficientRenderer", "ZmapLUTRenderer", "add_fixed_astigmatism_anchor"]
