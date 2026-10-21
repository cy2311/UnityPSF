from __future__ import annotations

import torch

from .assets import FDChannelAsset


def build_coord_channels(
    asset: FDChannelAsset,
    *,
    field_xy: torch.Tensor | tuple[float, float] | list[float],
    out_size: tuple[int, int] = (40, 40),
) -> torch.Tensor:
    coords = torch.as_tensor(field_xy, dtype=torch.float32)
    if coords.numel() != 2:
        raise ValueError("field_xy must contain exactly two values.")

    out_h = int(out_size[0])
    out_w = int(out_size[1])

    _, grid_w = asset.field_shape
    grid_h, _ = asset.field_shape
    cell_x = int(torch.clamp(torch.round(coords[0]), 0, grid_w - 1).item())
    cell_y = int(torch.clamp(torch.round(coords[1]), 0, grid_h - 1).item())

    x0 = asset.crop_left + cell_x * asset.cell_width_px
    y0 = asset.crop_top + cell_y * asset.cell_height_px

    xs = torch.linspace(float(x0 - asset.crop_left), float(x0 - asset.crop_left + out_w - 1), out_w, dtype=torch.float32)
    ys = torch.linspace(float(y0 - asset.crop_top), float(y0 - asset.crop_top + out_h - 1), out_h, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    x_norm = grid_x / max(1.0, float(asset.crop_width - 1))
    y_norm = grid_y / max(1.0, float(asset.crop_height - 1))
    return torch.stack((x_norm, y_norm), dim=0).contiguous()


def append_coord_channels(frames: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 3 or coords.ndim != 3:
        raise ValueError("frames and coords must both be 3D tensors shaped as (C,H,W).")
    if frames.shape[-2:] != coords.shape[-2:]:
        raise ValueError("frames and coords spatial shapes must match.")
    return torch.cat((frames, coords), dim=0)
