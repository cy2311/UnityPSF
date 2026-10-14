"""Spatial conditioning provider for direct PPG PCA maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class PPGFieldConditioning:
    """Expose a PPG coefficient field through the UnityPSF condition contract."""

    def __init__(self, *, coefficient_field: torch.Tensor, field_shape_yx: tuple[int, int]) -> None:
        if coefficient_field.ndim != 3 or int(coefficient_field.shape[0]) != 4:
            raise ValueError("PPG coefficient_field must have shape (4, grid_y, grid_x)")
        height, width = (int(value) for value in field_shape_yx)
        if min(height, width) <= 1:
            raise ValueError("PPG field_shape_yx must contain dimensions greater than one")
        self.coefficient_field = coefficient_field.to(dtype=torch.float32).contiguous()
        self._image_shape_hw = (height, width)
        self.coefficient_mean = self.coefficient_field.mean(dim=(1, 2), keepdim=True)
        self.coefficient_std = self.coefficient_field.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(1e-6)

    @classmethod
    def from_npz(cls, path: str | Path) -> "PPGFieldConditioning":
        with np.load(Path(path), allow_pickle=False) as payload:
            schema = str(payload["schema"].item())
            if schema != "ppg.pca_map.v1":
                raise ValueError(f"unsupported PPG map schema: {schema!r}")
            field = torch.as_tensor(payload["coefficient_field"], dtype=torch.float32)
            shape = tuple(int(value) for value in payload["field_shape_yx"])
        return cls(coefficient_field=field, field_shape_yx=shape)

    @property
    def image_shape_hw(self) -> tuple[int, int]:
        return self._image_shape_hw

    @property
    def condition_vector_dim(self) -> int:
        return 6

    def normalized_xy_from_xy(
        self,
        *,
        x0: int,
        y0: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        full_h, full_w = self.image_shape_hw
        cx = float(x0) + (float(width) - 1.0) / 2.0
        cy = float(y0) + (float(height) - 1.0) / 2.0
        values = torch.tensor(
            [2.0 * cx / float(full_w - 1) - 1.0, 2.0 * cy / float(full_h - 1) - 1.0],
            dtype=dtype,
            device=device,
        )
        return values

    def coefficient_vector_from_xy(
        self,
        *,
        x0: int,
        y0: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        xy = self.normalized_xy_from_xy(
            x0=x0,
            y0=y0,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        grid = xy.reshape(1, 1, 1, 2).to(dtype=torch.float32)
        field = self.coefficient_field.unsqueeze(0).to(device=grid.device)
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
        values = sampled[0, :, 0, 0]
        mean = self.coefficient_mean[:, 0, 0].to(device=values.device)
        std = self.coefficient_std[:, 0, 0].to(device=values.device)
        return ((values - mean) / std).to(dtype=dtype)

    def coefficient_vector_from_normalized_xy(
        self,
        normalized_xy: torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        xy = torch.as_tensor(normalized_xy, device=device, dtype=dtype).reshape(2)
        grid = xy.reshape(1, 1, 1, 2).to(dtype=torch.float32)
        field = self.coefficient_field.unsqueeze(0).to(device=grid.device)
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
        values = sampled[0, :, 0, 0]
        mean = self.coefficient_mean[:, 0, 0].to(device=values.device)
        std = self.coefficient_std[:, 0, 0].to(device=values.device)
        return ((values - mean) / std).to(dtype=dtype)

    def condition_vector_from_normalized_xy(
        self,
        normalized_xy: torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        xy = torch.as_tensor(normalized_xy, device=device, dtype=dtype).reshape(2)
        coeff = self.coefficient_vector_from_normalized_xy(xy, device=device, dtype=dtype)
        return torch.cat((xy, coeff), dim=0).contiguous()

    def condition_vector_from_xy(
        self,
        *,
        x0: int,
        y0: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        xy = self.normalized_xy_from_xy(
            x0=x0,
            y0=y0,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        coeff = self.coefficient_vector_from_normalized_xy(xy, device=device, dtype=dtype)
        return torch.cat((xy, coeff), dim=0).contiguous()


__all__ = ["PPGFieldConditioning"]
