from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class DirectGammaZernikeField:
    def __init__(
        self,
        *,
        gamma_nm: torch.Tensor | Sequence[Sequence[Sequence[float]]],
        mode_order: Sequence[tuple[int, int]],
    ) -> None:
        gamma = gamma_nm if torch.is_tensor(gamma_nm) else torch.as_tensor(gamma_nm)
        if gamma.ndim != 3 or any(int(size) <= 0 for size in gamma.shape):
            raise ValueError("gamma_nm must have shape (C, Px, Py) with non-empty dimensions.")
        if not torch.is_floating_point(gamma):
            raise TypeError("gamma_nm must use a floating-point dtype.")
        if not bool(torch.all(torch.isfinite(gamma)).item()):
            raise ValueError("gamma_nm must contain only finite values.")

        modes = tuple(_validate_mode(mode) for mode in mode_order)
        if len(modes) != int(gamma.shape[0]):
            raise ValueError("mode_order length must match the first gamma_nm dimension.")
        if len(set(modes)) != len(modes):
            raise ValueError("mode_order must not contain duplicate Zernike modes.")

        self.gamma_nm = gamma
        self.mode_order = modes

    def evaluate(
        self,
        x_normalized: torch.Tensor | Sequence[float] | float,
        y_normalized: torch.Tensor | Sequence[float] | float,
    ) -> torch.Tensor:
        x = torch.as_tensor(
            x_normalized,
            dtype=self.gamma_nm.dtype,
            device=self.gamma_nm.device,
        )
        y = torch.as_tensor(
            y_normalized,
            dtype=self.gamma_nm.dtype,
            device=self.gamma_nm.device,
        )
        x, y = torch.broadcast_tensors(x, y)
        if not bool(torch.all(torch.isfinite(x)).item()) or not bool(
            torch.all(torch.isfinite(y)).item()
        ):
            raise ValueError("Normalized coordinates must be finite.")

        x_basis = torch.stack(
            [_legendre(degree, x) for degree in range(int(self.gamma_nm.shape[1]))],
            dim=-1,
        )
        y_basis = torch.stack(
            [_legendre(degree, y) for degree in range(int(self.gamma_nm.shape[2]))],
            dim=-1,
        )
        return torch.einsum("cij,...i,...j->...c", self.gamma_nm, x_basis, y_basis)

    def coefficient_stack(self, *, image_shape_hw: tuple[int, int]) -> torch.Tensor:
        height, width = _validate_image_shape(image_shape_hw)
        y = -1.0 + 2.0 * torch.arange(
            height,
            dtype=self.gamma_nm.dtype,
            device=self.gamma_nm.device,
        ) / float(height)
        x = -1.0 + 2.0 * torch.arange(
            width,
            dtype=self.gamma_nm.dtype,
            device=self.gamma_nm.device,
        ) / float(width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return self.evaluate(xx, yy).permute(2, 0, 1).contiguous()

    def export_zmap(self, path: str | Path, *, image_shape_hw: tuple[int, int]) -> Path:
        output_path = Path(path)
        if output_path.suffix.lower() != ".npz":
            raise ValueError("Canonical zmap exports must use the .npz suffix.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stack = self.coefficient_stack(image_shape_hw=image_shape_hw)
        np.savez_compressed(
            output_path,
            zernike_maps_nm=stack.detach().cpu().numpy(),
            mode_order=np.asarray(self.mode_order, dtype=np.int64),
        )
        return output_path


def _validate_mode(mode: Sequence[int]) -> tuple[int, int]:
    if isinstance(mode, (str, bytes)) or not hasattr(mode, "__len__") or len(mode) != 2:
        raise ValueError("Each mode_order entry must be an (n, m) pair.")
    n, m = mode
    if (
        isinstance(n, (bool, np.bool_))
        or isinstance(m, (bool, np.bool_))
        or not isinstance(n, Integral)
        or not isinstance(m, Integral)
    ):
        raise ValueError("Zernike n and m values must be integers.")
    if n < 0 or abs(m) > n or (n - abs(m)) % 2 != 0:
        raise ValueError(f"Invalid Zernike mode: {(n, m)}")
    return int(n), int(m)


def _validate_image_shape(image_shape_hw: tuple[int, int]) -> tuple[int, int]:
    if (
        isinstance(image_shape_hw, (str, bytes))
        or not hasattr(image_shape_hw, "__len__")
        or len(image_shape_hw) != 2
    ):
        raise ValueError("image_shape_hw must contain height and width.")
    height, width = image_shape_hw
    if (
        isinstance(height, (bool, np.bool_))
        or isinstance(width, (bool, np.bool_))
        or not isinstance(height, Integral)
        or not isinstance(width, Integral)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("image_shape_hw dimensions must be positive integers.")
    return int(height), int(width)


def _legendre(degree: int, values: torch.Tensor) -> torch.Tensor:
    if degree == 0:
        return torch.ones_like(values)
    if degree == 1:
        return values
    previous_previous = torch.ones_like(values)
    previous = values
    for current_degree in range(2, degree + 1):
        current = (
            (2.0 * current_degree - 1.0) * values * previous
            - (current_degree - 1.0) * previous_previous
        ) / float(current_degree)
        previous_previous, previous = previous, current
    return previous


__all__ = ["DirectGammaZernikeField"]
