from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class NATComponent:
    px: int
    py: int
    n: int
    m: int
    weight: float


@dataclass(frozen=True)
class NATGamma:
    components: tuple[NATComponent, ...]


@dataclass(frozen=True)
class NATFieldConfig:
    aberrations: tuple[tuple[int, int], ...]
    gammas: tuple[NATGamma, ...]
    img_size_x: int
    img_size_y: int
    pixel_size_x_nm: float
    pixel_size_y_nm: float


@dataclass(frozen=True)
class NATCoeffStack:
    maps_nm: torch.Tensor
    mode_order: list[tuple[int, int]]
    image_shape_hw: tuple[int, int]


def default_order1_config(
    *,
    img_size_x: int = 1024,
    img_size_y: int = 1024,
    pixel_size_x_nm: float = 95.0,
    pixel_size_y_nm: float = 95.0,
) -> NATFieldConfig:
    aberrations = (
        (2, 0),
        (2, 2),
        (2, -2),
        (3, 1),
        (3, -1),
        (4, 0),
        (3, -3),
        (3, 3),
    )
    gammas = (
        NATGamma((NATComponent(0, 0, 2, 0, 1.0),)),
        NATGamma((NATComponent(1, 0, 2, 0, 1.0),)),
        NATGamma((NATComponent(0, 1, 2, 0, 1.0),)),
        NATGamma((NATComponent(2, 0, 2, 0, 1.0), NATComponent(0, 2, 2, 0, 1.0))),
        NATGamma((NATComponent(0, 0, 2, 2, 1.0),)),
        NATGamma((NATComponent(0, 0, 2, -2, 1.0),)),
        NATGamma((NATComponent(1, 0, 2, 2, 1.0), NATComponent(0, 1, 2, -2, 1.0))),
        NATGamma((NATComponent(0, 1, 2, 2, -1.0), NATComponent(1, 0, 2, -2, 1.0))),
    )
    return NATFieldConfig(
        aberrations=aberrations,
        gammas=gammas,
        img_size_x=int(img_size_x),
        img_size_y=int(img_size_y),
        pixel_size_x_nm=float(pixel_size_x_nm),
        pixel_size_y_nm=float(pixel_size_y_nm),
    )


def order1_13_config(
    *,
    img_size_x: int = 1024,
    img_size_y: int = 1024,
    pixel_size_x_nm: float = 95.0,
    pixel_size_y_nm: float = 95.0,
) -> NATFieldConfig:
    base = default_order1_config(
        img_size_x=img_size_x,
        img_size_y=img_size_y,
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    gammas = base.gammas + (
        NATGamma((NATComponent(0, 0, 3, 1, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, -1, 1.0),)),
        NATGamma((NATComponent(1, 0, 3, 1, 1.0), NATComponent(0, 1, 3, -1, 1.0))),
        NATGamma((NATComponent(0, 0, 4, 0, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, 3, 1.0), NATComponent(0, 0, 3, -3, 1.0))),
    )
    return NATFieldConfig(
        aberrations=base.aberrations,
        gammas=gammas,
        img_size_x=int(img_size_x),
        img_size_y=int(img_size_y),
        pixel_size_x_nm=float(pixel_size_x_nm),
        pixel_size_y_nm=float(pixel_size_y_nm),
    )


def order1_21_config(
    *,
    img_size_x: int = 1024,
    img_size_y: int = 1024,
    pixel_size_x_nm: float = 95.0,
    pixel_size_y_nm: float = 95.0,
) -> NATFieldConfig:
    base = default_order1_config(
        img_size_x=img_size_x,
        img_size_y=img_size_y,
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    gammas = base.gammas + (
        NATGamma((NATComponent(0, 0, 3, 1, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, -1, 1.0),)),
        NATGamma((NATComponent(1, 0, 3, 1, 1.0),)),
        NATGamma((NATComponent(0, 1, 3, 1, 1.0),)),
        NATGamma((NATComponent(1, 0, 3, -1, 1.0),)),
        NATGamma((NATComponent(0, 1, 3, -1, 1.0),)),
        NATGamma((NATComponent(0, 0, 4, 0, 1.0),)),
        NATGamma((NATComponent(1, 0, 4, 0, 1.0),)),
        NATGamma((NATComponent(0, 1, 4, 0, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, 3, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, -3, 1.0),)),
        NATGamma((NATComponent(1, 0, 3, 3, 1.0), NATComponent(0, 1, 3, -3, 1.0))),
        NATGamma((NATComponent(0, 1, 3, 3, -1.0), NATComponent(1, 0, 3, -3, 1.0))),
    )
    return NATFieldConfig(
        aberrations=base.aberrations,
        gammas=gammas,
        img_size_x=int(img_size_x),
        img_size_y=int(img_size_y),
        pixel_size_x_nm=float(pixel_size_x_nm),
        pixel_size_y_nm=float(pixel_size_y_nm),
    )


def build_named_nat_config(
    name: str,
    *,
    img_size_x: int = 1024,
    img_size_y: int = 1024,
    pixel_size_x_nm: float = 95.0,
    pixel_size_y_nm: float = 95.0,
) -> NATFieldConfig:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized in {"order1", "first_order", "j1"}:
        return default_order1_config(
            img_size_x=img_size_x,
            img_size_y=img_size_y,
            pixel_size_x_nm=pixel_size_x_nm,
            pixel_size_y_nm=pixel_size_y_nm,
        )
    if normalized in {"order1_13", "j1_13", "zernike13", "zernike_13"}:
        return order1_13_config(
            img_size_x=img_size_x,
            img_size_y=img_size_y,
            pixel_size_x_nm=pixel_size_x_nm,
            pixel_size_y_nm=pixel_size_y_nm,
        )
    if normalized in {"order1_21", "j1_21", "zernike21", "zernike_21"}:
        return order1_21_config(
            img_size_x=img_size_x,
            img_size_y=img_size_y,
            pixel_size_x_nm=pixel_size_x_nm,
            pixel_size_y_nm=pixel_size_y_nm,
        )
    raise ValueError(f"Unsupported NAT config name: {name!r}")


def get_fov_coordinates_torch(
    roixy_px: torch.Tensor | Sequence[Sequence[float]],
    *,
    img_size_x: int,
    img_size_y: int,
    pixel_size_x_nm: float,
    pixel_size_y_nm: float,
    local_x_nm: torch.Tensor | Sequence[float] | float | None = None,
    local_y_nm: torch.Tensor | Sequence[float] | float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    roixy = roixy_px if torch.is_tensor(roixy_px) else torch.as_tensor(roixy_px, dtype=dtype)
    roixy = roixy.to(dtype=dtype)
    if device is not None:
        roixy = roixy.to(device=device)
    if roixy.ndim != 2 or roixy.shape[1] != 2:
        raise ValueError("roixy_px must have shape (N, 2).")
    count = int(roixy.shape[0])
    local_x = _broadcast_optional_local(local_x_nm, count=count, dtype=dtype, device=roixy.device)
    local_y = _broadcast_optional_local(local_y_nm, count=count, dtype=dtype, device=roixy.device)
    x_nm = roixy[:, 0] * float(pixel_size_x_nm) + local_x
    y_nm = roixy[:, 1] * float(pixel_size_y_nm) + local_y
    physical_um = torch.stack([x_nm * 1e-3, y_nm * 1e-3], dim=1)
    normalized = torch.stack(
        [
            -1.0 + 2.0 * x_nm / (float(pixel_size_x_nm) * float(img_size_x)),
            -1.0 + 2.0 * y_nm / (float(pixel_size_y_nm) * float(img_size_y)),
        ],
        dim=1,
    )
    return normalized, physical_um


def evaluate_zernike_coefficients_torch(
    xn: torch.Tensor | Sequence[float],
    yn: torch.Tensor | Sequence[float],
    gamma_values: torch.Tensor | Sequence[float],
    config: NATFieldConfig,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    x = _as_1d_tensor(xn, dtype=dtype, device=device)
    y = _as_1d_tensor(yn, dtype=dtype, device=x.device)
    gamma = _as_1d_tensor(gamma_values, dtype=dtype, device=x.device)
    if x.shape != y.shape:
        raise ValueError("xn and yn must have matching shapes.")
    if gamma.shape[0] != len(config.gammas):
        raise ValueError("gamma_values length must match config.gammas.")
    coeffs = torch.zeros((x.shape[0], len(config.aberrations)), dtype=dtype, device=x.device)
    mode_index = {mode: idx for idx, mode in enumerate(config.aberrations)}
    for gamma_index, nat_gamma in enumerate(config.gammas):
        for component in nat_gamma.components:
            basis = _legendre_basis(component.px, x) * _legendre_basis(component.py, y)
            norm = ((1.0 + 2.0 * float(component.px)) * (1.0 + 2.0 * float(component.py))) ** 0.5
            coeffs[:, mode_index[(component.n, component.m)]] += gamma[gamma_index] * float(component.weight) * float(norm) * basis
    return coeffs


def evaluate_zernike_from_roi_positions_torch(
    roixy_px: torch.Tensor | Sequence[Sequence[float]],
    gamma_values: torch.Tensor | Sequence[float],
    config: NATFieldConfig,
    *,
    local_x_nm: torch.Tensor | Sequence[float] | float | None = None,
    local_y_nm: torch.Tensor | Sequence[float] | float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized, physical_um = get_fov_coordinates_torch(
        roixy_px,
        img_size_x=config.img_size_x,
        img_size_y=config.img_size_y,
        pixel_size_x_nm=config.pixel_size_x_nm,
        pixel_size_y_nm=config.pixel_size_y_nm,
        local_x_nm=local_x_nm,
        local_y_nm=local_y_nm,
        dtype=dtype,
        device=device,
    )
    coeffs = evaluate_zernike_coefficients_torch(
        normalized[:, 0],
        normalized[:, 1],
        gamma_values,
        config,
        dtype=dtype,
        device=normalized.device,
    )
    return coeffs, normalized, physical_um


def full_roi_coeff_stack_torch(
    gamma_values: torch.Tensor | Sequence[float],
    config: NATFieldConfig,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> NATCoeffStack:
    ys = torch.arange(int(config.img_size_y), dtype=dtype, device=device)
    xs = torch.arange(int(config.img_size_x), dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    roixy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    coeffs, _, _ = evaluate_zernike_from_roi_positions_torch(roixy, gamma_values, config, dtype=dtype, device=device)
    maps = coeffs.reshape(int(config.img_size_y), int(config.img_size_x), len(config.aberrations)).permute(2, 0, 1).contiguous()
    return NATCoeffStack(maps_nm=maps, mode_order=list(config.aberrations), image_shape_hw=(int(config.img_size_y), int(config.img_size_x)))


def _as_1d_tensor(
    value: torch.Tensor | Sequence[float],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=dtype)
    tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor.reshape(-1)


def _broadcast_optional_local(
    value: torch.Tensor | Sequence[float] | float | None,
    *,
    count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((count,), dtype=dtype, device=device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
    if tensor.shape[0] == 1 and count != 1:
        return tensor.expand(count)
    if tensor.shape[0] != count:
        raise ValueError("local_x_nm/local_y_nm must match roixy_px length.")
    return tensor


def _legendre_basis(order: int, values: torch.Tensor) -> torch.Tensor:
    if int(order) < 0:
        raise ValueError("Legendre order must be non-negative.")
    if int(order) == 0:
        return torch.ones_like(values)
    if int(order) == 1:
        return values
    p_nm2 = torch.ones_like(values)
    p_nm1 = values
    for degree in range(2, int(order) + 1):
        p_n = ((2.0 * degree - 1.0) * values * p_nm1 - (degree - 1.0) * p_nm2) / float(degree)
        p_nm2 = p_nm1
        p_nm1 = p_n
    return p_nm1


__all__ = [
    "NATCoeffStack",
    "NATComponent",
    "NATFieldConfig",
    "NATGamma",
    "build_named_nat_config",
    "default_order1_config",
    "order1_13_config",
    "order1_21_config",
    "evaluate_zernike_coefficients_torch",
    "evaluate_zernike_from_roi_positions_torch",
    "full_roi_coeff_stack_torch",
    "get_fov_coordinates_torch",
]
