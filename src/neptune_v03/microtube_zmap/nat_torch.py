from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .nat_field import NATFieldConfig, default_order1_config


def _as_tensor_1d(
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


def _legendre_basis_torch(order: int, values: torch.Tensor) -> torch.Tensor:
    if order < 0:
        raise ValueError("Legendre order must be non-negative.")
    if order == 0:
        return torch.ones_like(values)
    if order == 1:
        return values
    p_nm2 = torch.ones_like(values)
    p_nm1 = values
    for degree in range(2, int(order) + 1):
        p_n = ((2.0 * degree - 1.0) * values * p_nm1 - (degree - 1.0) * p_nm2) / float(degree)
        p_nm2 = p_nm1
        p_nm1 = p_n
    return p_nm1


def evaluate_legendre_2d_torch(
    orders: Sequence[tuple[int, int]],
    xn: torch.Tensor | Sequence[float],
    yn: torch.Tensor | Sequence[float],
    *,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    x = _as_tensor_1d(xn, dtype=dtype, device=device)
    y = _as_tensor_1d(yn, dtype=dtype, device=device)
    if x.shape != y.shape:
        raise ValueError("xn and yn must have matching shapes.")

    outputs: list[torch.Tensor] = []
    for px, py in orders:
        basis_x = _legendre_basis_torch(int(px), x)
        basis_y = _legendre_basis_torch(int(py), y)
        values = basis_x * basis_y
        if normalize:
            values = values * torch.sqrt(
                torch.as_tensor((1.0 + 2.0 * float(px)) * (1.0 + 2.0 * float(py)), dtype=dtype, device=x.device)
            )
        outputs.append(values)
    if not outputs:
        return torch.empty((0, x.shape[0]), dtype=dtype, device=x.device)
    return torch.stack(outputs, dim=0)


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

    count = roixy.shape[0]
    local_x = torch.zeros(count, dtype=dtype, device=roixy.device) if local_x_nm is None else _as_tensor_1d(local_x_nm, dtype=dtype, device=roixy.device)
    local_y = torch.zeros(count, dtype=dtype, device=roixy.device) if local_y_nm is None else _as_tensor_1d(local_y_nm, dtype=dtype, device=roixy.device)
    if local_x.shape[0] == 1 and count != 1:
        local_x = local_x.expand(count)
    if local_y.shape[0] == 1 and count != 1:
        local_y = local_y.expand(count)
    if local_x.shape[0] != count or local_y.shape[0] != count:
        raise ValueError("local_x_nm/local_y_nm must match roixy_px length.")

    x_nm = roixy[:, 0] * float(pixel_size_x_nm) + local_x
    y_nm = roixy[:, 1] * float(pixel_size_y_nm) + local_y
    physical_um = torch.stack([x_nm * 1e-3, y_nm * 1e-3], dim=1)

    normalized_x = -1.0 + 2.0 * x_nm / (float(pixel_size_x_nm) * float(img_size_x))
    normalized_y = -1.0 + 2.0 * y_nm / (float(pixel_size_y_nm) * float(img_size_y))
    normalized = torch.stack([normalized_x, normalized_y], dim=1)
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
    x = _as_tensor_1d(xn, dtype=dtype, device=device)
    y = _as_tensor_1d(yn, dtype=dtype, device=x.device)
    gammas = _as_tensor_1d(gamma_values, dtype=dtype, device=x.device)
    if gammas.shape[0] != len(config.gammas):
        raise ValueError("gamma_values length must match config.gammas.")

    aberration_index = {mode: idx for idx, mode in enumerate(config.aberrations)}
    orders: list[tuple[int, int]] = []
    gamma_indices: list[int] = []
    mode_indices: list[int] = []
    component_weights: list[float] = []
    for gamma_index, gamma in enumerate(config.gammas):
        for component in gamma.components:
            orders.append((component.px, component.py))
            gamma_indices.append(gamma_index)
            mode_indices.append(aberration_index[(component.n, component.m)])
            component_weights.append(float(component.weight))
    legendres = evaluate_legendre_2d_torch(orders, x, y, normalize=True, dtype=dtype, device=x.device)
    if not orders:
        return torch.zeros((x.shape[0], len(config.aberrations)), dtype=dtype, device=x.device)

    gamma_index_tensor = torch.as_tensor(gamma_indices, dtype=torch.long, device=x.device)
    mode_index_tensor = torch.as_tensor(mode_indices, dtype=torch.long, device=x.device)
    weight_tensor = torch.as_tensor(component_weights, dtype=dtype, device=x.device)
    component_values = legendres * (gammas[gamma_index_tensor] * weight_tensor)[:, None]
    mode_map = torch.nn.functional.one_hot(mode_index_tensor, num_classes=len(config.aberrations)).to(dtype=dtype, device=x.device)
    return component_values.transpose(0, 1) @ mode_map


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
    coefficients = evaluate_zernike_coefficients_torch(
        normalized[:, 0],
        normalized[:, 1],
        gamma_values,
        config,
        dtype=dtype,
        device=normalized.device,
    )
    return coefficients, normalized, physical_um


def absolute_pixel_positions_torch(
    *,
    roi_left_px: float,
    roi_top_px: float,
    emitter_x_px: torch.Tensor | Iterable[float],
    emitter_y_px: torch.Tensor | Iterable[float],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    x = _as_tensor_1d(tuple(emitter_x_px) if not torch.is_tensor(emitter_x_px) else emitter_x_px, dtype=dtype, device=device)
    y = _as_tensor_1d(tuple(emitter_y_px) if not torch.is_tensor(emitter_y_px) else emitter_y_px, dtype=dtype, device=x.device)
    if x.shape != y.shape:
        raise ValueError("emitter_x_px and emitter_y_px must have the same shape.")
    return torch.stack([x + float(roi_left_px), y + float(roi_top_px)], dim=1)


def _mode_index(config: NATFieldConfig, n: int, m: int) -> int:
    try:
        return config.aberrations.index((n, m))
    except ValueError as exc:
        raise ValueError(f"Required aberration mode {(n, m)} missing from config.") from exc


def _solve_least_squares(
    design: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float = 0.0,
) -> torch.Tensor:
    if design.ndim != 2:
        raise ValueError("design must be 2D.")
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.shape[0] != design.shape[0]:
        raise ValueError("targets must share the leading dimension with design.")
    if ridge > 0:
        gram = design.T @ design
        reg = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device) * float(ridge)
        rhs = design.T @ targets
        return torch.linalg.solve(gram + reg, rhs)
    return torch.linalg.pinv(design) @ targets


def build_nat_design_matrix_torch(
    xn: torch.Tensor | Sequence[float],
    yn: torch.Tensor | Sequence[float],
    config: NATFieldConfig,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    x = _as_tensor_1d(xn, dtype=dtype, device=device)
    y = _as_tensor_1d(yn, dtype=dtype, device=x.device)
    orders = [
        (component.px, component.py)
        for gamma in config.gammas
        for component in gamma.components
    ]
    legendres = evaluate_legendre_2d_torch(orders, x, y, normalize=True, dtype=dtype, device=x.device)
    n_points = x.shape[0]
    n_modes = len(config.aberrations)
    n_gammas = len(config.gammas)
    design = torch.zeros((n_points * n_modes, n_gammas), dtype=dtype, device=x.device)
    aberration_index = {mode: idx for idx, mode in enumerate(config.aberrations)}

    legendre_index = 0
    row_offsets = torch.arange(n_points, dtype=torch.long, device=x.device) * n_modes
    for gamma_index, gamma in enumerate(config.gammas):
        for component in gamma.components:
            mode_index = aberration_index[(component.n, component.m)]
            rows = row_offsets + int(mode_index)
            design[rows, gamma_index] += float(component.weight) * legendres[legendre_index]
            legendre_index += 1
    return design


def fit_nat_gammas_torch(
    xn: torch.Tensor | Sequence[float],
    yn: torch.Tensor | Sequence[float],
    zernike_coefficients: torch.Tensor | Sequence[Sequence[float]],
    config: NATFieldConfig,
    *,
    ridge: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    x = _as_tensor_1d(xn, dtype=dtype, device=device)
    y = _as_tensor_1d(yn, dtype=dtype, device=x.device)
    coeffs = zernike_coefficients if torch.is_tensor(zernike_coefficients) else torch.as_tensor(zernike_coefficients, dtype=dtype)
    coeffs = coeffs.to(device=x.device, dtype=dtype)
    if coeffs.ndim != 2:
        raise ValueError("zernike_coefficients must have shape (N, K).")
    if coeffs.shape[0] != x.shape[0]:
        raise ValueError("zernike_coefficients row count must match xn/yn.")
    if coeffs.shape[1] != len(config.aberrations):
        raise ValueError("zernike_coefficients column count must match config.aberrations.")
    design = build_nat_design_matrix_torch(
        x,
        y,
        config,
        dtype=dtype,
        device=x.device,
    )
    return _solve_least_squares(design, coeffs.reshape(-1), ridge=ridge).reshape(-1)


def invert_nat_order1_torch(
    xn: torch.Tensor | Sequence[float],
    yn: torch.Tensor | Sequence[float],
    zernike_coefficients: torch.Tensor | Sequence[Sequence[float]],
    config: NATFieldConfig | None = None,
    *,
    ridge: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    cfg = default_order1_config() if config is None else config
    return fit_nat_gammas_torch(
        xn,
        yn,
        zernike_coefficients,
        cfg,
        ridge=ridge,
        dtype=dtype,
        device=device,
    )


@dataclass(frozen=True)
class NATFieldObservation:
    normalized_coords: torch.Tensor
    coefficients_nm: torch.Tensor


class NATFieldModel(nn.Module):
    def __init__(
        self,
        config: NATFieldConfig | None = None,
        *,
        gamma_init: torch.Tensor | Sequence[float] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = default_order1_config() if config is None else config
        init = torch.zeros(len(self.config.gammas), dtype=dtype) if gamma_init is None else _as_tensor_1d(gamma_init, dtype=dtype, device=None)
        if init.shape[0] != len(self.config.gammas):
            raise ValueError("gamma_init length must match config.gammas.")
        self.gamma = nn.Parameter(init.to(dtype=dtype))
        if device is not None:
            self.to(device=device)

    def forward_normalized(self, normalized_xy: torch.Tensor) -> torch.Tensor:
        coords = normalized_xy.to(device=self.gamma.device, dtype=self.gamma.dtype)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("normalized_xy must have shape (N, 2).")
        return evaluate_zernike_coefficients_torch(
            coords[:, 0],
            coords[:, 1],
            self.gamma,
            self.config,
            dtype=self.gamma.dtype,
            device=self.gamma.device,
        )

    def forward_roi_positions(
        self,
        roixy_px: torch.Tensor | Sequence[Sequence[float]],
        *,
        local_x_nm: torch.Tensor | Sequence[float] | float | None = None,
        local_y_nm: torch.Tensor | Sequence[float] | float | None = None,
    ) -> NATFieldObservation:
        coefficients, normalized, _ = evaluate_zernike_from_roi_positions_torch(
            roixy_px,
            self.gamma,
            self.config,
            local_x_nm=local_x_nm,
            local_y_nm=local_y_nm,
            dtype=self.gamma.dtype,
            device=self.gamma.device,
        )
        return NATFieldObservation(normalized_coords=normalized, coefficients_nm=coefficients)

    @torch.no_grad()
    def initialize_from_order1_inversion(
        self,
        normalized_xy: torch.Tensor | Sequence[Sequence[float]],
        coefficients_nm: torch.Tensor | Sequence[Sequence[float]],
        *,
        ridge: float = 0.0,
    ) -> torch.Tensor:
        coords = normalized_xy if torch.is_tensor(normalized_xy) else torch.as_tensor(normalized_xy, dtype=self.gamma.dtype)
        coords = coords.to(device=self.gamma.device, dtype=self.gamma.dtype)
        coeffs = coefficients_nm if torch.is_tensor(coefficients_nm) else torch.as_tensor(coefficients_nm, dtype=self.gamma.dtype)
        coeffs = coeffs.to(device=self.gamma.device, dtype=self.gamma.dtype)
        gamma = invert_nat_order1_torch(
            coords[:, 0],
            coords[:, 1],
            coeffs,
            self.config,
            ridge=ridge,
            dtype=self.gamma.dtype,
            device=self.gamma.device,
        )
        self.gamma.copy_(gamma)
        return gamma


__all__ = [
    "NATFieldModel",
    "NATFieldObservation",
    "absolute_pixel_positions_torch",
    "build_nat_design_matrix_torch",
    "evaluate_legendre_2d_torch",
    "evaluate_zernike_coefficients_torch",
    "evaluate_zernike_from_roi_positions_torch",
    "fit_nat_gammas_torch",
    "get_fov_coordinates_torch",
    "invert_nat_order1_torch",
]
