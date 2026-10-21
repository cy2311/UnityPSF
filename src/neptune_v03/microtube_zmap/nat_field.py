from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


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


def _legendre_basis(order: int, values: np.ndarray) -> np.ndarray:
    coeffs = np.zeros(order + 1, dtype=np.float64)
    coeffs[order] = 1.0
    return np.polynomial.legendre.legval(values, coeffs)


def _evaluate_legendre_2d(
    orders: Sequence[tuple[int, int]],
    xn: np.ndarray,
    yn: np.ndarray,
) -> np.ndarray:
    x = np.asarray(xn, dtype=np.float64).reshape(-1)
    y = np.asarray(yn, dtype=np.float64).reshape(-1)
    out = np.zeros((len(orders), x.shape[0]), dtype=np.float64)
    for index, (px, py) in enumerate(orders):
        norm = np.sqrt((1.0 + 2.0 * float(px)) * (1.0 + 2.0 * float(py)))
        out[index] = norm * _legendre_basis(int(px), x) * _legendre_basis(int(py), y)
    return out


def get_fov_coordinates(
    roixy_px: np.ndarray,
    *,
    img_size_x: int,
    img_size_y: int,
    pixel_size_x_nm: float,
    pixel_size_y_nm: float,
    local_x_nm: np.ndarray | float | None = None,
    local_y_nm: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    roixy = np.asarray(roixy_px, dtype=np.float64)
    if roixy.ndim != 2 or roixy.shape[1] != 2:
        raise ValueError("roixy_px must have shape (N, 2).")

    count = roixy.shape[0]
    local_x = np.zeros(count, dtype=np.float64) if local_x_nm is None else np.asarray(local_x_nm, dtype=np.float64).reshape(-1)
    local_y = np.zeros(count, dtype=np.float64) if local_y_nm is None else np.asarray(local_y_nm, dtype=np.float64).reshape(-1)
    if local_x.shape[0] != count or local_y.shape[0] != count:
        raise ValueError("local_x_nm/local_y_nm must match roixy_px length.")

    x_physical_um = (roixy[:, 0] * float(pixel_size_x_nm) + local_x) * 1e-3
    y_physical_um = (roixy[:, 1] * float(pixel_size_y_nm) + local_y) * 1e-3

    x_size_nm = float(pixel_size_x_nm) * float(img_size_x)
    y_size_nm = float(pixel_size_y_nm) * float(img_size_y)
    x_normalized = -1.0 + 2.0 * (roixy[:, 0] * float(pixel_size_x_nm) + local_x) / x_size_nm
    y_normalized = -1.0 + 2.0 * (roixy[:, 1] * float(pixel_size_y_nm) + local_y) / y_size_nm

    normalized = np.stack([x_normalized, y_normalized], axis=1).astype(np.float32, copy=False)
    physical_um = np.stack([x_physical_um, y_physical_um], axis=1).astype(np.float32, copy=False)
    return normalized, physical_um


def evaluate_zernike_coefficients(
    xn: np.ndarray,
    yn: np.ndarray,
    gamma_values: Sequence[float],
    config: NATFieldConfig,
) -> np.ndarray:
    x = np.asarray(xn, dtype=np.float64).reshape(-1)
    y = np.asarray(yn, dtype=np.float64).reshape(-1)
    gammas = np.asarray(tuple(gamma_values), dtype=np.float64).reshape(-1)
    if gammas.shape[0] != len(config.gammas):
        raise ValueError("gamma_values length must match config.gammas.")

    orders = [
        (component.px, component.py)
        for gamma in config.gammas
        for component in gamma.components
    ]
    legendres = _evaluate_legendre_2d(orders, x, y)
    coefficients = np.zeros((x.shape[0], len(config.aberrations)), dtype=np.float64)
    aberration_index = {mode: idx for idx, mode in enumerate(config.aberrations)}

    legendre_index = 0
    for gamma_index, gamma in enumerate(config.gammas):
        for component in gamma.components:
            target = aberration_index[(component.n, component.m)]
            coefficients[:, target] += (
                gammas[gamma_index] * float(component.weight) * legendres[legendre_index]
            )
            legendre_index += 1

    return coefficients.astype(np.float32, copy=False)


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
    gamma_components: tuple[NATGamma, ...] = (
        NATGamma((NATComponent(0, 0, 2, 0, 1.0),)),
        NATGamma((NATComponent(1, 0, 2, 0, 1.0),)),
        NATGamma((NATComponent(0, 1, 2, 0, 1.0),)),
        NATGamma((NATComponent(2, 0, 2, 0, 1.0), NATComponent(0, 2, 2, 0, 1.0))),
        NATGamma((NATComponent(0, 0, 2, 2, 1.0),)),
        NATGamma((NATComponent(0, 0, 2, -2, 1.0),)),
        NATGamma((NATComponent(1, 0, 2, 2, 1.0), NATComponent(0, 1, 2, -2, 1.0))),
        NATGamma((NATComponent(0, 1, 2, 2, -1.0), NATComponent(1, 0, 2, -2, 1.0))),
        NATGamma(
            (
                NATComponent(2, 0, 2, 2, 1.0 / np.sqrt(5.0)),
                NATComponent(0, 2, 2, 2, -1.0 / np.sqrt(5.0)),
                NATComponent(1, 1, 2, -2, 1.0),
            )
        ),
        NATGamma((NATComponent(0, 0, 3, 1, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, -1, 1.0),)),
        NATGamma((NATComponent(1, 0, 3, 1, 1.0), NATComponent(0, 1, 3, -1, 1.0))),
        NATGamma((NATComponent(0, 0, 4, 0, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, -3, 1.0),)),
        NATGamma((NATComponent(1, 0, 3, -3, 1.0),)),
        NATGamma((NATComponent(0, 1, 3, -3, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, 3, 1.0),)),
        NATGamma((NATComponent(1, 0, 3, 3, 1.0),)),
        NATGamma((NATComponent(0, 1, 3, 3, 1.0),)),
    )
    return NATFieldConfig(
        aberrations=aberrations,
        gammas=gamma_components,
        img_size_x=int(img_size_x),
        img_size_y=int(img_size_y),
        pixel_size_x_nm=float(pixel_size_x_nm),
        pixel_size_y_nm=float(pixel_size_y_nm),
    )


def _aberrations_for_nat_order(order: int) -> tuple[tuple[int, int], ...]:
    order_i = int(order)
    if order_i < 0:
        raise ValueError("NAT order must be non-negative.")
    values: list[tuple[int, int]] = []
    for n in range(2, 2 * (order_i + 1) + 1):
        mmin = max(-n, n - 2 * (order_i + 1))
        mmax = min(n, 2 * (order_i + 1) - n)
        for m in range(mmin, mmax + 1, 2):
            values.append((int(n), int(m)))
    return tuple(values)


def default_order2_config(
    *,
    img_size_x: int = 1024,
    img_size_y: int = 1024,
    pixel_size_x_nm: float = 95.0,
    pixel_size_y_nm: float = 95.0,
) -> NATFieldConfig:
    # Mirrors fieldfit/set_parameters_scripts/set_parameters_Fu_3D_second_order_NAT.m, case j=2.
    aberrations = _aberrations_for_nat_order(2)
    gamma_components: tuple[NATGamma, ...] = (
        NATGamma((NATComponent(0, 0, 2, 2, 1.0),)),
        NATGamma((NATComponent(0, 0, 2, -2, 1.0),)),
        NATGamma((NATComponent(1, 0, 2, 2, 1.0),)),
        NATGamma((NATComponent(0, 1, 2, 2, 1.0),)),
        NATGamma((NATComponent(1, 0, 2, -2, 1.0),)),
        NATGamma((NATComponent(0, 1, 2, -2, 1.0),)),
        NATGamma((NATComponent(2, 0, 2, 2, 1.0), NATComponent(0, 2, 2, 2, -1.0))),
        NATGamma((NATComponent(1, 1, 2, 2, 1.0),)),
        NATGamma((NATComponent(2, 0, 2, -2, 1.0),)),
        NATGamma((NATComponent(1, 1, 2, -2, 1.0),)),
        NATGamma((NATComponent(0, 2, 2, -2, 1.0),)),
        NATGamma(
            (
                NATComponent(3, 0, 2, 2, 0.75 * np.sqrt(96.0 / 377.0)),
                NATComponent(1, 2, 2, 2, 1.25 * np.sqrt(672.0 / 377.0) / np.sqrt(15.0)),
                NATComponent(0, 3, 2, -2, 0.75 * np.sqrt(96.0 / 377.0)),
                NATComponent(2, 1, 2, -2, 1.25 * np.sqrt(672.0 / 377.0) / np.sqrt(15.0)),
            )
        ),
        NATGamma(
            (
                NATComponent(3, 0, 2, 2, 0.75 * np.sqrt(96.0 / 377.0)),
                NATComponent(1, 2, 2, 2, -3.75 * np.sqrt(672.0 / 377.0) / np.sqrt(15.0)),
                NATComponent(0, 3, 2, -2, -0.75 * np.sqrt(96.0 / 377.0)),
                NATComponent(2, 1, 2, -2, 3.75 * np.sqrt(672.0 / 377.0) / np.sqrt(15.0)),
            )
        ),
        NATGamma(
            (
                NATComponent(0, 3, 2, 2, 0.75 * np.sqrt(96.0 / 377.0)),
                NATComponent(2, 1, 2, 2, 3.75 * np.sqrt(672.0 / 377.0) / np.sqrt(15.0)),
                NATComponent(0, 3, 2, -2, -0.75 * np.sqrt(96.0 / 377.0)),
                NATComponent(2, 1, 2, -2, -3.75 * np.sqrt(672.0 / 377.0) / np.sqrt(15.0)),
            )
        ),
        NATGamma(
            (
                NATComponent(0, 3, 2, 2, 0.75 * np.sqrt(96.0 / 377.0)),
                NATComponent(2, 1, 2, 2, -1.25 * np.sqrt(672.0 / 377.0) / np.sqrt(15.0)),
                NATComponent(0, 3, 2, -2, 0.75 * np.sqrt(96.0 / 377.0)),
                NATComponent(2, 1, 2, -2, -1.25 * np.sqrt(672.0 / 377.0) / np.sqrt(15.0)),
            )
        ),
        NATGamma(
            (
                NATComponent(4, 0, 2, 2, np.sqrt(1.0 / 11.0)),
                NATComponent(0, 4, 2, 2, -np.sqrt(1.0 / 11.0)),
                NATComponent(3, 1, 2, -2, 7.0 * np.sqrt(9.0 / 11.0) / np.sqrt(21.0)),
                NATComponent(1, 3, 2, -2, 7.0 * np.sqrt(9.0 / 11.0) / np.sqrt(21.0)),
            )
        ),
        NATGamma((NATComponent(0, 0, 3, 1, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, -1, 1.0),)),
        NATGamma((NATComponent(1, 0, 3, 1, 1.0),)),
        NATGamma((NATComponent(0, 1, 3, -1, 1.0),)),
        NATGamma((NATComponent(0, 1, 3, 1, 1.0),)),
        NATGamma((NATComponent(1, 0, 3, -1, 1.0),)),
        NATGamma((NATComponent(2, 0, 3, 1, 1.0), NATComponent(0, 2, 3, 1, 1.0))),
        NATGamma(
            (
                NATComponent(2, 0, 3, 1, np.sqrt(3.0 / 7.0)),
                NATComponent(0, 2, 3, 1, -np.sqrt(3.0 / 7.0)),
                NATComponent(1, 1, 3, -1, np.sqrt(15.0 / 7.0)),
            )
        ),
        NATGamma((NATComponent(0, 2, 3, -1, 1.0), NATComponent(2, 0, 3, -1, 1.0))),
        NATGamma(
            (
                NATComponent(0, 2, 3, -1, np.sqrt(3.0 / 7.0)),
                NATComponent(2, 0, 3, -1, -np.sqrt(3.0 / 7.0)),
                NATComponent(1, 1, 3, 1, np.sqrt(15.0 / 7.0)),
            )
        ),
        NATGamma(
            (
                NATComponent(3, 0, 3, 1, 3.0 * np.sqrt(6.0 / 97.0)),
                NATComponent(0, 3, 3, -1, 3.0 * np.sqrt(6.0 / 97.0)),
                NATComponent(1, 2, 3, 1, 5.0 * np.sqrt(6.0 / 97.0)),
                NATComponent(2, 1, 3, -1, 5.0 * np.sqrt(6.0 / 97.0)),
            )
        ),
        NATGamma((NATComponent(0, 0, 3, 3, 1.0),)),
        NATGamma((NATComponent(0, 0, 3, -3, 1.0),)),
        NATGamma((NATComponent(0, 1, 3, 3, 1.0), NATComponent(1, 0, 3, -3, -1.0))),
        NATGamma((NATComponent(0, 1, 3, -3, 1.0), NATComponent(1, 0, 3, 3, 1.0))),
        NATGamma(
            (
                NATComponent(2, 0, 3, 3, np.sqrt(3.0 / 7.0)),
                NATComponent(0, 2, 3, 3, -np.sqrt(3.0 / 7.0)),
                NATComponent(1, 1, 3, -3, np.sqrt(15.0 / 7.0)),
            )
        ),
        NATGamma(
            (
                NATComponent(2, 0, 3, -3, -np.sqrt(3.0 / 7.0)),
                NATComponent(0, 2, 3, -3, np.sqrt(3.0 / 7.0)),
                NATComponent(1, 1, 3, 3, np.sqrt(15.0 / 7.0)),
            )
        ),
        NATGamma(
            (
                NATComponent(0, 1, 3, -3, np.sqrt(7.0 / 22.0)),
                NATComponent(1, 0, 3, 3, -np.sqrt(7.0 / 22.0)),
                NATComponent(3, 0, 3, 3, np.sqrt(3.0 / 22.0)),
                NATComponent(0, 3, 3, -3, -np.sqrt(3.0 / 22.0)),
                NATComponent(1, 2, 3, 3, -5.0 * np.sqrt(21.0 / 22.0) / np.sqrt(15.0)),
                NATComponent(2, 1, 3, -3, 5.0 * np.sqrt(21.0 / 22.0) / np.sqrt(15.0)),
            )
        ),
        NATGamma((NATComponent(0, 0, 4, 0, 1.0),)),
        NATGamma((NATComponent(0, 0, 4, 2, 1.0),)),
        NATGamma((NATComponent(0, 0, 4, -2, 1.0),)),
        NATGamma((NATComponent(1, 0, 4, 0, 1.0),)),
        NATGamma((NATComponent(0, 1, 4, 0, 1.0),)),
        NATGamma((NATComponent(1, 0, 4, 2, 1.0), NATComponent(0, 1, 4, -2, 1.0))),
        NATGamma((NATComponent(0, 1, 4, 2, 1.0), NATComponent(1, 0, 4, -2, -1.0))),
        NATGamma((NATComponent(2, 0, 4, 0, 1.0), NATComponent(0, 2, 4, 0, 1.0))),
        NATGamma(
            (
                NATComponent(2, 0, 4, 2, np.sqrt(3.0 / 7.0)),
                NATComponent(0, 2, 4, 2, -np.sqrt(3.0 / 7.0)),
                NATComponent(1, 1, 4, -2, np.sqrt(15.0 / 7.0)),
            )
        ),
        NATGamma((NATComponent(0, 0, 5, 1, 1.0),)),
        NATGamma((NATComponent(0, 0, 5, -1, 1.0),)),
        NATGamma((NATComponent(1, 0, 5, 1, 1.0), NATComponent(0, 1, 5, -1, 1.0))),
        NATGamma((NATComponent(0, 0, 6, 0, 1.0),)),
    )
    return NATFieldConfig(
        aberrations=aberrations,
        gammas=gamma_components,
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
    if normalized in {"order2", "second_order", "j2"}:
        return default_order2_config(
            img_size_x=img_size_x,
            img_size_y=img_size_y,
            pixel_size_x_nm=pixel_size_x_nm,
            pixel_size_y_nm=pixel_size_y_nm,
        )
    raise ValueError(f"Unsupported NAT config name: {name!r}")


def evaluate_zernike_from_roi_positions(
    roixy_px: np.ndarray,
    gamma_values: Sequence[float],
    config: NATFieldConfig,
    *,
    local_x_nm: np.ndarray | float | None = None,
    local_y_nm: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized, physical_um = get_fov_coordinates(
        roixy_px,
        img_size_x=config.img_size_x,
        img_size_y=config.img_size_y,
        pixel_size_x_nm=config.pixel_size_x_nm,
        pixel_size_y_nm=config.pixel_size_y_nm,
        local_x_nm=local_x_nm,
        local_y_nm=local_y_nm,
    )
    coeffs = evaluate_zernike_coefficients(normalized[:, 0], normalized[:, 1], gamma_values, config)
    return coeffs, normalized, physical_um


def absolute_pixel_positions(
    *,
    roi_left_px: float,
    roi_top_px: float,
    emitter_x_px: Iterable[float],
    emitter_y_px: Iterable[float],
) -> np.ndarray:
    x = np.asarray(tuple(emitter_x_px), dtype=np.float32)
    y = np.asarray(tuple(emitter_y_px), dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError("emitter_x_px and emitter_y_px must have the same shape.")
    return np.stack([x + float(roi_left_px), y + float(roi_top_px)], axis=1)


__all__ = [
    "NATComponent",
    "NATFieldConfig",
    "NATGamma",
    "absolute_pixel_positions",
    "build_named_nat_config",
    "default_order1_config",
    "default_order2_config",
    "evaluate_zernike_coefficients",
    "evaluate_zernike_from_roi_positions",
    "get_fov_coordinates",
]
