from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence

import torch


# Literature commonly reports this family as (l,p) = (1,1), (3,5),
# (5,9), (7,13). Internally each pair is stored as (radial p, azimuthal l).
CANONICAL_DH_LG_MODES = ((1, 1), (5, 3), (9, 5), (13, 7))


def laguerre_gaussian_basis(
    mode_order: Sequence[tuple[int, int]],
    x_pupil: torch.Tensor,
    y_pupil: torch.Tensor,
    *,
    waist: float,
) -> torch.Tensor:
    if x_pupil.shape != y_pupil.shape or x_pupil.ndim != 2:
        raise ValueError("x_pupil and y_pupil must be matching two-dimensional tensors.")
    if not 0.0 < float(waist) <= 1.0:
        raise ValueError("waist must be in (0, 1].")
    modes = tuple(_validate_lg_mode(mode) for mode in mode_order)
    if not modes:
        raise ValueError("mode_order must contain at least one LG mode.")

    rho = torch.sqrt(x_pupil.square() + y_pupil.square())
    phi = torch.atan2(y_pupil, x_pupil)
    aperture = rho < 1.0
    scaled_radius_squared = 2.0 * rho.square() / float(waist) ** 2
    basis = []
    for radial_order, azimuthal_order in modes:
        absolute_l = abs(azimuthal_order)
        radial = (
            (math.sqrt(2.0) * rho / float(waist)).pow(absolute_l)
            * _generalized_laguerre(radial_order, absolute_l, scaled_radius_squared)
            * torch.exp(-0.5 * scaled_radius_squared)
        )
        angular = torch.polar(torch.ones_like(phi), float(azimuthal_order) * phi)
        mode = torch.where(aperture, radial.to(angular.dtype) * angular, 0.0)
        rms = mode[aperture].abs().square().mean().sqrt().clamp_min(
            torch.finfo(x_pupil.dtype).eps
        )
        basis.append(mode / rms)
    return torch.stack(basis)


def lg_dh_carrier(
    basis: torch.Tensor,
    *,
    mode_order: Sequence[tuple[int, int]],
    weight_logits: torch.Tensor,
    phase_offsets_rad: torch.Tensor | None = None,
    rotation_rad: torch.Tensor | float,
) -> torch.Tensor:
    modes = tuple(_validate_lg_mode(mode) for mode in mode_order)
    if basis.ndim != 3 or basis.shape[0] != len(modes):
        raise ValueError("basis must have shape (mode_count, pupil_y, pupil_x).")
    logits = torch.as_tensor(weight_logits, dtype=basis.real.dtype, device=basis.device)
    if logits.shape != (len(modes),):
        raise ValueError("weight_logits must contain one value per LG mode.")
    if phase_offsets_rad is None:
        phase_offsets = torch.zeros_like(logits)
    else:
        phase_offsets = torch.as_tensor(
            phase_offsets_rad, dtype=basis.real.dtype, device=basis.device
        )
        if phase_offsets.shape != (len(modes),):
            raise ValueError("phase_offsets_rad must contain one value per LG mode.")
        phase_offsets = phase_offsets - phase_offsets[0]
    rotation = torch.as_tensor(rotation_rad, dtype=basis.real.dtype, device=basis.device)
    azimuthal_orders = torch.as_tensor(
        [mode[1] for mode in modes],
        dtype=basis.real.dtype,
        device=basis.device,
    )
    weights = torch.softmax(logits, dim=0)
    rotation_phase = torch.polar(
        torch.ones_like(azimuthal_orders),
        phase_offsets - azimuthal_orders * rotation,
    )
    field = torch.einsum("c,chw->hw", weights.to(basis.dtype) * rotation_phase, basis)
    aperture = basis.abs().sum(dim=0) > 0.0
    epsilon = torch.finfo(basis.real.dtype).eps
    stabilized = field + epsilon
    phase_only = stabilized / stabilized.abs().clamp_min(epsilon)
    return torch.where(aperture, phase_only, torch.ones_like(phase_only))


def _generalized_laguerre(order: int, alpha: int, values: torch.Tensor) -> torch.Tensor:
    if order == 0:
        return torch.ones_like(values)
    previous = torch.ones_like(values)
    current = 1.0 + float(alpha) - values
    if order == 1:
        return current
    for index in range(2, order + 1):
        following = (
            (2.0 * index - 1.0 + alpha - values) * current
            - float(index - 1 + alpha) * previous
        ) / float(index)
        previous, current = current, following
    return current


def _validate_lg_mode(mode: Sequence[int]) -> tuple[int, int]:
    if isinstance(mode, (str, bytes)) or not hasattr(mode, "__len__") or len(mode) != 2:
        raise ValueError("Each LG mode must be a (radial_order, azimuthal_order) pair.")
    radial_order, azimuthal_order = mode
    if (
        isinstance(radial_order, bool)
        or isinstance(azimuthal_order, bool)
        or not isinstance(radial_order, Integral)
        or not isinstance(azimuthal_order, Integral)
        or radial_order < 0
    ):
        raise ValueError(f"Invalid LG mode: {(radial_order, azimuthal_order)}")
    return int(radial_order), int(azimuthal_order)


__all__ = [
    "CANONICAL_DH_LG_MODES",
    "laguerre_gaussian_basis",
    "lg_dh_carrier",
]
