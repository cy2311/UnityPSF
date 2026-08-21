"""Small differentiable numerical primitives shared by DH fitting paths."""

from __future__ import annotations

import torch


def legendre_polynomial(degree: int, values: torch.Tensor) -> torch.Tensor:
    """Evaluate the degree-th Legendre polynomial without changing tensor metadata."""
    degree = int(degree)
    if degree < 0:
        raise ValueError("degree must be non-negative")
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


def normalized_cross_correlation(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compute per-sample NCC over the final two spatial dimensions."""
    if first.shape != second.shape or first.ndim < 2:
        raise ValueError("NCC inputs must have matching shapes and at least two dimensions")
    first_centered = first - first.mean(dim=(-2, -1), keepdim=True)
    second_centered = second - second.mean(dim=(-2, -1), keepdim=True)
    numerator = (first_centered * second_centered).sum(dim=(-2, -1))
    denominator = torch.sqrt(
        first_centered.square().sum(dim=(-2, -1))
        * second_centered.square().sum(dim=(-2, -1))
    ).clamp_min(1e-12)
    return numerator / denominator


__all__ = ["legendre_polynomial", "normalized_cross_correlation"]
