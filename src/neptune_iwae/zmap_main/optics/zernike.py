"""Zernike utilities shared by simulation and phase retrieval."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def noll_to_nm(j: int) -> Tuple[int, int]:
    """Convert Noll index to Zernike (n, m)."""
    if j < 1:
        raise ValueError("Noll index must be >= 1")
    n = int(math.ceil((-3.0 + math.sqrt(1.0 + 8.0 * j)) / 2.0))
    m = int(j - (n * (n + 1)) // 2 - 1)
    if (n % 2) != (m % 2):
        m += 1
    if (j % 2) == 1:
        m = -m
    return n, m


def zernike_polynomial(n: int, m: int, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Real-valued Zernike Z_n^m (Condon-Shortley)."""
    m_abs = abs(m)
    radial = torch.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** k) * math.comb(n - k, k) * math.comb(n - 2 * k, (n - m_abs) // 2 - k)
        radial = radial + coeff * rho ** (n - 2 * k)
    radial = radial * (rho <= 1).float()
    if m >= 0:
        return radial * torch.cos(m_abs * theta)
    return radial * torch.sin(m_abs * theta)


__all__ = ["noll_to_nm", "zernike_polynomial"]
