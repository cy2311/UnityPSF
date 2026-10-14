from __future__ import annotations

import numpy as np

from ppg.beads_alignment import apply_affine_z_shift, fit_affine_z_shift, stack_ncc


def test_affine_z_alignment_recovers_subpixel_bead_offset() -> None:
    coordinates = np.arange(21, dtype=np.float64) - 10.0
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    z_normalized = np.linspace(-1.0, 1.0, 7)
    canonical = []
    for angle in np.linspace(-0.7, 0.7, len(z_normalized)):
        dx = 2.4 * np.cos(angle)
        dy = 2.4 * np.sin(angle)
        left = np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2.0 * 1.15**2))
        right = np.exp(-((xx + dx) ** 2 + (yy + dy) ** 2) / (2.0 * 1.15**2))
        canonical.append(left + 0.82 * right)
    canonical = np.asarray(canonical)

    expected = np.asarray([1.10, -0.75, 0.35, -0.25])
    observed = apply_affine_z_shift(canonical, expected, z_normalized)
    fitted = fit_affine_z_shift(observed, canonical, z_normalized)
    aligned = apply_affine_z_shift(observed, -fitted, z_normalized)

    assert np.allclose(fitted, expected, atol=0.08)
    assert np.median(stack_ncc(observed, canonical)) < 0.90
    assert np.median(stack_ncc(aligned, canonical)) > 0.995
