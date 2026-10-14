"""Localization data contract used by the DH full-FOV physical update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndependentLocalizations:
    """Per-emitter observations used to project local PSFs into full frames."""

    frame_index: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    z_nm: np.ndarray
    photons_adu: np.ndarray
    background_adu: np.ndarray
    ncc: np.ndarray
    lobe_angle_rad: np.ndarray
    lobe_separation_px: np.ndarray


__all__ = ["IndependentLocalizations"]
