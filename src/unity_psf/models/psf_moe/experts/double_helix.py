"""Double-helix expert with lobe geometry auxiliary heads."""

from __future__ import annotations

from unity_psf.contracts.modality import PSFModality
from unity_psf.models.psf_moe.base import AdaptedPSFExpert


class DoubleHelixExpert(AdaptedPSFExpert):
    modality = PSFModality.DOUBLE_HELIX

    def __init__(self, feature_channels: int = 32) -> None:
        super().__init__(
            feature_channels,
            auxiliary_channels={"lobe_angle": 1, "lobe_separation": 1},
        )


__all__ = ["DoubleHelixExpert"]
