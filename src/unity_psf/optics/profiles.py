"""Named optical profiles used by modality-specific training contexts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AstigmatismAnchorProfile:
    """Calibration anchor for one wavelength-specific astigmatism setup."""

    name: str
    wavelength_nm: float
    anchor_nm: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "wavelength_nm": float(self.wavelength_nm),
            "anchor_nm": float(self.anchor_nm),
        }


@dataclass(frozen=True)
class CircularPSFAnchorProfile:
    """Circular 2D PSF anchor described by a physical spot radius."""

    name: str
    radius_nm: float
    radius_definition: str = "gaussian_sigma"

    def __post_init__(self) -> None:
        if float(self.radius_nm) <= 0.0:
            raise ValueError("circular PSF anchor radius_nm must be positive")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "radius_nm": float(self.radius_nm),
            "radius_definition": self.radius_definition,
        }


ASTIGMATISM_660NM_ANCHOR_PROFILE = AstigmatismAnchorProfile(
    name="astigmatism_660nm",
    wavelength_nm=660.0,
    anchor_nm=100.0,
)

ASTIGMATISM_660NM_LEGACY_99NM_ANCHOR_PROFILE = AstigmatismAnchorProfile(
    name="astigmatism_660nm_99nm",
    wavelength_nm=660.0,
    anchor_nm=99.0,
)

CIRCULAR_2D_50NM_ANCHOR_PROFILE = CircularPSFAnchorProfile(
    name="circular_2d_50nm",
    radius_nm=50.0,
)


def resolve_astigmatism_anchor_profile(name: str | None) -> AstigmatismAnchorProfile:
    """Resolve the only currently supported explicit astigmatism profile."""

    if name is None or not str(name).strip():
        return ASTIGMATISM_660NM_ANCHOR_PROFILE
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        ASTIGMATISM_660NM_ANCHOR_PROFILE.name: ASTIGMATISM_660NM_ANCHOR_PROFILE,
        "astigmatism_660nm": ASTIGMATISM_660NM_ANCHOR_PROFILE,
        "astigmatism_660nm_anchor_100": ASTIGMATISM_660NM_ANCHOR_PROFILE,
        "astigmatism_660nm_anchor100": ASTIGMATISM_660NM_ANCHOR_PROFILE,
        "astigmatism_660nm_100nm": ASTIGMATISM_660NM_ANCHOR_PROFILE,
        "astigmatism_660nm_anchor_99": ASTIGMATISM_660NM_LEGACY_99NM_ANCHOR_PROFILE,
        "astigmatism_660nm_anchor99": ASTIGMATISM_660NM_LEGACY_99NM_ANCHOR_PROFILE,
        "astigmatism_660nm_99nm": ASTIGMATISM_660NM_LEGACY_99NM_ANCHOR_PROFILE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown astigmatism anchor profile {name!r}") from exc


def resolve_circular_psf_anchor_profile(name: str | None) -> CircularPSFAnchorProfile:
    """Resolve the default circular 2D PSF anchor."""

    if name is None or not str(name).strip():
        return CIRCULAR_2D_50NM_ANCHOR_PROFILE
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        CIRCULAR_2D_50NM_ANCHOR_PROFILE.name: CIRCULAR_2D_50NM_ANCHOR_PROFILE,
        "circular_psf_50nm": CIRCULAR_2D_50NM_ANCHOR_PROFILE,
        "emitter_2d_circular_50nm": CIRCULAR_2D_50NM_ANCHOR_PROFILE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown circular PSF anchor profile {name!r}") from exc


__all__ = [
    "ASTIGMATISM_660NM_ANCHOR_PROFILE",
    "ASTIGMATISM_660NM_LEGACY_99NM_ANCHOR_PROFILE",
    "AstigmatismAnchorProfile",
    "CIRCULAR_2D_50NM_ANCHOR_PROFILE",
    "CircularPSFAnchorProfile",
    "resolve_circular_psf_anchor_profile",
    "resolve_astigmatism_anchor_profile",
]
