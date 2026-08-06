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


ASTIGMATISM_660NM_ANCHOR_PROFILE = AstigmatismAnchorProfile(
    name="astigmatism_660nm",
    wavelength_nm=660.0,
    anchor_nm=99.0,
)


def resolve_astigmatism_anchor_profile(name: str | None) -> AstigmatismAnchorProfile:
    """Resolve the only currently supported explicit astigmatism profile."""

    if name is None or not str(name).strip():
        return ASTIGMATISM_660NM_ANCHOR_PROFILE
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        ASTIGMATISM_660NM_ANCHOR_PROFILE.name: ASTIGMATISM_660NM_ANCHOR_PROFILE,
        "astigmatism_660nm": ASTIGMATISM_660NM_ANCHOR_PROFILE,
        "astigmatism_660nm_anchor_99": ASTIGMATISM_660NM_ANCHOR_PROFILE,
        "astigmatism_660nm_anchor99": ASTIGMATISM_660NM_ANCHOR_PROFILE,
        "astigmatism_660nm_99nm": ASTIGMATISM_660NM_ANCHOR_PROFILE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown astigmatism anchor profile {name!r}") from exc


__all__ = [
    "ASTIGMATISM_660NM_ANCHOR_PROFILE",
    "AstigmatismAnchorProfile",
    "resolve_astigmatism_anchor_profile",
]
