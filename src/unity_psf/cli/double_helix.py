"""Formal UnityPSF entry point for double-helix calibration."""

from __future__ import annotations

from .double_helix_calibration import main as _calibrate_main


def calibrate_double_helix() -> int:
    """Run the double-helix calibration CLI."""
    return int(_calibrate_main())


__all__ = ["calibrate_double_helix"]
