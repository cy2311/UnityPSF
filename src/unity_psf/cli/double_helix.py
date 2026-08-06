"""Formal UnityPSF entry points for double-helix calibration and evaluation."""

from __future__ import annotations

from .double_helix_calibration import main as _calibrate_main
from .double_helix_evaluation import main as _evaluate_main


def calibrate_double_helix() -> int:
    """Run the double-helix calibration CLI."""
    return int(_calibrate_main())


def evaluate_double_helix() -> int:
    """Run the double-helix evaluation CLI."""
    return int(_evaluate_main())


__all__ = ["calibrate_double_helix", "evaluate_double_helix"]
