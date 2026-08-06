"""Compatibility namespace for the pre-UnityPSF v0.4 package name."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.4.0"

# Keep legacy imports working while implementation ownership moves to UnityPSF.
_active_package = Path(__file__).resolve().parent.parent / "unity_psf"
if not _active_package.is_dir():
    raise ImportError(f"UnityPSF package is missing {_active_package}")
__path__ = [str(_active_package)]

__all__ = ["__version__"]
