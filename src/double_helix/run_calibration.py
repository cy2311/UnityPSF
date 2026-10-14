"""Deprecated compatibility wrapper for the UnityPSF calibration CLI."""

from unity_psf.cli.double_helix_calibration import *  # noqa: F403
from unity_psf.cli.double_helix_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
