"""Deprecated compatibility wrapper for the UnityPSF evaluation CLI."""

from unity_psf.cli.double_helix_evaluation import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
