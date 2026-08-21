"""Public CLI adapter for single-process joint training."""

from unity_psf.training.entrypoints.train_joint import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
