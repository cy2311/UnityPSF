"""Public CLI adapter for modality-owned multichannel training."""

from unity_psf.training.entrypoints.train_modality_joint import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
