"""Public entry point for high-fidelity localization training."""

from .high_fidelity.engine import main, parse_args, resume_epoch_training_config

__all__ = ["main", "parse_args", "resume_epoch_training_config"]


if __name__ == "__main__":
    raise SystemExit(main())
