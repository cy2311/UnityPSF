"""Compatibility entrypoint for modality-owned Expert Parallel training."""

from .train_modality_expert_parallel import main


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
