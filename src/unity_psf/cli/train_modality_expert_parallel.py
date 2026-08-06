"""Public CLI adapter for modality-owned Expert Parallel training."""

from unity_psf.training.entrypoints.train_modality_expert_parallel import (
    _heldout_eval_enabled,
    _metrics_from_progress,
    _read_completed_rank_statuses,
    main,
)

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
