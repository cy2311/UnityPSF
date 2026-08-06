"""Public CLI adapter for modality-owned multichannel training."""

from unity_psf.training.entrypoints.train_modality_joint import (
    _audit_formal_runtime_contracts,
    _build_modality_runtime,
    _channel_metadata,
    _config_path,
    _modality_groups,
    main,
)

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
