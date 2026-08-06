"""Public CLI adapter for single-process joint training."""

from unity_psf.training.entrypoints.train_joint import (
    _bind_instance,
    _instance_specs,
    _load_joint_config,
    _sha256,
    _visual_record,
    main,
)

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
