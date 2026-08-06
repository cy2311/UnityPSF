"""Compatibility entry point for high-fidelity localization training.

The implementation lives in :mod:`unity_psf.training.high_fidelity.engine`.
Only the established integration helpers are re-exported here.
"""

from .high_fidelity.engine import (
    _auto_build_roi_bank,
    _build_vector_roi_gamma_objective,
    _condition_store_batch_provider_overrides,
    _condition_store_from_runtime_config,
    _mapping,
    _peak_bootstrap_config,
    _physical_checkpoint_extra_fn,
    _posterior_photon_scale,
    _posterior_z_scale,
    _resolve_roi_bank_source,
    _roi_conditioning_context,
    _run_peak_zmap_bootstrap_if_enabled,
    _select_single_channel_roi_split,
    _single_channel_peak_domain,
    main,
    parse_args,
    resume_epoch_training_config,
)

__all__ = ["main", "parse_args", "resume_epoch_training_config"]


if __name__ == "__main__":
    raise SystemExit(main())
