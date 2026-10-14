from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from unity_psf.roi_library import ROIBank, ROIBankBuildConfig
from unity_psf.training.channel_context import sha256_file
from unity_psf.training.high_fidelity.gamma_projection import (
    build_vector_roi_gamma_objective as _build_vector_roi_gamma_objective,
    select_single_channel_roi_split as _select_single_channel_roi_split,
)
from unity_psf.training.high_fidelity.gamma_workflow import (
    build_roi_bank_gamma_hook as _build_roi_bank_gamma_hook,
    gamma_hook_bindings as _gamma_hook_bindings,
    roi_bank_gamma_route as _roi_bank_gamma_route,
)
from unity_psf.training.high_fidelity.peak_bootstrap import (
    peak_bootstrap_config as _peak_bootstrap_config,
    run_peak_zmap_bootstrap_if_enabled as _run_peak_zmap_bootstrap_if_enabled,
    single_channel_peak_domain as _single_channel_peak_domain,
)
from unity_psf.training.high_fidelity.physical_state import (
    physical_checkpoint_extra_fn as _physical_checkpoint_extra_fn,
    read_current_physical_state as _read_current_physical_state,
    runtime_dual_domain_coeff_maps as _runtime_dual_domain_coeff_maps,
    write_current_physical_state as _write_current_physical_state,
    write_initial_physical_state as _write_initial_physical_state,
)
from unity_psf.training.high_fidelity.roi_bank_builder import auto_build_roi_bank as _auto_build_roi_bank_impl
from unity_psf.training.high_fidelity.roi_bank_source import (
    ROIBankSource,
    camera_backward_from_training_config as _camera_backward_from_training_config,
    roi_conditioning_context as _roi_conditioning_context,
    resolve_roi_bank_source as _resolve_roi_bank_source,
)
from unity_psf.training.high_fidelity.roi_projection import (
    posterior_photon_scale as _posterior_photon_scale,
    posterior_z_scale as _posterior_z_scale,
    posterior_z_scale_nm as _posterior_z_scale_nm,
)
from unity_psf.training.high_fidelity.roi_workflow import build_roi_bank_from_loc_harvest_source
from unity_psf.training.high_fidelity.runtime_context import (
    condition_store_batch_provider_overrides as _condition_store_batch_provider_overrides,
    condition_store_from_runtime_config as _condition_store_from_runtime_config,
    mapping as _mapping,
)
from unity_psf.training.high_fidelity.runner import main, parse_args, resume_epoch_training_config


STAGE_NAME = "high_fidelity_localization"
_ROIBankSource = ROIBankSource




def _resume_identity_kwargs(
    runtime_config: Mapping[str, Any],
    *,
    config_base_dir: Path | None = None,
) -> dict[str, object]:
    instance = runtime_config.get("expert_instance")
    if not isinstance(instance, Mapping):
        return {}
    output: dict[str, object] = {}
    for source_key, target_key in (
        ("expert_type", "expected_expert_type"),
        ("instance_id", "expected_instance_id"),
        ("channel_id", "expected_channel_id"),
        ("parent_checkpoint_hash", "expected_parent_checkpoint_hash"),
    ):
        value = instance.get(source_key)
        if value is not None:
            output[target_key] = value
    prototype_ref = instance.get("prototype_ref")
    if prototype_ref is not None and "expected_parent_checkpoint_hash" not in output:
        reference = Path(str(prototype_ref))
        if config_base_dir is not None and not reference.is_absolute():
            reference = config_base_dir / reference
        if reference.is_file():
            output["expected_parent_checkpoint_hash"] = sha256_file(reference)
    return output


def _auto_build_roi_bank(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: ROIBankSource,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
) -> ROIBank:
    return _auto_build_roi_bank_impl(
        gamma_cfg,
        roi_source=roi_source,
        model=model,
        train_cfg=train_cfg,
        camera_backward_from_training_config=_camera_backward_from_training_config,
        posterior_photon_scale=_posterior_photon_scale,
        posterior_z_scale_nm=_posterior_z_scale_nm,
        roi_conditioning_context=_roi_conditioning_context,
        loc_harvest_builder=_auto_build_roi_bank_from_loc_harvest,
    )


def _auto_build_roi_bank_from_loc_harvest(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: ROIBankSource,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
    config: ROIBankBuildConfig,
    roi_size: int,
) -> ROIBank:
    from unity_psf.training.high_fidelity.roi_projection import (
        posterior_bg_scale,
        posterior_input_offset,
        posterior_input_scale,
    )

    return build_roi_bank_from_loc_harvest_source(
        gamma_cfg,
        roi_source=roi_source,
        model=model,
        train_cfg=train_cfg,
        config=config,
        roi_size=roi_size,
        input_offset=posterior_input_offset(train_cfg),
        input_scale=posterior_input_scale(train_cfg),
        photon_scale=_posterior_photon_scale(train_cfg),
        z_scale=_posterior_z_scale_nm(train_cfg),
        bg_scale=posterior_bg_scale(train_cfg),
        roi_conditioning_context=_roi_conditioning_context,
    )


if __name__ == "__main__":
    raise SystemExit(main())
