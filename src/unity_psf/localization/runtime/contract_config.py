"""Resolved localization runtime contract construction."""

from __future__ import annotations

from typing import Any, Mapping

from unity_psf.localization.scale_resolver import PhysicalScaleSpec
from unity_psf.localization.smlm_targets import SMLMTargetContract

from .paths import mapping


def resolved_contract(
    *,
    model_name: str,
    model_params: Mapping[str, Any],
    batch_provider: Mapping[str, Any],
    loss: Mapping[str, Any],
    legacy_loss_params: Mapping[str, Any],
    modality_contract: Mapping[str, Any],
    scales: PhysicalScaleSpec,
    target_contract: SMLMTargetContract,
    training_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    provider_params = mapping(batch_provider.get("params"), "batch_provider.params")
    loss_params = dict(mapping(loss.get("params", {}), "loss.params"))
    contract: dict[str, Any] = {
        "model": {
            "name": str(model_name),
            "output": "smlm_10ch"
            if str(model_name).startswith("active_smlm_") or str(model_name) in {"astigmatism_expert", "double_helix_expert", "emitter_2d_expert"}
            else "localization",
        },
        "batch_provider": {
            "name": str(batch_provider["name"]),
            "pxyz_target_order": target_contract.target_order,
        },
        "loss": {
            "name": str(loss["name"]),
            "params": loss_params,
            "legacy_params": dict(legacy_loss_params),
        },
        "training_runtime": dict(training_runtime),
        "physical_scales": scales.as_dict(),
        "target_contract": target_contract.as_dict(),
    }
    if str(model_name).startswith("active_smlm_"):
        contract["model"]["z_mu_activation"] = str(model_params.get("z_mu_activation", loss_params.get("z_activation", "tanh")))
    for key in (
        "batch_strategy",
        "sequence_count",
        "condition_dim",
        "condition_feature_dim",
        "condition_fields",
        "domain_count",
        "append_domain_onehot",
    ):
        if key in provider_params:
            contract["batch_provider"][key] = provider_params[key]
    if provider_params.get("dual_domain_coeff_maps"):
        contract["batch_provider"]["dual_domain_coeff_maps"] = provider_params["dual_domain_coeff_maps"]
    contract.update(dict(modality_contract))
    return contract
