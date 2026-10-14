"""Localization loss configuration resolution."""

from __future__ import annotations

from typing import Any, Mapping

from unity_psf.localization.scale_resolver import PhysicalScaleSpec, resolve_physical_scales
from .optimizer_config import (
    localization_overrides,
    normalize_z_activation,
)
from .paths import mapping
from .types import RuntimeLossConfig

ACTIVE_SMLM_LOSS_PARAM_KEYS = frozenset({"detection_weight", "pxyz_weight", "background_weight", "sigma_weight", "photon_scale", "z_scale", "z_activation", "sigma_min"})
SCALE_METADATA_KEYS = frozenset({"z_scale_unit"})
LEGACY_GMM_LOSS_PARAM_KEYS = frozenset({"gmm_target_chunk", "gmm_component_chunk", "gmm_backend"})
ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS = frozenset({"xyoffset", "ch_weight", "photon_scale", "z_scale", "disable_attr", "gmm_target_chunk", "gmm_component_chunk", "gmm_backend", "target_order", "eps"})

def resolve_loss_config(
    root_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    *,
    model_name: str,
    scales: PhysicalScaleSpec | None = None,
) -> RuntimeLossConfig:
    loss_cfg = train_cfg.get("loss")
    if isinstance(loss_cfg, Mapping) and "name" in loss_cfg:
        name = str(loss_cfg["name"])
        if model_name == "emitter_2d_expert" and name != "active_smlm_gmm_loss":
            raise ValueError("emitter_2d_expert requires active_smlm_gmm_loss with z disabled")
        if model_name == "astigmatism_expert" and name == "active_smlm_loss":
            target_order = str(
                mapping(train_cfg.get("online_generation"), "train.online_generation").get(
                    "pxyz_target_order", "legacy_iwae"
                )
            )
            if target_order != "v03":
                raise ValueError(
                    "astigmatism active_smlm_loss is incompatible with legacy_iwae targets; "
                    "use active_smlm_gmm_loss or set pxyz_target_order: v03"
                )
        if name == "active_smlm_gmm_loss":
            params = active_smlm_gmm_loss_params(train_cfg, mapping(loss_cfg.get("params", {}), "train.loss.params"), scales=scales)
            if model_name == "emitter_2d_expert":
                if int(params.get("disable_attr", 3)) != 3:
                    raise ValueError("emitter_2d loss must disable the z target")
                params["disable_attr"] = 3
            return {
                "name": name,
                "params": params,
            }
        return {
            "name": name,
            "params": dict(mapping(loss_cfg.get("params", {}), "train.loss.params")),
        }
    if model_name == "emitter_2d_expert":
        params = active_smlm_gmm_loss_params(train_cfg, legacy_gmm_loss_params(loss_cfg), scales=scales)
        params["disable_attr"] = 3
        return {"name": "active_smlm_gmm_loss", "params": params}
    if str(model_name).startswith("active_smlm_"):
        raw_loss_keys = set(loss_cfg) if isinstance(loss_cfg, Mapping) else set()
        legacy_gmm_params = legacy_gmm_loss_params(loss_cfg)
        if legacy_gmm_params and raw_loss_keys <= LEGACY_GMM_LOSS_PARAM_KEYS:
            return {"name": "active_smlm_gmm_loss", "params": active_smlm_gmm_loss_params(train_cfg, legacy_gmm_params, scales=scales)}
        params = active_smlm_loss_params(root_cfg, train_cfg, loss_cfg, scales=scales)
        return {"name": "active_smlm_loss", "params": params}
    if str(model_name) in {"astigmatism_expert", "double_helix_expert"}:
        return {
            "name": "active_smlm_gmm_loss",
            "params": active_smlm_gmm_loss_params(train_cfg, legacy_gmm_loss_params(loss_cfg), scales=scales),
        }
    return {"name": "localization_mse", "params": {}}

def active_smlm_loss_params(root_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any], loss_cfg: Any, *, scales: PhysicalScaleSpec | None = None) -> dict[str, Any]:
    raw_params = dict(loss_cfg) if isinstance(loss_cfg, Mapping) else {}
    unknown = sorted(set(raw_params) - ACTIVE_SMLM_LOSS_PARAM_KEYS - LEGACY_GMM_LOSS_PARAM_KEYS - SCALE_METADATA_KEYS)
    if unknown:
        raise ValueError(f"unknown active_smlm_loss params: {unknown}")
    params = {key: raw_params[key] for key in ACTIVE_SMLM_LOSS_PARAM_KEYS if key in raw_params}
    if scales is None:
        scales = resolve_physical_scales(train_cfg)
    if scales.photon_scale is not None:
        params["photon_scale"] = scales.photon_scale
    if scales.z_scale is not None:
        params["z_scale"] = scales.z_scale
    if "z_activation" not in params:
        overrides = localization_overrides(root_cfg, train_cfg)
        if "z_mu_activation" in overrides:
            params["z_activation"] = normalize_z_activation(overrides["z_mu_activation"])
    elif params["z_activation"] is not None:
        params["z_activation"] = normalize_z_activation(params["z_activation"])
    return params

def active_smlm_gmm_loss_params(train_cfg: Mapping[str, Any], raw_params: Mapping[str, Any], *, scales: PhysicalScaleSpec | None = None) -> dict[str, Any]:
    unknown = sorted(set(raw_params) - ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS - SCALE_METADATA_KEYS)
    if unknown:
        raise ValueError(f"unknown active_smlm_gmm_loss params: {unknown}")
    params = {key: raw_params[key] for key in ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS if key in raw_params}
    if scales is None:
        scales = resolve_physical_scales(train_cfg)
    if scales.photon_scale is not None:
        params["photon_scale"] = scales.photon_scale
    if scales.z_scale is not None:
        params["z_scale"] = scales.z_scale
    params.setdefault("target_order", "legacy_iwae")
    return params

def legacy_gmm_loss_params(loss_cfg: Any) -> dict[str, Any]:
    if not isinstance(loss_cfg, Mapping) or "name" in loss_cfg:
        return {}
    return {key: loss_cfg[key] for key in sorted(LEGACY_GMM_LOSS_PARAM_KEYS) if key in loss_cfg}
