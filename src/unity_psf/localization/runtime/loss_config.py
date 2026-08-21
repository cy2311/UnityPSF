from __future__ import annotations

from typing import Any, Mapping

from .conditioning_config import is_astigmatism_expert as _is_astigmatism_expert
from .optimizer_config import localization_overrides as _localization_overrides, normalize_z_activation as _normalize_z_activation
from .paths import mapping as _mapping


_ACTIVE_SMLM_LOSS_PARAM_KEYS = frozenset(
    {
        "detection_weight",
        "pxyz_weight",
        "background_weight",
        "sigma_weight",
        "photon_scale",
        "z_scale",
        "z_activation",
        "sigma_min",
    }
)
_LEGACY_GMM_LOSS_PARAM_KEYS = frozenset({"gmm_target_chunk", "gmm_component_chunk", "gmm_backend"})
_ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS = frozenset(
    {
        "xyoffset",
        "ch_weight",
        "photon_scale",
        "z_scale",
        "disable_attr",
        "gmm_target_chunk",
        "gmm_component_chunk",
        "gmm_backend",
        "target_order",
        "eps",
    }
)


def _loss_config(root_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any], *, model_name: str) -> dict[str, Any]:
    if model_name == "double_helix_expert":
        loss_cfg = train_cfg.get("loss")
        params = dict(loss_cfg.get("params", {})) if isinstance(loss_cfg, Mapping) else {}
        return {"name": "dh_direct_xyz_loss", "params": params}
    loss_cfg = train_cfg.get("loss")
    if isinstance(loss_cfg, Mapping) and "name" in loss_cfg:
        name = str(loss_cfg["name"])
        if model_name == "emitter_2d_expert" and name != "active_smlm_gmm_loss":
            raise ValueError("emitter_2d_expert requires active_smlm_gmm_loss with z disabled")
        if model_name == "astigmatism_expert" and name == "active_smlm_loss":
            target_order = str(
                _mapping(train_cfg.get("online_generation"), "train.online_generation").get(
                    "pxyz_target_order", "legacy_iwae"
                )
            )
            if target_order != "v03":
                raise ValueError(
                    "astigmatism active_smlm_loss is incompatible with legacy_iwae targets; "
                    "use active_smlm_gmm_loss or set pxyz_target_order: v03"
                )
        if name == "active_smlm_gmm_loss":
            params = _active_smlm_gmm_loss_params(train_cfg, _mapping(loss_cfg.get("params", {}), "train.loss.params"))
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
            "params": dict(_mapping(loss_cfg.get("params", {}), "train.loss.params")),
        }
    if model_name == "emitter_2d_expert":
        params = _active_smlm_gmm_loss_params(train_cfg, _legacy_gmm_loss_params(loss_cfg))
        params["disable_attr"] = 3
        return {"name": "active_smlm_gmm_loss", "params": params}
    if str(model_name).startswith("active_smlm_"):
        raw_loss_keys = set(loss_cfg) if isinstance(loss_cfg, Mapping) else set()
        legacy_gmm_params = _legacy_gmm_loss_params(loss_cfg)
        if legacy_gmm_params and raw_loss_keys <= _LEGACY_GMM_LOSS_PARAM_KEYS:
            return {"name": "active_smlm_gmm_loss", "params": _active_smlm_gmm_loss_params(train_cfg, legacy_gmm_params)}
        params = _active_smlm_loss_params(root_cfg, train_cfg, loss_cfg)
        return {"name": "active_smlm_loss", "params": params}
    if str(model_name) == "astigmatism_expert":
        return {
            "name": "active_smlm_gmm_loss",
            "params": _active_smlm_gmm_loss_params(train_cfg, _legacy_gmm_loss_params(loss_cfg)),
        }
    return {"name": "localization_mse", "params": {}}


def _active_smlm_loss_params(root_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any], loss_cfg: Any) -> dict[str, Any]:
    raw_params = dict(loss_cfg) if isinstance(loss_cfg, Mapping) else {}
    unknown = sorted(set(raw_params) - _ACTIVE_SMLM_LOSS_PARAM_KEYS - _LEGACY_GMM_LOSS_PARAM_KEYS)
    if unknown:
        raise ValueError(f"unknown active_smlm_loss params: {unknown}")
    params = {key: raw_params[key] for key in _ACTIVE_SMLM_LOSS_PARAM_KEYS if key in raw_params}
    if "photon_scale" not in params:
        scaling_cfg = train_cfg.get("scaling")
        normalization_cfg = train_cfg.get("normalization")
        if isinstance(scaling_cfg, Mapping) and "photon_max" in scaling_cfg:
            params["photon_scale"] = float(scaling_cfg["photon_max"])
        elif isinstance(normalization_cfg, Mapping) and "photon_scale" in normalization_cfg:
            params["photon_scale"] = float(normalization_cfg["photon_scale"])
    if "z_scale" not in params:
        scaling_cfg = train_cfg.get("scaling")
        if isinstance(scaling_cfg, Mapping) and "z_max" in scaling_cfg:
            params["z_scale"] = float(scaling_cfg["z_max"])
        else:
            train_params_cfg = train_cfg.get("train_params")
            if isinstance(train_params_cfg, Mapping) and "z_max" in train_params_cfg:
                params["z_scale"] = float(train_params_cfg["z_max"])
    if "z_activation" not in params:
        overrides = _localization_overrides(root_cfg, train_cfg)
        if "z_mu_activation" in overrides:
            params["z_activation"] = _normalize_z_activation(overrides["z_mu_activation"])
    elif params["z_activation"] is not None:
        params["z_activation"] = _normalize_z_activation(params["z_activation"])
    return params


def _active_smlm_gmm_loss_params(train_cfg: Mapping[str, Any], raw_params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(raw_params) - _ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS)
    if unknown:
        raise ValueError(f"unknown active_smlm_gmm_loss params: {unknown}")
    params = {key: raw_params[key] for key in _ACTIVE_SMLM_GMM_LOSS_PARAM_KEYS if key in raw_params}
    if "photon_scale" not in params:
        scaling_cfg = train_cfg.get("scaling")
        normalization_cfg = train_cfg.get("normalization")
        if isinstance(scaling_cfg, Mapping) and "photon_max" in scaling_cfg:
            params["photon_scale"] = float(scaling_cfg["photon_max"])
        elif isinstance(normalization_cfg, Mapping) and "photon_scale" in normalization_cfg:
            params["photon_scale"] = float(normalization_cfg["photon_scale"])
    if "z_scale" not in params:
        scaling_cfg = train_cfg.get("scaling")
        if isinstance(scaling_cfg, Mapping) and "z_max" in scaling_cfg:
            params["z_scale"] = float(scaling_cfg["z_max"])
        else:
            train_params_cfg = train_cfg.get("train_params")
            if isinstance(train_params_cfg, Mapping) and "z_max" in train_params_cfg:
                params["z_scale"] = float(train_params_cfg["z_max"])
    params.setdefault("target_order", "legacy_iwae")
    return params


def _legacy_gmm_loss_params(loss_cfg: Any) -> dict[str, Any]:
    if not isinstance(loss_cfg, Mapping) or "name" in loss_cfg:
        return {}
    return {key: loss_cfg[key] for key in sorted(_LEGACY_GMM_LOSS_PARAM_KEYS) if key in loss_cfg}
