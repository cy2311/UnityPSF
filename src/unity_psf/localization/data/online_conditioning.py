"""Condition-vector and domain helpers shared by online provider strategies."""

from __future__ import annotations

import torch

from unity_psf.localization.conditioning import condition_feature_order


_CONDITION_FIELD_ALIASES = {
    "zernike_0": "zernike_nm_mean:n2_m0",
    "zernike_1": "zernike_nm_mean:n3_m1",
    "field_x": "x_norm",
    "field_y": "y_norm",
}


def condition_feature_order_for_config(config) -> tuple[str, ...]:
    fields = tuple(str(item) for item in config.condition_fields)
    names = list(fields) if fields else list(condition_feature_order(include_domain_onehot=False, domain_count=0))
    if bool(config.append_domain_onehot):
        names.extend(f"domain_onehot:{idx}" for idx in range(int(config.domain_count)))
    return tuple(names)


def condition_feature_dim(config) -> int:
    if int(config.condition_feature_dim) > 0:
        return int(config.condition_feature_dim)
    if int(config.condition_dim) > 0:
        domain_terms = int(config.domain_count) if bool(config.append_domain_onehot) else 0
        return max(0, int(config.condition_dim) - domain_terms)
    return 8


def condition_vector_for_config(config, vector: torch.Tensor) -> torch.Tensor:
    fields = tuple(str(item) for item in config.condition_fields)
    if not fields:
        return match_feature_dim(vector, feature_dim=condition_feature_dim(config))
    if len(fields) != condition_feature_dim(config):
        raise ValueError("condition_fields length must equal condition_feature_dim")
    source_names = condition_feature_order(include_domain_onehot=False, domain_count=0)
    source_indices = {name: index for index, name in enumerate(source_names)}
    selected = []
    for field in fields:
        source_name = _CONDITION_FIELD_ALIASES.get(field, field)
        if source_name not in source_indices:
            raise ValueError(f"unsupported condition field {field!r}")
        selected.append(source_indices[source_name])
    if max(selected, default=-1) >= int(vector.shape[0]):
        raise ValueError("condition provider returned fewer features than the configured condition fields")
    return vector[selected].contiguous()


def domain_index(config, *, global_step: int) -> int:
    domain_count = int(config.domain_count)
    if domain_count <= 0:
        raise ValueError("domain_count must be positive")
    if str(config.domain_balance_mode) == "alternate_step":
        return int(global_step) % domain_count
    return 0


def match_feature_dim(vector: torch.Tensor, *, feature_dim: int) -> torch.Tensor:
    if int(feature_dim) == int(vector.shape[0]):
        return vector
    if int(feature_dim) < int(vector.shape[0]):
        return vector[: int(feature_dim)].contiguous()
    out = torch.zeros((int(feature_dim),), dtype=vector.dtype, device=vector.device)
    out[: int(vector.shape[0])] = vector
    return out
