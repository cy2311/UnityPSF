"""Typed boundary objects returned by localization runtime resolution."""

from __future__ import annotations

from typing import TypedDict


RuntimeParams = dict[str, object]


class RuntimeModelContract(TypedDict, total=False):
    name: str
    output: str
    z_mu_activation: str


class RuntimeBatchProviderContract(TypedDict, total=False):
    name: str
    pxyz_target_order: str
    batch_strategy: str
    sequence_count: int
    condition_dim: int
    condition_feature_dim: int
    condition_fields: list[str] | tuple[str, ...]
    domain_count: int
    append_domain_onehot: bool
    dual_domain_coeff_maps: list[dict[str, str]] | tuple[dict[str, str], ...]


class RuntimeLossContract(TypedDict, total=False):
    name: str
    params: RuntimeParams
    legacy_params: RuntimeParams


class RuntimeNamedConfig(TypedDict):
    name: str
    params: RuntimeParams


class RuntimeProviderConfig(TypedDict):
    name: str
    params: RuntimeParams


class RuntimeLossConfig(TypedDict):
    name: str
    params: RuntimeParams


class RuntimeContract(TypedDict, total=False):
    model_name: str
    model_params: RuntimeParams
    model: RuntimeModelContract
    batch_provider: RuntimeBatchProviderContract
    loss: RuntimeLossContract
    training_runtime: RuntimeParams
    input_frame_spec: dict[str, object]
    channel_layout: dict[str, object]
    expert_instance: dict[str, object]


class RuntimeEpochConfig(TypedDict):
    start: int
    stop: int


class RuntimeConfig(TypedDict, total=False):
    device: str
    model: RuntimeNamedConfig
    optimizer: RuntimeNamedConfig
    batch_provider: RuntimeProviderConfig
    loss: RuntimeLossConfig
    epochs: RuntimeEpochConfig
    resolved_contract: RuntimeContract
    input_frame_spec: dict[str, object]
    channel_layout: dict[str, object]
    expert_instance: dict[str, object]
    max_batches: int
    feedback: dict[str, str]
