from __future__ import annotations

from typing import Any, Mapping

from unity_psf.localization.conditioning import ConditioningProviderStore
from unity_psf.localization.online import OnlineBatchProviderConfig, build_online_batch_provider


def condition_store_from_runtime_config(runtime_config: Mapping[str, Any]) -> ConditioningProviderStore | None:
    batch_provider = runtime_config.get("batch_provider")
    if not isinstance(batch_provider, Mapping):
        return None
    params = batch_provider.get("params")
    if not isinstance(params, Mapping):
        return None
    entries = params.get("dual_domain_coeff_maps")
    if not isinstance(entries, (tuple, list)) or not entries:
        return None
    return ConditioningProviderStore.from_coeff_maps(tuple(entries))


def condition_store_batch_provider_overrides(condition_store: ConditioningProviderStore | None):
    if condition_store is None:
        return None

    def online_train_batch(params: dict[str, object]):
        return build_online_batch_provider(OnlineBatchProviderConfig(**params), condition_store=condition_store)

    return {"online_train_batch": online_train_batch}
