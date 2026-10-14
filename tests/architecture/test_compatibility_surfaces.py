from __future__ import annotations

from unity_psf.localization.data.online import (
    OnlineBatchProviderConfig,
    build_online_batch_provider,
)
from unity_psf.localization.online import (
    OnlineBatchProviderConfig as LegacyOnlineBatchProviderConfig,
    build_online_batch_provider as legacy_build_online_batch_provider,
)
from unity_psf.localization.runtime.config import (
    build_localization_runtime_config,
    resolve_localization_model_config,
)
from unity_psf.localization.runtime_config import (
    build_localization_runtime_config as legacy_build_localization_runtime_config,
    resolve_localization_model_config as legacy_resolve_localization_model_config,
)


def test_localization_shims_export_canonical_objects() -> None:
    assert LegacyOnlineBatchProviderConfig is OnlineBatchProviderConfig
    assert legacy_build_online_batch_provider is build_online_batch_provider
    assert legacy_build_localization_runtime_config is build_localization_runtime_config
    assert legacy_resolve_localization_model_config is resolve_localization_model_config
