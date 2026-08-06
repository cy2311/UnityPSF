"""Localization training data providers."""

from .online import OnlineBatchProviderConfig, build_online_batch_provider

__all__ = ["OnlineBatchProviderConfig", "build_online_batch_provider"]
