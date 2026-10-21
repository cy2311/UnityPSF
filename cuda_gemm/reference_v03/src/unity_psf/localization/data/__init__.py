"""Localization training data providers."""

from .online_types import OnlineBatchProviderConfig
from .online_provider import build_online_batch_provider

__all__ = ["OnlineBatchProviderConfig", "build_online_batch_provider"]
