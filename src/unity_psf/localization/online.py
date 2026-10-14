"""Legacy import shim for the online localization data provider.

New code should import from :mod:`unity_psf.localization.data.online`.
This module only preserves the former import path for external callers.
"""

from unity_psf.field_origin import build_sliding_window_origin_bank
from .data.online_types import OnlineBatchProviderConfig
from .data.online_provider import build_online_batch_provider

__all__ = [
    "OnlineBatchProviderConfig",
    "build_online_batch_provider",
    "build_sliding_window_origin_bank",
]
