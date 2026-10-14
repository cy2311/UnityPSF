"""Legacy import shim for localization runtime configuration.

New code should import from :mod:`unity_psf.localization.runtime.config`.
This module only preserves the former import path for external callers.
"""

from .runtime.config import build_localization_runtime_config, resolve_localization_model_config

__all__ = ["build_localization_runtime_config", "resolve_localization_model_config"]
