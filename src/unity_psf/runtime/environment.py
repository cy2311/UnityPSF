"""Runtime environment compatibility for UnityPSF v0.4."""

from __future__ import annotations

import os


def get_env(name: str, default: str | None = None) -> str | None:
    """Read UnityPSF variables, accepting both historical v0.4 and v0.3 names."""
    if name.startswith("UNITY_V04_"):
        suffix = name[len("UNITY_V04_") :]
        candidates = (name, f"NEPTUNE_V04_{suffix}", f"NEPTUNE_V03_{suffix}")
    elif name.startswith("NEPTUNE_V04_"):
        suffix = name[len("NEPTUNE_V04_") :]
        candidates = (f"UNITY_V04_{suffix}", name, f"NEPTUNE_V03_{suffix}")
    else:
        candidates = (name,)
    for candidate in candidates:
        value = os.environ.get(candidate)
        if value is not None:
            return value
    return default
