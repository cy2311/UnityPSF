"""Shared project-root resolution for double-helix entry points."""

from __future__ import annotations

import os
from pathlib import Path


def _project_root_from_environment() -> Path | None:
    for variable in ("UNITY_V04_ROOT", "NEPTUNE_V04_ROOT"):
        value = os.environ.get(variable)
        if value:
            return Path(value).expanduser().resolve()
    return None


def _project_root_from_working_directory() -> Path | None:
    candidate = Path.cwd().resolve()
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return None


PROJECT_ROOT = (
    _project_root_from_environment()
    or _project_root_from_working_directory()
    or Path(__file__).resolve().parents[2]
)
RACE_ROOT = PROJECT_ROOT.parent


__all__ = ["PROJECT_ROOT", "RACE_ROOT"]
