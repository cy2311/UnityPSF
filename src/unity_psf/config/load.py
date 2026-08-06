from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


Config = dict[str, Any]
PROJECT_REFERENCE_PREFIX = "project://"


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping: {config_path}")
    return dict(data)


def resolve_config_reference(value: str | Path, *, source_path: str | Path) -> Path:
    """Resolve a config reference relative to its file or explicit project root."""
    text = str(value)
    source = Path(source_path).resolve()
    if text.startswith(PROJECT_REFERENCE_PREFIX):
        relative = Path(text.removeprefix(PROJECT_REFERENCE_PREFIX))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid project config reference: {value!s}")
        project_root = _find_project_root(source.parent)
        if project_root is None:
            raise ValueError(f"could not find project root for config: {source}")
        return (project_root / relative).resolve()
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (source.parent / path).resolve()


def _find_project_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    merged: Config = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def materialize_config(base_path: str | Path, *override_paths: str | Path) -> Config:
    config = _materialize_config_paths(load_config(base_path), Path(base_path).resolve().parent)
    for override_path in override_paths:
        override = _materialize_config_paths(load_config(override_path), Path(override_path).resolve().parent)
        config = deep_merge(config, override)
    return config


def _materialize_config_paths(config: Config, base_dir: Path) -> Config:
    resolved = _expand_env_vars(deepcopy(config))
    train_cfg = resolved.get("train")
    if not isinstance(train_cfg, dict):
        return resolved
    online_cfg = train_cfg.get("online_generation")
    if not isinstance(online_cfg, dict):
        return resolved
    maps = online_cfg.get("dual_domain_coeff_maps")
    if not isinstance(maps, list):
        return resolved
    for item in maps:
        if not isinstance(item, dict):
            continue
        for key in ("coeff_maps_npz", "alternating_coeff_maps_npz", "path"):
            if key in item and item[key] not in (None, ""):
                item[key] = _resolve_config_path(str(item[key]), base_dir=base_dir)
    return resolved


def _resolve_config_path(value: str, *, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env_vars(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value
