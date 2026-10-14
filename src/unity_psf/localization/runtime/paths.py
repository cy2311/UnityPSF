"""Small, side-effect-free helpers for resolved runtime configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


PROJECT_REFERENCE_PREFIX = "project://"


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def optional_mapping(value: Any, label: str) -> Mapping[str, Any] | None:
    return None if value is None else mapping(value, label)


def range_from_config(value: Any, *, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    lo, hi = float(value[0]), float(value[1])
    if hi < lo:
        raise ValueError(f"{label} max must be greater than or equal to min")
    return lo, hi


def pair_from_config(value: Any, *, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    return float(value[0]), float(value[1])


def grid_size(value: Any) -> int | tuple[int, int]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return int(value)


def resolve_path(value: str, *, base_dir: Path | None) -> str:
    if value.startswith(PROJECT_REFERENCE_PREFIX):
        if base_dir is None:
            raise ValueError(f"project reference requires a config base directory: {value}")
        relative = Path(value.removeprefix(PROJECT_REFERENCE_PREFIX))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid project reference: {value}")
        for candidate in (base_dir.resolve(), *base_dir.resolve().parents):
            if (candidate / "pyproject.toml").is_file():
                return str((candidate / relative).resolve())
        raise ValueError(f"could not find project root for path: {value}")
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str((base_dir / path).resolve())
