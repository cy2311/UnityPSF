"""Small, side-effect-free helpers for resolved runtime configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


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
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str((base_dir / path).resolve())
