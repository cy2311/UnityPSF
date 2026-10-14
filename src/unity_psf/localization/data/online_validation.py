"""Validation helpers for the online batch provider configuration."""

from __future__ import annotations

from typing import Protocol

from unity_psf.runtime.environment import get_env


class OnlineValidationConfig(Protocol):
    @property
    def batch_size(self) -> int:
        ...

    @property
    def steps_per_epoch(self) -> int:
        ...

    @property
    def cached_window_cache_epochs(self) -> int:
        ...

    @property
    def cached_window_cache_frames(self) -> int:
        ...

    @property
    def cached_window_cache_storage(self) -> str:
        ...

    @property
    def field_origin_sampling_mode(self) -> str:
        ...

    @property
    def psf_type(self) -> str:
        ...

    @property
    def simulation_backend(self) -> str:
        ...


def _env_flag(name: str, *, default: bool) -> bool:
    raw = get_env(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def cached_window_precompute_enabled() -> bool:
    return _env_flag("UNITY_V04_CACHED_WINDOW_PRECOMPUTE", default=True)


def resolve_cached_window_cache_epochs(config: OnlineValidationConfig) -> int:
    value = int(config.cached_window_cache_epochs)
    if value <= 0:
        raise ValueError("cached_window_cache_epochs must be positive")
    return value


def resolve_cached_window_cache_frames(config: OnlineValidationConfig) -> int:
    value = int(config.cached_window_cache_frames)
    if value > 0:
        if value % int(config.batch_size) != 0:
            raise ValueError("cached_window_cache_frames must be divisible by batch_size")
        return value
    return int(config.steps_per_epoch) * int(config.batch_size)


def resolve_cached_window_cache_storage(config: OnlineValidationConfig) -> str:
    value = str(config.cached_window_cache_storage or "cpu").strip().lower()
    if value not in {"cpu", "gpu"}:
        raise ValueError("cached_window_cache_storage must be 'cpu' or 'gpu'")
    return value


def resolve_field_origin_sampling_mode(config: OnlineValidationConfig) -> str:
    raw = str(config.field_origin_sampling_mode or "grid").strip().lower()
    aliases = {
        "fixed_grid": "grid",
        "cached_grid": "grid",
        "center_grid": "cell_center_grid",
        "centered_grid": "cell_center_grid",
        "random_step": "per_step_random",
        "step_random": "per_step_random",
        "sliding": "sliding_window",
        "fd_deeploc": "sliding_window",
        "fd-deeploc": "sliding_window",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"grid", "cell_center_grid", "per_step_random", "sliding_window"}:
        raise ValueError(
            "field_origin_sampling_mode must be 'grid', 'cell_center_grid', "
            "'per_step_random', or 'sliding_window'"
        )
    return mode


def validate_online_simulation_modes(config: OnlineValidationConfig) -> None:
    if (
        str(config.psf_type).strip().lower() == "empirical_focal"
        and str(config.simulation_backend).strip().lower() != "native"
    ):
        raise ValueError("empirical_focal PSF requires simulation_backend='native'")


def online_item_seed(*, data_seed: int, epoch: int, item_index: int, items_per_epoch: int) -> int:
    epoch_i = int(epoch)
    item_i = int(item_index)
    count_i = int(items_per_epoch)
    seed_i = int(data_seed)
    if epoch_i <= 0 or item_i < 0 or count_i <= 0 or item_i >= count_i:
        raise ValueError("online seed coordinates are out of range")
    if seed_i < 0 or seed_i >= 2**31:
        raise ValueError("online data_seed must be in [0, 2**31)")
    global_item = (epoch_i - 1) * count_i + item_i
    if global_item >= 2**32:
        raise ValueError("online data stream exceeds the supported 2**32 items")
    return (seed_i << 32) | global_item


__all__ = [
    "cached_window_precompute_enabled",
    "online_item_seed",
    "resolve_cached_window_cache_epochs",
    "resolve_cached_window_cache_frames",
    "resolve_cached_window_cache_storage",
    "resolve_field_origin_sampling_mode",
    "validate_online_simulation_modes",
]
