"""Field-origin sampling geometry."""

from __future__ import annotations

import numpy as np
from unity_psf.field_origin import build_sliding_window_origin_bank
from unity_psf.localization.conditioning import FullResZernikeConditioning
from .online_types import OnlineBatchProviderConfig
from .online_validation import resolve_field_origin_sampling_mode


def cached_sequence_condition_origin(
    config: OnlineBatchProviderConfig,
    *,
    condition_provider: FullResZernikeConditioning,
    epoch: int,
    sequence_idx: int,
    sequence_count: int,
) -> tuple[int, int]:
    full_h, full_w = condition_provider.image_shape_hw
    roi_w = min(int(config.width), int(full_w))
    roi_h = min(int(config.height), int(full_h))
    grid = config.nat_grid_size
    if isinstance(grid, tuple):
        grid_x = max(1, int(grid[0]))
        grid_y = max(1, int(grid[1] if len(grid) > 1 else grid[0]))
    else:
        grid_x = grid_y = max(1, int(grid))
    cell_count = max(1, grid_x * grid_y)
    cell = (int(epoch) - 1) * int(sequence_count) + int(sequence_idx)
    cell = int(cell) % cell_count
    col = cell % grid_x
    row = cell // grid_x
    mode = resolve_field_origin_sampling_mode(config)
    x0 = grid_axis_origin(
        cell_index=col,
        grid_size=grid_x,
        full_size=int(full_w),
        roi_size=roi_w,
        centered=mode == "cell_center_grid",
    )
    y0 = grid_axis_origin(
        cell_index=row,
        grid_size=grid_y,
        full_size=int(full_h),
        roi_size=roi_h,
        centered=mode == "cell_center_grid",
    )
    return int(x0), int(y0)


def sliding_window_sequence_condition_origin(
    config: OnlineBatchProviderConfig,
    *,
    condition_provider: FullResZernikeConditioning,
    epoch: int,
    sequence_idx: int,
    sequence_count: int,
) -> tuple[tuple[int, int], int, int]:
    full_h, full_w = condition_provider.image_shape_hw
    roi_w = min(int(config.width), int(full_w))
    roi_h = min(int(config.height), int(full_h))
    origins = build_sliding_window_origin_bank(
        field_width_px=int(full_w),
        field_height_px=int(full_h),
        roi_width_px=roi_w,
        roi_height_px=roi_h,
        stride_px=int(config.field_origin_stride_px),
    )
    order = np.arange(len(origins), dtype=np.int64)
    seed = (int(config.seed) + 1_000_003 * int(epoch)) % (2**63 - 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    offset = ((int(epoch) - 1) * int(sequence_count) + int(sequence_idx)) % len(origins)
    origin_index = int(order[int(offset)])
    return tuple(int(v) for v in origins[origin_index]), origin_index, int(len(origins))


def random_sequence_condition_origin(
    config: OnlineBatchProviderConfig,
    *,
    condition_provider: FullResZernikeConditioning,
    epoch: int,
    sequence_idx: int,
    domain_index: int,
) -> tuple[int, int]:
    full_h, full_w = condition_provider.image_shape_hw
    roi_w = min(int(config.width), int(full_w))
    roi_h = min(int(config.height), int(full_h))
    max_x0 = max(0, int(full_w) - roi_w)
    max_y0 = max(0, int(full_h) - roi_h)
    seed = (
        int(config.seed)
        + 1_000_003 * int(epoch)
        + 97_531 * int(sequence_idx)
        + 12_289 * int(domain_index)
    ) % (2**63 - 1)
    rng = np.random.default_rng(seed)
    x0 = int(rng.integers(0, max_x0 + 1)) if max_x0 > 0 else 0
    y0 = int(rng.integers(0, max_y0 + 1)) if max_y0 > 0 else 0
    return x0, y0


def condition_roi_extent(axis_size: int, grid_size: int) -> int:
    return max(1, min(int(axis_size), int(axis_size) // max(1, int(grid_size))))


def grid_axis_origin(
    *,
    cell_index: int,
    grid_size: int,
    full_size: int,
    roi_size: int,
    centered: bool,
) -> int:
    max_origin = max(0, int(full_size) - int(roi_size))
    if centered:
        cell_center = (float(cell_index) + 0.5) * float(full_size) / float(grid_size)
        return int(min(max_origin, max(0, round(cell_center - float(roi_size) / 2.0))))
    if int(grid_size) <= 1:
        return 0
    return int(round(float(cell_index) * float(max_origin) / float(grid_size - 1)))


def condition_origin(
    config: OnlineBatchProviderConfig,
    *,
    sample_index: int,
    global_step: int,
    roi_width: int,
    roi_height: int,
) -> tuple[int, int]:
    grid = config.nat_grid_size
    if isinstance(grid, tuple):
        grid_x = max(1, int(grid[0]))
        grid_y = max(1, int(grid[1] if len(grid) > 1 else grid[0]))
    else:
        grid_x = grid_y = max(1, int(grid))
    cell = (int(global_step) * int(config.batch_size) + int(sample_index)) % (grid_x * grid_y)
    col = cell % grid_x
    row = cell // grid_x
    mode = resolve_field_origin_sampling_mode(config)
    x0 = grid_axis_origin(
        cell_index=col,
        grid_size=grid_x,
        full_size=int(config.width),
        roi_size=roi_width,
        centered=mode == "cell_center_grid",
    )
    y0 = grid_axis_origin(
        cell_index=row,
        grid_size=grid_y,
        full_size=int(config.height),
        roi_size=roi_height,
        centered=mode == "cell_center_grid",
    )
    return int(x0), int(y0)
