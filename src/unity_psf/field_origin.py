from __future__ import annotations

import math
from collections.abc import Sequence


def build_sliding_window_origin_bank(
    *,
    field_width_px: int,
    field_height_px: int,
    roi_width_px: int,
    roi_height_px: int,
    stride_px: int,
) -> tuple[tuple[int, int], ...]:
    width = int(field_width_px)
    height = int(field_height_px)
    roi_w = int(roi_width_px)
    roi_h = int(roi_height_px)
    stride = int(stride_px)
    if min(width, height, roi_w, roi_h, stride) <= 0:
        raise ValueError("field dimensions, ROI dimensions, and stride_px must be positive")
    if width < roi_w or height < roi_h:
        raise ValueError("field dimensions must be at least ROI dimensions")
    xs = _sliding_axis_origins(length=width, roi=roi_w, stride=stride)
    ys = _sliding_axis_origins(length=height, roi=roi_h, stride=stride)
    return tuple((int(x), int(y)) for y in ys for x in xs)


def sliding_window_origin_index_for_xy(
    xy_px: tuple[float, float],
    *,
    origins: Sequence[tuple[int, int]],
    roi_width_px: int,
    roi_height_px: int,
) -> int:
    if not origins:
        raise ValueError("origins must be non-empty")
    x, y = float(xy_px[0]), float(xy_px[1])
    roi_w = float(roi_width_px)
    roi_h = float(roi_height_px)
    containing = [
        (idx, origin)
        for idx, origin in enumerate(origins)
        if float(origin[0]) <= x < float(origin[0]) + roi_w and float(origin[1]) <= y < float(origin[1]) + roi_h
    ]
    candidates = containing or list(enumerate(origins))
    best_idx, _best_origin = min(
        candidates,
        key=lambda item: (
            _center_distance_sq(xy_px, item[1], roi_width_px=roi_width_px, roi_height_px=roi_height_px),
            int(item[0]),
        ),
    )
    return int(best_idx)


def _sliding_axis_origins(*, length: int, roi: int, stride: int) -> tuple[int, ...]:
    max_origin = max(0, int(length) - int(roi))
    if max_origin == 0:
        return (0,)
    count = int(math.floor(max_origin / int(stride))) + 1
    values = [min(index * int(stride), max_origin) for index in range(count)]
    if values[-1] != max_origin:
        values.append(max_origin)
    return tuple(dict.fromkeys(int(v) for v in values))


def _center_distance_sq(
    xy_px: tuple[float, float],
    origin_xy_px: tuple[int, int],
    *,
    roi_width_px: int,
    roi_height_px: int,
) -> float:
    cx = float(origin_xy_px[0]) + float(roi_width_px) / 2.0
    cy = float(origin_xy_px[1]) + float(roi_height_px) / 2.0
    dx = float(xy_px[0]) - cx
    dy = float(xy_px[1]) - cy
    return float(dx * dx + dy * dy)
