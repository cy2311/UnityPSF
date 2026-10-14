from __future__ import annotations

from typing import Mapping


def build_liteloc_subfov_tiles(
    *,
    field_height: int,
    field_width: int,
    context_size: int,
    valid_core_size: int,
) -> list[dict[str, int]]:
    if int(valid_core_size) <= 0:
        raise ValueError("valid_core_size must be positive")
    if int(context_size) < int(valid_core_size):
        raise ValueError("context_size must be >= valid_core_size")
    if (int(context_size) - int(valid_core_size)) % 2 != 0:
        raise ValueError("context_size - valid_core_size must be even")
    if int(field_height) < int(context_size) or int(field_width) < int(context_size):
        raise ValueError("field dimensions must be at least context_size")

    cut_edge = (int(context_size) - int(valid_core_size)) // 2
    max_patch_y0 = int(field_height) - int(context_size)
    max_patch_x0 = int(field_width) - int(context_size)

    def bounds(length: int) -> list[int]:
        values = {0, int(length)}
        values.update(range(int(valid_core_size), int(length), int(valid_core_size)))
        return sorted(values)

    tiles: list[dict[str, int]] = []
    y_bounds = bounds(field_height)
    x_bounds = bounds(field_width)
    for valid_y0, valid_y1 in zip(y_bounds[:-1], y_bounds[1:]):
        for valid_x0, valid_x1 in zip(x_bounds[:-1], x_bounds[1:]):
            patch_y0 = min(max(int(valid_y0) - cut_edge, 0), max_patch_y0)
            patch_x0 = min(max(int(valid_x0) - cut_edge, 0), max_patch_x0)
            tiles.append(
                {
                    "tile_index": len(tiles),
                    "patch_y0": int(patch_y0),
                    "patch_x0": int(patch_x0),
                    "keep_y0": int(valid_y0) - int(patch_y0),
                    "keep_x0": int(valid_x0) - int(patch_x0),
                    "keep_h": int(valid_y1) - int(valid_y0),
                    "keep_w": int(valid_x1) - int(valid_x0),
                    "valid_y0": int(valid_y0),
                    "valid_x0": int(valid_x0),
                    "valid_y1": int(valid_y1),
                    "valid_x1": int(valid_x1),
                }
            )
    return tiles


def emitter_in_valid_core(*, x_patch: float, y_patch: float, tile: Mapping[str, int]) -> bool:
    return (
        int(tile["keep_x0"]) <= float(x_patch) < int(tile["keep_x0"]) + int(tile["keep_w"])
        and int(tile["keep_y0"]) <= float(y_patch) < int(tile["keep_y0"]) + int(tile["keep_h"])
    )


def tile_local_to_field_coordinates(
    *,
    x_patch: float,
    y_patch: float,
    tile: Mapping[str, int],
    crop_left: int,
    crop_top: int,
) -> dict[str, float]:
    x_crop = float(tile["patch_x0"]) + float(x_patch)
    y_crop = float(tile["patch_y0"]) + float(y_patch)
    return {
        "x_crop": x_crop,
        "y_crop": y_crop,
        "x_full": x_crop + float(crop_left),
        "y_full": y_crop + float(crop_top),
    }


__all__ = [
    "build_liteloc_subfov_tiles",
    "emitter_in_valid_core",
    "tile_local_to_field_coordinates",
]
