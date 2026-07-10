import numpy as np
import pytest

from neptune_v03.infer_recon.tiling import (
    build_liteloc_subfov_tiles,
    emitter_in_valid_core,
    tile_local_to_field_coordinates,
)


@pytest.mark.parametrize("height,width", [(400, 400), (600, 1200), (413, 617)])
def test_valid_cores_cover_every_field_pixel_exactly_once(height: int, width: int) -> None:
    tiles = build_liteloc_subfov_tiles(
        field_height=height,
        field_width=width,
        context_size=96,
        valid_core_size=80,
    )
    coverage = np.zeros((height, width), dtype=np.int16)

    for tile in tiles:
        coverage[tile["valid_y0"] : tile["valid_y1"], tile["valid_x0"] : tile["valid_x1"]] += 1
        assert 0 <= tile["patch_y0"] <= height - 96
        assert 0 <= tile["patch_x0"] <= width - 96
        assert tile["keep_y0"] + tile["keep_h"] <= 96
        assert tile["keep_x0"] + tile["keep_w"] <= 96

    assert np.all(coverage == 1)


def test_half_open_boundary_assigns_emitter_to_one_tile() -> None:
    tiles = build_liteloc_subfov_tiles(
        field_height=160,
        field_width=160,
        context_size=96,
        valid_core_size=80,
    )
    field_x = 80.0
    field_y = 40.0
    owners = []

    for tile in tiles:
        x_patch = field_x - tile["patch_x0"]
        y_patch = field_y - tile["patch_y0"]
        if emitter_in_valid_core(x_patch=x_patch, y_patch=y_patch, tile=tile):
            owners.append(tile["tile_index"])

    assert owners == [1]


def test_tile_local_coordinates_map_to_crop_and_full_tiff_without_offset() -> None:
    tiles = build_liteloc_subfov_tiles(
        field_height=400,
        field_width=400,
        context_size=96,
        valid_core_size=80,
    )
    tile = tiles[-1]
    x_patch = float(tile["keep_x0"]) + 12.25
    y_patch = float(tile["keep_y0"]) + 7.75

    coordinates = tile_local_to_field_coordinates(
        x_patch=x_patch,
        y_patch=y_patch,
        tile=tile,
        crop_left=700,
        crop_top=400,
    )

    assert coordinates["x_crop"] == pytest.approx(tile["valid_x0"] + 12.25)
    assert coordinates["y_crop"] == pytest.approx(tile["valid_y0"] + 7.75)
    assert coordinates["x_full"] == pytest.approx(700 + tile["valid_x0"] + 12.25)
    assert coordinates["y_full"] == pytest.approx(400 + tile["valid_y0"] + 7.75)


def test_subfov_contract_requires_fixed_context_patches() -> None:
    with pytest.raises(ValueError, match="at least context_size"):
        build_liteloc_subfov_tiles(
            field_height=80,
            field_width=400,
            context_size=96,
            valid_core_size=80,
        )

    with pytest.raises(ValueError, match="even"):
        build_liteloc_subfov_tiles(
            field_height=400,
            field_width=400,
            context_size=95,
            valid_core_size=80,
        )
