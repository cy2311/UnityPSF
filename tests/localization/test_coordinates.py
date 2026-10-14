from __future__ import annotations

import pytest

from unity_psf.localization.coordinates import pixel_index_image_extent


def test_pixel_index_extent_keeps_array_centers_at_integer_coordinates() -> None:
    assert pixel_index_image_extent(96, 128) == (-0.5, 127.5, 95.5, -0.5)


@pytest.mark.parametrize("height,width", [(0, 8), (8, 0), (-1, 8)])
def test_pixel_index_extent_rejects_empty_images(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        pixel_index_image_extent(height, width)
