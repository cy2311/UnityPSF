from __future__ import annotations


def pixel_index_image_extent(height: int, width: int) -> tuple[float, float, float, float]:
    """Return the Matplotlib extent for zero-based pixel-index coordinates.

    UnityPSF emitter coordinates use pixel indices as continuous coordinates:
    array element ``(row, col)`` is centered at ``(row, col)``.  The half-pixel
    bounds therefore belong at the image edges, not in the emitter coordinates.
    """
    height_i = int(height)
    width_i = int(width)
    if height_i <= 0 or width_i <= 0:
        raise ValueError("image dimensions must be positive")
    return (-0.5, width_i - 0.5, height_i - 0.5, -0.5)


__all__ = ["pixel_index_image_extent"]
