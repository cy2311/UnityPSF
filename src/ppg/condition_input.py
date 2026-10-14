"""Build PPG FiLM inputs with the same field coordinates as the renderer."""

from __future__ import annotations

from pathlib import Path

import torch

from unity_psf.localization.ppg_conditioning import PPGFieldConditioning


PPG_CONDITION_FIELDS = (
    "field_x",
    "field_y",
    "ppg_pca_0",
    "ppg_pca_1",
    "ppg_pca_2",
    "ppg_pca_3",
)


def load_ppg_conditioning(path: str | Path) -> PPGFieldConditioning:
    return PPGFieldConditioning.from_npz(path)


def ppg_model_input(
    images: torch.Tensor,
    *,
    provider: PPGFieldConditioning | None,
    origin_xy_px: tuple[int, int] = (0, 0),
    source_field_shape_yx: tuple[int, int] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Attach one field condition per image window.

    ``source_field_shape_yx`` is the coordinate frame of the input TIFF.  It
    may differ from the 640x640 PPG map; using normalized window centers keeps
    sample-TIFF detection and simulated training on the same field contract.
    """
    if provider is None:
        return images
    if images.ndim != 4:
        raise ValueError(f"images must have shape (N,C,H,W), got {tuple(images.shape)}")
    source_h, source_w = provider.image_shape_hw if source_field_shape_yx is None else (
        int(source_field_shape_yx[0]),
        int(source_field_shape_yx[1]),
    )
    if min(source_h, source_w) <= 1:
        raise ValueError("source_field_shape_yx must contain dimensions greater than one")
    x0, y0 = (int(origin_xy_px[0]), int(origin_xy_px[1]))
    height, width = int(images.shape[-2]), int(images.shape[-1])
    cx = float(x0) + (float(width) - 1.0) / 2.0
    cy = float(y0) + (float(height) - 1.0) / 2.0
    normalized_xy = torch.tensor(
        [2.0 * cx / float(source_w - 1) - 1.0, 2.0 * cy / float(source_h - 1) - 1.0],
        device=images.device,
        dtype=images.dtype,
    )
    condition = provider.condition_vector_from_normalized_xy(
        normalized_xy,
        device=images.device,
        dtype=images.dtype,
    ).unsqueeze(0).expand(int(images.shape[0]), -1).contiguous()
    return images, condition


__all__ = ["PPG_CONDITION_FIELDS", "load_ppg_conditioning", "ppg_model_input"]
