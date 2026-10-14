from __future__ import annotations

from pathlib import Path

import torch

from unity_psf.field_origin import build_sliding_window_origin_bank
from unity_psf.localization.ppg_conditioning import PPGFieldConditioning
from ppg.condition_input import ppg_model_input


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PPG_MAP = PROJECT_ROOT / "configs/calibration/ppg/ppg_map_initial.npz"


def test_ppg_provider_uses_full_map_shape_and_interpolates_four_coefficients() -> None:
    provider = PPGFieldConditioning.from_npz(PPG_MAP)

    assert provider.image_shape_hw == (640, 640)
    assert len(
        build_sliding_window_origin_bank(
            field_width_px=provider.image_shape_hw[1],
            field_height_px=provider.image_shape_hw[0],
            roi_width_px=96,
            roi_height_px=96,
            stride_px=88,
        )
    ) == 64

    top_left = provider.condition_vector_from_xy(x0=0, y0=0, height=96, width=96)
    center = provider.condition_vector_from_xy(x0=272, y0=272, height=96, width=96)
    assert top_left.shape == (6,)
    assert torch.isfinite(top_left).all()
    assert not torch.allclose(top_left, center)


def test_ppg_model_input_maps_sample_field_coordinates_to_provider() -> None:
    provider = PPGFieldConditioning.from_npz(PPG_MAP)
    images = torch.zeros((3, 3, 96, 96))
    conditioned = ppg_model_input(
        images,
        provider=provider,
        origin_xy_px=(208, 208),
        source_field_shape_yx=(512, 512),
    )
    assert isinstance(conditioned, tuple)
    assert conditioned[0] is images
    assert conditioned[1].shape == (3, 6)
    assert torch.isfinite(conditioned[1]).all()
