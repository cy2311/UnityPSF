from __future__ import annotations

import torch
import pytest

from unity_psf.roi_library import ROIBankDomain
from unity_psf.training.high_fidelity.raw_tiff_inference import (
    build_model_raw_tiff_infer_fn,
    iter_inference_tiles,
    spatial_integration,
)


def test_iter_inference_tiles_preserves_overlap_and_coverage() -> None:
    tiles = iter_inference_tiles(300, 260, tile_size=128, overlap_px=16)

    assert tiles[0] == {
        "x0": 0,
        "x1": 128,
        "y0": 0,
        "y1": 128,
        "keep_x0": 0,
        "keep_x1": 120,
        "keep_y0": 0,
        "keep_y1": 120,
    }
    assert tiles[-1]["x1"] == 260
    assert tiles[-1]["y1"] == 300
    assert tiles[-1]["keep_x1"] == 260
    assert tiles[-1]["keep_y1"] == 300
    assert all(tile["keep_x0"] < tile["keep_x1"] for tile in tiles)
    assert all(tile["keep_y0"] < tile["keep_y1"] for tile in tiles)


def test_iter_inference_tiles_uses_single_tile_when_input_fits() -> None:
    assert iter_inference_tiles(32, 24, tile_size=128, overlap_px=16) == [
        {
            "x0": 0,
            "x1": 24,
            "y0": 0,
            "y1": 32,
            "keep_x0": 0,
            "keep_x1": 24,
            "keep_y0": 0,
            "keep_y1": 32,
        }
    ]


def test_spatial_integration_keeps_probability_shape_and_range() -> None:
    probability = torch.zeros((1, 5, 5), dtype=torch.float32)
    probability[0, 2, 2] = 0.9

    integrated = spatial_integration(probability, raw_threshold=0.3, split_threshold=0.6)

    assert integrated.shape == probability.shape
    assert float(integrated.max()) <= 1.0
    assert float(integrated.min()) >= 0.0
    assert float(integrated[0, 2, 2]) > 0.0


def test_model_raw_tiff_inference_preserves_output_contract_and_training_state() -> None:
    class Model(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            output = torch.zeros((1, 10, images.shape[-2], images.shape[-1]), dtype=torch.float32)
            output[:, 0, 2, 3] = 0.9
            output[:, 1, 2, 3] = 2.0
            output[:, 4, 2, 3] = 0.1
            output[:, 9] = 0.25
            return output

    model = Model()
    model.train()
    infer = build_model_raw_tiff_infer_fn(
        model=model,
        threshold=0.3,
        max_emitters=2,
        expected_channels=3,
        photon_scale=100.0,
        z_scale=1.0,
    )

    result = infer(
        domain=ROIBankDomain("left", crop_left=0, crop_top=0, crop_width=8, crop_height=8),
        frame_window=(4, 7),
        raw_domain_frames_photon=torch.ones((3, 8, 8)).numpy(),
    )

    assert model.training is True
    assert result.metadata == {
        "domain": "left",
        "frame_window": (4, 7),
        "source": "loc_infer_raw_tiff",
    }
    assert result.background_mu.shape == (8, 8)
    assert float(result.background_mu[0, 0]) == 0.25
    assert len(result.emitters) == 1
    assert result.emitters[0].mu_xy_px == (3.0, 2.0)
    assert result.emitters[0].mu_photons == 200.0
    assert result.emitters[0].mu_z_nm == pytest.approx(100.0)
