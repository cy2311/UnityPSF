import torch

from neptune_v03.localization.legacy_decode import (
    decode_liteloc_eval_emitters,
    decode_liteloc_formal_infer_emitters,
    liteloc_spatial_integration_probability,
)
from neptune_v03.localization.smlm_output import SMLMOutputChannels


def _liteloc_reference_spatial_integration(p: torch.Tensor) -> torch.Tensor:
    p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))[:, None]
    pool = torch.nn.functional.max_pool2d(p_clip, 3, 1, padding=1)
    max_mask1 = torch.eq(p[:, None], pool).float()
    filt = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=p.dtype,
        device=p.device,
    ).view(1, 1, 3, 3)
    conv = torch.nn.functional.conv2d(p[:, None], filt, padding=1, bias=None)
    p_ps1 = max_mask1 * conv
    p_copy = p * (1 - max_mask1[:, 0])
    max_mask2 = torch.where(p_copy > 0.6, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:, None]
    p_ps2 = max_mask2 * conv
    return (p_ps1 + p_ps2)[:, 0]


def test_liteloc_spatial_integration_matches_reference_without_clamp() -> None:
    p = torch.zeros((1, 5, 5), dtype=torch.float32)
    p[0, 2, 2] = 0.8
    p[0, 2, 1] = 0.65
    p[0, 1, 2] = 0.45
    p[0, 4, 4] = 0.31

    actual = liteloc_spatial_integration_probability(p)
    expected = _liteloc_reference_spatial_integration(p)

    torch.testing.assert_close(actual, expected)
    assert float(actual.max()) > 1.0


def test_liteloc_candidate_and_adjacent_thresholds_are_strict() -> None:
    p = torch.zeros((1, 7, 7), dtype=torch.float32)
    p[0, 1, 1] = 0.3
    p[0, 3, 3] = 0.8
    p[0, 3, 4] = 0.6

    actual = liteloc_spatial_integration_probability(p)

    assert float(actual[0, 1, 1]) == 0.0
    assert float(actual[0, 3, 4]) == 0.0


def test_eval_and_formal_infer_use_liteloc_context_thresholds() -> None:
    y_out = torch.zeros((1, SMLMOutputChannels.count, 3, 5), dtype=torch.float32)
    y_out[0, SMLMOutputChannels.p, 1, 1] = 0.4
    y_out[0, SMLMOutputChannels.p, 1, 3] = 0.7

    evaluated = decode_liteloc_eval_emitters(y_out)
    inferred = decode_liteloc_formal_infer_emitters(y_out)

    torch.testing.assert_close(evaluated.probability, torch.tensor([0.4, 0.7]))
    assert int(inferred.probability.numel()) == 0


def test_liteloc_decode_uses_direct_candidate_pixel_values_and_physical_units() -> None:
    y_out = torch.zeros((1, SMLMOutputChannels.count, 3, 3), dtype=torch.float32)
    y_out[0, SMLMOutputChannels.p, 1, 1] = 0.8
    y_out[0, SMLMOutputChannels.p, 1, 2] = 0.4
    y_out[0, SMLMOutputChannels.x_mu, 1, 1] = 0.2
    y_out[0, SMLMOutputChannels.y_mu, 1, 1] = -0.1
    y_out[0, SMLMOutputChannels.z_mu, 1, 1] = 0.25
    y_out[0, SMLMOutputChannels.photons_mu, 1, 1] = 0.5
    y_out[0, SMLMOutputChannels.x_sigma, 1, 1] = 0.3
    y_out[0, SMLMOutputChannels.y_sigma, 1, 1] = 0.4
    y_out[0, SMLMOutputChannels.z_sigma, 1, 1] = 0.2
    y_out[0, SMLMOutputChannels.photons_sigma, 1, 1] = 0.1
    y_out[0, SMLMOutputChannels.x_mu, 1, 2] = -0.45

    emitters = decode_liteloc_formal_infer_emitters(
        y_out,
        z_scale=0.6,
        photon_scale=31000.0,
    )

    torch.testing.assert_close(emitters.xyz_px_nm, torch.tensor([[1.7, 1.4, 150.0]]))
    torch.testing.assert_close(emitters.photons, torch.tensor([15500.0]))
    torch.testing.assert_close(emitters.sigma_xy_px, torch.tensor([[0.3, 0.4]]))
    torch.testing.assert_close(emitters.sigma_z_nm, torch.tensor([120.0]))
    torch.testing.assert_close(emitters.sigma_photons, torch.tensor([3100.0]))
