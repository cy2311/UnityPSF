import torch

from neptune_v03.localization.posterior import select_posterior_emitters
from neptune_v03.localization.smlm_output import SMLMOutputChannels


def _reference_spatial_integration(p: torch.Tensor, *, raw_th: float, split_th: float) -> torch.Tensor:
    filt = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=p.dtype,
    ).view(1, 1, 3, 3)
    conv = torch.nn.functional.conv2d(p.unsqueeze(1), filt, padding=1)
    p_clip = torch.where(p > raw_th, p, torch.zeros_like(p))
    pool = torch.nn.functional.max_pool2d(p_clip.unsqueeze(1), 3, 1, padding=1)
    max_mask1 = torch.eq(p.unsqueeze(1), pool)
    p_ps1 = max_mask1.to(p.dtype) * conv
    p_copy = p.unsqueeze(1) * (1.0 - max_mask1.to(p.dtype))
    p_ps2 = torch.where(p_copy > split_th, torch.ones_like(p_copy), torch.zeros_like(p_copy)) * conv
    return torch.clamp(p_ps1 + p_ps2, 0.0, 1.0).squeeze(1)


def test_posterior_selector_preserves_configurable_legacy_behavior() -> None:
    y_out = torch.zeros((2, SMLMOutputChannels.count, 4, 4), dtype=torch.float32)
    y_out[0, SMLMOutputChannels.p, 1, 1] = 0.55
    y_out[0, SMLMOutputChannels.p, 1, 2] = 0.35
    y_out[0, SMLMOutputChannels.p, 3, 3] = 0.9
    y_out[1, SMLMOutputChannels.p, 2, 2] = 0.8
    y_out[:, SMLMOutputChannels.x_mu] = 0.1
    y_out[:, SMLMOutputChannels.y_mu] = -0.2
    y_out[:, SMLMOutputChannels.z_mu] = 0.25
    y_out[:, SMLMOutputChannels.photons_mu] = 0.5

    selected = select_posterior_emitters(
        y_out,
        threshold=0.5,
        max_emitters=1,
        photon_scale=1000.0,
        z_scale=0.6,
        candidate_threshold=0.4,
        split_threshold=0.7,
    )

    expected_logits = _reference_spatial_integration(y_out[:, 0], raw_th=0.4, split_th=0.7)
    torch.testing.assert_close(selected.logits, expected_logits)
    torch.testing.assert_close(selected.xyzph[0, 0], torch.tensor([3.6, 3.3, 150.0, 500.0]))
    torch.testing.assert_close(selected.xyzph[1, 0], torch.tensor([2.6, 2.3, 150.0, 500.0]))
    assert selected.mask.tolist() == [[True], [True]]
    assert selected.metadata["candidate_threshold"] == 0.4
    assert selected.metadata["split_threshold"] == 0.7
