import torch

from unity_psf.optics.zmap_renderer import add_fixed_astigmatism_anchor


def test_fixed_astigmatism_anchor_adds_only_the_positive_astig_mode() -> None:
    modes = ((2, 0), (2, 2), (2, -2))
    coeffs = torch.zeros((2, 3), dtype=torch.float32)

    result = add_fixed_astigmatism_anchor(
        coeffs,
        mode_order=modes,
        anchor_nm=99.0,
    )

    assert torch.equal(coeffs, torch.zeros_like(coeffs))
    assert torch.all(result[:, 0] == 0.0)
    assert torch.all(result[:, 1] == 99.0)
    assert torch.all(result[:, 2] == 0.0)


def test_fixed_astigmatism_anchor_is_a_noop_when_disabled() -> None:
    coeffs = torch.randn((3, 4), dtype=torch.float32)

    result = add_fixed_astigmatism_anchor(
        coeffs,
        mode_order=((2, 0), (2, 2), (2, -2)),
        anchor_nm=None,
    )

    assert result is coeffs
