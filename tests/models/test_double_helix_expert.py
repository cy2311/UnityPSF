from __future__ import annotations

import torch

from unity_psf.contracts.modality import PSFModality
from unity_psf.data.normalization import LunarInputNormalization
from unity_psf.localization.smlm_output import SMLMOutputChannels, decode_smlm_output
from unity_psf.models.psf_moe.experts.double_helix import DoubleHelixExpert


def _expert() -> DoubleHelixExpert:
    torch.manual_seed(17)
    return DoubleHelixExpert(
        nch_in=3,
        depth_shared=1,
        depth_union=1,
        nfeatures_init=4,
        nfeatures_inter=4,
        input_normalization={
            "name": "lunar",
            "background_range": [110.0, 110.0],
            "photon_range": [0.0, 31000.0],
        },
    )


def test_double_helix_expert_owns_complete_ten_channel_localizer() -> None:
    expert = _expert()
    raw = torch.full((2, 3, 8, 8), 730.0)

    output = expert(raw)

    assert output.shape == (2, SMLMOutputChannels.count, 8, 8)
    assert decode_smlm_output(output).raw.shape == output.shape
    assert expert.input_normalization == LunarInputNormalization(
        background_range=(110.0, 110.0),
        photon_range=(0.0, 31000.0),
    )
    assert not any(name.startswith("stem.") for name, _ in expert.named_parameters())

    output.square().mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in expert.parameters()
        if parameter.requires_grad
    )


def test_double_helix_checkpoint_round_trip_preserves_gmm_contract() -> None:
    expert = _expert().eval()
    metadata = expert.checkpoint_metadata()
    restored = DoubleHelixExpert(**dict(metadata.model_config)).eval()
    restored.load_state_dict(expert.state_dict(), strict=True)
    images = torch.randn(2, 3, 8, 8) * 20.0 + 110.0

    with torch.no_grad():
        expected = expert(images)
        actual = restored(images)

    assert torch.equal(expected, actual)
    assert metadata.expert_type is PSFModality.DOUBLE_HELIX
    assert metadata.model_name == "double_helix_expert"
    assert metadata.model_config["nch_in"] == 3
    assert metadata.model_config["input_normalization"]["name"] == "lunar"
