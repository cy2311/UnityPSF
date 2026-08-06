from __future__ import annotations

import pytest
import torch

from unity_psf.contracts.modality import PSFModality
from unity_psf.localization.film import FiLMConditionedDoubleUNet
from unity_psf.localization.smlm_output import SMLMOutputChannels, decode_smlm_output
from unity_psf.models.psf_moe import PSFMoE
from unity_psf.models.psf_moe.experts.astigmatism import AstigmatismExpert, LegacyAstigmatismExpert


def _build_expert() -> AstigmatismExpert:
    torch.manual_seed(17)
    return AstigmatismExpert(
        nch_in=3,
        depth_shared=1,
        depth_union=1,
        nfeatures_init=4,
        nfeatures_inter=4,
        condition_dim=4,
        film_hidden_dim=4,
    )


def test_astigmatism_expert_owns_complete_film_localizer_and_schema() -> None:
    expert = _build_expert()

    assert isinstance(expert.backbone, FiLMConditionedDoubleUNet)
    assert expert.condition_dim == 4
    assert expert.condition_schema == {
        "name": "astigmatism_film_v1",
        "fields": ["zernike_0", "zernike_1", "field_x", "field_y"],
        "dimension": 4,
    }
    assert not any(name.startswith("stem.") for name, _ in expert.named_parameters())

    metadata = expert.checkpoint_metadata()
    assert metadata.expert_type is PSFModality.ASTIGMATISM
    assert metadata.condition_schema == expert.condition_schema
    assert metadata.model_config["nch_in"] == 3


def test_astigmatism_expert_returns_decoder_compatible_ten_channel_output_and_gradients() -> None:
    expert = _build_expert()
    images = torch.randn(2, 3, 8, 8)
    conditions = torch.randn(2, 4)

    output = expert(images, conditions)
    assert output.shape == (2, SMLMOutputChannels.count, 8, 8)
    decoded = decode_smlm_output(output)
    assert decoded.raw.shape == output.shape

    output.square().mean().backward()
    film_gradients = [parameter.grad for name, parameter in expert.named_parameters() if "film_modulator" in name]
    assert film_gradients
    assert all(gradient is not None for gradient in film_gradients)


def test_astigmatism_expert_state_round_trip_is_deterministic() -> None:
    expert = _build_expert().eval()
    metadata = expert.checkpoint_metadata()
    restored = AstigmatismExpert(**dict(metadata.model_config)).eval()
    restored.load_state_dict(expert.state_dict(), strict=True)
    images = torch.randn(2, 3, 8, 8)
    conditions = torch.randn(2, 4)

    with torch.no_grad():
        expected = expert(images, conditions)
        actual = restored(images, conditions)

    assert torch.equal(expected, actual)


def test_astigmatism_expert_requires_explicit_image_and_condition_contract() -> None:
    expert = _build_expert()
    images = torch.randn(2, 3, 8, 8)

    with pytest.raises(TypeError, match="conditions"):
        expert(images)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="condition_dim"):
        expert(images, torch.zeros(2, 3))
    with pytest.raises(ValueError, match="nch_in"):
        expert(torch.randn(2, 2, 8, 8), torch.zeros(2, 4))
    with pytest.raises(TypeError, match="condition_fields"):
        AstigmatismExpert(condition_fields="zernike")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="activation"):
        AstigmatismExpert(activation=torch.nn.PReLU())  # type: ignore[arg-type]


def test_legacy_psf_moe_keeps_feature_adapter_separate_from_complete_expert() -> None:
    model = PSFMoE(in_channels=3, feature_channels=4)
    assert isinstance(model.experts[PSFModality.ASTIGMATISM.value], LegacyAstigmatismExpert)
    images = torch.randn(1, 3, 8, 8)
    output = model(images, PSFModality.ASTIGMATISM)
    assert output.detection_logits.shape == (1, 8, 8)
