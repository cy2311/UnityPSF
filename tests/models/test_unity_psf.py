from __future__ import annotations

import pytest
import torch
from torch import nn

from unity_psf.contracts import (
    InputFrameSpec,
    ModalityChannelState,
    ModalityExpertState,
    ModalityJointCheckpointMetadata,
    save_modality_joint_checkpoint,
)
from unity_psf.models import UnityPSF
from unity_psf.models.psf_moe.experts.emitter_2d import Emitter2DExpert
from unity_psf.models.psf_moe.router import ModalityRouter


class _TinyEmitter(nn.Module):
    def __init__(self, marker: float) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(marker))
        self.calls = 0

    def forward(
        self,
        images: torch.Tensor,
        conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        condition_value = 0.0 if conditions is None else conditions[:, :1].reshape(-1, 1, 1, 1)
        return images.new_ones((images.shape[0], 10, *images.shape[-2:])) * (
            self.marker + condition_value
        )


class _TinyAstigmatism(nn.Module):
    condition_dim = 2

    def __init__(self, marker: float) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(marker))
        self.calls = 0

    def forward(self, images: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return images.new_ones((images.shape[0], 10, *images.shape[-2:])) * (
            self.marker + conditions[:, :1].reshape(-1, 1, 1, 1)
        )


def _model() -> UnityPSF:
    return UnityPSF(
        {
            "emitter_2d": _TinyEmitter(1.0),
            "astigmatism": _TinyAstigmatism(2.0),
        },
        channel_states={
            "emitter_2d": {
                "main": {"physical_state": {"anchor": "zero_aberration"}},
            },
            "astigmatism": {
                "left": {
                    "physical_state": {"peak_zmap": "left-zmap"},
                    "calibration": {"version": "left-v1"},
                },
                "right": {
                    "physical_state": {"peak_zmap": "right-zmap"},
                    "calibration": {"version": "right-v1"},
                },
            },
        },
    )


def test_modality_router_resolves_only_the_psf_modality() -> None:
    router = ModalityRouter(("emitter_2d", "astigmatism"))

    assert router.resolve("2d").value == "emitter_2d"
    assert router.resolve("astig").value == "astigmatism"
    with pytest.raises(ValueError, match="unsupported UnityPSF modality"):
        router.resolve("double_helix")


def test_unity_psf_routes_left_and_right_to_the_same_modality_expert() -> None:
    model = _model()
    images = torch.zeros(2, 3, 8, 8)
    left_conditions = torch.tensor([[0.5, 0.0], [1.0, 0.0]])
    right_conditions = torch.tensor([[1.5, 0.0], [2.0, 0.0]])

    left = model.localize(
        images,
        modality="astigmatism",
        channel_id="left",
        conditions=left_conditions,
    )
    right = model.localize(
        images,
        modality="astigmatism",
        channel_id="right",
        conditions=right_conditions,
    )

    assert left.channel_id == "left"
    assert right.channel_id == "right"
    assert torch.allclose(left.raw[:, 0, 0, 0], torch.tensor([2.5, 3.0]))
    assert torch.allclose(right.raw[:, 0, 0, 0], torch.tensor([3.5, 4.0]))
    assert model.experts["astigmatism"].calls == 2
    assert model.experts["emitter_2d"].calls == 0
    assert model.activation_audit() == {"astigmatism": 2}


def test_unity_psf_keeps_channel_physical_and_calibration_states_independent() -> None:
    model = _model()

    left = model.channel_state("astigmatism", "left")
    right = model.channel_state("astigmatism", "right")

    assert left["physical_state"]["peak_zmap"] == "left-zmap"
    assert right["physical_state"]["peak_zmap"] == "right-zmap"
    assert left["calibration"]["version"] == "left-v1"
    assert right["calibration"]["version"] == "right-v1"
    assert left is not right


def test_unity_psf_rejects_unknown_channel_inside_the_selected_modality() -> None:
    model = _model()

    with pytest.raises(ValueError, match="unsupported channel"):
        model.localize(
            torch.zeros(1, 3, 8, 8),
            modality="astigmatism",
            channel_id="missing",
            conditions=torch.zeros(1, 2),
        )


def test_unity_psf_2d_output_marks_z_invalid_and_accepts_optional_conditions() -> None:
    model = _model()
    images = torch.zeros(1, 3, 8, 8)

    result = model.localize(images, modality="emitter_2d", channel_id="main")

    assert result.z_valid is False
    assert torch.count_nonzero(result.decoded.z_mu) == 0
    conditioned = model(images, modality="emitter_2d", channel_id="main", conditions=torch.zeros(1, 2))
    assert conditioned.shape == result.raw.shape


def test_unity_psf_astigmatism_requires_channel_local_conditions() -> None:
    model = _model()
    images = torch.zeros(1, 3, 8, 8)

    with pytest.raises(ValueError, match="requires FiLM conditions"):
        model(images, modality="astigmatism", channel_id="left")
    with pytest.raises(ValueError, match="condition batch size"):
        model(
            images,
            modality="astigmatism",
            channel_id="left",
            conditions=torch.zeros(2, 2),
        )


def test_unity_psf_describe_reports_two_experts_and_their_channels() -> None:
    description = _model().describe()

    assert description["model_family"] == "UnityPSF"
    assert description["supported_modalities"] == ["emitter_2d", "astigmatism"]
    assert description["experts"] == ["astigmatism", "emitter_2d"]
    assert description["supported_channels"] == {
        "astigmatism": ["left", "right"],
        "emitter_2d": ["main"],
    }


def test_unity_psf_loads_v2_checkpoint_with_one_network_and_nested_channels(tmp_path) -> None:
    expert = Emitter2DExpert(
        nch_in=3,
        depth_shared=1,
        depth_union=1,
        nfeatures_init=4,
        nfeatures_inter=4,
        film_hidden_dim=4,
    ).eval()
    metadata = expert.checkpoint_metadata()
    checkpoint = tmp_path / "unitypsf_joint.ckpt"
    save_modality_joint_checkpoint(
        checkpoint,
        metadata=ModalityJointCheckpointMetadata(supported_modalities=("emitter_2d",)),
        experts=(
            ModalityExpertState(
                modality="emitter_2d",
                model_class=f"{type(expert).__module__}.{type(expert).__name__}",
                model_config=metadata.model_config,
                input_frame_spec=InputFrameSpec(input_frame_channels=3),
                condition_schema=metadata.condition_schema,
                model_state_dict=expert.state_dict(),
                channel_states=(
                    ModalityChannelState(
                        channel_id="left",
                        physical_state={"anchor": "left"},
                        calibration={"version": "left-v1"},
                    ),
                    ModalityChannelState(
                        channel_id="right",
                        physical_state={"anchor": "right"},
                        calibration={"version": "right-v1"},
                    ),
                ),
            ),
        ),
    )

    restored = UnityPSF.from_checkpoint(checkpoint).eval()
    images = torch.randn(1, 3, 8, 8)

    with torch.no_grad():
        expected = expert(images)
        actual_left = restored(images, modality="emitter_2d", channel_id="left")
        actual_right = restored(images, modality="emitter_2d", channel_id="right")

    assert torch.equal(actual_left, expected)
    assert torch.equal(actual_right, expected)
    assert list(restored.experts) == ["emitter_2d"]
    assert restored.describe()["supported_channels"] == {"emitter_2d": ["left", "right"]}
