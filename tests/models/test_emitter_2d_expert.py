from __future__ import annotations

from pathlib import Path

import torch

from unity_psf.contracts import (
    JointCheckpointMetadata,
    JointExpertKey,
    JointExpertState,
    save_joint_checkpoint,
)
from unity_psf.localization.film import FiLMConditionedDoubleUNet
from unity_psf.localization.smlm_output import SMLMOutputChannels
from unity_psf.models import UnityPSF
from unity_psf.models.psf_moe.experts.astigmatism import AstigmatismExpert
from unity_psf.models.psf_moe.experts.emitter_2d import Emitter2DExpert, Emitter2DGMMLoss


def _emitter() -> Emitter2DExpert:
    torch.manual_seed(23)
    return Emitter2DExpert(
        nch_in=3,
        depth_shared=1,
        depth_union=1,
        nfeatures_init=4,
        nfeatures_inter=4,
        film_hidden_dim=4,
    )


def _astigmatism() -> AstigmatismExpert:
    return AstigmatismExpert(
        nch_in=3,
        depth_shared=1,
        depth_union=1,
        nfeatures_init=4,
        nfeatures_inter=4,
        condition_dim=4,
        film_hidden_dim=4,
    )


def _joint_state(key: str, expert: torch.nn.Module) -> JointExpertState:
    metadata = expert.checkpoint_metadata()
    return JointExpertState(
        key=JointExpertKey.parse(key),
        instance_id=key.replace(":", "-"),
        model_class=f"{type(expert).__module__}.{type(expert).__name__}",
        model_config=metadata.model_config,
        input_frame_spec=metadata.input_frame_spec,
        condition_schema=metadata.condition_schema,
        model_state_dict=expert.state_dict(),
    )


def test_emitter_2d_owns_complete_film_backbone_and_masks_z() -> None:
    expert = _emitter()
    images = torch.randn(2, 3, 8, 8)

    output = expert(images)

    assert isinstance(expert.backbone, FiLMConditionedDoubleUNet)
    assert output.shape == (2, SMLMOutputChannels.count, 8, 8)
    assert torch.count_nonzero(output[:, SMLMOutputChannels.z_mu]) == 0
    assert torch.allclose(output[:, SMLMOutputChannels.z_sigma], torch.full((2, 8, 8), 0.1))
    assert expert.condition_schema["name"] == "emitter_2d_film_v1"
    assert expert.checkpoint_metadata().expert_type.value == "emitter_2d"


def test_emitter_2d_accepts_optional_conditions_and_trains_film() -> None:
    expert = _emitter()
    images = torch.randn(2, 3, 8, 8)
    conditions = torch.randn(2, expert.condition_dim)

    expert(images, conditions).square().mean().backward()

    film_gradients = [
        parameter.grad
        for name, parameter in expert.named_parameters()
        if "film_modulator" in name
    ]
    assert film_gradients
    assert all(gradient is not None for gradient in film_gradients)


def test_emitter_2d_loss_is_invariant_to_target_z() -> None:
    expert = _emitter().eval()
    output = expert(torch.randn(1, 3, 8, 8))
    detect = torch.zeros(1, 8, 8)
    detect[:, 3, 4] = 1.0
    targets_a = torch.tensor([[[100.0, 4.0, 3.0, -900.0]]])
    targets_b = targets_a.clone()
    targets_b[..., 3] = 900.0
    mask = torch.ones(1, 1, dtype=torch.bool)
    background = torch.zeros(1, 8, 8)
    loss = Emitter2DGMMLoss(gmm_component_chunk=64)

    value_a = loss.forward(output, detect, targets_a, mask, background)
    value_b = loss.forward(output, detect, targets_b, mask, background)

    assert torch.equal(value_a, value_b)


def test_one_joint_checkpoint_loads_2d_main_and_independent_astig_channels(tmp_path: Path) -> None:
    emitter = _emitter().eval()
    left = _astigmatism().eval()
    right = _astigmatism().eval()
    with torch.no_grad():
        next(right.parameters()).add_(0.25)
    checkpoint = tmp_path / "unitypsf_joint.ckpt"
    save_joint_checkpoint(
        checkpoint,
        metadata=JointCheckpointMetadata(),
        experts=(
            _joint_state("emitter_2d:main", emitter),
            _joint_state("astigmatism:left", left),
            _joint_state("astigmatism:right", right),
        ),
    )

    restored = UnityPSF.from_checkpoint(checkpoint).eval()
    images = torch.randn(1, 3, 8, 8)
    conditions = torch.randn(1, 4)

    with torch.no_grad():
        expected_2d = emitter(images)
        expected_left = left(images, conditions)
        actual_2d = restored(images, modality="emitter_2d", channel_id="main")
        actual_left = restored(images, modality="astigmatism", channel_id="left", conditions=conditions)
    assert torch.equal(expected_2d, actual_2d)
    assert torch.equal(expected_left, actual_left)
    assert (
        next(restored.experts["astigmatism:left"].parameters()).data_ptr()
        != next(restored.experts["astigmatism:right"].parameters()).data_ptr()
    )
