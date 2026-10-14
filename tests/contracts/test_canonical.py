from __future__ import annotations

import pytest
import torch

from unity_psf.contracts.canonical import (
    CANONICAL_CHECKPOINT_SCHEMA,
    CANONICAL_TRAINING_ENTRYPOINT,
    load_canonical_release_checkpoint,
)
from unity_psf.contracts import (
    InputFrameSpec,
    ModalityChannelState,
    ModalityExpertState,
    ModalityJointCheckpointMetadata,
    save_modality_joint_checkpoint,
)


def test_canonical_training_and_release_contract_are_explicit() -> None:
    assert CANONICAL_TRAINING_ENTRYPOINT == "unity_psf.cli.train_selection"
    assert CANONICAL_CHECKPOINT_SCHEMA == "unity_psf.joint_checkpoint.v2"


def test_canonical_release_loader_rejects_resume_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "resume.ckpt"
    save_modality_joint_checkpoint(
        checkpoint,
        metadata=ModalityJointCheckpointMetadata(
            checkpoint_role="resume",
            supported_modalities=("emitter_2d",),
            supported_channels_per_modality={"emitter_2d": ("left",)},
        ),
        experts=(
            ModalityExpertState(
                modality="emitter_2d",
                model_class="tests.Expert",
                model_config={"condition_dim": 0},
                input_frame_spec=InputFrameSpec(input_frame_channels=3),
                condition_schema={"dimension": 0},
                model_state_dict={"weight": torch.ones(1)},
                channel_states=(ModalityChannelState(channel_id="left"),),
            ),
        ),
        training_state={"emitter_2d": {"epoch": 1}},
    )
    with pytest.raises(ValueError, match="checkpoint_role='release'"):
        load_canonical_release_checkpoint(checkpoint)


def test_canonical_release_loader_rejects_release_without_entrypoint_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "release.ckpt"
    save_modality_joint_checkpoint(
        checkpoint,
        metadata=ModalityJointCheckpointMetadata(
            supported_modalities=("emitter_2d",),
            supported_channels_per_modality={"emitter_2d": ("left",)},
        ),
        experts=(
            ModalityExpertState(
                modality="emitter_2d",
                model_class="tests.Expert",
                model_config={"condition_dim": 0},
                input_frame_spec=InputFrameSpec(input_frame_channels=3),
                condition_schema={"dimension": 0},
                model_state_dict={"weight": torch.ones(1)},
                channel_states=(ModalityChannelState(channel_id="left"),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="training entrypoint"):
        load_canonical_release_checkpoint(checkpoint)
