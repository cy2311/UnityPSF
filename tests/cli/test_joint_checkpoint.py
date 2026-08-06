from __future__ import annotations

import json
from pathlib import Path

import torch

from unity_psf.cli.joint_checkpoint import main
from unity_psf.contracts import (
    CheckpointMetadata,
    InputFrameSpec,
    MeasurementChannelSpec,
    ModalityChannelState,
    ModalityExpertState,
    ModalityJointCheckpointMetadata,
    save_checkpoint,
    save_modality_joint_checkpoint,
)
from unity_psf.models import UnityPSF
from unity_psf.models.psf_moe.experts.astigmatism import AstigmatismExpert
from unity_psf.models.psf_moe.experts.emitter_2d import Emitter2DExpert


def _model_config(*, condition_fields: tuple[str, ...]) -> dict[str, object]:
    return {
        "nch_in": 3,
        "depth_shared": 1,
        "depth_union": 1,
        "nfeatures_init": 4,
        "nfeatures_inter": 4,
        "condition_dim": len(condition_fields),
        "condition_fields": list(condition_fields),
        "film_hidden_dim": 4,
    }


def _instance_checkpoint(
    path: Path,
    *,
    expert_type: str,
    channel_id: str,
    seed: int,
) -> Path:
    torch.manual_seed(seed)
    if expert_type == "emitter_2d":
        config = _model_config(condition_fields=("field_x", "field_y"))
        model = Emitter2DExpert(**config)
    else:
        config = _model_config(condition_fields=("zernike_0", "zernike_1", "field_x", "field_y"))
        model = AstigmatismExpert(**config)
    metadata = CheckpointMetadata(
        model_name=f"{expert_type}_expert",
        checkpoint_role="instance",
        expert_type=expert_type,
        model_config=config,
        input_frame_spec=InputFrameSpec(input_frame_channels=3),
        instance_id=channel_id,
        channel_spec=MeasurementChannelSpec(channel_id=channel_id),
        condition_schema={"fields": config["condition_fields"], "dimension": config["condition_dim"]},
        parent_checkpoint_hash=f"{seed:064x}",
    )
    return save_checkpoint(
        path,
        model_state_dict=model.state_dict(),
        metadata=metadata,
        extra={
            "physical_state": {"channel_id": channel_id, "version": seed},
            "calibration": {"anchor_nm": 99.0},
        },
    )


def test_checkpoint_cli_assembles_one_loadable_dual_modality_multichannel_file(
    tmp_path: Path,
    capsys,
) -> None:
    emitter = _instance_checkpoint(
        tmp_path / "emitter.pt",
        expert_type="emitter_2d",
        channel_id="main",
        seed=1,
    )
    left = _instance_checkpoint(
        tmp_path / "left.pt",
        expert_type="astigmatism",
        channel_id="left",
        seed=2,
    )
    right = _instance_checkpoint(
        tmp_path / "right.pt",
        expert_type="astigmatism",
        channel_id="right",
        seed=3,
    )
    output = tmp_path / "unitypsf_joint.ckpt"

    assert main(["assemble", "--output", str(output), str(emitter), str(left), str(right)]) == 0
    assembled = json.loads(capsys.readouterr().out)
    assert assembled["instances"] == ["astigmatism:left", "astigmatism:right", "emitter_2d:main"]
    assert output.is_file()

    assert main(["verify", str(output)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] is True
    assert len(verified["sha256"]) == 64

    model = UnityPSF.from_checkpoint(output)
    images = torch.rand(1, 3, 8, 8)
    assert model.localize(images, modality="emitter_2d", channel_id="main").raw.shape == (1, 10, 8, 8)
    conditions = torch.zeros(1, 4)
    assert model.localize(images, modality="astigmatism", channel_id="left", conditions=conditions).raw.shape == (1, 10, 8, 8)
    assert model.localize(images, modality="astigmatism", channel_id="right", conditions=conditions).raw.shape == (1, 10, 8, 8)
    assert model.physical_states["astigmatism:left"] != model.physical_states["astigmatism:right"]


def test_checkpoint_cli_verifies_modality_routed_v2_checkpoint(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "modality_joint.ckpt"
    save_modality_joint_checkpoint(
        checkpoint,
        metadata=ModalityJointCheckpointMetadata(
            supported_modalities=("emitter_2d",),
            supported_channels_per_modality={"emitter_2d": ("left",)},
        ),
        experts=(
            ModalityExpertState(
                modality="emitter_2d",
                model_class="tests.Emitter2DExpert",
                model_config={"condition_dim": 2},
                input_frame_spec=InputFrameSpec(input_frame_channels=3),
                condition_schema={"dimension": 2},
                model_state_dict={"weight": torch.tensor([1.0])},
                channel_states=(ModalityChannelState(channel_id="left"),),
            ),
        ),
    )

    assert main(["verify", str(checkpoint)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["schema_version"] == "unity_psf.joint_checkpoint.v2"
