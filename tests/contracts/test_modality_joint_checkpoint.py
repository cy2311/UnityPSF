from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from unity_psf.contracts import (
    InputFrameSpec,
    JointCheckpointMetadata,
    JointExpertKey,
    JointExpertState,
    ModalityChannelState,
    ModalityExpertState,
    ModalityJointCheckpointMetadata,
    build_modality_joint_checkpoint,
    load_legacy_joint_checkpoint_read_only,
    load_modality_joint_checkpoint,
    save_joint_checkpoint,
    save_modality_joint_checkpoint,
)


def _modality_expert(modality: str) -> ModalityExpertState:
    condition_dimension = 2 if modality == "emitter_2d" else 4
    return ModalityExpertState(
        modality=modality,
        model_class=f"tests.{modality}.Expert",
        model_config={"condition_dim": condition_dimension},
        input_frame_spec=InputFrameSpec(input_frame_channels=3),
        condition_schema={"dimension": condition_dimension},
        model_state_dict={"weight": torch.tensor([1.0])},
        channel_states=(
            ModalityChannelState(
                channel_id="left",
                physical_state={"peak_zmap": torch.tensor([1.0])},
                calibration={"version": "left-v1"},
            ),
            ModalityChannelState(
                channel_id="right",
                physical_state={"peak_zmap": torch.tensor([2.0])},
                calibration={"version": "right-v1"},
            ),
        ),
    )


def _legacy_expert(modality: str, channel_id: str) -> JointExpertState:
    key = JointExpertKey(modality, channel_id)
    return JointExpertState(
        key=key,
        instance_id=key.storage_key.replace(":", "-"),
        model_class="tests.LegacyExpert",
        model_config={"condition_dim": 2},
        input_frame_spec=InputFrameSpec(input_frame_channels=3),
        condition_schema={"dimension": 2},
        model_state_dict={"weight": torch.tensor([1.0])},
        physical_state={"peak_zmap": torch.tensor([1.0])},
        calibration={"version": "legacy-v1"},
    )


def test_modality_joint_checkpoint_stores_one_network_per_modality() -> None:
    payload = build_modality_joint_checkpoint(
        metadata=ModalityJointCheckpointMetadata(supported_modalities=("astigmatism",)),
        experts=(_modality_expert("astigmatism"),),
    )

    assert payload["checkpoint_schema"] == "unity_psf.joint_checkpoint.v2"
    assert set(payload["experts"]) == {"astigmatism"}
    assert "model_state_dict" in payload["experts"]["astigmatism"]
    assert set(payload["channel_states"]["astigmatism"]) == {"left", "right"}
    assert payload["router"] == {
        "type": "deterministic",
        "mode": "hard_top1",
        "key": "modality",
    }


def test_modality_joint_checkpoint_round_trip_verifies_nested_channel_integrity(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "unitypsf_joint.ckpt"
    save_modality_joint_checkpoint(
        destination,
        metadata=ModalityJointCheckpointMetadata(supported_modalities=("astigmatism",)),
        experts=(_modality_expert("astigmatism"),),
    )

    payload = load_modality_joint_checkpoint(destination)
    payload["channel_states"]["astigmatism"]["left"]["physical_state"]["peak_zmap"][0] += 1
    torch.save(payload, destination)

    with pytest.raises(ValueError, match="integrity hash mismatch"):
        load_modality_joint_checkpoint(destination)


def test_legacy_joint_checkpoint_import_is_read_only(tmp_path: Path) -> None:
    destination = tmp_path / "legacy_joint.ckpt"
    save_joint_checkpoint(
        destination,
        metadata=JointCheckpointMetadata(supported_modalities=("astigmatism",)),
        experts=(
            _legacy_expert("astigmatism", "left"),
            _legacy_expert("astigmatism", "right"),
        ),
    )
    before = hashlib.sha256(destination.read_bytes()).hexdigest()

    payload = load_legacy_joint_checkpoint_read_only(destination)

    after = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert payload["checkpoint_schema"] == "unity_psf.joint_checkpoint.v1"
    assert set(payload["experts"]) == {"astigmatism:left", "astigmatism:right"}
    assert after == before
