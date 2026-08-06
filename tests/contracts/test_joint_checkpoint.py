from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unity_psf.contracts.joint_checkpoint import (
    JOINT_CHECKPOINT_SCHEMA_VERSION,
    JointCheckpointMetadata,
    JointExpertKey,
    JointExpertState,
    build_joint_checkpoint,
    load_joint_checkpoint,
    save_joint_checkpoint,
    validate_joint_checkpoint,
)


def _expert(modality: str, channel_id: str, value: float) -> JointExpertState:
    return JointExpertState(
        key=JointExpertKey(modality, channel_id),
        instance_id=f"{modality}-{channel_id}",
        model_class="tests.TinyExpert",
        model_config={"width": 2},
        input_frame_spec={"input_frame_channels": 3, "frame_order": "temporal"},
        condition_schema={"name": f"{modality}_v1", "fields": []},
        model_state_dict={"weight": torch.full((2, 2), value)},
        physical_state={"peak_zmap": torch.full((2, 2), value + 10.0), "gamma_version": 3},
        calibration={"pixel_size_nm": 99.0},
        provenance={"source": f"fixture-{channel_id}"},
    )


def _metadata(role: str = "release") -> JointCheckpointMetadata:
    return JointCheckpointMetadata(
        checkpoint_role=role,
        supported_modalities=("emitter_2d", "astigmatism"),
        code_version="0.4.0-test",
    )


def test_release_joint_checkpoint_round_trip_is_one_self_contained_file(tmp_path: Path) -> None:
    experts = (
        _expert("emitter_2d", "main", 1.0),
        _expert("astigmatism", "left", 2.0),
        _expert("astigmatism", "right", 3.0),
    )
    destination = tmp_path / "unitypsf_joint.ckpt"

    save_joint_checkpoint(destination, metadata=_metadata(), experts=experts)
    payload = load_joint_checkpoint(destination)

    assert destination.is_file()
    assert payload["checkpoint_schema"] == JOINT_CHECKPOINT_SCHEMA_VERSION
    assert set(payload["experts"]) == {
        "emitter_2d:main",
        "astigmatism:left",
        "astigmatism:right",
    }
    assert "training_state" not in payload
    assert not torch.equal(
        payload["experts"]["astigmatism:left"]["physical_state"]["peak_zmap"],
        payload["experts"]["astigmatism:right"]["physical_state"]["peak_zmap"],
    )


def test_resume_joint_checkpoint_requires_training_state_for_every_instance() -> None:
    experts = (_expert("astigmatism", "left", 2.0), _expert("astigmatism", "right", 3.0))
    with pytest.raises(ValueError, match="every expert instance"):
        build_joint_checkpoint(
            metadata=JointCheckpointMetadata(
                checkpoint_role="resume",
                supported_modalities=("astigmatism",),
            ),
            experts=experts,
            training_state={"astigmatism:left": {"optimizer": {}, "global_step": 4}},
        )


def test_release_joint_checkpoint_rejects_optimizer_state() -> None:
    with pytest.raises(ValueError, match="release.*training_state"):
        build_joint_checkpoint(
            metadata=_metadata(),
            experts=(_expert("emitter_2d", "main", 1.0), _expert("astigmatism", "left", 2.0)),
            training_state={"emitter_2d:main": {"optimizer": {}}},
        )


def test_joint_checkpoint_rejects_duplicate_instance_keys() -> None:
    with pytest.raises(ValueError, match="duplicate expert instance key"):
        build_joint_checkpoint(
            metadata=JointCheckpointMetadata(supported_modalities=("astigmatism",)),
            experts=(_expert("astigmatism", "left", 1.0), _expert("astigmatism", "left", 2.0)),
        )


def test_joint_checkpoint_integrity_detects_nested_tensor_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "unitypsf_joint.ckpt"
    save_joint_checkpoint(
        destination,
        metadata=_metadata(),
        experts=(_expert("emitter_2d", "main", 1.0), _expert("astigmatism", "left", 2.0)),
    )
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    payload["experts"]["astigmatism:left"]["physical_state"]["peak_zmap"][0, 0] += 1
    torch.save(payload, destination)

    with pytest.raises(ValueError, match="integrity hash mismatch"):
        load_joint_checkpoint(destination)


def test_validator_rejects_declared_but_absent_modality() -> None:
    payload = build_joint_checkpoint(
        metadata=_metadata(),
        experts=(_expert("emitter_2d", "main", 1.0), _expert("astigmatism", "left", 2.0)),
    )
    payload["metadata"]["supported_modalities"].append("double_helix")

    with pytest.raises(ValueError, match="supported_modalities"):
        validate_joint_checkpoint(payload, verify_integrity=False)
