from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unity_psf.contracts import (
    CheckpointMetadata,
    InputFrameSpec,
    MeasurementChannelSpec,
    build_checkpoint,
    detect_checkpoint_format,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_payload,
)


def _metadata(*, role: str = "prototype", parent_hash: str | None = None) -> CheckpointMetadata:
    return CheckpointMetadata(
        checkpoint_role=role,
        expert_type="astigmatism",
        model_config={"feature_channels": 8, "condition_dim": 4},
        input_frame_spec=InputFrameSpec(input_frame_channels=3),
        instance_id="astig-left" if role == "instance" else None,
        channel_spec=MeasurementChannelSpec("left"),
        condition_schema={"name": "film_v1", "fields": ["x", "y", "z", "photon"]},
        parent_checkpoint_hash=parent_hash,
        code_version="0.4.0",
    )


def test_v2_prototype_round_trip_uses_training_field_names(tmp_path: Path) -> None:
    metadata = _metadata()
    payload = build_checkpoint(
        {"weight": torch.ones(2)},
        metadata=metadata,
        optimizer_state={"state": {}},
        scheduler_state={"last_epoch": 0},
    )

    assert payload["checkpoint_schema"] == "unity_psf.checkpoint.v2"
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert "scheduler_state_dict" in payload
    assert "model_state" not in payload
    assert detect_checkpoint_format(payload) == "v2"
    assert validate_checkpoint_payload(payload).checkpoint_role == "prototype"

    path = save_checkpoint(tmp_path / "prototype.ckpt", {"weight": torch.ones(2)}, metadata=metadata)
    restored = load_checkpoint(path)
    assert detect_checkpoint_format(restored) == "v2"
    assert restored["metadata"]["expert_type"] == "astigmatism"
    assert tuple(restored["metadata"]["input_frame_spec"]["frame_order"] if isinstance(restored["metadata"]["input_frame_spec"]["frame_order"], list) else (restored["metadata"]["input_frame_spec"]["frame_order"],)) == ("temporal",)


def test_instance_metadata_requires_parent_checkpoint_hash() -> None:
    with pytest.raises(ValueError, match="parent_checkpoint_hash"):
        _metadata(role="instance")

    metadata = _metadata(role="instance", parent_hash="a" * 64)
    assert metadata.instance_id == "astig-left"
    assert metadata.parent_checkpoint_hash == "a" * 64


def test_legacy_training_and_v1_payloads_are_read_only_compatible(tmp_path: Path) -> None:
    legacy_training = {
        "epoch": 4,
        "step_count": 2,
        "global_step": 8,
        "model_state_dict": {"weight": torch.ones(1)},
        "optimizer_state_dict": {},
    }
    legacy_path = tmp_path / "legacy-training.ckpt"
    torch.save(legacy_training, legacy_path)
    restored_training = load_checkpoint(legacy_path)
    assert detect_checkpoint_format(restored_training) == "legacy.training"
    assert "checkpoint_schema" not in restored_training

    v1_payload = {
        "checkpoint_schema": "unity_psf.checkpoint.v1",
        "metadata": {
            "schema_version": "unity_psf.checkpoint.v1",
            "model_family": "UnityPSF",
            "model_name": "psf_moe",
            "experts": ["emitter_2d", "astigmatism", "double_helix"],
        },
        "model_state": {"weight": torch.ones(1)},
    }
    v1_path = tmp_path / "legacy-v1.ckpt"
    torch.save(v1_payload, v1_path)
    restored_v1 = load_checkpoint(v1_path)
    assert detect_checkpoint_format(restored_v1) == "legacy.v1"
    assert validate_checkpoint_payload(restored_v1).schema_version == "unity_psf.checkpoint.v1"
