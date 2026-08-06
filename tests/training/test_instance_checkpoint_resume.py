from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from unity_psf.contracts.checkpoint import CheckpointMetadata, save_checkpoint
from unity_psf.contracts.modality import InputFrameSpec, MeasurementChannelSpec, PSFModality
from unity_psf.runtime.layout import ensure_run_layout
from unity_psf.training.loop import (
    EpochTrainingConfig,
    TrainingBatch,
    TrainingConfig,
    initialize_model_from_checkpoint,
    load_training_checkpoint,
    resume_training_checkpoint,
    train_epochs,
    train_one_epoch,
)


def _metadata(*, instance_id: str = "left", channel_id: str = "left", parent_hash: str | None = None) -> CheckpointMetadata:
    return CheckpointMetadata(
        model_name="astigmatism_expert",
        checkpoint_role="instance",
        expert_type=PSFModality.ASTIGMATISM,
        model_config={"in_features": 2},
        input_frame_spec=InputFrameSpec(input_frame_channels=3),
        instance_id=instance_id,
        channel_spec=MeasurementChannelSpec(channel_id),
        condition_schema={"name": "film_v1", "fields": ["z"]},
        parent_checkpoint_hash=parent_hash or "a" * 64,
        experts=(PSFModality.ASTIGMATISM.value,),
        shared_feature_channels=4,
    )


def _prototype_metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        model_name="astigmatism_expert",
        checkpoint_role="prototype",
        expert_type=PSFModality.ASTIGMATISM,
        model_config={"in_features": 2},
        input_frame_spec=InputFrameSpec(input_frame_channels=3),
        condition_schema={"name": "film_v1", "fields": ["z"]},
        experts=(PSFModality.ASTIGMATISM.value,),
        shared_feature_channels=4,
    )


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))


def _batch_provider(epoch: int):
    del epoch
    inputs = torch.randn(4, 2)
    targets = torch.randn(4, 1)
    return [TrainingBatch(inputs=inputs, targets=targets)]


def test_instance_checkpoint_saves_and_restores_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    metadata = _metadata()
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    uninterrupted = _model()
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=0.01)
    uninterrupted_scheduler = torch.optim.lr_scheduler.StepLR(uninterrupted_optimizer, step_size=1, gamma=0.9)
    uninterrupted_layout = ensure_run_layout(tmp_path, "uninterrupted")
    train_epochs(
        model=uninterrupted,
        optimizer=uninterrupted_optimizer,
        scheduler=uninterrupted_scheduler,
        batch_provider=_batch_provider,
        layout=uninterrupted_layout,
        config=EpochTrainingConfig(start_epoch=1, stop_epoch=2, checkpoint_metadata=metadata),
    )
    uninterrupted_torch_rng = torch.get_rng_state().clone()
    uninterrupted_numpy_rng = np.random.get_state()
    uninterrupted_python_rng = random.getstate()

    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    interrupted = _model()
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=0.01)
    interrupted_scheduler = torch.optim.lr_scheduler.StepLR(interrupted_optimizer, step_size=1, gamma=0.9)
    interrupted_layout = ensure_run_layout(tmp_path, "interrupted")
    first = train_epochs(
        model=interrupted,
        optimizer=interrupted_optimizer,
        scheduler=interrupted_scheduler,
        batch_provider=_batch_provider,
        layout=interrupted_layout,
        config=EpochTrainingConfig(start_epoch=1, stop_epoch=1, checkpoint_metadata=metadata),
    )[-1]

    resumed = _model()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.9)
    resume = resume_training_checkpoint(
        first.checkpoint_path,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_metadata=metadata,
    )
    assert resume.checkpoint_format == "v2"
    assert resume.metadata is not None
    assert resume.metadata.instance_id == "left"
    assert resume.global_step == 1
    train_epochs(
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        batch_provider=_batch_provider,
        layout=interrupted_layout,
        config=EpochTrainingConfig(
            start_epoch=2,
            stop_epoch=2,
            global_step_start=resume.global_step,
            checkpoint_metadata=metadata,
        ),
    )

    for name, value in uninterrupted.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    assert uninterrupted_scheduler.state_dict() == resumed_scheduler.state_dict()
    assert uninterrupted_optimizer.state_dict().keys() == resumed_optimizer.state_dict().keys()
    assert uninterrupted_optimizer.state_dict()["param_groups"] == resumed_optimizer.state_dict()["param_groups"]
    uninterrupted_states = uninterrupted_optimizer.state_dict()["state"]
    resumed_states = resumed_optimizer.state_dict()["state"]
    assert uninterrupted_states.keys() == resumed_states.keys()
    for parameter_id, uninterrupted_state in uninterrupted_states.items():
        for key, value in uninterrupted_state.items():
            resumed_value = resumed_states[parameter_id][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, resumed_value)
            else:
                assert value == resumed_value
    assert torch.equal(uninterrupted_torch_rng, torch.get_rng_state())
    resumed_numpy_rng = np.random.get_state()
    assert uninterrupted_numpy_rng[0] == resumed_numpy_rng[0]
    assert np.array_equal(uninterrupted_numpy_rng[1], resumed_numpy_rng[1])
    assert uninterrupted_numpy_rng[2:] == resumed_numpy_rng[2:]
    assert uninterrupted_python_rng == random.getstate()


def test_instance_checkpoint_records_lineage_and_physical_reference(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    layout = ensure_run_layout(tmp_path, "lineage")
    result = train_one_epoch(
        model=model,
        optimizer=optimizer,
        batches=[TrainingBatch(inputs=torch.ones(4, 2), targets=torch.zeros(4, 1))],
        layout=layout,
        config=TrainingConfig(
            epoch=1,
            checkpoint_metadata=_metadata(),
            checkpoint_extra_fn=lambda: {"physical_state": {"state_hash": "physical-hash"}},
        ),
    )

    payload = torch.load(result.checkpoint_path, map_location="cpu")
    assert payload["checkpoint_schema"] == "unity_psf.checkpoint.v2"
    assert payload["metadata"]["expert_type"] == "astigmatism"
    assert payload["metadata"]["instance_id"] == "left"
    assert payload["metadata"]["channel_spec"]["channel_id"] == "left"
    assert payload["metadata"]["parent_checkpoint_hash"] == "a" * 64
    assert payload["physical_state"] == {"state_hash": "physical-hash"}
    assert "rng_state" in payload


def test_left_checkpoint_cannot_resume_right_instance(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    result = train_one_epoch(
        model=model,
        optimizer=optimizer,
        batches=[TrainingBatch(inputs=torch.ones(4, 2), targets=torch.zeros(4, 1))],
        layout=ensure_run_layout(tmp_path, "left"),
        config=TrainingConfig(epoch=1, checkpoint_metadata=_metadata(instance_id="left", channel_id="left")),
    )

    with pytest.raises(ValueError, match="instance_id|channel_id"):
        right_model = _model()
        resume_training_checkpoint(
            result.checkpoint_path,
            model=right_model,
            optimizer=torch.optim.AdamW(right_model.parameters(), lr=0.01),
            expected_metadata=_metadata(instance_id="right", channel_id="right"),
        )


def test_instance_resume_requires_complete_identity_and_matching_parent(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    metadata = _metadata(parent_hash="b" * 64)
    result = train_one_epoch(
        model=model,
        optimizer=optimizer,
        batches=[TrainingBatch(inputs=torch.ones(4, 2), targets=torch.zeros(4, 1))],
        layout=ensure_run_layout(tmp_path, "lineage"),
        config=TrainingConfig(epoch=1, checkpoint_metadata=metadata),
    )

    unbound_model = _model()
    unbound_optimizer = torch.optim.AdamW(unbound_model.parameters(), lr=0.01)
    unbound_scheduler = torch.optim.lr_scheduler.StepLR(unbound_optimizer, step_size=1)
    with pytest.raises(ValueError, match="expert_type.*parent_checkpoint_hash"):
        resume_training_checkpoint(
            result.checkpoint_path,
            model=unbound_model,
            optimizer=unbound_optimizer,
            scheduler=unbound_scheduler,
        )

    right_model = _model()
    right_optimizer = torch.optim.AdamW(right_model.parameters(), lr=0.01)
    right_scheduler = torch.optim.lr_scheduler.StepLR(right_optimizer, step_size=1)
    with pytest.raises(ValueError, match="parent_checkpoint_hash"):
        resume_training_checkpoint(
            result.checkpoint_path,
            model=right_model,
            optimizer=right_optimizer,
            scheduler=right_scheduler,
            expected_metadata=_metadata(parent_hash="c" * 64),
        )


def test_weights_only_initialization_does_not_load_instance_optimizer(tmp_path: Path) -> None:
    source = _model()
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
    loss = source(torch.ones(4, 2)).square().mean()
    loss.backward()
    source_optimizer.step()
    prototype_path = tmp_path / "prototype.ckpt"
    save_checkpoint(
        prototype_path,
        source.state_dict(),
        metadata=_prototype_metadata(),
        optimizer_state=source_optimizer.state_dict(),
        extra={"epoch": 4, "global_step": 9},
    )

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.02)
    initialize_model_from_checkpoint(
        prototype_path,
        model=target,
        expected_expert_type=PSFModality.ASTIGMATISM,
    )
    assert all(torch.equal(source.state_dict()[name], target.state_dict()[name]) for name in source.state_dict())
    assert target_optimizer.state == {}
    assert target_optimizer.param_groups[0]["lr"] == 0.02


def test_prototype_file_is_not_modified_by_instance_training(tmp_path: Path) -> None:
    prototype_path = tmp_path / "astigmatism_base.ckpt"
    prototype = _model()
    save_checkpoint(
        prototype_path,
        prototype.state_dict(),
        metadata=_prototype_metadata(),
    )
    before = hashlib.sha256(prototype_path.read_bytes()).digest()
    instance = _model()
    optimizer = torch.optim.AdamW(instance.parameters(), lr=0.01)
    train_one_epoch(
        model=instance,
        optimizer=optimizer,
        batches=[TrainingBatch(inputs=torch.ones(4, 2), targets=torch.zeros(4, 1))],
        layout=ensure_run_layout(tmp_path, "instance"),
        config=TrainingConfig(epoch=1, checkpoint_metadata=_metadata()),
    )
    assert hashlib.sha256(prototype_path.read_bytes()).digest() == before


def test_legacy_training_checkpoint_remains_loadable(tmp_path: Path) -> None:
    source = _model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
    result = train_one_epoch(
        model=source,
        optimizer=optimizer,
        batches=[TrainingBatch(inputs=torch.ones(4, 2), targets=torch.zeros(4, 1))],
        layout=ensure_run_layout(tmp_path, "legacy"),
        config=TrainingConfig(epoch=1),
    )
    restored = _model()
    resume = load_training_checkpoint(
        result.checkpoint_path,
        model=restored,
        optimizer=torch.optim.AdamW(restored.parameters(), lr=0.01),
    )
    assert resume.checkpoint_format == "legacy.training"
    assert resume.global_step == 1
