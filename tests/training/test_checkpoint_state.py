import random
from pathlib import Path

import numpy as np
import pytest
import torch

from unity_psf.contracts import CheckpointMetadata, load_modality_joint_checkpoint
from unity_psf.runtime.layout import ensure_run_layout
from unity_psf.training.checkpoint_state import build_training_state_snapshot, capture_rng_state, restore_rng_state
from unity_psf.training.loop import TrainingBatch, TrainingConfig, resume_training_checkpoint, train_one_epoch
from unity_psf.training.modality_joint import (
    ModalityChannelStream,
    ModalityTrainingRuntime,
    commit_modality_joint_checkpoint,
    restore_modality_training_shard,
    save_modality_training_shard,
)


def test_training_snapshot_has_one_canonical_state_inventory() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler(device="cpu", init_scale=128)

    snapshot = build_training_state_snapshot(
        model=model, optimizer=optimizer, scheduler=scheduler, amp_scaler=scaler,
        progress={"epoch": 3, "step_count": 7, "global_step": 11},
    )

    assert set(snapshot) == {
        "epoch", "step_count", "global_step", "model_state_dict", "optimizer", "scheduler", "amp_scaler", "rng_state",
    }
    assert snapshot["optimizer"] == optimizer.state_dict()
    assert snapshot["scheduler"] == scheduler.state_dict()
    assert snapshot["amp_scaler"] == scaler.state_dict()


@pytest.mark.parametrize("metadata", [None, CheckpointMetadata()])
def test_single_checkpoint_adapts_shared_snapshot_without_losing_state(
    tmp_path: Path, metadata: CheckpointMetadata | None,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    result = train_one_epoch(
        model=model, optimizer=optimizer, scheduler=scheduler,
        batches=[TrainingBatch(torch.ones(1, 1), torch.zeros(1, 1))],
        layout=ensure_run_layout(tmp_path, "single"),
        config=TrainingConfig(epoch=3, checkpoint_metadata=metadata),
    )
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 3
    assert payload["step_count"] == 1
    assert payload["global_step"] == 1
    assert payload["optimizer_state_dict"] == optimizer.state_dict()
    assert payload["scheduler_state_dict"] == scheduler.state_dict()
    assert "optimizer" not in payload
    assert "amp_scaler" not in payload
    assert "scaler_state_dict" not in payload
    assert ("checkpoint_schema" in payload) is (metadata is not None)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.random()
    np.random.random()
    torch.rand(3)
    resume_training_checkpoint(result.checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


class _CheckpointModel(torch.nn.Linear):
    def checkpoint_metadata(self):
        return CheckpointMetadata(expert_type="astigmatism", model_config={"in_features": 1})


def _runtime(*, scaler=None):
    model = _CheckpointModel(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return ModalityTrainingRuntime(
        modality="astigmatism", model=model, optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1), amp_scaler=scaler,
        channels={
            "left": ModalityChannelStream(
                channel_id="left", batches=lambda epoch: (), loss_fn=lambda output, batch: output.sum(),
                physical_state={"version": 1}, calibration={}, provenance={},
            ),
        }, step_budgets={"left": 1},
    )


@pytest.mark.parametrize("save_route", ["shard", "joint_resume"])
@pytest.mark.parametrize("scaler_enabled", [False, True])
def test_joint_resume_and_shard_have_the_same_training_inventory(
    tmp_path: Path, save_route: str, scaler_enabled: bool,
) -> None:
    scaler = torch.amp.GradScaler(device="cpu", enabled=scaler_enabled, init_scale=128)
    runtime = _runtime(scaler=scaler)
    progress = {"epoch": 2, "optimizer_steps": 5, "channel_progress": {"left": {"steps": 5}}, "status": "running"}
    path = tmp_path / "resume.ckpt"
    if save_route == "shard":
        save_modality_training_shard(path, runtime=runtime, **progress)
    else:
        commit_modality_joint_checkpoint(
            path, runtimes={"astigmatism": runtime}, completed_modalities=("astigmatism",),
            role="resume", training_progress={"astigmatism": progress},
        )
    payload = load_modality_joint_checkpoint(path)
    state = payload["training_state"]["astigmatism"]
    assert set(state) == set(progress) | {"optimizer", "scheduler", "amp_scaler", "rng_state"}
    assert state["optimizer"] == runtime.optimizer.state_dict()
    assert state["scheduler"] == runtime.scheduler.state_dict()
    assert state["amp_scaler"] == (scaler.state_dict() if scaler_enabled else None)
    assert payload["channel_states"]["astigmatism"]["left"]["physical_state"] == {"version": 1}
    expected = (random.random(), np.random.random(), torch.rand(3))
    restored = _runtime(scaler=torch.amp.GradScaler(device="cpu", enabled=scaler_enabled, init_scale=2))
    restored_state = restore_modality_training_shard(path, runtime=restored)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert restored_state.epoch == 2
    assert restored_state.optimizer_steps == 5
    if scaler_enabled:
        assert restored.amp_scaler.get_scale() == 128


def test_release_does_not_capture_training_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(**kwargs):
        raise AssertionError("release must not construct a training snapshot")

    monkeypatch.setattr("unity_psf.training.modality_checkpoint.build_training_state_snapshot", forbidden)
    runtime = _runtime()
    path = tmp_path / "release.ckpt"
    commit_modality_joint_checkpoint(
        path, runtimes={"astigmatism": runtime}, completed_modalities=("astigmatism",),
    )
    assert "training_state" not in load_modality_joint_checkpoint(path)


def test_snapshot_preserves_explicit_rng_and_copies_progress() -> None:
    model = torch.nn.Linear(1, 1)
    saved_rng = capture_rng_state()
    progress = {"rng_state": saved_rng, "channels": {"left": {"steps": 3}}}
    torch.rand(3)
    snapshot = build_training_state_snapshot(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), progress=progress,
        amp_scaler=torch.amp.GradScaler(device="cpu", enabled=False),
    )
    assert torch.equal(snapshot["rng_state"]["torch_cpu"], saved_rng["torch_cpu"])
    progress["channels"]["left"]["steps"] = 100
    assert snapshot["channels"]["left"]["steps"] == 3
    assert snapshot["scheduler"] is None
    assert snapshot["amp_scaler"] is None


def test_rng_restore_reads_legacy_numpy_tuple() -> None:
    saved_rng = capture_rng_state()
    saved_rng["numpy"] = np.random.get_state()
    expected = np.random.random()
    restore_rng_state(saved_rng)
    assert np.random.random() == expected


def test_checkpoint_extra_rng_consumption_is_not_included_in_saved_rng(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    expected = capture_rng_state()

    def extra():
        random.random()
        np.random.random()
        torch.rand(1)
        return {}

    result = train_one_epoch(
        model=model, optimizer=optimizer,
        batches=[TrainingBatch(torch.ones(1, 1), torch.zeros(1, 1))],
        layout=ensure_run_layout(tmp_path, "rng-order"),
        config=TrainingConfig(epoch=1, checkpoint_extra_fn=extra),
    )
    payload = torch.load(result.checkpoint_path, weights_only=False)
    assert payload["rng_state"]["python"] == expected["python"]
    assert payload["rng_state"]["numpy"] == expected["numpy"]
    assert torch.equal(payload["rng_state"]["torch_cpu"], expected["torch_cpu"])


def test_checkpoint_extra_rejects_reserved_training_state_keys(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="reserved checkpoint field"):
        train_one_epoch(
            model=model, optimizer=optimizer,
            batches=[TrainingBatch(torch.ones(1, 1), torch.zeros(1, 1))],
            layout=ensure_run_layout(tmp_path, "reserved"),
            config=TrainingConfig(epoch=1, checkpoint_extra_fn=lambda: {"optimizer": "bad"}),
        )
