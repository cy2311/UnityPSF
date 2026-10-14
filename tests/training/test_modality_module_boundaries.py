from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from unity_psf.contracts import CheckpointMetadata, load_modality_joint_checkpoint
from unity_psf.training import modality_checkpoint, modality_epoch, modality_evaluation
from unity_psf.training import modality_joint, modality_physical_state, modality_types


@pytest.mark.parametrize(
    "module,names",
    [
        (modality_types, (
            "ModalityTrainingBatch", "ModalityChannelStream", "ModalityTrainingRuntime",
            "ModalityEpochResult", "ModalityTrainingShardState", "ModalityHeldoutEvalResult",
            "ModalityBatchProvider", "ModalityLossFn", "PhysicalStateSnapshot",
            "PhysicalStateRestore", "HeldoutEvalFn",
        )),
        (modality_epoch, ("train_modality_epoch",)),
        (modality_evaluation, ("evaluate_modality_heldout",)),
        (modality_checkpoint, (
            "commit_modality_joint_checkpoint", "save_modality_training_shard",
            "restore_modality_training_shard", "assemble_modality_joint_checkpoint_from_shards",
        )),
    ],
)
def test_facade_reexports_canonical_objects(module, names) -> None:
    for name in names:
        assert getattr(modality_joint, name) is getattr(module, name)


def test_facade_preserves_private_helper_aliases() -> None:
    for module, names in (
        (modality_evaluation, ("normalize_heldout_metrics", "aggregate_heldout_metrics")),
        (modality_physical_state, ("physical_provenance_registry", "snapshot_channel_physical_state")),
    ):
        for name in names:
            assert getattr(modality_joint, f"_{name}") is getattr(module, name)


def test_modality_modules_have_public_dependencies_and_bounded_size() -> None:
    for module, limit in (
        (modality_joint, 150), (modality_types, 350), (modality_epoch, 350),
        (modality_evaluation, 350), (modality_checkpoint, 350), (modality_physical_state, 350),
    ):
        source = inspect.getsource(module)
        assert len(source.splitlines()) <= limit
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert all(not alias.name.startswith("_") for alias in node.names)
                if module is not modality_joint:
                    assert node.module != modality_joint.__name__


class _Expert(torch.nn.Linear):
    def checkpoint_metadata(self):
        return CheckpointMetadata(expert_type="astigmatism", model_config={})


def test_checkpoint_physical_callbacks_keep_snapshot_and_restore_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    model = _Expert(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)

    def snapshot_physical():
        events.append("physical_snapshot")
        return {"condition_store_version": 1}

    def restore_physical(state):
        events.append("physical_restore")
        assert state == {"condition_store_version": 1}

    stream = modality_types.ModalityChannelStream(
        channel_id="left", batches=lambda epoch: (), loss_fn=lambda output, batch: output.sum(),
        physical_state={}, calibration={}, provenance={},
        snapshot_physical_state=snapshot_physical, restore_physical_state=restore_physical,
    )
    runtime = modality_types.ModalityTrainingRuntime(
        modality="astigmatism", model=model, optimizer=optimizer, scheduler=scheduler,
        amp_scaler=scaler,
        channels={"left": stream}, step_budgets={"left": 1},
    )
    original_snapshot = modality_checkpoint.build_training_state_snapshot

    def snapshot_training(**kwargs):
        events.append("training_snapshot")
        return original_snapshot(**kwargs)

    monkeypatch.setattr(modality_checkpoint, "build_training_state_snapshot", snapshot_training)
    path = tmp_path / "resume.ckpt"
    modality_joint.save_modality_training_shard(
        path, runtime=runtime, epoch=2, optimizer_steps=3,
        channel_progress={"left": {"steps": 3}}, status="running",
    )
    assert events == ["training_snapshot", "physical_snapshot"]
    payload = load_modality_joint_checkpoint(path)
    assert payload["checkpoint_schema"] == "unity_psf.joint_checkpoint.v2"

    events.clear()
    original_model_restore = model.load_state_dict
    original_optimizer_restore = optimizer.load_state_dict
    original_scheduler_restore = scheduler.load_state_dict
    original_scaler_restore = scaler.load_state_dict
    original_rng_restore = modality_checkpoint.restore_rng_state

    def restore_model(*args, **kwargs):
        events.append("model")
        return original_model_restore(*args, **kwargs)

    def restore_optimizer(*args, **kwargs):
        events.append("optimizer")
        return original_optimizer_restore(*args, **kwargs)

    def restore_rng(state):
        events.append("rng")
        return original_rng_restore(state)

    def restore_scheduler(state):
        events.append("scheduler")
        return original_scheduler_restore(state)

    def restore_scaler(state):
        events.append("scaler")
        return original_scaler_restore(state)

    monkeypatch.setattr(model, "load_state_dict", restore_model)
    monkeypatch.setattr(optimizer, "load_state_dict", restore_optimizer)
    monkeypatch.setattr(scheduler, "load_state_dict", restore_scheduler)
    monkeypatch.setattr(scaler, "load_state_dict", restore_scaler)
    monkeypatch.setattr(modality_checkpoint, "restore_rng_state", restore_rng)
    state = modality_joint.restore_modality_training_shard(path, runtime=runtime)
    assert events == [
        "model", "optimizer", "scheduler", "scaler", "physical_restore", "physical_snapshot", "rng",
    ]
    assert state.epoch == 2
    assert state.optimizer_steps == 3
