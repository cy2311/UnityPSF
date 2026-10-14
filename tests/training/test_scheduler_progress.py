from pathlib import Path

import pytest
import torch

from unity_psf.contracts import CheckpointMetadata, load_modality_joint_checkpoint
from unity_psf.runtime.layout import ensure_run_layout
from unity_psf.training.loop import (
    EpochTrainingConfig,
    TrainingBatch,
    TrainingConfig,
    resume_training_checkpoint,
    train_epochs,
    train_one_epoch,
)
from unity_psf.training.modality_joint import (
    ModalityChannelStream,
    ModalityTrainingBatch,
    ModalityTrainingRuntime,
    restore_modality_training_shard,
    save_modality_training_shard,
    train_modality_epoch,
)
from unity_psf.training.modality_progress import (
    accumulate_epoch,
    channel_progress,
    empty_metrics,
    metrics_from_progress,
)


@pytest.mark.parametrize("route", ["generic", "epochs", "modality"])
@pytest.mark.parametrize("unit", ["optimizer_step", "epoch"])
@pytest.mark.parametrize("updates", [(False, False), (False, True), (True, True)])
def test_scheduler_and_progress_follow_successful_updates(
    tmp_path: Path, route: str, unit: str, updates: tuple[bool, bool],
) -> None:
    model = torch.nn.Conv2d(1, 1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler(device="cpu", init_scale=2.0**127)
    images = torch.ones(1, 1, 2, 2)
    multipliers = tuple(1.0 if update else 1e30 for update in updates)
    batch_results = []
    epoch_results = []
    publications = []

    def checkpoint_extra():
        publications.append(scheduler.last_epoch)
        return {"scheduler_at_publication": scheduler.last_epoch}

    if route == "modality":
        channels = {
            channel: ModalityChannelStream(
                channel_id=channel,
                batches=lambda epoch, channel=channel, multiplier=multiplier: [
                    ModalityTrainingBatch(images, multiplier, channel)
                ],
                loss_fn=lambda output, batch: output.mean() * batch.target,
                physical_state={}, calibration={}, provenance={},
            )
            for channel, multiplier in zip(("left", "right"), multipliers)
        }
        result = train_modality_epoch(
            runtime=ModalityTrainingRuntime(
                modality="astigmatism", model=model, optimizer=optimizer,
                channels=channels, step_budgets={"left": 1, "right": 1},
                scheduler=scheduler, scheduler_step_unit=unit, amp_scaler=scaler,
            ), epoch=1,
        )
        assert result.optimizer_steps_by_channel == dict(zip(channels, map(int, updates)))
        assert result.skipped_optimizer_steps_by_channel == {
            channel: int(not update) for channel, update in zip(channels, updates)
        }
        assert result.optimizer_skip_reasons_by_channel == {
            channel: {} if update else {"amp_scale_reduced": 1}
            for channel, update in zip(channels, updates)
        }
    else:
        batches = [TrainingBatch(images, torch.tensor(multiplier)) for multiplier in multipliers]
        layout = ensure_run_layout(tmp_path, "progress")
        arguments = dict(
            model=model, optimizer=optimizer, scheduler=scheduler, layout=layout,
            loss_fn=lambda network, batch: network(batch.inputs).mean() * batch.targets,
            on_batch_end=batch_results.append,
        )
        if route == "generic":
            result = train_one_epoch(
                **arguments, batches=batches,
                config=TrainingConfig(
                    epoch=1, global_step_start=7, amp_scaler=scaler,
                    scheduler_step_unit=unit, checkpoint_extra_fn=checkpoint_extra,
                ),
            )
        else:
            result = train_epochs(
                **arguments, batch_provider=lambda epoch: batches,
                config=EpochTrainingConfig(
                    start_epoch=1, stop_epoch=1, global_step_start=7,
                    amp_scaler=scaler, scheduler_step_unit=unit,
                ), checkpoint_extra_fn=checkpoint_extra, on_epoch_end=epoch_results.append,
            )[0]
            assert epoch_results == [result]
        assert result.step_count == 2
        assert result.global_step == 9
        assert [item.global_step for item in batch_results] == [8, 9]
        assert [item.optimizer_step_performed for item in batch_results] == list(updates)
        payload = torch.load(result.checkpoint_path, weights_only=False)
        assert payload["global_step"] == 9
        assert payload["scheduler_state_dict"] == scheduler.state_dict()
        assert publications == [scheduler.last_epoch]

    assert result.attempted_optimizer_steps == 2
    assert result.optimizer_steps == sum(updates)
    assert result.skipped_optimizer_steps == 2 - sum(updates)
    expected_advances = sum(updates) if unit == "optimizer_step" else int(any(updates))
    assert scheduler.last_epoch == expected_advances
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.5**expected_advances)


@pytest.mark.parametrize("unit", ["optimizer_step", "epoch"])
def test_modality_skip_counters_and_progress_survive_shard_resume(tmp_path: Path, unit: str) -> None:
    class CheckpointModel(torch.nn.Conv2d):
        def checkpoint_metadata(self):
            return CheckpointMetadata(expert_type="astigmatism", model_config={"in_channels": 1})

    def build():
        model = CheckpointModel(1, 1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        channels = {
            channel: ModalityChannelStream(
                channel_id=channel,
                batches=lambda epoch, channel=channel, multiplier=multiplier: [
                    ModalityTrainingBatch(torch.ones(1, 1, 2, 2), multiplier, channel)
                ] * 2,
                loss_fn=lambda output, batch: output.mean() * batch.target,
                physical_state={}, calibration={}, provenance={},
            )
            for channel, multiplier in (("left", 1e30), ("right", 1.0))
        }
        return ModalityTrainingRuntime(
            modality="astigmatism", model=model, optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
            scheduler_step_unit=unit,
            amp_scaler=torch.amp.GradScaler(device="cpu", init_scale=2.0**127),
            channels=channels, step_budgets={"left": 2, "right": 2},
        )

    uninterrupted = build()
    full_metrics = empty_metrics(uninterrupted)
    for epoch in (1, 2):
        accumulate_epoch(full_metrics, train_modality_epoch(runtime=uninterrupted, epoch=epoch))

    interrupted = build()
    metrics = empty_metrics(interrupted)
    accumulate_epoch(metrics, train_modality_epoch(runtime=interrupted, epoch=1))
    progress = channel_progress(metrics)
    assert metrics["attempted_optimizer_steps"] == 4
    assert metrics["optimizer_steps"] == metrics["skipped_optimizer_steps"] == 2
    assert progress["left"]["optimizer_skip_reasons"] == {"amp_scale_reduced": 2}
    assert progress["right"]["optimizer_skip_reasons"] == {}
    path = save_modality_training_shard(
        tmp_path / "resume.ckpt", runtime=interrupted, epoch=metrics["epochs_completed"],
        optimizer_steps=metrics["optimizer_steps"], channel_progress=progress, status="running",
    )
    saved = load_modality_joint_checkpoint(path)["training_state"]["astigmatism"]
    assert saved["channel_progress"] == progress
    assert saved["epoch"] == 1
    assert saved["optimizer_steps"] == 2

    resumed = build()
    state = restore_modality_training_shard(path, runtime=resumed)
    assert state.status == "running"
    assert state.channel_progress == progress
    restored_metrics = metrics_from_progress(
        resumed, epoch=state.epoch, optimizer_steps=state.optimizer_steps, progress=state.channel_progress,
    )
    assert restored_metrics == metrics
    assert channel_progress(restored_metrics) == progress
    accumulate_epoch(restored_metrics, train_modality_epoch(runtime=resumed, epoch=state.epoch + 1))
    assert restored_metrics == full_metrics
    assert restored_metrics["attempted_optimizer_steps"] == 8
    assert restored_metrics["optimizer_steps"] == restored_metrics["skipped_optimizer_steps"] == 4
    for channel, updates, skips in (("left", 0, 4), ("right", 4, 0)):
        values = restored_metrics["channels"][channel]
        assert values["steps"] == values["samples"] == len(values["losses"]) == 4
        assert values["optimizer_steps"] == updates
        assert values["skipped_optimizer_steps"] == skips
        assert values["optimizer_skip_reasons"] == ({"amp_scale_reduced": skips} if skips else {})
    assert torch.equal(resumed.model.weight, uninterrupted.model.weight)
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    assert resumed.amp_scaler.state_dict() == uninterrupted.amp_scaler.state_dict()
    assert resumed.optimizer.param_groups == uninterrupted.optimizer.param_groups
    resumed_momentum = next(iter(resumed.optimizer.state.values()))["momentum_buffer"]
    full_momentum = next(iter(uninterrupted.optimizer.state.values()))["momentum_buffer"]
    assert torch.equal(resumed_momentum, full_momentum)


def test_epoch_scheduler_is_not_advanced_twice_across_epochs(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    results = train_epochs(
        model=model, optimizer=optimizer, scheduler=scheduler,
        batch_provider=lambda epoch: [TrainingBatch(torch.ones(1, 1), torch.zeros(1, 1))],
        layout=ensure_run_layout(tmp_path, "epochs"),
        config=EpochTrainingConfig(start_epoch=1, stop_epoch=2, scheduler_step_unit="epoch"),
    )
    assert scheduler.last_epoch == 2
    assert [result.optimizer_steps for result in results] == [1, 1]
    assert [result.global_step for result in results] == [1, 2]


@pytest.mark.parametrize("unit", ["optimizer_step", "epoch"])
def test_scheduler_resume_matches_uninterrupted_training(tmp_path: Path, unit: str) -> None:
    def build():
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        scaler = torch.amp.GradScaler(device="cpu", init_scale=2.0**127)
        return model, optimizer, scheduler, scaler

    def run(components, run_id, start, stop, global_step=0):
        model, optimizer, scheduler, scaler = components
        return train_epochs(
            model=model, optimizer=optimizer, scheduler=scheduler,
            batch_provider=lambda epoch: [
                TrainingBatch(torch.ones(1, 1), torch.tensor(multiplier))
                for multiplier in (1e30, 1.0)
            ],
            layout=ensure_run_layout(tmp_path, run_id),
            config=EpochTrainingConfig(
                start_epoch=start, stop_epoch=stop, amp_scaler=scaler,
                global_step_start=global_step, scheduler_step_unit=unit,
            ),
            loss_fn=lambda network, batch: network(batch.inputs).mean() * batch.targets,
        )

    uninterrupted = build()
    full_results = run(uninterrupted, "full", 1, 2)
    interrupted = build()
    saved = run(interrupted, "split", 1, 1)[0]
    resumed = build()
    model, optimizer, scheduler, scaler = resumed
    state = resume_training_checkpoint(
        saved.checkpoint_path, model=model, optimizer=optimizer,
        scheduler=scheduler, amp_scaler=scaler,
    )
    resumed_result = run(resumed, "split", state.epoch + 1, 2, state.global_step)[0]
    assert torch.equal(model.weight, uninterrupted[0].weight)
    assert scheduler.state_dict() == uninterrupted[2].state_dict()
    assert scaler.state_dict() == uninterrupted[3].state_dict()
    assert optimizer.param_groups[0]["lr"] == uninterrupted[1].param_groups[0]["lr"]
    assert resumed_result.global_step == full_results[-1].global_step == 4
    assert resumed_result.optimizer_steps == full_results[-1].optimizer_steps == 1
    assert resumed_result.skipped_optimizer_steps == 1


def test_batch_budget_counts_attempts_even_when_every_update_is_skipped(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    results = train_epochs(
        model=model, optimizer=optimizer, scheduler=scheduler,
        batch_provider=lambda epoch: [TrainingBatch(torch.ones(1, 1), torch.tensor(1e30))] * 4,
        layout=ensure_run_layout(tmp_path, "budget"),
        config=EpochTrainingConfig(
            start_epoch=1, stop_epoch=3, max_batches=3, scheduler_step_unit="epoch",
            amp_scaler=torch.amp.GradScaler(device="cpu", init_scale=2.0**127),
        ),
        loss_fn=lambda network, batch: network(batch.inputs).mean() * batch.targets,
    )
    assert len(results) == 1
    assert results[0].global_step == results[0].attempted_optimizer_steps == 3
    assert results[0].optimizer_steps == 0
    assert results[0].skipped_optimizer_steps == 3
    assert scheduler.last_epoch == 0


@pytest.mark.parametrize("route", ["generic", "epochs"])
@pytest.mark.parametrize("with_scheduler", [False, True])
def test_invalid_scheduler_policy_fails_before_training(
    tmp_path: Path, route: str, with_scheduler: bool,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model.eval()
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1) if with_scheduler else None
    scaler = torch.amp.GradScaler(device="cpu", init_scale=128)
    scaler_before = scaler.state_dict()
    calls = []
    layout = ensure_run_layout(tmp_path, "invalid")

    def provider(epoch):
        calls.append("provider")
        yield TrainingBatch(torch.ones(1, 1), torch.zeros(1, 1))

    with pytest.raises(ValueError, match="scheduler_step_unit"):
        if route == "generic":
            train_one_epoch(
                model=model, optimizer=optimizer, scheduler=scheduler,
                batches=provider(1), layout=layout,
                config=TrainingConfig(epoch=1, amp_scaler=scaler, scheduler_step_unit="invalid"),
                on_batch_end=lambda result: calls.append("batch"),
            )
        else:
            train_epochs(
                model=model, optimizer=optimizer, scheduler=scheduler,
                batch_provider=provider, layout=layout,
                config=EpochTrainingConfig(
                    start_epoch=1, stop_epoch=1, amp_scaler=scaler,
                    scaler_state_dict={**scaler_before, "scale": 2.0},
                    scheduler_step_unit="invalid",
                ),
                on_batch_end=lambda result: calls.append("batch"),
                on_epoch_end=lambda result: calls.append("epoch"),
            )
    assert torch.equal(model.weight, before)
    assert not model.training
    assert not optimizer.state
    assert scaler.state_dict() == scaler_before
    assert calls == []
    assert not list(layout.checkpoints_dir.iterdir())
    assert scheduler is None or scheduler.last_epoch == 0
