from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from unity_psf.contracts import (
    CheckpointMetadata,
    InputFrameSpec,
    PSFModality,
    load_modality_joint_checkpoint,
)
from unity_psf.localization.legacy_decode import LegacyEmitterSet, evaluate_legacy_localizations
from unity_psf.models import UnityPSF
from unity_psf.runtime import ensure_run_layout
from unity_psf.training.modality_joint import (
    ModalityChannelStream,
    ModalityTrainingBatch,
    ModalityTrainingRuntime,
    assemble_modality_joint_checkpoint_from_shards,
    commit_modality_joint_checkpoint,
    evaluate_modality_heldout,
    restore_modality_training_shard,
    save_modality_training_shard,
    train_modality_epoch,
)
from unity_psf.training.modality_runtime import build_modality_runtime


class _TrainableExpert(nn.Module):
    def __init__(self, modality: str, value: float = 1.0) -> None:
        super().__init__()
        self.modality = modality
        self.value = nn.Parameter(torch.tensor(value))

    def forward(
        self,
        images: torch.Tensor,
        conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.value * torch.ones_like(images)

    def checkpoint_metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            expert_type=self.modality,
            model_config={"condition_dim": 1},
            input_frame_spec=InputFrameSpec(input_frame_channels=1),
            condition_schema={"fields": ["physical_condition"], "dimension": 1},
            experts=(self.modality,),
        )


def _batch(channel_id: str, target: float, *, sample_count: int = 1) -> ModalityTrainingBatch:
    return ModalityTrainingBatch(
        images=torch.ones(sample_count, 1, 2, 2),
        conditions=torch.zeros(sample_count, 1),
        target=torch.tensor(target),
        channel_id=channel_id,
    )


def _loss(output: torch.Tensor, batch: ModalityTrainingBatch) -> torch.Tensor:
    return (output.mean() - batch.target).square()


def _stream(
    channel_id: str,
    batches: tuple[ModalityTrainingBatch, ...],
    *,
    peak_zmap: str | None = None,
) -> ModalityChannelStream:
    return ModalityChannelStream(
        channel_id=channel_id,
        batches=lambda epoch: batches,
        loss_fn=_loss,
        physical_state={"peak_zmap": peak_zmap or f"{channel_id}.npy", "gamma": channel_id},
        calibration={"version": f"{channel_id}-v1"},
        provenance={"dataset": f"{channel_id}.tif"},
    )


def _runtime(
    modality: str,
    *,
    left_batches: tuple[ModalityTrainingBatch, ...],
    right_batches: tuple[ModalityTrainingBatch, ...],
    step_budgets: dict[str, int],
) -> ModalityTrainingRuntime:
    model = _TrainableExpert(modality)
    return ModalityTrainingRuntime(
        modality=modality,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        channels={
            "left": _stream("left", left_batches),
            "right": _stream("right", right_batches),
        },
        step_budgets=step_budgets,
    )


def test_one_modality_runtime_owns_one_model_and_optimizer_for_both_channels() -> None:
    runtime = _runtime(
        "astigmatism",
        left_batches=(_batch("left", 0.0), _batch("left", 0.0)),
        right_batches=(_batch("right", 2.0),),
        step_budgets={"left": 2, "right": 1},
    )

    optimizer_parameters = {
        id(parameter)
        for group in runtime.optimizer.param_groups
        for parameter in group["params"]
    }

    assert set(runtime.channels) == {"left", "right"}
    assert optimizer_parameters == {id(parameter) for parameter in runtime.model.parameters()}
    assert all(not hasattr(channel, "model") for channel in runtime.channels.values())
    assert all(not hasattr(channel, "optimizer") for channel in runtime.channels.values())


def test_public_runtime_builder_resolves_shared_astigmatism_channels(
    tmp_path: Path,
) -> None:
    config_path = (
        Path(__file__).parents[2]
        / "configs/modalities/astigmatism/astigmatism_single_channel_smoke.yaml"
    )
    joint_path = tmp_path / "joint.yaml"
    joint_path.write_text("schema_version: unitypsf.joint_training.v1\n", encoding="utf-8")
    entries = (
        (
            "astigmatism:left",
            {"config": str(config_path), "model_seed": 101, "data_seed": 201, "step_budget": 1},
        ),
        (
            "astigmatism:right",
            {"config": str(config_path), "model_seed": 101, "data_seed": 202, "step_budget": 1},
        ),
    )

    runtime, runtime_configs, formal_evidence = build_modality_runtime(
        PSFModality.ASTIGMATISM,
        entries,
        joint_path=joint_path,
        run_layout=ensure_run_layout(tmp_path, "resolved-runtime"),
        device="cpu",
    )

    assert set(runtime.channels) == {"left", "right"}
    assert set(runtime_configs) == {"left", "right"}
    assert formal_evidence is None
    assert runtime_configs["left"]["expert_instance"]["channel_id"] == "left"
    assert runtime_configs["right"]["expert_instance"]["channel_id"] == "right"
    assert {
        channel_id: stream.provenance["config"]
        for channel_id, stream in runtime.channels.items()
    } == {"left": str(config_path), "right": str(config_path)}


def test_balanced_channels_update_the_same_model_and_report_independent_metrics() -> None:
    runtime = _runtime(
        "astigmatism",
        left_batches=(_batch("left", 0.0, sample_count=2),),
        right_batches=(_batch("right", 2.0, sample_count=3),),
        step_budgets={"left": 1, "right": 1},
    )

    result = train_modality_epoch(runtime=runtime, epoch=4)

    assert result.schedule == ("left", "right")
    assert result.step_counts == {"left": 1, "right": 1}
    assert result.sample_counts == {"left": 2, "right": 3}
    assert result.optimizer_steps == 2
    assert result.losses_by_channel["left"] == pytest.approx((1.0,))
    assert result.losses_by_channel["right"] == pytest.approx((1.44,))
    assert result.mean_loss == pytest.approx(1.22)
    assert runtime.model.value.item() == pytest.approx(1.04)


def test_round_robin_schedule_is_auditable_for_unequal_channel_budgets() -> None:
    runtime = _runtime(
        "astigmatism",
        left_batches=(_batch("left", 0.0), _batch("left", 0.0)),
        right_batches=(_batch("right", 2.0),),
        step_budgets={"left": 2, "right": 1},
    )

    result = train_modality_epoch(runtime=runtime, epoch=1)

    assert result.schedule == ("left", "right", "left")
    assert result.step_counts == {"left": 2, "right": 1}


def test_amp_overflow_does_not_count_optimizer_step_or_advance_epoch_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SkippingGradScaler:
        def __init__(self, *args, **kwargs) -> None:
            self.current_scale = 65536.0

        def is_enabled(self) -> bool:
            return True

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

        def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
            return None

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            return None

        def update(self) -> None:
            self.current_scale /= 2.0

        def get_scale(self) -> float:
            return self.current_scale

    monkeypatch.setattr(torch.amp, "GradScaler", SkippingGradScaler)
    model = _TrainableExpert("astigmatism")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    runtime = ModalityTrainingRuntime(
        modality="astigmatism",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_unit="epoch",
        amp_enabled=True,
        channels={
            "left": _stream("left", (_batch("left", 0.0),)),
            "right": _stream("right", (_batch("right", 2.0),)),
        },
        step_budgets={"left": 1, "right": 1},
    )
    initial_scheduler_epoch = scheduler.last_epoch

    result = train_modality_epoch(runtime=runtime, epoch=1)

    assert result.optimizer_steps == 0
    assert result.optimizer_steps_by_channel == {"left": 0, "right": 0}
    assert result.skipped_optimizer_steps_by_channel == {"left": 1, "right": 1}
    assert result.attempted_optimizer_steps == 2
    assert result.skipped_optimizer_steps == 2
    assert scheduler.last_epoch == initial_scheduler_epoch
    assert model.value.item() == pytest.approx(1.0)

    scaler = runtime.amp_scaler
    second_result = train_modality_epoch(runtime=runtime, epoch=2)

    assert runtime.amp_scaler is scaler
    assert scaler is not None and scaler.get_scale() == 4096.0
    assert second_result.optimizer_steps == 0


def test_non_finite_modality_loss_fails_before_checkpoint_progress() -> None:
    runtime = _runtime(
        "astigmatism",
        left_batches=(_batch("left", float("nan")),),
        right_batches=(_batch("right", 2.0),),
        step_budgets={"left": 1, "right": 1},
    )

    with pytest.raises(FloatingPointError, match="non-finite loss.*left"):
        train_modality_epoch(runtime=runtime, epoch=1)


def test_channel_state_and_metrics_remain_independent_inside_shared_runtime() -> None:
    runtime = _runtime(
        "astigmatism",
        left_batches=(_batch("left", 0.0),),
        right_batches=(_batch("right", 2.0),),
        step_budgets={"left": 1, "right": 1},
    )

    left = runtime.channels["left"]
    right = runtime.channels["right"]
    result = train_modality_epoch(runtime=runtime, epoch=1)

    assert left.physical_state is not right.physical_state
    assert left.calibration is not right.calibration
    assert left.provenance is not right.provenance
    assert left.physical_state["peak_zmap"] == "left.npy"
    assert right.physical_state["peak_zmap"] == "right.npy"
    assert set(result.losses_by_channel) == {"left", "right"}
    assert result.losses_by_channel["left"] != result.losses_by_channel["right"]


def test_heldout_metrics_are_recomputed_from_counts_instead_of_averaged() -> None:
    def eval_result(
        *,
        eval_loss: float,
        true_positive: int,
        false_positive: int,
        false_negative: int,
        lateral_sq_sum: float,
        axial_sq_sum: float,
        photon_relative_sum: float,
        samples: int,
    ):
        return {
            "metrics": {
                "eval_loss": eval_loss,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "lateral_sq_error_nm2_sum": lateral_sq_sum,
                "axial_sq_error_nm2_sum": axial_sq_sum,
                "photon_relative_error_sum": photon_relative_sum,
                "matched_photons": true_positive,
                "sample_count": samples,
                "route_count": samples,
            },
            "artifacts": {},
        }

    model = _TrainableExpert("astigmatism")
    streams = {
        "left": ModalityChannelStream(
            channel_id="left",
            batches=lambda epoch: (_batch("left", 0.0),),
            loss_fn=_loss,
            physical_state={},
            calibration={},
            provenance={},
            heldout_eval=lambda model: eval_result(
                eval_loss=1.0,
                true_positive=1,
                false_positive=0,
                false_negative=0,
                lateral_sq_sum=100.0,
                axial_sq_sum=400.0,
                photon_relative_sum=0.1,
                samples=2,
            ),
        ),
        "right": ModalityChannelStream(
            channel_id="right",
            batches=lambda epoch: (_batch("right", 0.0),),
            loss_fn=_loss,
            physical_state={},
            calibration={},
            provenance={},
            heldout_eval=lambda model: eval_result(
                eval_loss=3.0,
                true_positive=3,
                false_positive=1,
                false_negative=2,
                lateral_sq_sum=1200.0,
                axial_sq_sum=3600.0,
                photon_relative_sum=0.9,
                samples=6,
            ),
        ),
    }
    runtime = ModalityTrainingRuntime(
        modality="astigmatism",
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        channels=streams,
        step_budgets={"left": 1, "right": 1},
    )

    result = evaluate_modality_heldout(runtime)

    assert set(result.channels) == {"left", "right"}
    assert result.modality["eval_loss"] == pytest.approx(2.5)
    assert result.modality["precision"] == pytest.approx(4 / 5)
    assert result.modality["recall"] == pytest.approx(4 / 6)
    assert result.modality["Jaccard"] == pytest.approx(4 / 7)
    assert result.modality["RMSE_XY_nm"] == pytest.approx((1300 / 4) ** 0.5)
    assert result.modality["RMSE_Z_nm"] == pytest.approx((4000 / 4) ** 0.5)
    assert result.modality["photon_relative_error"] == pytest.approx(1.0 / 4)
    assert result.modality["sample_count"] == 8
    assert result.modality["route_count"] == 8


def test_emitter_2d_heldout_metrics_mark_axial_rmse_not_applicable() -> None:
    model = _TrainableExpert("emitter_2d")
    stream = ModalityChannelStream(
        channel_id="left",
        batches=lambda epoch: (_batch("left", 0.0),),
        loss_fn=_loss,
        physical_state={},
        calibration={},
        provenance={},
        heldout_eval=lambda model: {
            "metrics": {
                "eval_loss": 1.0,
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 0,
                "lateral_sq_error_nm2_sum": 25.0,
                "axial_sq_error_nm2_sum": 999.0,
                "photon_relative_error_sum": 0.2,
                "matched_photons": 1,
                "sample_count": 1,
                "route_count": 1,
            },
            "artifacts": {},
        },
    )
    runtime = ModalityTrainingRuntime(
        modality="emitter_2d",
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        channels={"left": stream},
        step_budgets={"left": 1},
    )

    result = evaluate_modality_heldout(runtime)

    assert result.channels["left"]["RMSE_Z_nm"] is None
    assert result.modality["RMSE_Z_nm"] is None


def test_zero_match_heldout_errors_are_unavailable_not_zero() -> None:
    model = _TrainableExpert("astigmatism")
    stream = ModalityChannelStream(
        channel_id="left",
        batches=lambda epoch: (_batch("left", 0.0),),
        loss_fn=_loss,
        physical_state={},
        calibration={},
        provenance={},
        heldout_eval=lambda model: {
            "metrics": {
                "eval_loss": 1.0,
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 3,
                "lateral_sq_error_nm2_sum": 0.0,
                "axial_sq_error_nm2_sum": 0.0,
                "photon_relative_error_sum": 0.0,
                "matched_photons": 0,
                "sample_count": 2,
                "route_count": 1,
            },
            "artifacts": {},
        },
    )
    runtime = ModalityTrainingRuntime(
        modality="astigmatism",
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        channels={"left": stream},
        step_budgets={"left": 1},
    )

    result = evaluate_modality_heldout(runtime)

    for metrics in (result.channels["left"], result.modality):
        assert metrics["RMSE_XY_nm"] is None
        assert metrics["RMSE_Z_nm"] is None
        assert metrics["photon_relative_error"] is None

    empty = LegacyEmitterSet(
        batch_index=torch.empty(0, dtype=torch.long),
        probability=torch.empty(0),
        xyz_px_nm=torch.empty((0, 3)),
        photons=torch.empty(0),
        sigma_xy_px=torch.empty((0, 2)),
    )
    target = LegacyEmitterSet(
        batch_index=torch.zeros(1, dtype=torch.long),
        probability=torch.ones(1),
        xyz_px_nm=torch.zeros((1, 3)),
        photons=torch.ones(1),
        sigma_xy_px=torch.zeros((1, 2)),
    )
    legacy = evaluate_legacy_localizations(empty, target).to_dict()
    assert legacy["rmse_lat"] is None
    assert legacy["rmse_ax"] is None
    assert legacy["photon_relative_error"] is None


def test_legacy_eval_exposes_exact_aggregation_and_photon_statistics() -> None:
    pred = LegacyEmitterSet(
        batch_index=torch.tensor([0, 0, 0]),
        probability=torch.ones(3),
        xyz_px_nm=torch.tensor(
            [[1.0, 1.0, 10.0], [4.0, 4.0, 40.0], [7.0, 7.0, 70.0]]
        ),
        photons=torch.tensor([80.0, 240.0, 50.0]),
        sigma_xy_px=torch.zeros(3, 2),
    )
    target = LegacyEmitterSet(
        batch_index=torch.tensor([0, 0, 0]),
        probability=torch.ones(3),
        xyz_px_nm=torch.tensor(
            [[1.0, 1.0, 0.0], [4.0, 4.0, 20.0], [20.0, 20.0, 0.0]]
        ),
        photons=torch.tensor([100.0, 200.0, 100.0]),
        sigma_xy_px=torch.zeros(3, 2),
    )

    metrics = evaluate_legacy_localizations(
        pred,
        target,
        dist_tol_xy_px=1.0,
        dist_tol_z_nm=50.0,
        match_dims=3,
    ).to_dict()

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["lateral_sq_error_nm2_sum"] == pytest.approx(0.0)
    assert metrics["axial_sq_error_nm2_sum"] == pytest.approx(500.0)
    assert metrics["matched_photons"] == 2
    assert metrics["photon_relative_error_sum"] == pytest.approx(0.4)
    assert metrics["photon_relative_error"] == pytest.approx(0.2)


def test_commit_writes_one_v2_model_state_per_modality_with_nested_channels(
    tmp_path: Path,
) -> None:
    emitter = _runtime(
        "emitter_2d",
        left_batches=(_batch("left", 0.0),),
        right_batches=(_batch("right", 2.0),),
        step_budgets={"left": 1, "right": 1},
    )
    astigmatism = _runtime(
        "astigmatism",
        left_batches=(_batch("left", 0.0),),
        right_batches=(_batch("right", 2.0),),
        step_budgets={"left": 1, "right": 1},
    )
    destination = tmp_path / "unitypsf_joint.ckpt"

    commit_modality_joint_checkpoint(
        destination,
        runtimes={"emitter_2d": emitter, "astigmatism": astigmatism},
        completed_modalities=("emitter_2d", "astigmatism"),
        role="release",
        provenance={"training": "modality_joint"},
    )

    payload = load_modality_joint_checkpoint(destination)
    assert payload["checkpoint_schema"] == "unity_psf.joint_checkpoint.v2"
    assert set(payload["experts"]) == {"emitter_2d", "astigmatism"}
    assert all("model_state_dict" in expert for expert in payload["experts"].values())
    assert set(payload["channel_states"]["emitter_2d"]) == {"left", "right"}
    assert set(payload["channel_states"]["astigmatism"]) == {"left", "right"}
    assert payload["channel_states"]["astigmatism"]["left"]["physical_state"]["peak_zmap"] == "left.npy"
    assert payload["channel_states"]["astigmatism"]["right"]["calibration"]["version"] == "right-v1"
    assert payload["channel_states"]["astigmatism"]["left"]["provenance"]["dataset"] == "left.tif"


def test_modality_joint_cli_trains_dual_modality_dual_channel_cpu_smoke(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from unity_psf.cli.train_modality_joint import main as train_modality_joint

    config_root = Path(__file__).parents[2] / "configs"
    joint_config = tmp_path / "dual_modality_dual_channel.yaml"
    joint_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "unitypsf.joint_training.v1",
                "execution": "round_robin",
                "epochs": 1,
                "instances": [
                    {
                        "key": "emitter_2d:left",
                        "config": str(config_root / "modalities/emitter_2d/emitter_2d_single_channel_smoke.yaml"),
                        "model_seed": 101,
                        "data_seed": 201,
                        "step_budget": 1,
                    },
                    {
                        "key": "emitter_2d:right",
                        "config": str(config_root / "modalities/emitter_2d/emitter_2d_single_channel_smoke.yaml"),
                        "model_seed": 101,
                        "data_seed": 202,
                        "step_budget": 1,
                    },
                    {
                        "key": "astigmatism:left",
                        "config": str(config_root / "modalities/astigmatism/astigmatism_single_channel_smoke.yaml"),
                        "model_seed": 102,
                        "data_seed": 203,
                        "step_budget": 1,
                    },
                    {
                        "key": "astigmatism:right",
                        "config": str(config_root / "modalities/astigmatism/astigmatism_single_channel_smoke.yaml"),
                        "model_seed": 102,
                        "data_seed": 204,
                        "step_budget": 1,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert train_modality_joint(
        [
            "--config",
            str(joint_config),
            "--run-root",
            str(tmp_path),
            "--run-id",
            "modality-joint-smoke",
        ]
    ) == 0

    command_result = json.loads(capsys.readouterr().out)
    run_dir = tmp_path / "modality-joint-smoke"
    checkpoint_path = run_dir / "checkpoints" / "unitypsf_joint.ckpt"
    summary_path = run_dir / "metrics" / "joint_training_summary.json"
    assert command_result["status"] == "complete"
    assert command_result["checkpoint"] == str(checkpoint_path)
    assert checkpoint_path.is_file()
    assert summary_path.is_file()

    payload = load_modality_joint_checkpoint(checkpoint_path)
    assert set(payload["experts"]) == {"emitter_2d", "astigmatism"}
    assert set(payload["channel_states"]["emitter_2d"]) == {"left", "right"}
    assert set(payload["channel_states"]["astigmatism"]) == {"left", "right"}
    for modality in ("emitter_2d", "astigmatism"):
        for channel_id in ("left", "right"):
            physical_state = payload["channel_states"][modality][channel_id][
                "physical_state"
            ]
            assert physical_state["schema_version"] == "unitypsf.channel_physical_state.v1"
            assert physical_state["condition_store_version"] == 0
            assert physical_state["expert_instance"] == {
                "expert_type": modality,
                "instance_id": channel_id,
                "channel_id": channel_id,
            }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary["modalities"]) == {"emitter_2d", "astigmatism"}
    for modality in ("emitter_2d", "astigmatism"):
        modality_metrics = summary["modalities"][modality]
        assert modality_metrics["optimizer_steps"] == 2
        assert modality_metrics["schedule"] == ["left", "right"]
        assert set(modality_metrics["channels"]) == {"left", "right"}
        assert {
            channel_id: metrics["steps"]
            for channel_id, metrics in modality_metrics["channels"].items()
        } == {"left": 1, "right": 1}
    assert summary["smoke_activation_counts"] == {
        "astigmatism": 2,
        "emitter_2d": 2,
    }

    model = UnityPSF.from_checkpoint(checkpoint_path)
    images = torch.rand(1, 3, 8, 8)
    for channel_id in ("left", "right"):
        model.localize(
            images,
            modality="emitter_2d",
            channel_id=channel_id,
            conditions=torch.zeros(1, 2),
        )
        model.localize(
            images,
            modality="astigmatism",
            channel_id=channel_id,
            conditions=torch.zeros(1, 4),
        )
    assert model.activation_audit() == {"astigmatism": 2, "emitter_2d": 2}


def _runtime_with_scheduler(modality: str) -> ModalityTrainingRuntime:
    model = _TrainableExpert(modality)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return ModalityTrainingRuntime(
        modality=modality,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        channels={
            "left": _stream("left", (_batch("left", 0.0),)),
            "right": _stream("right", (_batch("right", 2.0),)),
        },
        step_budgets={"left": 1, "right": 1},
    )


def test_modality_shard_restores_model_optimizer_scheduler_and_channel_progress(
    tmp_path: Path,
) -> None:
    source = _runtime_with_scheduler("astigmatism")
    object.__setattr__(
        source,
        "amp_scaler",
        torch.amp.GradScaler(device="cpu", enabled=True, init_scale=128.0),
    )
    train_modality_epoch(runtime=source, epoch=3)
    expected_value = source.model.value.detach().clone()
    expected_momentum = next(iter(source.optimizer.state.values()))["momentum_buffer"].clone()
    expected_scheduler = source.scheduler.state_dict()
    channel_progress = {
        "left": {"steps": 4, "samples": 8},
        "right": {"steps": 3, "samples": 6},
    }
    shard = tmp_path / "astigmatism.resume.ckpt"

    save_modality_training_shard(
        shard,
        runtime=source,
        epoch=3,
        optimizer_steps=7,
        channel_progress=channel_progress,
        status="complete",
        provenance={"training_signature": "signature-v1"},
    )
    restored = _runtime_with_scheduler("astigmatism")
    object.__setattr__(
        restored,
        "amp_scaler",
        torch.amp.GradScaler(device="cpu", enabled=True, init_scale=2.0),
    )
    state = restore_modality_training_shard(
        shard,
        runtime=restored,
        expected_provenance={"training_signature": "signature-v1"},
    )

    assert state.modality.value == "astigmatism"
    assert state.epoch == 3
    assert state.optimizer_steps == 7
    assert state.channel_progress == channel_progress
    assert state.status == "complete"
    assert torch.equal(restored.model.value, expected_value)
    assert torch.equal(
        next(iter(restored.optimizer.state.values()))["momentum_buffer"],
        expected_momentum,
    )
    assert restored.scheduler.state_dict() == expected_scheduler
    assert restored.amp_scaler is not None
    assert restored.amp_scaler.get_scale() == 128.0

    with pytest.raises(ValueError, match="provenance"):
        restore_modality_training_shard(
            shard,
            runtime=_runtime_with_scheduler("astigmatism"),
            expected_provenance={"training_signature": "changed-signature"},
        )


def test_modality_shard_restores_channel_physical_provider_before_resuming(
    tmp_path: Path,
) -> None:
    provider_state = {
        "condition_store_version": 5,
        "condition_fingerprint": "restored-condition",
    }

    def source_stream(channel_id: str) -> ModalityChannelStream:
        return ModalityChannelStream(
            channel_id=channel_id,
            batches=lambda epoch: (_batch(channel_id, 0.0),),
            loss_fn=_loss,
            physical_state={"condition_store_version": 0},
            calibration={"version": f"{channel_id}-v1"},
            provenance={"dataset": f"{channel_id}.tif"},
            snapshot_physical_state=lambda: dict(provider_state),
        )

    source_model = _TrainableExpert("astigmatism")
    source = ModalityTrainingRuntime(
        modality="astigmatism",
        model=source_model,
        optimizer=torch.optim.SGD(source_model.parameters(), lr=0.1),
        channels={
            "left": source_stream("left"),
            "right": source_stream("right"),
        },
        step_budgets={"left": 1, "right": 1},
    )
    shard = tmp_path / "astigmatism.resume.ckpt"
    save_modality_training_shard(
        shard,
        runtime=source,
        epoch=1,
        optimizer_steps=2,
        channel_progress={
            "left": {"steps": 1, "samples": 1},
            "right": {"steps": 1, "samples": 1},
        },
        status="running",
    )

    restored_provider_states: dict[str, dict[str, object]] = {}

    def restored_stream(channel_id: str) -> ModalityChannelStream:
        live_state = {
            "condition_store_version": 0,
            "condition_fingerprint": "initial-condition",
        }

        def restore(state) -> None:
            live_state.clear()
            live_state.update(state)
            restored_provider_states[channel_id] = dict(live_state)

        return ModalityChannelStream(
            channel_id=channel_id,
            batches=lambda epoch: (_batch(channel_id, 0.0),),
            loss_fn=_loss,
            physical_state=dict(live_state),
            calibration={"version": f"{channel_id}-v1"},
            provenance={"dataset": f"{channel_id}.tif"},
            snapshot_physical_state=lambda: dict(live_state),
            restore_physical_state=restore,
        )

    restored_model = _TrainableExpert("astigmatism")
    restored = ModalityTrainingRuntime(
        modality="astigmatism",
        model=restored_model,
        optimizer=torch.optim.SGD(restored_model.parameters(), lr=0.1),
        channels={
            "left": restored_stream("left"),
            "right": restored_stream("right"),
        },
        step_budgets={"left": 1, "right": 1},
    )

    restore_modality_training_shard(shard, runtime=restored)

    assert restored_provider_states == {
        "left": provider_state,
        "right": provider_state,
    }
    assert restored.channels["left"].physical_state == provider_state
    assert restored.channels["right"].physical_state == provider_state


def test_modality_shard_rejects_physical_restore_hook_version_drift(
    tmp_path: Path,
) -> None:
    source = _runtime_with_scheduler("astigmatism")
    source.channels["left"].physical_state["condition_store_version"] = 4
    shard = tmp_path / "astigmatism.resume.ckpt"
    save_modality_training_shard(
        shard,
        runtime=source,
        epoch=1,
        optimizer_steps=2,
        channel_progress={
            "left": {"steps": 1, "samples": 1},
            "right": {"steps": 1, "samples": 1},
        },
        status="running",
    )

    restored = _runtime_with_scheduler("astigmatism")
    left = restored.channels["left"]
    object.__setattr__(
        left,
        "restore_physical_state",
        lambda state: None,
    )
    object.__setattr__(
        left,
        "snapshot_physical_state",
        lambda: {"condition_store_version": 3},
    )

    with pytest.raises(ValueError, match="condition_store_version"):
        restore_modality_training_shard(shard, runtime=restored)


def test_modality_shard_rejects_modality_and_channel_inventory_mismatches(
    tmp_path: Path,
) -> None:
    source = _runtime_with_scheduler("astigmatism")
    shard = tmp_path / "astigmatism.resume.ckpt"
    save_modality_training_shard(
        shard,
        runtime=source,
        epoch=1,
        optimizer_steps=2,
        channel_progress={
            "left": {"steps": 1, "samples": 1},
            "right": {"steps": 1, "samples": 1},
        },
        status="running",
    )

    with pytest.raises(ValueError, match="modality"):
        restore_modality_training_shard(
            shard,
            runtime=_runtime_with_scheduler("emitter_2d"),
        )

    model = _TrainableExpert("astigmatism")
    channel_mismatch = ModalityTrainingRuntime(
        modality="astigmatism",
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        channels={"left": _stream("left", (_batch("left", 0.0),))},
        step_budgets={"left": 1},
    )
    with pytest.raises(ValueError, match="channel"):
        restore_modality_training_shard(shard, runtime=channel_mismatch)


def test_completed_modality_shards_atomically_assemble_one_v2_release(
    tmp_path: Path,
) -> None:
    shard_paths: list[Path] = []
    for modality in ("emitter_2d", "astigmatism"):
        runtime = _runtime_with_scheduler(modality)
        shard = tmp_path / f"{modality}.resume.ckpt"
        save_modality_training_shard(
            shard,
            runtime=runtime,
            epoch=2,
            optimizer_steps=4,
            channel_progress={
                "left": {"steps": 2, "samples": 2},
                "right": {"steps": 2, "samples": 2},
            },
            status="complete",
        )
        shard_paths.append(shard)
    destination = tmp_path / "unitypsf_joint.ckpt"

    assemble_modality_joint_checkpoint_from_shards(
        destination,
        shard_paths=shard_paths,
        required_modalities=("emitter_2d", "astigmatism"),
        provenance={"execution": "expert_parallel"},
    )

    payload = load_modality_joint_checkpoint(destination)
    assert payload["metadata"]["checkpoint_role"] == "release"
    assert "training_state" not in payload
    assert set(payload["experts"]) == {"emitter_2d", "astigmatism"}
    assert set(payload["channel_states"]["emitter_2d"]) == {"left", "right"}
    assert set(payload["channel_states"]["astigmatism"]) == {"left", "right"}
    assert payload["provenance"]["execution"] == "expert_parallel"
    assert all(path.is_file() for path in shard_paths)
    assert not tuple(tmp_path.glob(f".{destination.name}.*.tmp"))
