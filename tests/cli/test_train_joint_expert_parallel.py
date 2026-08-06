from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from unity_psf.contracts import PSFModality, load_modality_joint_checkpoint
from unity_psf.runtime import ensure_run_layout
from unity_psf.training import atomic_write_json


PROJECT_ROOT = Path(__file__).parents[2]
EMITTER_CONFIG_PATH = PROJECT_ROOT / "configs" / "modalities" / "emitter_2d" / "emitter_2d_single_channel_smoke.yaml"
ASTIGMATISM_CONFIG_PATH = PROJECT_ROOT / "configs" / "modalities" / "astigmatism" / "astigmatism_single_channel_smoke.yaml"


def test_two_rank_smoke_trains_one_shared_multichannel_expert_per_modality(
    tmp_path: Path,
) -> None:
    eval_configs = {}
    for modality, source, match_dims, eval_seed in (
        ("emitter_2d", EMITTER_CONFIG_PATH, 2, 9101),
        ("astigmatism", ASTIGMATISM_CONFIG_PATH, 3, 9201),
    ):
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["train"]["eval"] = {
            "enabled": True,
            "source": "online_generation",
            "seed": eval_seed,
            "batch_count": 1,
            "batch_size": 1,
            "match_dims": match_dims,
            "dist_tol_xy_nm": 250.0,
            "dist_tol_z_nm": None if match_dims == 2 else 500.0,
        }
        eval_config_path = tmp_path / f"{modality}_heldout.yaml"
        eval_config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        eval_configs[modality] = eval_config_path
    config_path = tmp_path / "dual-modality-dual-channel.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "unitypsf.joint_training.v1",
                "execution": "expert_parallel",
                "rank_assignment": "one_modality_per_rank",
                "epochs": 1,
                "instances": [
                    {
                        "key": "emitter_2d:left",
                        "config": str(eval_configs["emitter_2d"]),
                        "model_seed": 101,
                        "data_seed": 111,
                        "step_budget": 1,
                    },
                    {
                        "key": "emitter_2d:right",
                        "config": str(eval_configs["emitter_2d"]),
                        "model_seed": 101,
                        "data_seed": 112,
                        "step_budget": 1,
                    },
                    {
                        "key": "astigmatism:left",
                        "config": str(eval_configs["astigmatism"]),
                        "model_seed": 201,
                        "data_seed": 211,
                        "step_budget": 1,
                    },
                    {
                        "key": "astigmatism:right",
                        "config": str(eval_configs["astigmatism"]),
                        "model_seed": 201,
                        "data_seed": 212,
                        "step_budget": 1,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
            "UNITYPSF_ATTEMPT_ID": "step3-smoke-attempt",
        }
    )
    command = [
        sys.executable,
        "-m",
        "unity_psf.cli.train_joint_expert_parallel",
        "--config",
        str(config_path),
        "--run-root",
        str(tmp_path),
        "--run-id",
        "expert-parallel-smoke",
        "--allow-cpu-smoke",
    ]
    for rank in (1, 0):
        rank_environment = {
            **environment,
            "RANK": str(rank),
            "WORLD_SIZE": "2",
            "LOCAL_RANK": str(rank),
        }
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=rank_environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    run_dir = tmp_path / "expert-parallel-smoke"
    checkpoint = run_dir / "checkpoints" / "unitypsf_joint.ckpt"
    summary = json.loads((run_dir / "metrics" / "joint_training_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["execution"] == "expert_parallel"
    assert summary["world_size"] == 2
    assert isinstance(summary["cuda_confirmed"], bool)
    assert (run_dir / "report" / "report.html").is_file()
    assert (run_dir / "figures" / "09_heldout_metrics_scorecard.png").is_file()
    rank_statuses = [
        json.loads((run_dir / "metadata" / f"rank_{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert {item["owned_modality"] for item in rank_statuses} == {
        "emitter_2d",
        "astigmatism",
    }
    assert all(set(item["channels"]) == {"left", "right"} for item in rank_statuses)
    assert {item["attempt_id"] for item in rank_statuses} == {"step3-smoke-attempt"}
    assert len({item["training_signature"] for item in rank_statuses}) == 1
    for status in rank_statuses:
        shard = load_modality_joint_checkpoint(status["modality_checkpoint"])
        assert shard["provenance"]["training_signature"] == status["training_signature"]

    payload = load_modality_joint_checkpoint(checkpoint)
    assert set(payload["experts"]) == {"emitter_2d", "astigmatism"}
    assert set(payload["channel_states"]["emitter_2d"]) == {"left", "right"}
    assert set(payload["channel_states"]["astigmatism"]) == {"left", "right"}
    assert set(summary["modalities"]) == {"emitter_2d", "astigmatism"}
    assert summary["rank_assignments"] == {
        "0": "emitter_2d",
        "1": "astigmatism",
    }
    assert summary["attempt_id"] == "step3-smoke-attempt"
    assert summary["smoke_activation_counts"] == {
        "astigmatism": 2,
        "emitter_2d": 2,
    }
    for modality in ("emitter_2d", "astigmatism"):
        assert set(summary["modalities"][modality]["channels"]) == {"left", "right"}
        assert summary["modalities"][modality]["optimizer_steps"] == 2
        heldout = summary["modalities"][modality]["heldout_history"][-1]
        assert heldout["epoch"] == 1
        assert set(heldout["channels"]) == {"left", "right"}
        for channel_id in ("left", "right"):
            metrics = heldout["channels"][channel_id]
            assert {
                "eval_loss",
                "precision",
                "recall",
                "Jaccard",
                "RMSE_XY_nm",
                "RMSE_Z_nm",
                "photon_relative_error",
                "sample_count",
                "route_count",
                "optimizer_steps",
            } <= set(metrics)
            assert metrics["sample_count"] == 1
            assert metrics["route_count"] == 1
        if modality == "emitter_2d":
            assert heldout["modality"]["RMSE_Z_nm"] is None
        elif heldout["modality"]["true_positive"] == 0:
            assert heldout["modality"]["RMSE_Z_nm"] is None
        else:
            assert isinstance(heldout["modality"]["RMSE_Z_nm"], float)
    validation_summary = json.loads(
        (run_dir / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert all(
        item["status"] == "heldout-evaluated"
        for item in validation_summary["instances"].values()
    )
    assert set(validation_summary["modality_metrics"]) == {
        "emitter_2d",
        "astigmatism",
    }
    for item in validation_summary["instances"].values():
        assert item["heldout_metrics"] is not None
        assert {
            "precision",
            "recall",
            "Jaccard",
            "RMSE_XY_nm",
            "RMSE_Z_nm",
            "photon_relative_error",
        } == set(item["heldout_metrics"])


def test_expert_parallel_training_source_does_not_depend_on_distributed_barrier() -> None:
    import unity_psf.cli.train_joint_expert_parallel as expert_parallel
    import unity_psf.cli.train_modality_expert_parallel as implementation

    source = inspect.getsource(expert_parallel) + inspect.getsource(implementation)

    assert "dist.barrier(" not in source
    assert ".barrier(" not in source
    assert "init_process_group" not in source


def test_rank_zero_fails_fast_when_current_attempt_reports_failed_rank(
    tmp_path: Path,
) -> None:
    from unity_psf.cli.train_modality_expert_parallel import _read_completed_rank_statuses

    layout = ensure_run_layout(tmp_path, "failed-rank")
    atomic_write_json(
        layout.metadata_dir / "rank_1.json",
        {
            "rank": 1,
            "owned_modality": "astigmatism",
            "status": "failed",
            "attempt_id": "attempt-2",
            "training_signature": "signature-2",
            "error": "synthetic failure",
        },
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        _read_completed_rank_statuses(
            layout,
            assignments=(PSFModality.EMITTER_2D, PSFModality.ASTIGMATISM),
            attempt_id="attempt-2",
            training_signature="signature-2",
            timeout_seconds=1.0,
        )


def test_partial_channel_heldout_eval_is_rejected() -> None:
    from unity_psf.cli.train_modality_expert_parallel import _heldout_eval_enabled

    runtime = SimpleNamespace(
        channels={
            "left": SimpleNamespace(heldout_eval=lambda model: {}),
            "right": SimpleNamespace(heldout_eval=None),
        }
    )

    with pytest.raises(ValueError, match="all channels or none"):
        _heldout_eval_enabled(runtime)


def test_resume_rejects_divergent_modality_heldout_history_copies() -> None:
    from unity_psf.cli.train_modality_expert_parallel import _metrics_from_progress

    runtime = SimpleNamespace(channels={"left": object(), "right": object()})
    progress = {
        "left": {"modality_heldout_history": [{"epoch": 1}]},
        "right": {"modality_heldout_history": [{"epoch": 2}]},
    }

    with pytest.raises(ValueError, match="modality held-out history"):
        _metrics_from_progress(runtime, epoch=1, optimizer_steps=2, progress=progress)
