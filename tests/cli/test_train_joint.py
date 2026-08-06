from __future__ import annotations

import json
from pathlib import Path

import torch

from unity_psf.cli.train_joint import main
from unity_psf.contracts import load_joint_checkpoint
from unity_psf.models import UnityPSF


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "experiments" / "unitypsf_dual_modality_multichannel_smoke.yaml"


def test_joint_training_cli_trains_three_routes_and_releases_one_checkpoint(
    tmp_path: Path,
    capsys,
) -> None:
    assert main(
        [
            "--config",
            str(CONFIG_PATH),
            "--run-root",
            str(tmp_path),
            "--run-id",
            "dual-smoke",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    checkpoint_path = tmp_path / "dual-smoke" / "checkpoints" / "unitypsf_joint.ckpt"
    summary_path = tmp_path / "dual-smoke" / "metrics" / "joint_training_summary.json"
    report_path = tmp_path / "dual-smoke" / "report" / "report.html"

    assert result["status"] == "complete"
    assert result["checkpoint"] == str(checkpoint_path)
    assert checkpoint_path.is_file()
    assert summary_path.is_file()
    assert report_path.is_file()
    assert (tmp_path / "dual-smoke" / "figures" / "00_input_audit.png").is_file()
    assert (tmp_path / "dual-smoke" / "figures" / "08_cross_modality_scorecard.png").is_file()
    assert (tmp_path / "dual-smoke" / "figures" / "07_physical_state_astigmatism_left.png").is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {"emitter_2d:main", "astigmatism:left", "astigmatism:right"}
    assert set(summary["instances"]) == expected
    assert {key: value["steps"] for key, value in summary["instances"].items()} == {
        key: 1 for key in expected
    }
    assert summary["schedule"] == ["emitter_2d:main", "astigmatism:left", "astigmatism:right"]
    assert summary["smoke_activation_counts"] == {key: 1 for key in expected}

    payload = load_joint_checkpoint(checkpoint_path)
    assert set(payload["experts"]) == expected
    assert payload["integrity"]["expert_sha256"]["astigmatism:left"] != payload["integrity"]["expert_sha256"]["astigmatism:right"]

    model = UnityPSF.from_checkpoint(checkpoint_path)
    images = torch.rand(1, 3, 8, 8)
    assert model.localize(images, modality="emitter_2d", channel_id="main").z_valid is False
    conditions = torch.zeros(1, 4)
    assert model.localize(images, modality="astigmatism", channel_id="left", conditions=conditions).z_valid is True
    assert model.localize(images, modality="astigmatism", channel_id="right", conditions=conditions).z_valid is True
