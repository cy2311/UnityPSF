from __future__ import annotations

from pathlib import Path
import sys

from unity_psf.contracts import CheckpointMetadata, load_checkpoint
from unity_psf.training.run_localization import main


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "modalities" / "emitter_2d" / "emitter_2d_single_channel_smoke.yaml"


def test_localization_entrypoint_trains_configured_expert_and_writes_instance_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unity-psf-train-localization",
            "--config",
            str(CONFIG_PATH),
            "--run-root",
            str(tmp_path),
            "--run-name",
            "emitter-main",
            "--seed",
            "23",
        ],
    )

    assert main() == 0

    checkpoint = load_checkpoint(tmp_path / "emitter-main" / "checkpoints" / "checkpoint_latest.pt")
    metadata = CheckpointMetadata.from_dict(checkpoint["metadata"])
    assert metadata.model_name == "emitter_2d_expert"
    assert metadata.checkpoint_role == "instance"
    assert metadata.expert_type.value == "emitter_2d"
    assert metadata.instance_id == "main"
    assert metadata.channel_spec is not None
    assert metadata.channel_spec.channel_id == "main"
    assert metadata.parent_checkpoint_hash is not None
    assert any(key.startswith("network.") for key in checkpoint["model_state_dict"])
