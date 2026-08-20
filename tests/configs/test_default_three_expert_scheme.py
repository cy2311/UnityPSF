from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]


def test_default_three_expert_entrypoint_and_contract() -> None:
    script = ROOT / "scripts/train/unitypsf_default_3expert.sbatch"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "--gres=gpu:3" in text
    assert "unitypsf_three_modality_physical_update_300epoch.sbatch" in text

    joint = yaml.safe_load(
        (ROOT / "configs/experiments/unitypsf_three_modality_raw_tiff_300epoch.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert joint["execution"] == "expert_parallel"
    assert joint["rank_assignment"] == "one_modality_per_rank"
    assert [item["key"] for item in joint["instances"]] == [
        "emitter_2d:left",
        "emitter_2d:right",
        "astigmatism:left",
        "astigmatism:right",
        "double_helix:main",
    ]


def test_default_dh_performance_contract() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/modalities/double_helix/double_helix_raw_tiff_300epoch.yaml").read_text(
            encoding="utf-8"
        )
    )
    online = config["train"]["online_generation"]
    assert online["batch_strategy"] == "cached_window"
    assert online["vector_batch_size"] == 1024
    assert online["amp_enabled"] is True
    assert online["amp_dtype"] == "float16"
    assert online["simulation_backend"] == "lut"
    assert online["psf_type"] == "vector"
    assert config["train"]["loss"]["name"] == "dh_direct_xyz_loss"
