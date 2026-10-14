from pathlib import Path

import yaml

from unity_psf.config import load_config
import unity_psf.localization.runtime.config as runtime_config_module
from unity_psf.training.entrypoints.canonical_runtime import _bind_instance


ROOT = Path(__file__).parents[2]


def test_default_three_expert_entrypoint_and_contract() -> None:
    script = ROOT / "scripts/train/unitypsf_default_3expert.sbatch"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "--gres=gpu:3" in text
    assert "-m unity_psf.cli.train_selection" in text
    assert "--route emitter_2d:left,right" in text
    assert "--route astigmatism:left,right" in text
    assert "--route double_helix:main" in text
    assert "--require-cuda" in text

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
    assert online["simulation_backend"] == "cuda_gemm"
    assert online["psf_type"] == "vector"
    assert config["train"]["loss"]["name"] == "active_smlm_gmm_loss"
    assert config["train"]["loss"]["params"]["target_order"] == "v03"
    assert online["pxyz_target_order"] == "v03"
    assert config["train"]["model"]["name"] == "active_smlm_soft_moe_double_unet"


def test_formal_inputs_use_project_local_calibration_paths() -> None:
    astig = load_config(ROOT / "configs/modalities/astigmatism/astigmatism_dual_channel_300epoch.yaml")
    emitter = load_config(ROOT / "configs/modalities/emitter_2d/emitter_2d_dual_channel_300epoch.yaml")
    dh = load_config(ROOT / "configs/modalities/double_helix/double_helix_raw_tiff_300epoch.yaml")
    paths = [
        *(item["coeff_maps_npz"] for item in astig["train"]["online_generation"]["dual_domain_coeff_maps"]),
        *(item["coeff_maps_npz"] for item in emitter["train"]["online_generation"]["dual_domain_coeff_maps"]),
        dh["train"]["online_generation"]["dual_domain_coeff_maps"][0]["coeff_maps_npz"],
        dh["train"]["online_generation"]["pupil_carrier_complex_npz"],
    ]
    assert all(path.startswith("project://.local/formal-inputs/") for path in paths)


def test_formal_dh_runtime_contract_accepts_v03_targets(monkeypatch) -> None:
    from unity_psf.training.entrypoints.canonical_runtime import _audit_formal_runtime_contracts

    path = ROOT / "configs/modalities/double_helix/double_helix_raw_tiff_300epoch.yaml"
    monkeypatch.setattr(runtime_config_module, "_pupil_carrier_complex", lambda value, *, base_dir: object())
    runtime = runtime_config_module.build_localization_runtime_config(
        _bind_instance(load_config(path), "double_helix:main", device="cpu"),
        config_base_dir=path.parent,
    )
    evidence = _audit_formal_runtime_contracts("double_helix", {"main": runtime})
    assert evidence["main"]["target_order"] == "v03"

