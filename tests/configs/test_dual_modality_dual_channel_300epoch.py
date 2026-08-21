from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import pytest
import yaml

from unity_psf.config import load_config, resolve_config_reference
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.runtime import ensure_run_layout
from unity_psf.training import build_trainer_runtime
from unity_psf.training.channel_context import sha256_file
from unity_psf.training.joint_config import bind_instance, instance_specs, load_joint_config
from unity_psf.training.modality_runtime import build_modality_runtime, modality_groups


CONFIG_DIR = Path(__file__).parents[2] / "configs"
FORMAL_DENSITY_UM2 = 0.5
FORMAL_BACKGROUND_PHOTONS = 110.0
FORMAL_PHOTON_MEAN_SIGMA = [20000.0, 1000.0]
FORMAL_PHOTON_RANGE = [0.0, 31000.0]


def _write_test_coeff_map(path: Path, value: float) -> Path:
    mode_order = np.asarray(
        [(2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0), (3, -3), (3, 3)],
        dtype=np.int64,
    )
    maps = np.full((len(mode_order), 96, 96), float(value), dtype=np.float32)
    np.savez_compressed(path, zernike_maps_nm=maps, mode_order=mode_order)
    return path


def _load(name: str) -> dict:
    matches = list(CONFIG_DIR.rglob(name))
    assert len(matches) == 1, matches
    value = yaml.safe_load(matches[0].read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_formal_joint_config_declares_dual_channels_and_two_rank_assignment() -> None:
    config = _load("unitypsf_dual_modality_dual_channel_300epoch.yaml")

    assert config["execution"] == "expert_parallel"
    assert config["rank_assignment"] == "one_modality_per_rank"
    assert config["epochs"] == 300
    assert [item["key"] for item in config["instances"]] == [
        "emitter_2d:left",
        "emitter_2d:right",
        "astigmatism:left",
        "astigmatism:right",
    ]
    assert {item["step_budget"] for item in config["instances"]} == {417}
    assert config["instances"][0]["model_seed"] == config["instances"][1]["model_seed"]
    assert config["instances"][2]["model_seed"] == config["instances"][3]["model_seed"]
    assert len({item["data_seed"] for item in config["instances"]}) == 4


def test_three_modality_dh_contract_matches_astigmatism_epoch_exposure() -> None:
    joint = _load("unitypsf_three_modality_raw_tiff_300epoch.yaml")
    dh = _load("double_helix_raw_tiff_300epoch.yaml")["train"]
    astigmatism = _load("astigmatism_dual_channel_300epoch.yaml")["train"]

    dh_instance = next(item for item in joint["instances"] if item["key"] == "double_helix:main")
    assert dh_instance["step_budget"] == 417
    assert dh["batch_size"] == 24
    online = dh["online_generation"]
    assert online["steps_per_epoch"] == 417
    assert dh["batch_size"] * online["steps_per_epoch"] == 10_008
    assert astigmatism["batch_size"] * astigmatism["online_generation"]["steps_per_epoch"] == 10_008
    assert dh["online_generation"]["simulation_backend"] == "lut"
    assert dh["online_generation"]["psf_type"] == "vector"
    assert dh["online_generation"]["vector_psf_size"] == 21
    assert "bead_01_complex_pupil.npz" in dh["online_generation"]["pupil_carrier_complex_npz"]
    assert "target_npz_path" not in dh["online_generation"]


def test_gpu_gate_reuses_formal_instance_contracts_with_amp_warmup_steps() -> None:
    config = _load("unitypsf_dual_modality_dual_channel_gpu_smoke.yaml")

    assert config["epochs"] == 1
    assert {item["config"] for item in config["instances"]} == {
        "project://configs/modalities/emitter_2d/emitter_2d_dual_channel_300epoch.yaml",
        "project://configs/modalities/astigmatism/astigmatism_dual_channel_300epoch.yaml",
    }
    assert {item["key"]: item["step_budget"] for item in config["instances"]} == {
        "emitter_2d:left": 8,
        "emitter_2d:right": 8,
        "astigmatism:left": 8,
        "astigmatism:right": 8,
    }
    assert config["metadata"]["formal_contract_gate"] is True
    assert [item["key"] for item in config["instances"]] == [
        "emitter_2d:left",
        "emitter_2d:right",
        "astigmatism:left",
        "astigmatism:right",
    ]


def test_formal_2d_config_uses_sample_derived_field_dependent_vector_lut() -> None:
    config = _load("emitter_2d_single_channel_300epoch.yaml")
    train = config["train"]
    online = train["online_generation"]

    assert config["optical"]["pixel_size_nm_x"] == 101.11
    assert config["optical"]["pixel_size_nm_y"] == 98.83
    assert train["model"]["params"]["condition_dim"] == 8
    assert online["conditioning_profile"] == "zero"
    assert online["z_range"] == [-0.1, 0.1]
    assert online["psf_type"] == "vector"
    assert online["simulation_backend"] == "lut"
    assert online["vector_psf_size"] == 15
    assert online["nat_grid_z_steps"] == 21
    assert online["lut_simulation"]["field_mode"] == "global_field"
    assert online["field_origin_sampling_mode"] == "sliding_window"
    assert online["field_origin_stride_px"] == 88
    assert "empirical_psf_path" not in online
    assert online["batch_strategy"] == "cached_window"
    assert online["simulation_output_device"] == "renderer"
    assert train["model"]["params"]["disabled_attr"] == [3]
    assert [channel["id"] for channel in train["channel_layout"]["channels"]] == ["left"]
    assert [channel["crop"] for channel in train["channel_layout"]["channels"]] == [
        [0, 0, 600, 1200]
    ]
    sample_root = Path(config["metadata"]["real_sample_root"])
    raw_sample = Path(config["metadata"]["real_sample_acquisition"])
    assert sample_root.is_dir()
    assert raw_sample.is_dir()
    assert config["metadata"]["real_sample_frames"] == 8000
    assert config["metadata"]["psf_initialization"] == (
        "explicit_zero_zernike_anchor_then_origami_sample_peak_nat"
    )
    serialized = next(CONFIG_DIR.rglob("emitter_2d_single_channel_300epoch.yaml")).read_text(encoding="utf-8").lower()
    assert "bead" not in serialized
    assert "psf_aligned" not in serialized
    assert "neptune_v0.3" not in serialized
    assert "origami_2d_right" not in serialized
    coeff_maps = online["dual_domain_coeff_maps"]
    assert [entry["name"] for entry in coeff_maps] == ["left"]
    assert all("unity/output/unitypsf/calibration" in entry["coeff_maps_npz"] for entry in coeff_maps)


def test_formal_astigmatism_config_keeps_left_right_channel_contract() -> None:
    config = _load("astigmatism_dual_channel_300epoch.yaml")
    train = config["train"]

    assert train["model"]["name"] == "astigmatism_expert"
    assert train["model"]["params"]["condition_dim"] == 8
    assert train["loss"] == {
        "name": "active_smlm_gmm_loss",
        "params": {
            "photon_scale": 31000.0,
            "z_scale": 0.6,
            "gmm_target_chunk": 4,
            "gmm_component_chunk": 64,
            "gmm_backend": "mixture_same_family",
            "target_order": "legacy_iwae",
        },
    }
    assert train["optimizer"] == {
        "name": "adamw",
        "params": {"lr": 0.0006, "weight_decay": 0.1},
    }
    online = train["online_generation"]
    assert online["conditioning_profile"] == "astigmatism_660nm"
    assert online["z_range"] == [-0.6, 0.6]
    assert online["batch_strategy"] == "cached_window"
    assert online["simulation_backend"] == "lut"
    assert online["simulation_output_device"] == "renderer"
    assert online["vector_psf_size"] == 25
    assert online["nat_grid_size"] == [2, 2]
    assert online["lut_simulation"]["field_mode"] == "global_field"
    assert online["field_origin_sampling_mode"] == "sliding_window"
    assert online["field_origin_stride_px"] == 88
    assert online["grad_clip_norm"] == 0.03
    assert online["amp_enabled"] is True
    assert online["amp_dtype"] == "float16"
    assert [channel["id"] for channel in train["channel_layout"]["channels"]] == ["left", "right"]
    assert {channel["anchor_profile"] for channel in train["channel_layout"]["channels"]} == {
        "astigmatism_660nm"
    }
    coeff_maps = online["dual_domain_coeff_maps"]
    assert [entry["name"] for entry in coeff_maps] == ["left", "right"]
    assert len({entry["coeff_maps_npz"] for entry in coeff_maps}) == 2
    assert all(Path(entry["coeff_maps_npz"]).is_file() for entry in coeff_maps)
    assert Path(config["metadata"]["real_sample_tiff"]).is_file()


def test_formal_instance_configs_use_temporal_camera_simulation() -> None:
    for name in ("emitter_2d_single_channel_300epoch.yaml", "astigmatism_dual_channel_300epoch.yaml"):
        config = _load(name)
        camera = config["camera"]
        online = config["train"]["online_generation"]

        assert camera["qe"] == 0.9
        assert camera["spurious_charge"] == 0.002
        assert camera["em_gain"] == 1.0
        assert camera["e_per_adu"] == 1.020784562122306
        if name.startswith("emitter_2d"):
            assert camera["read_sigma"] == pytest.approx(3.3170624029816382**0.5)
            assert camera["baseline"] == 99.78090423787435
        else:
            assert camera["read_sigma"] == 0.0
            assert camera["baseline"] == 398.6
        assert online["batch_strategy"] == "cached_window"
        assert online["simulation_backend"] == "lut"
        assert online["sequence_count"] == 64
        expected_cached_sequences = 2 if name.startswith("astigmatism") else 4
        assert online["cached_window_max_gpu_sequences"] == expected_cached_sequences
        assert online["lifetime_avg"] == 1.0
        assert online["warmup_frames"] == 6.0
        assert config["metadata"]["formal_training_contract"] == "unitypsf.formal.v1"


def test_formal_instance_configs_share_density_background_and_photon_contract() -> None:
    configs = [
        _load("emitter_2d_single_channel_300epoch.yaml"),
        _load("astigmatism_dual_channel_300epoch.yaml"),
    ]

    for config in configs:
        train = config["train"]
        online = train["online_generation"]
        assert online["emitter_density_um2"] == FORMAL_DENSITY_UM2
        assert online["background_range"] == [
            FORMAL_BACKGROUND_PHOTONS,
            FORMAL_BACKGROUND_PHOTONS,
        ]
        assert online["photon_mean_sigma"] == FORMAL_PHOTON_MEAN_SIGMA
        assert online["photon_range"] == FORMAL_PHOTON_RANGE
        assert train["loss"]["params"]["photon_scale"] == 31000.0
        assert train["scaling"]["photon_max"] == 31000.0
        assert train["scaling"]["bg_max"] == 240.0


def test_formal_runtime_audit_rejects_optimizer_loss_and_physical_state_drift(tmp_path: Path) -> None:
    from unity_psf.training.modality_runtime import audit_formal_runtime_contracts

    joint_path = CONFIG_DIR / "experiments/unitypsf_dual_modality_dual_channel_300epoch.yaml"
    specs = instance_specs(load_joint_config(joint_path, expected_execution="expert_parallel"))
    astigmatism_configs = {}
    for key, spec in specs.items():
        if not key.startswith("astigmatism:"):
            continue
        instance_path = resolve_config_reference(spec["config"], source_path=joint_path)
        bound = bind_instance(load_config(instance_path), key, device="cpu")
        astigmatism_configs[key.split(":", maxsplit=1)[1]] = build_localization_runtime_config(
            bound,
            config_base_dir=CONFIG_DIR,
            seed=3,
        )

    evidence = audit_formal_runtime_contracts("astigmatism", astigmatism_configs)

    assert set(evidence) == {"left", "right"}
    assert all(item["optimizer"] == "AdamW" for item in evidence.values())
    assert all(item["scheduler"] == "StepLR" for item in evidence.values())
    assert all(item["loss"] == "active_smlm_gmm_loss" for item in evidence.values())
    assert all(item["target_order"] == "legacy_iwae" for item in evidence.values())
    assert len({item["coefficient_map_sha256"] for item in evidence.values()}) == 2

    wrong_optimizer = deepcopy(astigmatism_configs)
    wrong_optimizer["left"]["optimizer"] = {"name": "sgd", "params": {"lr": 0.001}}
    with pytest.raises(ValueError, match="AdamW"):
        audit_formal_runtime_contracts("astigmatism", wrong_optimizer)

    wrong_lr = deepcopy(astigmatism_configs)
    wrong_lr["left"]["optimizer"]["params"]["lr"] = 0.001
    with pytest.raises(ValueError, match="lr=0.0006"):
        audit_formal_runtime_contracts("astigmatism", wrong_lr)

    wrong_loss = deepcopy(astigmatism_configs)
    wrong_loss["left"]["loss"] = {"name": "active_smlm_loss", "params": {}}
    with pytest.raises(ValueError, match="GMM"):
        audit_formal_runtime_contracts("astigmatism", wrong_loss)

    wrong_gmm = deepcopy(astigmatism_configs)
    wrong_gmm["left"]["loss"]["params"]["gmm_backend"] = "manual"
    with pytest.raises(ValueError, match="mixture_same_family"):
        audit_formal_runtime_contracts("astigmatism", wrong_gmm)

    wrong_scheduler = deepcopy(astigmatism_configs)
    wrong_scheduler["left"]["resolved_contract"]["training_runtime"]["scheduler"]["params"]["step_size"] = 1000
    with pytest.raises(ValueError, match="StepLR"):
        audit_formal_runtime_contracts("astigmatism", wrong_scheduler)

    empty_physics = deepcopy(astigmatism_configs)
    empty_physics["right"]["batch_provider"]["params"]["dual_domain_coeff_maps"] = ()
    with pytest.raises(ValueError, match="coefficient map"):
        audit_formal_runtime_contracts("astigmatism", empty_physics)

    emitter_configs = {}
    for key, spec in specs.items():
        if not key.startswith("emitter_2d:"):
            continue
        instance_path = resolve_config_reference(spec["config"], source_path=joint_path)
        bound = bind_instance(load_config(instance_path), key, device="cpu")
        emitter_configs[key.split(":", maxsplit=1)[1]] = build_localization_runtime_config(
            bound,
            config_base_dir=CONFIG_DIR,
            seed=3,
        )
    path = _write_test_coeff_map(tmp_path / "emitter_left.npz", 1.0)
    emitter_configs["left"]["batch_provider"]["params"]["dual_domain_coeff_maps"] = (
        {"name": "left", "coeff_maps_npz": str(path)},
    )

    emitter_evidence = audit_formal_runtime_contracts("emitter_2d", emitter_configs)

    assert {item["psf_type"] for item in emitter_evidence.values()} == {"vector"}
    assert {item["simulation_backend"] for item in emitter_evidence.values()} == {"lut"}
    assert {tuple(item["z_range_um"]) for item in emitter_evidence.values()} == {(-0.1, 0.1)}
    assert set(emitter_evidence) == {"left", "right"}
    assert {item["emitter_density_um2"] for item in emitter_evidence.values()} == {0.5}
    assert {tuple(item["background_range_photons"]) for item in emitter_evidence.values()} == {
        (110.0, 110.0)
    }
    assert all(
        item["target_active_emitters_per_frame"] == pytest.approx(46.0463675904)
        for item in emitter_evidence.values()
    )
    assert all(item["recenter_mode"] == "fd_deeploc_exact_recenter" for item in emitter_evidence.values())
    assert all(
        item["train_background_adu"] == pytest.approx(196.76512958133938)
        for item in emitter_evidence.values()
    )

    assert {item["emitter_density_um2"] for item in evidence.values()} == {0.5}
    assert {tuple(item["background_range_photons"]) for item in evidence.values()} == {
        (110.0, 110.0)
    }
    assert all(
        item["target_active_emitters_per_frame"] == pytest.approx(46.0463675904)
        for item in evidence.values()
    )
    assert all(item["recenter_mode"] == "fd_deeploc_exact_recenter" for item in evidence.values())
    assert all(
        item["train_background_adu"] == pytest.approx(495.58422534346505)
        for item in evidence.values()
    )

    wrong_emitter_psf = deepcopy(emitter_configs)
    wrong_emitter_psf["left"]["batch_provider"]["params"]["psf_type"] = "empirical_focal"
    with pytest.raises(ValueError, match="vector LUT"):
        audit_formal_runtime_contracts("emitter_2d", wrong_emitter_psf)

    wrong_density = deepcopy(emitter_configs)
    wrong_density["left"]["batch_provider"]["params"]["emitter_density_um2"] = None
    with pytest.raises(ValueError, match="density=0.5"):
        audit_formal_runtime_contracts("emitter_2d", wrong_density)

    wrong_background = deepcopy(astigmatism_configs)
    wrong_background["right"]["batch_provider"]["params"]["background_range"] = (40.0, 120.0)
    with pytest.raises(ValueError, match="background=110"):
        audit_formal_runtime_contracts("astigmatism", wrong_background)


def test_formal_runtime_audit_uses_modality_specific_channel_contracts() -> None:
    from unity_psf.training.modality_runtime import audit_formal_runtime_contracts

    joint_path = CONFIG_DIR / "experiments/unitypsf_dual_modality_dual_channel_300epoch.yaml"
    specs = instance_specs(load_joint_config(joint_path, expected_execution="expert_parallel"))
    runtime_configs = {}
    for key, spec in specs.items():
        instance_path = resolve_config_reference(spec["config"], source_path=joint_path)
        bound = bind_instance(load_config(instance_path), key, device="cpu")
        runtime_configs[key] = build_localization_runtime_config(
            bound,
            config_base_dir=CONFIG_DIR,
            seed=int(spec["data_seed"]),
        )

    emitter_evidence = audit_formal_runtime_contracts(
        "emitter_2d",
        {
            "left": runtime_configs["emitter_2d:left"],
            "right": runtime_configs["emitter_2d:right"],
        },
    )

    assert set(emitter_evidence) == {"left", "right"}
    with pytest.raises(ValueError, match="left and right"):
        audit_formal_runtime_contracts(
            "emitter_2d",
            {"left": runtime_configs["emitter_2d:left"]},
        )
    with pytest.raises(ValueError, match="left and right"):
        audit_formal_runtime_contracts(
            "astigmatism",
            {"left": runtime_configs["astigmatism:left"]},
        )


def test_formal_sampling_and_recenter_contract_enters_channel_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    from unity_psf.training.modality_runtime import channel_metadata

    joint_path = CONFIG_DIR / "experiments/unitypsf_dual_modality_dual_channel_300epoch.yaml"
    specs = instance_specs(load_joint_config(joint_path, expected_execution="expert_parallel"))
    spec = specs["emitter_2d:left"]
    config_path = resolve_config_reference(spec["config"], source_path=joint_path)
    bound = bind_instance(load_config(config_path), "emitter_2d:left", device="cpu")
    runtime_config = build_localization_runtime_config(
        bound,
        config_base_dir=CONFIG_DIR,
        seed=int(spec["data_seed"]),
    )
    coefficient_map = _write_test_coeff_map(tmp_path / "metadata_left.npz", 1.0)
    runtime_config["batch_provider"]["params"]["dual_domain_coeff_maps"] = (
        {"name": "left", "coeff_maps_npz": str(coefficient_map)},
    )
    physical_state_path = tmp_path / "current_physical_state.json"
    physical_state_path.write_text('{"schema_version": "test"}\n', encoding="utf-8")

    _, calibration, provenance = channel_metadata(
        runtime_config,
        physical_context=SimpleNamespace(physical_state_path=physical_state_path),
        config_path=config_path,
        key="emitter_2d:left",
        data_seed=int(spec["data_seed"]),
        experiment_metadata=bound["metadata"],
    )

    assert provenance["emitter_density_um2"] == 0.5
    assert provenance["target_active_emitters_per_frame"] == pytest.approx(46.0463675904)
    assert provenance["train_background_photons"] == 110.0
    assert provenance["photon_mean"] == 20000.0
    assert provenance["photon_sigma"] == 1000.0
    assert provenance["photon_range"] == [0.0, 31000.0]
    assert calibration["camera_baseline_adu"] == pytest.approx(99.78090423787435)
    assert calibration["train_background_adu"] == pytest.approx(196.76512958133938)
    assert calibration["recenter_mode"] == "fd_deeploc_exact_recenter"
    assert calibration["psf_type"] == "vector"
    assert calibration["coefficient_map"] == str(coefficient_map.resolve())
    assert calibration["coefficient_map_sha256"] == sha256_file(coefficient_map)


def test_formal_joint_config_builds_all_dual_channel_training_runtimes(tmp_path: Path) -> None:
    joint_path = CONFIG_DIR / "experiments/unitypsf_dual_modality_dual_channel_300epoch.yaml"
    specs = instance_specs(load_joint_config(joint_path, expected_execution="expert_parallel"))

    for key, spec in specs.items():
        instance_path = resolve_config_reference(spec["config"], source_path=joint_path)
        bound = bind_instance(load_config(instance_path), key, device="cpu")
        bound["train"]["online_generation"]["simulation_backend"] = "native"
        bound["train"]["online_generation"]["simulation_output_device"] = "cpu"
        if key.startswith("emitter_2d:"):
            channel_id = key.split(":", maxsplit=1)[1]
            path = _write_test_coeff_map(
                tmp_path / f"runtime_{channel_id}.npz",
                1.0 if channel_id == "left" else 2.0,
            )
            bound["train"]["online_generation"]["dual_domain_coeff_maps"] = [
                {"name": channel_id, "coeff_maps_npz": str(path)}
            ]
        runtime_config = build_localization_runtime_config(bound, config_base_dir=CONFIG_DIR, seed=3)
        runtime = build_trainer_runtime(
            runtime_config,
            layout=ensure_run_layout(tmp_path, key.replace(":", "_")),
            model_registry=build_localization_model_registry(),
        )

        assert runtime_config["expert_instance"]["channel_id"] == key.split(":", maxsplit=1)[1]
        assert runtime.config.stop_epoch == 300
        assert runtime.config.scheduler_step_unit == "epoch"
        assert runtime.config.grad_clip_norm == 0.03
        assert runtime.config.amp_enabled is True
        assert isinstance(runtime.optimizer, torch.optim.AdamW)
        assert runtime.loss_fn is not None
        assert runtime_config["loss"]["name"] == "active_smlm_gmm_loss"
        assert runtime_config["loss"]["params"]["target_order"] == "legacy_iwae"
        if key.startswith("astigmatism:"):
            entries = runtime_config["batch_provider"]["params"]["dual_domain_coeff_maps"]
            assert len(entries) == 1
            assert entries[0]["name"] == key.split(":", maxsplit=1)[1]
        else:
            provider = runtime_config["batch_provider"]["params"]
            assert provider["psf_type"] == "vector"
            assert provider["simulation_backend"] == "native"
            entries = provider["dual_domain_coeff_maps"]
            assert len(entries) == 1
            assert entries[0]["name"] == key.split(":", maxsplit=1)[1]


def test_formal_runtime_builder_audits_dual_channel_emitter_and_astigmatism(
    tmp_path: Path,
) -> None:
    joint_path = CONFIG_DIR / "experiments/unitypsf_dual_modality_dual_channel_gpu_smoke.yaml"
    specs = instance_specs(load_joint_config(joint_path, expected_execution="expert_parallel"))
    grouped = modality_groups(specs)

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    fake_runtime = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_provider=lambda epoch: (),
        loss_fn=SimpleNamespace(from_output=lambda output, target: output),
        config=SimpleNamespace(
            scheduler_step_unit="epoch",
            grad_clip_norm=0.03,
            amp_enabled=True,
            amp_dtype=torch.float16,
        ),
    )

    with patch("unity_psf.training.modality_runtime.build_trainer_runtime", return_value=fake_runtime), patch(
        "unity_psf.training.modality_runtime.build_runtime_batch_provider",
        return_value=lambda epoch: (),
    ), patch(
        "unity_psf.training.modality_runtime.build_runtime_loss",
        return_value=SimpleNamespace(from_output=lambda output, target: output),
    ), patch(
        "unity_psf.training.modality_runtime.prepare_instance_runtime",
        side_effect=lambda runtime, runtime_config, config_base_dir: (runtime, None),
    ), patch(
        "unity_psf.training.modality_runtime.build_localizer_eval_provider",
        return_value=None,
    ):
        for modality, entries in grouped.items():
            runtime, runtime_configs, evidence = build_modality_runtime(
                modality,
                entries,
                joint_path=joint_path,
                run_layout=ensure_run_layout(tmp_path, modality.value),
                device="cpu",
            )
            assert set(runtime.channels) == {"left", "right"}
            assert set(runtime_configs) == {"left", "right"}
            assert evidence is not None
            assert set(evidence) == {"left", "right"}
            assert runtime.model is model
            assert runtime.optimizer is optimizer


def test_formal_slurm_entrypoint_requires_two_gpus_and_no_cpu_fallback() -> None:
    script = (CONFIG_DIR.parent / "scripts" / "train" / "unitypsf_dual_modality_dual_channel_2gpu_300epoch.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --gres=gpu:2" in script
    assert 'PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/guest/Others/main/race/unity}"' in script
    assert "--nproc-per-node=2" in script
    assert "--allow-cpu-smoke" not in script
    assert "CUDA unavailable: formal training refuses CPU fallback" in script
    assert "unitypsf_dual_modality_dual_channel_300epoch.yaml" in script
    assert '--coordination-timeout-seconds "${UNITYPSF_COORDINATION_TIMEOUT_SECONDS:-21600}"' in script
