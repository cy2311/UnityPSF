from __future__ import annotations

from pathlib import Path

import pytest

from unity_psf.config import materialize_config
from unity_psf.config.materialize import _apply_assignment


ROOT = Path(__file__).parents[2]
CONFIGS = ROOT / "configs"


def test_formal_cuda_gemm_override_materializes_without_launcher_yaml(monkeypatch, tmp_path: Path) -> None:
    raw_tiff = tmp_path / "raw.tif"
    left_map = tmp_path / "left.npz"
    right_map = tmp_path / "right.npz"
    for path in (raw_tiff, left_map, right_map):
        path.touch()
    monkeypatch.setenv("UNITYPSF_RAW_TIFF", str(raw_tiff))
    monkeypatch.setenv("UNITYPSF_LEFT_MAP", str(left_map))
    monkeypatch.setenv("UNITYPSF_RIGHT_MAP", str(right_map))

    config = materialize_config(
        CONFIGS / "base/microtube.yaml",
        CONFIGS / "overrides/standard_roi_gamma.yaml",
    )
    online = config["train"]["online_generation"]
    gamma = config["train"]["roi_bank_gamma"]

    assert config["phase_retrieval"]["input"]["tiff_path"] == str(raw_tiff)
    assert online["simulation_backend"] == "cuda_gemm"
    assert online["batch_strategy"] == "cached_window"
    assert "cached_window_cache_epochs" not in online
    assert online["cached_window_prefetch_batches"] == 1
    assert online["photon_range"] == [0.0, 31000.0]
    assert online["photon_mean_sigma"] == [20000.0, 1000.0]
    assert online["condition_feature_dim"] == 13
    assert online["condition_dim"] == 15
    assert online["expert_mode"] == "soft_moe"
    assert online["fixed_astig_anchor_nm"] is None
    assert gamma["start_epoch"] == 30
    assert gamma["update_interval_epochs"] == 10
    assert gamma["roi_library_max_rois"] == 2048
    assert "roi_bank_max_rois" not in gamma
    assert gamma["target_projected_emitters"] == 5000
    assert not {
        "warmup_epochs",
        "roi_library_start_epoch",
        "roi_library_size",
        "roi_physical_update_rois",
        "roi_physical_update_target_emitters",
        "roi_physical_update_interval_epochs",
    } & set(config["metadata"])

    for name in ("unitypsf_cuda_gemm_from_scratch.sbatch", "unitypsf_cuda_gemm_roi_gamma.sbatch"):
        launcher = (ROOT / "scripts/train" / name).read_text(encoding="utf-8")
        assert "cat <<YAML" not in launcher
        assert "yaml.safe_load" not in launcher
        assert "assert train" not in launcher
        assert "configs/base/microtube.yaml" in launcher
        assert "configs/overrides/" in launcher

    budget_launcher = (ROOT / "scripts/train/standard_roi_gamma_batch_budget.sbatch").read_text(encoding="utf-8")
    assert "APPEND_DOMAIN_ONEHOT" not in budget_launcher
    assert "DOMAIN_BALANCE_MODE" not in budget_launcher


def test_roi_library_limit_can_be_overridden_without_stale_alias(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UNITYPSF_RAW_TIFF", str(tmp_path / "raw.tif"))
    override = tmp_path / "roi_limit.yaml"
    override.write_text("train:\n  roi_bank_gamma:\n    roi_library_max_rois: 32\n", encoding="utf-8")

    config = materialize_config(
        CONFIGS / "base/microtube.yaml",
        CONFIGS / "overrides/standard_roi_gamma.yaml",
        override,
    )
    gamma = config["train"]["roi_bank_gamma"]

    assert gamma["roi_library_max_rois"] == 32
    assert "roi_bank_max_rois" not in gamma


def test_roi_bank_builder_prefers_canonical_limit(monkeypatch) -> None:
    from unity_psf.training.high_fidelity import engine

    monkeypatch.setattr(engine, "_auto_build_roi_bank_from_loc_harvest", lambda *args, **kwargs: kwargs["config"])
    source = engine._ROIBankSource(
        mode="raw_tiff",
        raw_path="unused.tif",
        candidate_mode="dense_tile_temporal",
        frame_range=None,
        domains=None,
    )

    config = engine._auto_build_roi_bank(
        {"roi_library_max_rois": 32, "roi_bank_max_rois": 2048},
        roi_source=source,
        model=None,
        train_cfg={},
    )

    assert config.max_rois == 32

    legacy_config = engine._auto_build_roi_bank(
        {"roi_bank_max_rois": 16},
        roi_source=source,
        model=None,
        train_cfg={},
    )
    assert legacy_config.max_rois == 16


def test_materialize_resolves_project_references_from_override_source() -> None:
    config = materialize_config(
        CONFIGS / "base/microtube.yaml",
        CONFIGS / "modalities/emitter_2d/emitter_2d_dual_channel_300epoch.yaml",
    )
    entries = config["train"]["online_generation"]["dual_domain_coeff_maps"]
    assert entries
    assert all(str(item["coeff_maps_npz"]).startswith(str(ROOT / ".local/formal-inputs")) for item in entries)


def test_from_scratch_override_resolves_environment_paths(monkeypatch, tmp_path: Path) -> None:
    values = {
        "UNITYPSF_RAW_TIFF": tmp_path / "raw.tif",
        "UNITYPSF_LEFT_MAP": tmp_path / "left.npz",
        "UNITYPSF_RIGHT_MAP": tmp_path / "right.npz",
    }
    for name, path in values.items():
        path.touch()
        monkeypatch.setenv(name, str(path))

    config = materialize_config(
        CONFIGS / "base/microtube.yaml",
        CONFIGS / "overrides/cuda_gemm_from_scratch.yaml",
    )
    online = config["train"]["online_generation"]
    assert config["phase_retrieval"]["input"]["tiff_path"] == str(values["UNITYPSF_RAW_TIFF"])
    assert [item["coeff_maps_npz"] for item in online["dual_domain_coeff_maps"]] == [
        str(values["UNITYPSF_LEFT_MAP"]),
        str(values["UNITYPSF_RIGHT_MAP"]),
    ]


def test_runtime_set_applies_yaml_scalar_types() -> None:
    config: dict = {"train": {"online_generation": {}}}
    _apply_assignment(config, "train.epochs=3")
    _apply_assignment(config, "train.online_generation.enabled=true")
    _apply_assignment(config, "train.online_generation.name=profile")
    assert config == {
        "train": {
            "epochs": 3,
            "online_generation": {"enabled": True, "name": "profile"},
        }
    }


def test_materialize_rejects_unresolved_environment_variables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("UNITYPSF_MISSING_TEST_VALUE", raising=False)
    override = tmp_path / "override.yaml"
    override.write_text(
        "phase_retrieval:\n  input:\n    tiff_path: ${UNITYPSF_MISSING_TEST_VALUE}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unresolved environment variable"):
        materialize_config(CONFIGS / "base/microtube.yaml", override)
