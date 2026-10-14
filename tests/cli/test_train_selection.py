from __future__ import annotations

from pathlib import Path

import pytest

from unity_psf.cli.train_selection import (
    _write_plan,
    build_plan,
    parse_routes,
    validate_cuda_capacity,
)
from unity_psf.contracts.canonical import (
    CANONICAL_CONTRACT_HASH_ENV,
    CANONICAL_LAUNCH_ENV,
    CANONICAL_TRAINING_ENTRYPOINT,
    validate_canonical_launch,
)


def test_route_selection_accepts_single_dual_and_three_modalities() -> None:
    assert parse_routes(["emitter_2d:left,right", "double_helix:main"]) == {
        "emitter_2d": ("left", "right"), "double_helix": ("main",)
    }
    assert parse_routes(["astigmatism:right"]) == {"astigmatism": ("right",)}
    with pytest.raises(ValueError, match="main"):
        parse_routes(["double_helix:left"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_routes(["emitter_2d:left,left"])


def test_plan_uses_circular_2d_and_astigmatism_anchors_by_default(tmp_path: Path) -> None:
    plan = build_plan(
        routes=parse_routes(["emitter_2d:left,right", "astigmatism:right"]),
        maps={}, dh_ppg_map=None, epochs=2, steps_per_epoch=2, batch_size=1,
        run_root=tmp_path, run_id="selection", smoke=True,
    )
    assert len(plan["joint"]["instances"]) == 3
    assert plan["required_gpus"] == 2
    for channel in ("left", "right"):
        key = f"emitter_2d:{channel}"
        config = plan["configs"][key]
        entries = config["train"]["online_generation"]["dual_domain_coeff_maps"]
        assert entries == []
        assert config["train"]["online_generation"]["conditioning_profile"] == "circular_2d_50nm"
        assert config["train"]["online_generation"]["circular_psf_radius_nm"] == 50.0
        assert config["metadata"]["physical_initialization"] == "anchor_only"
        assert config["metadata"]["anchor_profile"] == "circular_2d_50nm"
        assert f"{key}:coefficient_map" not in plan["inputs"]
    astig = plan["configs"]["astigmatism:right"]
    assert astig["train"]["online_generation"]["dual_domain_coeff_maps"] == []
    assert astig["train"]["online_generation"]["conditioning_profile"] == "astigmatism_660nm"
    assert astig["metadata"]["physical_initialization"] == "anchor_only"
    assert astig["metadata"]["anchor_nm"] == 100.0


def test_plan_can_pin_shared_qualification_input_size(tmp_path: Path) -> None:
    plan = build_plan(
        routes=parse_routes(["astigmatism:left"]), maps={}, dh_ppg_map=None,
        epochs=2, steps_per_epoch=2, batch_size=8, input_size=64,
        run_root=tmp_path, run_id="sized", smoke=False,
    )
    config = plan["configs"]["astigmatism:left"]
    assert config["train"]["online_generation"]["height"] == 64
    assert config["train"]["online_generation"]["width"] == 64
    assert config["train"]["channel_layout"]["frame_size"] == [64, 64]


def test_plan_dh_requires_approved_pca_and_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PPG"):
        build_plan(
            routes=parse_routes(["double_helix:main"]), maps={},
            dh_ppg_map=tmp_path / "missing.npz", epochs=1, steps_per_epoch=1,
            batch_size=1, run_root=tmp_path, run_id="dh", smoke=True,
        )


def test_resume_rejects_changed_resolved_inputs(tmp_path: Path) -> None:
    kwargs = dict(
        routes=parse_routes(["emitter_2d:left"]), maps={}, dh_ppg_map=None,
        epochs=1, steps_per_epoch=2, batch_size=1, run_root=tmp_path,
        run_id="resume", smoke=False,
    )
    plan = build_plan(**kwargs)
    joint = _write_plan(plan, resume=False)
    assert joint.is_file()
    assert _write_plan(build_plan(**kwargs), resume=True) == joint
    changed = build_plan(**{**kwargs, "steps_per_epoch": 3})
    with pytest.raises(ValueError, match="changed"):
        _write_plan(changed, resume=True)


def test_require_cuda_rejects_missing_devices_and_cpu_fallback() -> None:
    with pytest.raises(ValueError, match="requires 3 CUDA devices"):
        validate_cuda_capacity(required_gpus=3, available_gpus=0, require_cuda=True)
    assert validate_cuda_capacity(required_gpus=2, available_gpus=2, require_cuda=True) == 2
    assert validate_cuda_capacity(required_gpus=2, available_gpus=0, require_cuda=False) == 0


def test_cuda_smoke_script_covers_all_routes_without_cpu_fallback() -> None:
    script = Path(__file__).parents[2] / "scripts/train/unitypsf_selection_cuda_smoke.sbatch"
    text = script.read_text(encoding="utf-8")
    assert "--route emitter_2d:left,right" in text
    assert "--route astigmatism:left,right" in text
    assert "--route double_helix:main" in text
    assert "--require-cuda" in text
    assert "--allow-cpu-smoke" not in text


def test_selection_launches_only_the_internal_canonical_executor() -> None:
    source = Path(__file__).parents[2] / "src/unity_psf/cli/train_selection.py"
    text = source.read_text(encoding="utf-8")
    assert "unity_psf.training.entrypoints.train_modality_expert_parallel" in text
    assert "unity_psf.cli.train_joint_expert_parallel" not in text
    assert "CANONICAL_LAUNCH_ENV" in text
    assert "CANONICAL_CONTRACT_HASH_ENV" in text


def test_training_scripts_cannot_bypass_the_selection_cli() -> None:
    scripts_root = Path(__file__).parents[2] / "scripts/train"
    offenders = []
    for script in scripts_root.glob("*.sbatch"):
        text = script.read_text(encoding="utf-8")
        if "unity_psf.training.entrypoints.train_modality_expert_parallel" in text:
            offenders.append(script.name)
    assert offenders == []


def test_internal_executor_requires_canonical_cli_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CANONICAL_LAUNCH_ENV, raising=False)
    monkeypatch.delenv(CANONICAL_CONTRACT_HASH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="unity-psf-train"):
        validate_canonical_launch("a" * 64)

    monkeypatch.setenv(CANONICAL_LAUNCH_ENV, CANONICAL_TRAINING_ENTRYPOINT)
    monkeypatch.setenv(CANONICAL_CONTRACT_HASH_ENV, "b" * 64)
    with pytest.raises(RuntimeError, match="contract hash"):
        validate_canonical_launch("a" * 64)

    monkeypatch.setenv(CANONICAL_CONTRACT_HASH_ENV, "a" * 64)
    validate_canonical_launch("a" * 64)
