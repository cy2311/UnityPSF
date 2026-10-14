from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from unity_psf.training.formal_gate import INPUT_NAMES, make_smoke_config, stage_inputs, verify_inputs, verify_gpu_run, verify_release_gate


def _sources(tmp_path: Path) -> dict[str, str]:
    result = {}
    for name in INPUT_NAMES:
        path = tmp_path / f"source_{name}.npz"
        if name == "double_helix_pupil":
            np.savez(path, complex_pupil=np.ones((128, 128), dtype=np.complex64))
        else:
            np.savez(path, zernike_maps_nm=np.zeros((2, 4, 4), dtype=np.float32), mode_order=np.zeros((2, 2), dtype=np.int64))
        result[name] = str(path)
    return result


def test_stage_and_verify_formal_inputs_detects_changed_artifact(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    stage_inputs(sources, tmp_path / "staged")
    manifest = verify_inputs(tmp_path / "staged")
    assert set(manifest["inputs"]) == set(INPUT_NAMES)
    assert all(len(item["sha256"]) == 64 for item in manifest["inputs"].values())

    np.savez(tmp_path / "staged/emitter_2d_left.npz", zernike_maps_nm=np.ones((2, 4, 4)), mode_order=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_inputs(tmp_path / "staged")


def test_stage_rejects_wrong_pupil_sampling(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    np.savez(sources["double_helix_pupil"], complex_pupil=np.ones((64, 64), dtype=np.complex64))
    with pytest.raises(ValueError, match="128 x 128"):
        stage_inputs(sources, tmp_path / "staged")


def test_verify_reports_missing_input(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"inputs": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing formal input"):
        verify_inputs(tmp_path)


def test_gpu_smoke_config_keeps_all_formal_modalities_with_amp_warmup(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "configs/experiments/unitypsf_three_modality_raw_tiff_300epoch.yaml"
    generated = make_smoke_config(source, tmp_path)
    joint = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert joint["epochs"] == 1
    assert len(joint["instances"]) == 5
    assert all(item["step_budget"] == 8 for item in joint["instances"])
    for item in joint["instances"]:
        config = yaml.safe_load(Path(item["config"]).read_text(encoding="utf-8"))
        assert config["train"]["batch_size"] == 1
        assert config["train"]["online_generation"]["steps_per_epoch"] == 8
        assert all(Path(entry["coeff_maps_npz"]).is_absolute() for entry in config["train"]["online_generation"]["dual_domain_coeff_maps"])


def test_gpu_run_gate_rejects_cpu_rank(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metadata = tmp_path / "metadata"
    checkpoints = tmp_path / "checkpoints"
    report = tmp_path / "report"
    for directory in (metrics, metadata, checkpoints, report):
        directory.mkdir()
    (checkpoints / "unitypsf_joint.ckpt").write_bytes(b"checkpoint")
    (report / "report.html").write_text("report", encoding="utf-8")
    summary = {"status": "complete", "world_size": 3, "cuda_confirmed": True, "attempt_id": "attempt-1", "resolved_training_contract_hash": "signature-1", "rank_assignments": {"0": "emitter_2d", "1": "astigmatism", "2": "double_helix"},
               "modalities": {name: {"optimizer_steps": 1} for name in ("emitter_2d", "astigmatism", "double_helix")}}
    summary["checkpoint_sha256"] = hashlib.sha256((checkpoints / "unitypsf_joint.ckpt").read_bytes()).hexdigest()
    (metrics / "joint_training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for rank, modality in enumerate(("emitter_2d", "astigmatism", "double_helix")):
        (metadata / f"rank_{rank}.json").write_text(json.dumps({"rank": rank, "owned_modality": modality, "attempt_id": "attempt-1", "resolved_training_contract_hash": "signature-1", "device": f"cuda:{rank}", "cuda_confirmed": True}), encoding="utf-8")
    stage_inputs(_sources(tmp_path), tmp_path / "inputs")
    (metadata / "formal_inputs_verified.json").write_text((tmp_path / "inputs/manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    receipt = verify_gpu_run(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    verify_release_gate(receipt_path, tmp_path / "inputs")
    (metadata / "rank_1.json").write_text(json.dumps({"device": "cpu", "cuda_confirmed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="rank 1.*CUDA"):
        verify_gpu_run(tmp_path)
    np.savez(tmp_path / "inputs/emitter_2d_left.npz", zernike_maps_nm=np.ones((2, 4, 4)), mode_order=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_release_gate(receipt_path, tmp_path / "inputs")
