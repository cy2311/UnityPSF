"""Stage and verify external calibration inputs used by formal training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
import yaml

from unity_psf.config import resolve_config_reference


INPUT_NAMES = (
    "emitter_2d_left",
    "emitter_2d_right",
    "astigmatism_left",
    "astigmatism_right",
    "double_helix_map",
    "double_helix_pupil",
)
MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_npz(name: str, path: Path) -> None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if name == "double_helix_pupil":
                key = "complex_pupil" if "complex_pupil" in payload else "carrier_complex"
                if key not in payload:
                    raise ValueError("must contain complex_pupil or carrier_complex")
                shape = tuple(int(value) for value in payload[key].shape)
                if shape != (128, 128):
                    raise ValueError(f"must be 128 x 128, got {shape}")
            else:
                if "zernike_maps_nm" not in payload:
                    raise ValueError("must contain zernike_maps_nm")
                maps = payload["zernike_maps_nm"]
                if maps.ndim != 3 or not np.issubdtype(maps.dtype, np.number) or not np.isfinite(maps).all():
                    raise ValueError("zernike_maps_nm must be a finite three-dimensional numeric array")
                if "mode_order" not in payload or payload["mode_order"].shape != (maps.shape[0], 2):
                    raise ValueError("mode_order must have one (n, m) pair per Zernike map")
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid formal input {name} ({path}): {exc}") from exc


def stage_inputs(sources: Mapping[str, str | Path], destination: str | Path) -> dict[str, object]:
    missing = [name for name in INPUT_NAMES if name not in sources]
    if missing:
        raise ValueError(f"missing formal input source(s): {', '.join(missing)}")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, str]] = {}
    for name in INPUT_NAMES:
        source = Path(sources[name]).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"formal input source does not exist: {name}: {source}")
        _validate_npz(name, source)
        staged = target / f"{name}.npz"
        shutil.copy2(source, staged)
        entries[name] = {"path": staged.name, "source": str(source), "sha256": _sha256(staged)}
    manifest = {"schema_version": "unitypsf.formal_inputs.v1", "inputs": entries}
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_inputs(destination: str | Path) -> dict[str, object]:
    target = Path(destination)
    manifest_path = target / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"formal input manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("inputs") if isinstance(manifest, dict) else None
    if not isinstance(entries, dict):
        raise ValueError("formal input manifest has no inputs mapping")
    for name in INPUT_NAMES:
        entry = entries.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"missing formal input: {name}")
        if entry.get("path") != f"{name}.npz":
            raise ValueError(f"formal input path is invalid: {name}")
        path = target / f"{name}.npz"
        if not path.is_file():
            raise ValueError(f"missing formal input: {name}: {path}")
        _validate_npz(name, path)
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(f"SHA256 mismatch for formal input: {name}")
    return manifest


def make_smoke_config(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source).resolve()
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    joint = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if joint.get("execution") != "expert_parallel" or len(joint.get("instances", ())) != 5:
        raise ValueError("GPU smoke requires the five-instance Expert Parallel config")
    joint["epochs"] = 1
    generated_configs: dict[Path, Path] = {}
    for instance in joint["instances"]:
        original = resolve_config_reference(instance["config"], source_path=source_path)
        if original not in generated_configs:
            config = yaml.safe_load(original.read_text(encoding="utf-8"))
            train = config["train"]
            train["epochs"] = 1
            train["batch_size"] = 1
            online = train["online_generation"]
            online["steps_per_epoch"] = 8
            online["sequence_count"] = 1
            online["cached_window_cache_frames"] = 4
            for entry in online.get("dual_domain_coeff_maps", ()):
                entry["coeff_maps_npz"] = str(resolve_config_reference(entry["coeff_maps_npz"], source_path=original))
            if online.get("pupil_carrier_complex_npz"):
                online["pupil_carrier_complex_npz"] = str(resolve_config_reference(online["pupil_carrier_complex_npz"], source_path=original))
            generated = target / original.name
            generated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            generated_configs[original] = generated
        instance["config"] = str(generated_configs[original])
        instance["step_budget"] = 8
    path = target / "joint_gpu_smoke.yaml"
    path.write_text(yaml.safe_dump(joint, sort_keys=False), encoding="utf-8")
    return path


def verify_gpu_run(run_dir: str | Path) -> dict[str, object]:
    run = Path(run_dir)
    summary_path = run / "metrics/joint_training_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"GPU smoke summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("world_size") != 3 or summary.get("cuda_confirmed") is not True:
        raise ValueError("GPU smoke did not complete with three CUDA ranks")
    if set(summary.get("modalities", {})) != {"emitter_2d", "astigmatism", "double_helix"}:
        raise ValueError("GPU smoke did not train all three modalities")
    if summary.get("rank_assignments") != {"0": "emitter_2d", "1": "astigmatism", "2": "double_helix"}:
        raise ValueError("GPU smoke rank assignments do not match the formal route")
    if not summary.get("attempt_id") or not summary.get("resolved_training_contract_hash"):
        raise ValueError("GPU smoke attempt or contract hash is missing")
    for name, metrics in summary["modalities"].items():
        if int(metrics.get("optimizer_steps", 0)) <= 0:
            raise ValueError(f"GPU smoke recorded no optimizer step for {name}")
    for rank, modality in enumerate(("emitter_2d", "astigmatism", "double_helix")):
        status_path = run / "metadata" / f"rank_{rank}.json"
        if not status_path.is_file():
            raise ValueError(f"GPU smoke rank {rank} status is missing")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("cuda_confirmed") is not True or status.get("device") != f"cuda:{rank}":
            raise ValueError(f"GPU smoke rank {rank} did not confirm CUDA")
        if (status.get("rank") != rank or status.get("owned_modality") != modality
                or status.get("attempt_id") != summary["attempt_id"]
                or status.get("resolved_training_contract_hash")
                != summary["resolved_training_contract_hash"]):
            raise ValueError(f"GPU smoke rank {rank} identity does not match summary")
    for artifact in (run / "checkpoints/unitypsf_joint.ckpt", run / "report/report.html"):
        if not artifact.is_file():
            raise ValueError(f"GPU smoke artifact is missing: {artifact}")
    if summary.get("checkpoint_sha256") != _sha256(run / "checkpoints/unitypsf_joint.ckpt"):
        raise ValueError("GPU smoke checkpoint SHA256 does not match summary")
    input_path = run / "metadata/formal_inputs_verified.json"
    if not input_path.is_file():
        raise ValueError("GPU smoke verified input manifest is missing")
    return {
        "schema_version": "unitypsf.gpu_gate.v1",
        "run_dir": str(run.resolve()),
        "attempt_id": summary["attempt_id"],
        "resolved_training_contract_hash": summary["resolved_training_contract_hash"],
        "checkpoint_sha256": _sha256(run / "checkpoints/unitypsf_joint.ckpt"),
        "formal_inputs": json.loads(input_path.read_text(encoding="utf-8")),
        "cuda_confirmed": True,
        "modalities": sorted(summary["modalities"]),
    }


def verify_release_gate(receipt_path: str | Path, inputs: str | Path) -> dict[str, object]:
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    current_inputs = verify_inputs(inputs)
    if receipt.get("cuda_confirmed") is not True or receipt.get("formal_inputs") != current_inputs:
        raise ValueError("GPU gate receipt does not match current formal inputs")
    run = Path(receipt["run_dir"])
    if _sha256(run / "checkpoints/unitypsf_joint.ckpt") != receipt.get("checkpoint_sha256"):
        raise ValueError("GPU gate checkpoint SHA256 mismatch")
    current = verify_gpu_run(run)
    if current != receipt:
        raise ValueError("GPU gate receipt does not match the smoke run")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--destination", type=Path, required=True)
    for name in INPUT_NAMES:
        stage.add_argument(f"--{name.replace('_', '-')}", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--destination", type=Path, required=True)
    smoke = subparsers.add_parser("smoke-config")
    smoke.add_argument("--source", type=Path, required=True)
    smoke.add_argument("--destination", type=Path, required=True)
    gpu = subparsers.add_parser("verify-gpu-run")
    gpu.add_argument("--run-dir", type=Path, required=True)
    release = subparsers.add_parser("verify-release-gate")
    release.add_argument("--receipt", type=Path, required=True)
    release.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        sources = {name: getattr(args, name) for name in INPUT_NAMES}
        manifest = stage_inputs(sources, args.destination)
    elif args.command == "verify":
        manifest = verify_inputs(args.destination)
    elif args.command == "smoke-config":
        print(make_smoke_config(args.source, args.destination))
        return 0
    elif args.command == "verify-gpu-run":
        manifest = verify_gpu_run(args.run_dir)
    else:
        manifest = verify_release_gate(args.receipt, args.destination)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
