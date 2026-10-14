from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import shutil

import torch


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(root: Path) -> dict:
    from torch.utils.cpp_extension import CUDA_HOME

    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError("qualification requires three CUDA devices; no CPU fallback")
    if torch.__version__ != "2.7.1+cu118" or torch.version.cuda != "11.8":
        raise RuntimeError("qualification environment differs from the recorded PyTorch baseline")
    toolkit = Path(CUDA_HOME or "")
    sources = sorted((root / "src").rglob("*.py")) + sorted((root / "src").rglob("*.cu")) + sorted((root / "src").rglob("*.cpp"))
    hashes = {str(path.relative_to(root)): file_hash(path) for path in sources}
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": str(toolkit),
        "nvcc": subprocess.check_output([str(toolkit / "bin/nvcc"), "--version"], text=True),
        "gpu_driver": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cudnn": torch.backends.cudnn.version(),
        "qualification_script_sha256": file_hash(Path(__file__).resolve()),
        "qualification_trace_sha256": file_hash(Path(__file__).with_name('qualification_trace.py')),
        "determinism": {
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "python_numpy_initial_seed": "4100 + rank; common epoch-0 checkpoint restores all RNG",
        },
        "source_hashes": hashes,
        "source_manifest_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "toolkit_torch_minor_mismatch": "11.7 toolkit / cu118 PyTorch retained, not eliminated",
    }


def worker() -> None:
    from unity_psf.training.entrypoints import expert_parallel_worker
    from qualification_trace import install_trace, seed_all

    receipts = Path(os.environ['UNITYPSF_QUALIFICATION_RECEIPTS'])
    recorded = json.loads((receipts / 'environment.json').read_text())
    if recorded['qualification_script_sha256'] != file_hash(Path(__file__).resolve()) or (
        recorded['qualification_trace_sha256'] != file_hash(Path(__file__).with_name('qualification_trace.py'))
    ):
        raise ValueError('qualification implementation changed during execution')
    seed_all(4100 + int(os.environ['RANK']))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    original_save = expert_parallel_worker.save_modality_training_shard
    original_restore = expert_parallel_worker.restore_modality_training_shard

    def restore_and_check(path, **kwargs):
        from unity_psf.contracts import load_modality_joint_checkpoint
        from unity_psf.training.checkpoint_state import capture_rng_state

        saved = load_modality_joint_checkpoint(path)
        restored = original_restore(path, **kwargs)
        runtime = kwargs["runtime"]
        modality = runtime.modality.value
        state = saved["training_state"][modality]
        results = {"tensor_count": 0, "non_bitwise_tensors": 0, "non_bitwise_scalars": 0,
                   "max_abs_delta": 0.0, "structural_mismatches": [], "numeric_mismatches": []}
        compare_values(saved["experts"][modality]["model_state_dict"], runtime.model.state_dict(), "model", results)
        compare_values(state["optimizer"], runtime.optimizer.state_dict(), "optimizer", results)
        compare_values(state["scheduler"], runtime.scheduler.state_dict(), "scheduler", results)
        compare_values(state["amp_scaler"], runtime.amp_scaler.state_dict(), "amp_scaler", results)
        compare_values(state["rng_state"], capture_rng_state(), "rng_state", results)
        compare_values(state["channel_progress"], dict(restored.channel_progress), "channel_progress", results)
        results["status"] = "passed" if not any(results[key] for key in (
            "numeric_mismatches", "structural_mismatches", "non_bitwise_tensors", "non_bitwise_scalars",
        )) else "failed"
        prefix = 'initial-restore' if state['epoch'] == 0 else 'restore'
        output = Path(os.environ["UNITYPSF_QUALIFICATION_RECEIPTS"]) / f"{prefix}-{os.environ['UNITYPSF_QUALIFICATION_LABEL']}-{modality}.json"
        output.write_text(json.dumps(results, indent=2))
        if results["status"] != "passed":
            raise RuntimeError(f"immediate restored state differs for {modality}")
        return restored

    def save_then_stop(*args, **kwargs):
        result = original_save(*args, **kwargs)
        if kwargs['epoch'] > 0:
            receipts = Path(os.environ['UNITYPSF_QUALIFICATION_RECEIPTS'])
            label = os.environ['UNITYPSF_QUALIFICATION_LABEL']
            modality = kwargs['runtime'].modality.value
            destination = receipts / 'epochs' / label
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args[0], destination / f'{modality}-{kwargs["epoch"]}.ckpt')
        if os.environ.get("UNITYPSF_QUALIFICATION_INTERRUPT") == "1" and kwargs["epoch"] == 1:
            run_dir = Path(args[0]).parent.parent
            marker_dir = run_dir / "metadata" / "qualification_interrupt"
            marker_dir.mkdir(parents=True, exist_ok=True)
            (marker_dir / f"{os.environ['RANK']}.ready").write_text("checkpoint_saved")
            deadline = time.monotonic() + 120.0
            while len(list(marker_dir.glob("*.ready"))) != 3:
                if time.monotonic() > deadline:
                    raise RuntimeError("not all ranks saved the interruption checkpoint")
                time.sleep(0.1)
            print(json.dumps({"event": "qualification_checkpoint_interrupt", "epoch": 1}), flush=True)
            raise SystemExit(73)
        return result

    expert_parallel_worker.save_modality_training_shard = save_then_stop
    expert_parallel_worker.restore_modality_training_shard = restore_and_check
    install_trace(
        expert_parallel_worker,
        receipt_dir=Path(os.environ['UNITYPSF_QUALIFICATION_RECEIPTS']),
        label=os.environ['UNITYPSF_QUALIFICATION_LABEL'], resumed='--resume' in sys.argv,
    )
    from unity_psf.training.entrypoints.expert_parallel_cli import main

    raise SystemExit(main(sys.argv[2:]))


def compare_values(left, right, path: str, results: dict) -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            results["structural_mismatches"].append(path)
            return
        left = left.detach().cpu()
        right = right.detach().cpu()
        results["tensor_count"] += 1
        if not torch.equal(left, right):
            if left.is_floating_point() or left.is_complex():
                delta = float((left - right).abs().max().item()) if left.numel() else 0.0
                results["max_abs_delta"] = max(results["max_abs_delta"], delta)
                if not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
                    results["numeric_mismatches"].append(path)
            else:
                results["numeric_mismatches"].append(path)
            results["non_bitwise_tensors"] += 1
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            results["structural_mismatches"].append(path)
            return
        for key in left:
            compare_values(left[key], right[key], f"{path}/{key}", results)
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            results["structural_mismatches"].append(path)
            return
        for index, value in enumerate(left):
            compare_values(value, right[index], f"{path}/{index}", results)
    elif left != right:
        if isinstance(left, float) and isinstance(right, float) and abs(left - right) <= 1e-6 + 1e-5 * abs(left):
            results["non_bitwise_scalars"] += 1
        else:
            results["numeric_mismatches"].append(path)


def main() -> int:
    from unity_psf.cli.train_selection import _write_plan, build_plan, parse_routes, verify_short_run
    from unity_psf.contracts import load_modality_joint_checkpoint
    from unity_psf.contracts.canonical import CANONICAL_CONTRACT_HASH_ENV, CANONICAL_LAUNCH_ENV, CANONICAL_TRAINING_ENTRYPOINT

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--astigmatism-map", action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    receipt_dir = args.run_root.resolve() / args.run_id
    receipt_dir.mkdir(parents=True, exist_ok=False)
    env = environment(root)
    (receipt_dir / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True))
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=(receipt_dir / "pip-freeze.txt").open("w"), check=True)
    routes = parse_routes(["emitter_2d:left,right", "astigmatism:left,right", "double_helix:main"])
    maps = {}
    for item in args.astigmatism_map:
        route, path = item.split("=", 1)
        if route not in ("astigmatism:left", "astigmatism:right"):
            raise ValueError(f"invalid astigmatism map route: {route}")
        maps[route] = Path(path).expanduser().resolve()
    if set(maps) != {"astigmatism:left", "astigmatism:right"}:
        raise ValueError("aligned qualification requires both astigmatism coefficient maps")
    plans = {}
    for label in ("continuous", "interrupted"):
        plan = build_plan(
            routes=routes, maps=maps, dh_ppg_map=None, epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size,
            input_size=args.input_size, run_root=args.run_root,
            run_id=f"{args.run_id}-{label}", smoke=False,
        )
        if label == 'interrupted':
            if plan['configs'] != plans['continuous']['configs'] or plan['inputs'] != plans['continuous']['inputs']:
                raise ValueError('paired runs differ in scientific configuration or input manifest')
        joint = _write_plan(plan, resume=False)
        plans[label] = plan
        launch_env = os.environ.copy()
        launch_env['PYTHONHASHSEED'] = '4100'
        launch_env['UNITYPSF_QUALIFICATION_LABEL'] = label
        launch_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        launch_env["UNITYPSF_QUALIFICATION_INTERRUPT"] = "1" if label == "interrupted" else "0"
        launch_env["UNITYPSF_QUALIFICATION_RECEIPTS"] = str(receipt_dir)
        launch_env[CANONICAL_LAUNCH_ENV] = CANONICAL_TRAINING_ENTRYPOINT
        launch_env[CANONICAL_CONTRACT_HASH_ENV] = plan["contract"].contract_hash
        base = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=3"]
        entry = [str(Path(__file__).resolve()), "worker"]
        command = base + entry + ["--config", str(joint), "--run-root", str(args.run_root.resolve()), "--run-id", plan["contract"].run_id]
        with (receipt_dir / f"{label}.log").open("w") as handle:
            result = subprocess.run(command, env=launch_env, stdout=handle, stderr=subprocess.STDOUT)
        if label == "continuous" and result.returncode != 0:
            raise RuntimeError(f"continuous run failed: {result.returncode}")
        if label == "interrupted":
            if result.returncode == 0:
                raise RuntimeError("interrupted run unexpectedly completed")
            for modality in routes:
                shard = load_modality_joint_checkpoint(Path(plan["run_dir"]) / "checkpoints" / f"{modality}.resume.ckpt")
                state = shard["training_state"][modality]
                if state["epoch"] != 1 or state["status"] != "running":
                    raise RuntimeError("interruption did not preserve an epoch-1 running checkpoint")
            with (receipt_dir / "resume.log").open("w") as handle:
                launch_env["UNITYPSF_QUALIFICATION_INTERRUPT"] = "0"
                result = subprocess.run(command + ["--resume"],
                                        env=launch_env, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise RuntimeError(f"resume failed: {result.returncode}")
        gate = verify_short_run(Path(plan["run_dir"]), expected_routes=routes)
        (receipt_dir / f"{label}-gate.json").write_text(json.dumps(gate, indent=2))
        if label == 'continuous':
            initial_manifest = {modality: file_hash(receipt_dir / 'initial' / f'{modality}.ckpt')
                                for modality in routes}
            (receipt_dir / 'initial-manifest.json').write_text(json.dumps(initial_manifest, indent=2))
    comparisons = {}
    from qualification_trace import compare_traces, gate_status, state_hash
    trace_results = compare_traces(receipt_dir, routes, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
    (receipt_dir / 'initial-state-and-step-trace.json').write_text(json.dumps(trace_results, indent=2))
    epoch_results = {}
    for modality in routes:
        epoch_results[modality] = {}
        for epoch in range(1, args.epochs + 1):
            checkpoints = [load_modality_joint_checkpoint(receipt_dir / 'epochs' / label / f'{modality}-{epoch}.ckpt')
                           for label in ('continuous', 'interrupted')]
            result = {"tensor_count": 0, "non_bitwise_tensors": 0, "non_bitwise_scalars": 0,
                      "max_abs_delta": 0.0, "structural_mismatches": [], "numeric_mismatches": []}
            for key in ('experts', 'training_state'):
                compare_values(checkpoints[0][key], checkpoints[1][key], key, result)
            physical = [{channel: {key: checkpoint['channel_states'][modality][channel][key]
                                   for key in ('physical_state', 'calibration')}
                         for channel in routes[modality]} for checkpoint in checkpoints]
            compare_values(physical[0], physical[1], 'physical_and_calibration', result)
            result['strict_byte_status'] = 'passed' if all(
                state_hash(checkpoints[0][key]) == state_hash(checkpoints[1][key])
                for key in ('experts', 'training_state')
            ) and state_hash(physical[0]) == state_hash(physical[1]) else 'failed'
            epoch_results[modality][epoch] = result
    (receipt_dir / 'epoch-equivalence.json').write_text(json.dumps(epoch_results, indent=2))
    for modality in routes:
        shards = [load_modality_joint_checkpoint(Path(plans[label]["run_dir"]) / "checkpoints" / f"{modality}.resume.ckpt") for label in ("continuous", "interrupted")]
        results = {"tensor_count": 0, "non_bitwise_tensors": 0, "non_bitwise_scalars": 0, "max_abs_delta": 0.0,
                   "structural_mismatches": [], "numeric_mismatches": []}
        for key in ("experts", "training_state"):
            compare_values(shards[0][key], shards[1][key], key, results)
        for channel in routes[modality]:
            for key in ("physical_state", "calibration"):
                compare_values(shards[0]["channel_states"][modality][channel][key], shards[1]["channel_states"][modality][channel][key],
                               f"channel_states/{channel}/{key}", results)
        results['strict_byte_status'] = 'passed' if all(
            state_hash(shards[0][key]) == state_hash(shards[1][key])
            for key in ('experts', 'training_state')
        ) and all(state_hash(shards[0]['channel_states'][modality][channel][key])
                  == state_hash(shards[1]['channel_states'][modality][channel][key])
                  for channel in routes[modality] for key in ('physical_state', 'calibration')) else 'failed'
        comparisons[modality] = results
    binaries = sorted((root / ".local/cache/torch_extensions").rglob("*.so"))
    receipt = {"comparisons": comparisons, "rtol": 1e-5, "atol": 1e-6,
               "comparison_mode": "exact_value_equality",
               "interpretation": "common epoch-0 checkpoint; step and epoch boundaries compared separately",
               "trace_results": trace_results,
               "extension_binary_hashes": {str(path): file_hash(path) for path in binaries},
               "tolerance_status": "passed" if all(not item["structural_mismatches"] and not item["numeric_mismatches"] for item in comparisons.values()) else "failed",
               "status": gate_status(trace_results, epoch_results, comparisons)}
    (receipt_dir / "resume-equivalence.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker()
    else:
        raise SystemExit(main())
