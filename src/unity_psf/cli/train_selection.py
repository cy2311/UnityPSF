"""Select measurement routes and launch the modality Expert Parallel trainer."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from unity_psf.config import load_config
from unity_psf.contracts.canonical import (
    CANONICAL_CONTRACT_HASH_ENV,
    CANONICAL_LAUNCH_ENV,
    CANONICAL_TRAINING_ENTRYPOINT,
    load_canonical_release_checkpoint,
)
from unity_psf.contracts import (
    ResolvedTrainingContract,
    ResolvedTrainingInstance,
    load_modality_joint_checkpoint,
)


PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = {
    "emitter_2d": PROJECT / "configs/modalities/emitter_2d/emitter_2d_dual_channel_300epoch.yaml",
    "astigmatism": PROJECT / "configs/modalities/astigmatism/astigmatism_dual_channel_300epoch.yaml",
    "double_helix": PROJECT / "configs/modalities/double_helix/ppg_physical.yaml",
}
DEFAULT_PPG = PROJECT / "configs/calibration/ppg/ppg_map_initial.npz"
DEFAULT_2D_ANCHOR_PROFILE = "circular_2d_50nm"
DEFAULT_2D_ANCHOR_RADIUS_NM = 50.0
DEFAULT_ASTIGMATISM_ANCHOR_PROFILE = "astigmatism_660nm"
DEFAULT_ASTIGMATISM_ANCHOR_NM = 100.0
SEEDS = {"emitter_2d": (4101, 4111), "astigmatism": (4201, 4211), "double_helix": (4301, 4311)}


def validate_cuda_capacity(*, required_gpus: int, available_gpus: int, require_cuda: bool) -> int:
    """Validate the device contract before launching torchrun."""

    required = int(required_gpus)
    available = int(available_gpus)
    if require_cuda and available < required:
        raise ValueError(
            f"requires {required} CUDA devices for the selected routes; found {available}"
        )
    if available and available < required:
        raise ValueError(
            f"selected routes need {required} CUDA devices; found {available}"
        )
    return available


def parse_routes(values: Sequence[str]) -> dict[str, tuple[str, ...]]:
    if not values:
        raise ValueError("at least one --route is required")
    routes: dict[str, tuple[str, ...]] = {}
    for value in values:
        modality, separator, channel_text = value.partition(":")
        if not separator or modality not in TEMPLATES:
            raise ValueError(f"invalid route {value!r}; expected modality:channel[,channel]")
        if modality in routes:
            raise ValueError(f"duplicate modality route: {modality}")
        channels = tuple(channel_text.split(","))
        allowed = {"main"} if modality == "double_helix" else {"left", "right"}
        if not channels or any(channel not in allowed for channel in channels):
            raise ValueError(f"{modality} channels must be {sorted(allowed)}")
        if len(channels) != len(set(channels)):
            raise ValueError(f"duplicate channel in route {value!r}")
        routes[modality] = channels
    return routes


def parse_maps(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        route, separator, path = value.partition("=")
        if not separator or not path:
            raise ValueError(f"invalid --map {value!r}; expected modality:channel=PATH")
        selected = parse_routes([route])
        modality, channels = next(iter(selected.items()))
        if modality == "double_helix" or len(channels) != 1:
            raise ValueError("--map is only for one 2D or Astigmatism channel")
        if route in result:
            raise ValueError(f"duplicate map for {route}")
        result[route] = Path(path).expanduser().resolve()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_ppg(path: Path) -> tuple[Path, Path]:
    manifest = path.with_suffix(".json")
    if not path.is_file() or not manifest.is_file():
        raise ValueError(f"PPG map and beads manifest are required: {path}, {manifest}")
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"].item()) != "ppg.pca_map.v1":
            raise ValueError(f"PPG map has unsupported schema: {path}")
        for name in ("pca_mean", "pca_components", "coefficient_field", "gamma"):
            if name not in data or not np.isfinite(data[name]).all():
                raise ValueError(f"PPG map has invalid {name}: {path}")
        if data["pca_components"].shape[0] != 4 or data["gamma"].shape != (9, 4):
            raise ValueError(f"PPG map has incompatible PCA/gamma dimensions: {path}")
    calibration = json.loads(manifest.read_text(encoding="utf-8"))
    if calibration.get("schema") != "ppg.beads_gamma36_manifest.v1" or calibration.get("status") != "approved_default":
        raise ValueError(f"PPG beads manifest is not approved: {manifest}")
    if float(calibration.get("heldout_ncc_median", 0.0)) < 0.85:
        raise ValueError(f"PPG beads calibration failed heldout NCC gate: {manifest}")
    return path, manifest


def build_plan(
    *, routes: Mapping[str, tuple[str, ...]], maps: Mapping[str, Path],
    dh_ppg_map: Path | None, epochs: int, steps_per_epoch: int, batch_size: int | None,
    run_root: Path, run_id: str, smoke: bool, input_size: int | None = None,
) -> dict[str, Any]:
    if not routes or epochs <= 0 or steps_per_epoch <= 0 or (batch_size is not None and batch_size <= 0):
        raise ValueError("routes, epochs, steps_per_epoch, and batch_size must be positive")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be one safe directory name")
    selected = {f"{modality}:{channel}" for modality, channels in routes.items() for channel in channels}
    if set(maps) - selected:
        raise ValueError(f"--map refers to unselected routes: {sorted(set(maps) - selected)}")
    if dh_ppg_map is not None and "double_helix" not in routes:
        raise ValueError("--dh-ppg-map requires double_helix:main")
    if input_size is not None and input_size <= 0:
        raise ValueError("input_size must be positive")
    inputs: dict[str, dict[str, str]] = {}
    ppg = None
    if "double_helix" in routes:
        ppg, manifest = _validate_ppg((dh_ppg_map or DEFAULT_PPG).expanduser().resolve())
        inputs["double_helix:main:ppg_map"] = {"path": str(ppg), "sha256": _sha256(ppg)}
        inputs["double_helix:main:beads_manifest"] = {"path": str(manifest), "sha256": _sha256(manifest)}
    configs: dict[str, dict[str, Any]] = {}
    instances: list[dict[str, Any]] = []
    resolved = (run_root.expanduser().resolve() / run_id / "resolved")
    for modality, channels in routes.items():
        source = TEMPLATES[modality]
        for channel in channels:
            key = f"{modality}:{channel}"
            config = deepcopy(load_config(source))
            train = config["train"]
            online = train["online_generation"]
            train["epochs"] = epochs
            if batch_size is not None:
                train["batch_size"] = batch_size
            online["steps_per_epoch"] = steps_per_epoch
            if input_size is not None:
                online["height"] = input_size
                online["width"] = input_size
                train["channel_layout"]["frame_size"] = [input_size, input_size]
                for layout_channel in train["channel_layout"].get("channels", ()):
                    layout_channel["crop"] = [0, 0, input_size, input_size]
            config.setdefault("metadata", {})["input_size_override"] = input_size
            if smoke:
                evaluation = train.get("eval")
                if isinstance(evaluation, dict):
                    evaluation["batch_count"] = 1
                    evaluation["batch_size"] = 1
            metadata = config.setdefault("metadata", {})
            metadata["selection_route"] = key
            for layout_channel in train.get("channel_layout", {}).get("channels", ()):
                if isinstance(layout_channel, dict) and layout_channel.get("id") == channel:
                    layout_channel["anchor_profile"] = (
                        DEFAULT_2D_ANCHOR_PROFILE
                        if modality == "emitter_2d"
                        else DEFAULT_ASTIGMATISM_ANCHOR_PROFILE
                    )
                    break
            if modality == "double_helix":
                assert ppg is not None
                online["ppg_map_path"] = str(ppg)
                metadata["physical_initialization"] = "beads_ppg_pca"
                train.pop("ppg_physical_update", None)
            else:
                selected_map = maps.get(key)
                if selected_map is None:
                    online["dual_domain_coeff_maps"] = []
                    if modality == "emitter_2d":
                        online["conditioning_profile"] = DEFAULT_2D_ANCHOR_PROFILE
                        online["circular_psf_radius_nm"] = DEFAULT_2D_ANCHOR_RADIUS_NM
                    else:
                        online["conditioning_profile"] = DEFAULT_ASTIGMATISM_ANCHOR_PROFILE
                    metadata["physical_initialization"] = "anchor_only"
                    metadata["anchor_profile"] = online["conditioning_profile"]
                    if modality == "emitter_2d":
                        metadata["anchor_radius_nm"] = DEFAULT_2D_ANCHOR_RADIUS_NM
                    else:
                        metadata["anchor_nm"] = DEFAULT_ASTIGMATISM_ANCHOR_NM
                else:
                    if not selected_map.is_file():
                        raise ValueError(f"coefficient map is missing: {selected_map}")
                    with np.load(selected_map, allow_pickle=False) as data:
                        if "zernike_maps_nm" not in data or "mode_order" not in data:
                            raise ValueError(f"invalid coefficient map: {selected_map}")
                        if modality == "emitter_2d" and not np.any(data["zernike_maps_nm"]):
                            raise ValueError(f"2D anchor coefficient map is all zero: {selected_map}")
                    online["dual_domain_coeff_maps"] = [{"name": channel, "coeff_maps_npz": str(selected_map)}]
                    metadata["physical_initialization"] = "coefficient_map"
                    inputs[f"{key}:coefficient_map"] = {"path": str(selected_map), "sha256": _sha256(selected_map)}
                    metadata["anchor_profile"] = (
                        "origami_sample_peak_nat" if modality == "emitter_2d" else online["conditioning_profile"]
                    )
            configs[key] = config
            model_seed, data_seed = SEEDS[modality]
            data_seed += {"left": 0, "right": 1, "main": 0}[channel]
            instances.append({
                "key": key, "config": str(resolved / f"{modality}_{channel}.yaml"),
                "model_seed": model_seed, "data_seed": data_seed,
                "step_budget": steps_per_epoch,
            })
    resolved_instances = tuple(
        ResolvedTrainingInstance(
            key=item["key"],
            modality=item["key"].split(":", 1)[0],
            channel=item["key"].split(":", 1)[1],
            config=item["config"],
            model_seed=int(item["model_seed"]),
            data_seed=int(item["data_seed"]),
            step_budget=int(item["step_budget"]),
        )
        for item in instances
    )
    contract = ResolvedTrainingContract(
        run_id=run_id,
        run_dir=str(resolved.parent),
        epochs=epochs,
        required_gpus=len(routes),
        smoke=smoke,
        instances=resolved_instances,
        configs=configs,
        inputs=inputs,
    )
    return {
        "contract": contract,
        "joint": contract.joint_config(resolved),
        "configs": configs,
        "inputs": inputs,
        "required_gpus": contract.required_gpus,
        "run_dir": contract.run_dir,
        "smoke": contract.smoke,
    }


def _write_plan(plan: Mapping[str, Any], *, resume: bool) -> Path:
    resolved = Path(plan["run_dir"]) / "resolved"
    contract = plan.get("contract")
    if not isinstance(contract, ResolvedTrainingContract):
        raise TypeError("selection plan must carry a ResolvedTrainingContract")
    joint_path = resolved / "joint.yaml"
    documents = {
        joint_path: yaml.safe_dump(contract.joint_config(resolved), sort_keys=False),
        resolved / "training_contract.json": json.dumps(
            contract.to_dict(), indent=2, sort_keys=True
        ) + "\n",
        resolved / "inputs.json": json.dumps(
            contract.to_dict(include_hash=False)["inputs"], indent=2, sort_keys=True
        ) + "\n",
    }
    serialized_configs = contract.to_dict(include_hash=False)["configs"]
    for key, config in serialized_configs.items():
        documents[resolved / (key.replace(":", "_") + ".yaml")] = yaml.safe_dump(config, sort_keys=False)
    if resume:
        if any(not path.is_file() or path.read_text(encoding="utf-8") != content for path, content in documents.items()):
            raise ValueError("resolved configuration or input manifest changed; cannot resume")
    else:
        if resolved.exists():
            raise FileExistsError(f"run already has resolved configuration: {resolved}")
        resolved.mkdir(parents=True)
        for path, content in documents.items():
            path.write_text(content, encoding="utf-8")
    return joint_path


def verify_short_run(
    run_dir: Path, *, expected_routes: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Check every selected channel after a short training run."""

    import torch

    from unity_psf.models import UnityPSF

    summary = json.loads((run_dir / "metrics/joint_training_summary.json").read_text(encoding="utf-8"))
    contract_path = run_dir / "resolved/training_contract.json"
    if not contract_path.is_file():
        raise ValueError("short run resolved training contract is missing")
    contract = ResolvedTrainingContract.from_dict(
        json.loads(contract_path.read_text(encoding="utf-8"))
    )
    expected = {modality: set(channels) for modality, channels in expected_routes.items()}
    if summary.get("status") != "complete" or set(summary.get("modalities", {})) != set(expected):
        raise ValueError("short run modality inventory or status does not match the plan")
    contract_hash = summary.get("resolved_training_contract_hash")
    if contract_hash != contract.contract_hash:
        raise ValueError("short run summary contract identity does not match the resolved contract")
    checkpoint = run_dir / "checkpoints/unitypsf_joint.ckpt"
    if not checkpoint.is_file() or _sha256(checkpoint) != summary.get("checkpoint_sha256"):
        raise ValueError("short run release checkpoint hash does not match")
    load_canonical_release_checkpoint(checkpoint)
    model = UnityPSF.from_checkpoint(checkpoint, device="cpu")
    actual = {name: set(channels) for name, channels in model.describe()["supported_channels"].items()}
    if actual != expected:
        raise ValueError("short run checkpoint routes do not match the plan")
    for modality, values in summary["modalities"].items():
        channels = values.get("channels", {})
        if set(channels) != expected[modality]:
            raise ValueError(f"short run channel inventory differs for {modality}")
        rank = next(
            (rank for rank, owner in summary.get("rank_assignments", {}).items() if owner == modality),
            None,
        )
        if rank is None:
            raise ValueError(f"short run rank is missing for {modality}")
        status = json.loads((run_dir / "metadata" / f"rank_{rank}.json").read_text(encoding="utf-8"))
        if (
            status.get("status") != "complete"
            or status.get("resolved_training_contract_hash") != contract_hash
        ):
            raise ValueError(f"short run rank status is invalid for {modality}")
        shard = load_modality_joint_checkpoint(run_dir / "checkpoints" / f"{modality}.resume.ckpt")
        if set(shard["channel_states"].get(modality, {})) != expected[modality]:
            raise ValueError(f"short run resume shard routes differ for {modality}")
        if (
            shard["provenance"].get("resolved_training_contract_hash") != contract_hash
        ):
            raise ValueError(f"short run resume signature differs for {modality}")
        progress = shard["training_state"][modality]["channel_progress"]
        for channel, metrics in channels.items():
            label = f"{modality}:{channel}"
            losses = metrics.get("losses", ())
            if len(losses) != int(metrics.get("steps", -1)) or not losses or any(
                not math.isfinite(float(loss)) for loss in losses
            ):
                raise ValueError(f"non-finite loss or missing losses for {label}")
            if int(metrics.get("optimizer_steps", 0)) <= 0:
                raise ValueError(f"no optimizer update for {label}")
            history = metrics.get("heldout_history", ())
            if not history or not math.isfinite(float(history[-1].get("eval_loss", float("nan")))):
                raise ValueError(f"heldout evaluation is missing or non-finite for {label}")
            if int(progress[channel]["optimizer_steps"]) != int(metrics["optimizer_steps"]):
                raise ValueError(f"resume progress differs for {label}")
    for expert in model.experts.values():
        if any(not bool(torch.isfinite(parameter).all()) for parameter in expert.parameters()):
            raise ValueError("short run checkpoint has non-finite model parameters")
    receipt = {
        "status": "passed",
        "channels": sum(map(len, expected.values())),
        "resolved_training_contract_hash": contract_hash,
        "checkpoint_sha256": summary["checkpoint_sha256"],
    }
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select UnityPSF modalities and measurement channels")
    parser.add_argument("action", choices=("plan", "smoke", "run", "resume"))
    parser.add_argument("--route", action="append", required=True, help="modality:left,right or double_helix:main")
    parser.add_argument("--map", action="append", default=[], help="modality:channel=coefficient_map.npz")
    parser.add_argument("--dh-ppg-map", type=Path, help="beads calibrated PPG/PCA NPZ with adjacent JSON manifest")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=417)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "output/unitypsf")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="refuse CPU smoke fallback and require one CUDA device per selected modality",
    )
    args = parser.parse_args(argv)
    try:
        smoke = args.action == "smoke"
        plan = build_plan(
            routes=parse_routes(args.route), maps=parse_maps(args.map), dh_ppg_map=args.dh_ppg_map,
            epochs=1 if smoke else args.epochs,
            steps_per_epoch=min(args.steps_per_epoch, 8) if smoke else args.steps_per_epoch,
            batch_size=(args.batch_size or 1) if smoke else args.batch_size,
            run_root=args.run_root, run_id=args.run_id, smoke=smoke,
        )
        if args.action == "plan":
            print(json.dumps({"routes": [item["key"] for item in plan["joint"]["instances"]],
                              "initialization": {
                                  key: {
                                      "physical_initialization": config["metadata"]["physical_initialization"],
                                      "anchor_profile": config["metadata"].get("anchor_profile"),
                                      "anchor_nm": config["metadata"].get("anchor_nm"),
                                      "anchor_radius_nm": config["metadata"].get("anchor_radius_nm"),
                                  }
                                  for key, config in plan["configs"].items()
                              },
                              "required_gpus": plan["required_gpus"], "epochs": plan["joint"]["epochs"],
                              "step_budget": plan["joint"]["instances"][0]["step_budget"],
                              "inputs": plan["inputs"], "run_dir": plan["run_dir"],
                              "contract_hash": plan["contract"].contract_hash}, indent=2))
            return 0
        import torch

        validate_cuda_capacity(
            required_gpus=plan["required_gpus"],
            available_gpus=torch.cuda.device_count(),
            require_cuda=args.require_cuda or args.action in {"run", "resume"},
        )
        joint_path = _write_plan(plan, resume=args.action == "resume")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc-per-node={plan['required_gpus']}", "-m",
                   "unity_psf.training.entrypoints.train_modality_expert_parallel",
                   "--config", str(joint_path),
                   "--run-root", str(args.run_root.expanduser().resolve()), "--run-id", args.run_id]
        if smoke and not args.require_cuda:
            command.append("--allow-cpu-smoke")
        if args.action == "resume":
            command.append("--resume")
        launch_environment = os.environ.copy()
        launch_environment[CANONICAL_LAUNCH_ENV] = CANONICAL_TRAINING_ENTRYPOINT
        launch_environment[CANONICAL_CONTRACT_HASH_ENV] = plan["contract"].contract_hash
        return_code = subprocess.run(command, check=False, env=launch_environment).returncode
        if return_code == 0 and smoke:
            from unity_psf.training import atomic_write_json

            receipt = verify_short_run(Path(plan["run_dir"]), expected_routes=parse_routes(args.route))
            atomic_write_json(Path(plan["run_dir"]) / "metadata/short_run_gate.json", receipt)
            print(json.dumps({"event": "short_run_gate", **receipt}, sort_keys=True))
        return return_code
    except (ValueError, FileExistsError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
