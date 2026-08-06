from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from unity_psf.config import load_config
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.localization.legacy_decode import (
    LegacyEmitterSet,
    decode_legacy_targets,
    decode_liteloc_eval_emitters,
)
from unity_psf.localization.training_adapter import LocalizationTrainBatch, localization_batch_to_device
from unity_psf.runtime import ensure_run_layout
from unity_psf.training import build_trainer_runtime, load_training_checkpoint
from unity_psf.training.localizer_eval import (
    _concat_emitter_sets,
    _model_input_batch_size,
    _offset_emitter_set,
    build_localizer_eval_provider,
)
from unity_psf.training.run_high_fidelity import (
    _condition_store_batch_provider_overrides,
    _condition_store_from_runtime_config,
)

from .evaluate_z_bins import RunZBinResult, evaluate_z_bins, write_z_bin_package


Z_EDGES_NM = (-2000.0, -1500.0, -1000.0, -500.0, 0.0, 500.0, 1000.0, 1500.0, 2000.0)


@dataclass(frozen=True)
class RunSpec:
    name: str
    density_um2: float
    config_path: Path
    checkpoint_path: Path
    physical_state_path: Path


def parse_run_specs(values: Sequence[Sequence[str]]) -> tuple[RunSpec, ...]:
    return tuple(
        RunSpec(
            name=str(value[0]),
            density_um2=float(value[1]),
            config_path=Path(value[2]),
            checkpoint_path=Path(value[3]),
            physical_state_path=Path(value[4]),
        )
        for value in values
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained double-helix localizers by ground-truth z bin.")
    parser.add_argument(
        "--run",
        nargs=5,
        action="append",
        metavar=("NAME", "DENSITY", "CONFIG", "CHECKPOINT", "PHYSICAL_STATE"),
        required=True,
    )
    parser.add_argument(
        "--evaluation-density-um2",
        type=float,
        help="Density of the fixed evaluation set when it differs from the checkpoint training density.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal z-bin evaluation requires CUDA")
    specs = parse_run_specs(args.run)
    results = [
        evaluate_run(
            spec,
            runtime_root=args.output_dir / ".runtime",
            evaluation_density_um2=args.evaluation_density_um2,
        )
        for spec in specs
    ]
    artifacts = write_z_bin_package(results, args.output_dir)
    print(
        json.dumps(
            {
                "status": "completed",
                "device": "cuda:0",
                "gpu_name": torch.cuda.get_device_name(0),
                "summary_json": str(artifacts.summary_json),
                "metrics_png": str(artifacts.metrics_png),
                "heatmap_png": str(artifacts.heatmap_png),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def evaluate_run(
    spec: RunSpec,
    *,
    runtime_root: Path,
    evaluation_density_um2: float | None = None,
) -> RunZBinResult:
    for path in (spec.config_path, spec.checkpoint_path, spec.physical_state_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = load_config(spec.config_path)
    physical_state = json.loads(spec.physical_state_path.read_text(encoding="utf-8"))
    coeff_maps = _mapping(physical_state, "physical_state").get("coeff_maps")
    if not isinstance(coeff_maps, list) or not coeff_maps:
        raise ValueError("physical state must contain a non-empty coeff_maps list")
    config = _with_physical_coeff_maps(config, coeff_maps)
    if evaluation_density_um2 is not None:
        config = _with_evaluation_density(config, evaluation_density_um2)
    config_base_dir = spec.config_path.resolve().parent
    train_cfg = _mapping(config.get("train"), "train")
    runtime_config = build_localization_runtime_config(
        config,
        config_base_dir=config_base_dir,
        seed=int(_mapping(train_cfg.get("eval"), "train.eval").get("seed", 100000)),
    )
    condition_store = _condition_store_from_runtime_config(runtime_config)
    if condition_store is None:
        raise RuntimeError("double-helix evaluation requires physical conditioning maps")
    layout = ensure_run_layout(runtime_root, spec.name)
    runtime = build_trainer_runtime(
        runtime_config,
        layout=layout,
        model_registry=build_localization_model_registry(),
        batch_provider_overrides=_condition_store_batch_provider_overrides(condition_store),
    )
    device = torch.device(str(runtime_config["device"]))
    if device.type != "cuda":
        raise RuntimeError(f"formal z-bin evaluation requires a CUDA runtime, got {device}")
    checkpoint_state = load_training_checkpoint(
        spec.checkpoint_path,
        model=runtime.model,
        map_location=device,
    )
    runtime.model.eval()
    eval_provider = build_localizer_eval_provider(
        train_cfg,
        config_base_dir=config_base_dir,
        root_config=config,
        condition_store=condition_store,
    )
    if eval_provider is None:
        raise RuntimeError("resolved config does not enable online evaluation")
    pred, target = _infer_emitters(
        runtime.model,
        list(eval_provider()),
        device=device,
        train_cfg=train_cfg,
    )
    contract = _legacy_eval_contract(train_cfg, config)
    evaluation = evaluate_z_bins(
        pred,
        target,
        z_edges_nm=Z_EDGES_NM,
        **contract,
    )
    return RunZBinResult(
        name=spec.name,
        density_um2=spec.density_um2,
        checkpoint_path=str(spec.checkpoint_path.resolve()),
        checkpoint_epoch=checkpoint_state.epoch,
        checkpoint_global_step=checkpoint_state.global_step,
        physical_state_path=str(spec.physical_state_path.resolve()),
        evaluation_density_um2=evaluation_density_um2,
        config_path=str(spec.config_path.resolve()),
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device),
        provenance={
            "checkpoint_sha256": _sha256(spec.checkpoint_path),
            "physical_state_sha256": _sha256(spec.physical_state_path),
            "physical_coeff_map_sha256": {
                str(item["name"]): _sha256(Path(str(item["coeff_maps_npz"])))
                for item in coeff_maps
            },
            "checkpoint_embedded_physical_state": checkpoint_state.physical_state,
            "evaluated_physical_state": physical_state,
            "physical_pairing_contract": _physical_pairing_contract(
                checkpoint_state.physical_state,
                physical_state,
            ),
            "training_density_um2": spec.density_um2,
            "evaluation_density_um2": spec.density_um2 if evaluation_density_um2 is None else evaluation_density_um2,
            "eval_seed": int(_mapping(train_cfg.get("eval"), "train.eval").get("seed", 100000)),
            "decode_threshold": 0.3,
            "matching_contract": contract,
        },
        evaluation=evaluation,
    )


def _infer_emitters(
    model: torch.nn.Module,
    batches: Sequence[object],
    *,
    device: torch.device,
    train_cfg: Mapping[str, Any],
) -> tuple[LegacyEmitterSet, LegacyEmitterSet]:
    scaling = train_cfg.get("scaling") if isinstance(train_cfg.get("scaling"), Mapping) else {}
    loss_cfg = train_cfg.get("loss") if isinstance(train_cfg.get("loss"), Mapping) else {}
    loss_params = loss_cfg.get("params") if isinstance(loss_cfg.get("params"), Mapping) else {}
    photon_scale = float(scaling["photon_max"]) if "photon_max" in scaling else None
    z_scale = float(scaling["z_max"]) if "z_max" in scaling else None
    target_order = str(loss_params.get("target_order", "legacy_iwae"))
    pred_sets: list[LegacyEmitterSet] = []
    target_sets: list[LegacyEmitterSet] = []
    batch_offset = 0
    with torch.no_grad():
        for batch in batches:
            loc_batch = batch.inputs
            if not isinstance(loc_batch, LocalizationTrainBatch):
                raise TypeError("online eval batch must contain LocalizationTrainBatch inputs")
            loc_batch = localization_batch_to_device(loc_batch, device)
            output = model(loc_batch.model_input)
            pred = decode_liteloc_eval_emitters(output, photon_scale=photon_scale, z_scale=z_scale)
            target = decode_legacy_targets(
                loc_batch.pxyz_tar,
                loc_batch.mask_tar,
                target_order=target_order,
                photon_scale=photon_scale,
                z_scale=z_scale,
            )
            pred_sets.append(_offset_emitter_set(pred, batch_offset))
            target_sets.append(_offset_emitter_set(target, batch_offset))
            batch_offset += _model_input_batch_size(loc_batch.model_input)
    return _concat_emitter_sets(pred_sets), _concat_emitter_sets(target_sets)


def _legacy_eval_contract(train_cfg: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, object]:
    eval_cfg = train_cfg.get("eval") if isinstance(train_cfg.get("eval"), Mapping) else {}
    online_cfg = train_cfg.get("online_generation") if isinstance(train_cfg.get("online_generation"), Mapping) else {}
    optical_cfg = config.get("optical") if isinstance(config.get("optical"), Mapping) else {}
    if "dist_tol_xy_nm" in eval_cfg:
        dist_tol_xy_nm = float(eval_cfg["dist_tol_xy_nm"])
    elif "dist_tolr_nm" in eval_cfg:
        dist_tol_xy_nm = float(eval_cfg["dist_tolr_nm"])
    elif "dist_tolr" in eval_cfg:
        dist_tol_xy_nm = float(eval_cfg["dist_tolr"])
    else:
        dist_tol_xy_nm = 250.0
    dist_tol_z_nm = eval_cfg.get("dist_tol_z_nm", eval_cfg.get("dist_tolz_nm", eval_cfg.get("dist_tolz", 500.0)))
    return {
        "dist_tol_xy_px": None,
        "dist_tol_xy_nm": dist_tol_xy_nm,
        "dist_tol_z_nm": None if dist_tol_z_nm is None else float(dist_tol_z_nm),
        "pixel_size_nm_x": float(eval_cfg.get("pixel_size_nm_x", online_cfg.get("pixel_size_nm_x", optical_cfg.get("pixel_size_nm_x", 1.0)))),
        "pixel_size_nm_y": float(eval_cfg.get("pixel_size_nm_y", online_cfg.get("pixel_size_nm_y", optical_cfg.get("pixel_size_nm_y", 1.0)))),
        "match_dims": int(eval_cfg.get("match_dims", 3)),
    }


def _with_physical_coeff_maps(config: Mapping[str, Any], coeff_maps: list[object]) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    train_cfg = dict(_mapping(updated.get("train"), "train"))
    online_cfg = dict(_mapping(train_cfg.get("online_generation"), "train.online_generation"))
    online_cfg["dual_domain_coeff_maps"] = copy.deepcopy(coeff_maps)
    train_cfg["online_generation"] = online_cfg
    updated["train"] = train_cfg
    return updated


def _with_evaluation_density(config: Mapping[str, Any], density_um2: float) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    train_cfg = dict(_mapping(updated.get("train"), "train"))
    online_cfg = dict(_mapping(train_cfg.get("online_generation"), "train.online_generation"))
    online_cfg["emitter_density_um2"] = float(density_um2)
    train_cfg["online_generation"] = online_cfg
    updated["train"] = train_cfg
    return updated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_pairing_contract(
    checkpoint_state: Mapping[str, Any] | None,
    evaluated_state: Mapping[str, Any],
) -> str:
    checkpoint_paths = _physical_coeff_map_paths(checkpoint_state)
    evaluated_paths = _physical_coeff_map_paths(evaluated_state)
    if checkpoint_paths == evaluated_paths:
        return "final_network_plus_final_physical_state_endpoint"
    return "cross_density_checkpoint_on_fixed_evaluation_physical_state"


def _physical_coeff_map_paths(state: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(state, Mapping):
        return ()
    entries = state.get("coeff_maps")
    if not isinstance(entries, list):
        return ()
    return tuple(
        str(item["coeff_maps_npz"])
        for item in entries
        if isinstance(item, Mapping) and "coeff_maps_npz" in item
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
