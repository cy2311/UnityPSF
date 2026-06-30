from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
V03_ROOT = REPO_ROOT / "neptune_v0.3"
IWAE_ROOT = REPO_ROOT / "neptune_iwae"


def _install_v03_path() -> None:
    path = str(V03_ROOT / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


def _install_old_path() -> None:
    for path in reversed((str(IWAE_ROOT), str(V03_ROOT / "src"))):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _stats(value: Any) -> dict[str, Any]:
    if value is None:
        return {"present": False}
    if isinstance(value, tuple):
        value = value[0]
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array)
    result: dict[str, Any] = {
        "present": True,
        "shape": [int(v) for v in array.shape],
        "dtype": str(array.dtype),
    }
    if array.size == 0:
        return result
    numeric = array.astype(np.float64, copy=False)
    result.update(
        {
            "min": float(np.nanmin(numeric)),
            "max": float(np.nanmax(numeric)),
            "mean": float(np.nanmean(numeric)),
            "std": float(np.nanstd(numeric)),
            "p01": float(np.nanpercentile(numeric, 1)),
            "p50": float(np.nanpercentile(numeric, 50)),
            "p99": float(np.nanpercentile(numeric, 99)),
            "sum": float(np.nansum(numeric)),
        }
    )
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return {
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.ndarray):
        if value.size <= 16:
            return value.tolist()
        return {
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _model_image(model_input: Any) -> torch.Tensor:
    if isinstance(model_input, (tuple, list)):
        first = model_input[0]
        if torch.is_tensor(first):
            return first
    if torch.is_tensor(model_input):
        return model_input
    raise TypeError(f"unsupported model_input container: {type(model_input)!r}")


def _condition_stats(model_input: Any) -> dict[str, Any]:
    if not isinstance(model_input, (tuple, list)) or len(model_input) < 2:
        return {"present": False}
    return _stats(model_input[1])


def _loc_batch(batch: Any) -> Any:
    if hasattr(batch, "model_input"):
        return batch
    if hasattr(batch, "inputs") and hasattr(batch.inputs, "model_input"):
        return batch.inputs
    return batch


def _target_xyzph(batch: Any) -> torch.Tensor:
    loc_batch = _loc_batch(batch)
    pxyz = loc_batch.pxyz_tar if hasattr(loc_batch, "pxyz_tar") else loc_batch[3]
    return pxyz.detach().cpu() if torch.is_tensor(pxyz) else torch.as_tensor(pxyz)


def _mask(batch: Any) -> torch.Tensor:
    loc_batch = _loc_batch(batch)
    mask = loc_batch.mask_tar if hasattr(loc_batch, "mask_tar") else loc_batch[4]
    return mask.detach().cpu().to(dtype=torch.bool) if torch.is_tensor(mask) else torch.as_tensor(mask, dtype=torch.bool)


def _summarize_batch(name: str, batch: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    loc_batch = _loc_batch(batch)
    model_input = loc_batch.model_input if hasattr(loc_batch, "model_input") else loc_batch[0]
    image = _model_image(model_input).detach().cpu().to(dtype=torch.float32)
    detect = loc_batch.detect_tar if hasattr(loc_batch, "detect_tar") else loc_batch[1]
    bkg = loc_batch.bkg_tar if hasattr(loc_batch, "bkg_tar") else loc_batch[2]
    pxyz = _target_xyzph(batch).to(dtype=torch.float32)
    mask = _mask(batch)
    active = pxyz[mask] if bool(mask.any()) else torch.zeros((0, pxyz.shape[-1]), dtype=torch.float32)
    center = image[:, int(image.shape[1] // 2)] if image.ndim == 4 else image
    report = {
        "name": name,
        "metadata": _jsonable(dict(metadata or getattr(batch, "metadata", {}) or {})),
        "model_input": _stats(image),
        "model_input_center": _stats(center),
        "condition": _condition_stats(model_input),
        "detect_tar": _stats(detect),
        "bkg_tar": _stats(bkg),
        "pxyz_tar": _stats(pxyz),
        "mask_tar": _stats(mask.to(dtype=torch.float32)),
        "active_emitter_count_total": int(mask.sum().item()),
        "active_emitter_count_per_sample": [int(v) for v in mask.sum(dim=1).tolist()] if mask.ndim >= 2 else [],
        "active_xyzph": _stats(active),
        "adjacent_window_overlap": _adjacent_window_overlap(image),
    }
    if active.numel() > 0 and active.shape[-1] >= 4:
        report["active_columns"] = {
            "col0": _stats(active[:, 0]),
            "col1": _stats(active[:, 1]),
            "col2": _stats(active[:, 2]),
            "col3": _stats(active[:, 3]),
        }
        report["pxyz_order_hint"] = {
            "old_reference_order": "phot,x,y,z before v0.3 target conversion",
            "v03_order": "x,y,z,phot",
            "columns_are_reported_raw_from_this_batch": True,
        }
    return report


def _adjacent_window_overlap(image: torch.Tensor) -> dict[str, Any]:
    image = image.detach().cpu().to(dtype=torch.float32)
    if image.ndim != 4 or image.shape[0] < 2 or image.shape[1] < 2:
        return {"checked": False, "reason": "expected [batch, channels, height, width] with batch>=2/channels>=2"}
    pairs = []
    for idx in range(int(image.shape[0]) - 1):
        left = image[idx, 1]
        right = image[idx + 1, 0]
        diff = (left - right).abs()
        pairs.append(
            {
                "sample_i_center_vs_sample_i_plus_1_left": [int(idx), int(idx + 1)],
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
                "allclose_1e_6": bool(torch.allclose(left, right, atol=1e-6, rtol=0.0)),
            }
        )
    return {"checked": True, "pairs": pairs}


def _save_sample_npz(path: Path, batch: Any) -> None:
    loc_batch = _loc_batch(batch)
    model_input = loc_batch.model_input if hasattr(loc_batch, "model_input") else loc_batch[0]
    image = _model_image(model_input).detach().cpu().numpy()
    condition = None
    if isinstance(model_input, (tuple, list)) and len(model_input) >= 2 and torch.is_tensor(model_input[1]):
        condition = model_input[1].detach().cpu().numpy()
    detect = loc_batch.detect_tar if hasattr(loc_batch, "detect_tar") else loc_batch[1]
    bkg = loc_batch.bkg_tar if hasattr(loc_batch, "bkg_tar") else loc_batch[2]
    pxyz = loc_batch.pxyz_tar if hasattr(loc_batch, "pxyz_tar") else loc_batch[3]
    mask = loc_batch.mask_tar if hasattr(loc_batch, "mask_tar") else loc_batch[4]
    payload = {
        "model_input": image[: min(4, image.shape[0])],
        "detect_tar": detect.detach().cpu().numpy()[: min(4, image.shape[0])],
        "bkg_tar": bkg.detach().cpu().numpy()[: min(4, image.shape[0])],
        "pxyz_tar": pxyz.detach().cpu().numpy()[: min(4, image.shape[0])],
        "mask_tar": mask.detach().cpu().numpy()[: min(4, image.shape[0])],
    }
    if condition is not None:
        payload["condition"] = condition[: min(4, condition.shape[0])]
    np.savez_compressed(path, **payload)


def _old_tuple_to_batch(batch_tuple: Any, metadata: dict[str, Any] | None = None) -> Any:
    class _Batch:
        pass

    batch = _Batch()
    batch.model_input, batch.detect_tar, batch.bkg_tar, batch.pxyz_tar, batch.mask_tar, *_rest = batch_tuple
    batch.metadata = dict(metadata or {})
    return batch


def _dump_v03(config_path: Path, *, seed: int, batch_size: int, num_batches: int) -> tuple[list[Any], dict[str, Any]]:
    _install_v03_path()
    from neptune_v03.config import load_config
    from neptune_v03.localization import build_localization_model_registry, build_localization_runtime_config
    from neptune_v03.runtime import ensure_run_layout
    from neptune_v03.training import build_trainer_runtime

    cfg = load_config(config_path)
    runtime_cfg = build_localization_runtime_config(cfg, config_base_dir=config_path.parent, seed=int(seed))
    runtime_cfg["device"] = "cpu"
    runtime_cfg["optimizer"]["params"]["lr"] = 0.0
    runtime_cfg["batch_provider"]["params"]["steps_per_epoch"] = int(num_batches)
    runtime_cfg["batch_provider"]["params"]["batch_size"] = int(batch_size)
    layout = ensure_run_layout(V03_ROOT / ".local/tmp/diagnostics/training_batch_parity", "v03_runtime", stage_names=("batch_dump",))
    runtime = build_trainer_runtime(runtime_cfg, layout=layout, model_registry=build_localization_model_registry())
    batches = [item.inputs for item in runtime.batch_provider(1)[: int(num_batches)]]
    return batches, {"runtime_batch_provider": runtime_cfg["batch_provider"], "resolved_contract": runtime_cfg.get("resolved_contract", {})}


def _dump_old(
    config_path: Path,
    *,
    seed: int,
    output_dir: Path,
    batch_size: int,
    num_batches: int,
    sequence_count: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    _install_old_path()
    from neptune_core.cached_window_train import (
        _build_lut_generate_dataset_fn,
        _build_sequence_domain_indices,
        build_cached_train_dataloader,
    )
    from neptune_core.online_config import OnlinelocalizationTrainConfig
    from neptune_core.online_train import (
        _build_nat_phase_payload,
        _build_subregion_context,
        _build_training_frame_proc,
        _resolve_dual_domain_coeff_maps,
        _resolve_nat_coeff_maps_path,
    )
    from neptune_core.localization_runtime import build_online_training_runtime
    from neptune_core.localization_runtime.data_factory import BkgFrameRescalar, TargetProcess

    cfg = _load_config(config_path)
    train_cfg = cfg.get("train") or {}
    online_cfg = train_cfg.get("online_generation") or {}
    train_config = OnlinelocalizationTrainConfig.from_pipeline_config(
        cfg,
        pipeline_config_path=config_path,
        checkpoint_init=None,
        output_dir=output_dir,
        device_override="cpu",
        batch_size_override=int(batch_size),
        steps_per_epoch_override=1,
    )
    dual_domain_coeff_maps = _resolve_dual_domain_coeff_maps(cfg, pipeline_config_path=config_path)
    coeff_maps_npz = _resolve_nat_coeff_maps_path(cfg, pipeline_config_path=config_path)
    if coeff_maps_npz is None and dual_domain_coeff_maps:
        coeff_maps_npz = dual_domain_coeff_maps[0][1]
    phase_payload = _build_nat_phase_payload(cfg, coeff_maps_npz=coeff_maps_npz)
    subregion_context = _build_subregion_context(
        cfg,
        coeff_maps_npz=coeff_maps_npz,
        conditioning_enabled=train_config.conditioning_enabled,
        dual_domain_coeff_maps=dual_domain_coeff_maps,
    )
    condition_channels = 0
    if subregion_context is not None:
        condition_channels = int(subregion_context.condition_channels)
    runtime = build_online_training_runtime(
        cfg,
        checkpoint_init=None,
        output_dir=output_dir,
        device=torch.device("cpu"),
        epochs=train_config.epochs,
        batch_size=train_config.batch_size,
        window_size=train_config.window_size,
        condition_channels=condition_channels,
        conditioning_mode=train_config.online_generation.conditioning_mode,
    )
    frame_proc = _build_training_frame_proc(cfg, runtime)
    bkg_proc = BkgFrameRescalar(scale=runtime.param.Scaling.bkg_max)
    target_proc = TargetProcess(
        disable_attr=runtime.param.DataFilter.disabled_attributes,
        phot_max=runtime.param.Scaling.phot_max,
        z_max=runtime.param.Scaling.z_max,
    )
    generate_dataset_fn = _build_lut_generate_dataset_fn(cfg, pipeline_config_path=config_path)
    sequence_count = int(sequence_count or max(1, int(online_cfg.get("sequence_count", 0) or 0) or int(online_cfg.get("sequence_window_chunks", 1) or 1)))
    center_samples_per_epoch = int(train_config.batch_size) * int(num_batches)
    domain_indices = _build_sequence_domain_indices(
        sequence_count=sequence_count,
        epoch=0,
        mode=str(online_cfg.get("domain_balance_mode", "fixed")),
    )
    torch.manual_seed(int(seed))
    _dataset, loader, cache_meta = build_cached_train_dataloader(
        cfg,
        phase_payload=phase_payload,
        frame_proc=frame_proc,
        bkg_frame_proc=bkg_proc,
        tar_proc=target_proc,
        window_size=train_config.window_size,
        batch_size=train_config.batch_size,
        center_samples_per_epoch=center_samples_per_epoch,
        sequence_count=sequence_count,
        seed=int(seed),
        generate_dataset_fn=generate_dataset_fn,
        pin_memory=False,
        subregion_context=subregion_context,
        conditioning_mode=train_config.online_generation.conditioning_mode,
        nat_simulation_mode=str(online_cfg.get("nat_simulation_mode", "grid_map")),
        nat_grid_size=online_cfg.get("nat_grid_size", 2),
        nat_grid_z_steps=int(online_cfg.get("nat_grid_z_steps", 41)),
        append_domain_onehot=bool(online_cfg.get("append_domain_onehot", False)),
        forced_domain_indices=domain_indices,
    )
    batches = []
    for idx, batch_tuple in enumerate(loader):
        if idx >= int(num_batches):
            break
        batches.append(_old_tuple_to_batch(batch_tuple, {**cache_meta, "loader_batch_index": int(idx)}))
    ordered_preview = _summarize_old_dataset_order(_dataset, limit=min(int(batch_size), 8))
    return batches, {
        "cache_meta": cache_meta,
        "dual_domain_coeff_maps": [(n, str(p)) for n, p in dual_domain_coeff_maps],
        "ordered_dataset_preview": ordered_preview,
    }


def _summarize_old_dataset_order(dataset: Any, *, limit: int) -> dict[str, Any]:
    images = []
    targets = []
    for idx in range(min(int(limit), len(dataset))):
        item = dataset[idx]
        images.append(_model_image(item[0] if not isinstance(item[0], tuple) else item[0][0]))
        targets.append(item[1])
    if not images:
        return {"present": False}
    image = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in images], dim=0)
    return {
        "present": True,
        "num_windows": int(image.shape[0]),
        "model_input": _stats(image),
        "adjacent_window_overlap": _adjacent_window_overlap(image),
        "target_tuple_lengths": [len(t) if isinstance(t, (tuple, list)) else None for t in targets],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump old vs v0.3 same-seed training batch parity stats.")
    parser.add_argument(
        "--old-config",
        default=str(
            IWAE_ROOT
            / "output/microtube_real_tiff_switch_train_anchor99_roi_gamma_long_default_interval5_steps100_epoch300_3052/resolved_config.json"
        ),
    )
    parser.add_argument(
        "--v03-config",
        default=str(V03_ROOT / ".local/tmp/parity/resolved_v03_current_route_epoch30_3174.yaml"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--old-sequence-count", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(V03_ROOT / ".local/tmp/diagnostics/training_batch_parity"),
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    old_batch, old_meta = _dump_old(
        Path(args.old_config).resolve(),
        seed=int(args.seed),
        output_dir=out / "old_runtime",
        batch_size=int(args.batch_size),
        num_batches=int(args.num_batches),
        sequence_count=(None if int(args.old_sequence_count) <= 0 else int(args.old_sequence_count)),
    )
    v03_batch, v03_meta = _dump_v03(
        Path(args.v03_config).resolve(),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        num_batches=int(args.num_batches),
    )
    if old_batch:
        _save_sample_npz(out / "old_batch_sample.npz", old_batch[0])
    if v03_batch:
        _save_sample_npz(out / "v03_batch_sample.npz", v03_batch[0])
    report = {
        "schema_version": "training_batch_parity.v2",
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "old_config": str(Path(args.old_config).resolve()),
        "v03_config": str(Path(args.v03_config).resolve()),
        "old": {
            "run": _jsonable(old_meta),
            "batches": [
                _summarize_batch(f"old_batch_{idx}", batch, getattr(batch, "metadata", {}))
                for idx, batch in enumerate(old_batch)
            ],
        },
        "v03": {
            "run": _jsonable(v03_meta),
            "batches": [
                _summarize_batch(f"v03_batch_{idx}", batch, getattr(batch, "metadata", {}))
                for idx, batch in enumerate(v03_batch)
            ],
        },
        "artifacts": {
            "old_batch_npz": str(out / "old_batch_sample.npz"),
            "v03_batch_npz": str(out / "v03_batch_sample.npz"),
        },
    }
    (out / "training_batch_parity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out / "training_batch_parity_report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
