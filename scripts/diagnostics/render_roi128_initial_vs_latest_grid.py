#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from neptune_v03.config import load_config
from neptune_v03.localization import build_localization_model_registry, build_localization_runtime_config
from neptune_v03.runtime import ensure_run_layout
from neptune_v03.training import build_trainer_runtime
from neptune_v03.training.loop import load_training_checkpoint
from neptune_v03.training.run_high_fidelity import (
    _auto_build_roi_bank,
    _build_vector_roi_gamma_objective,
    _mapping,
    _resolve_roi_bank_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ROI128 raw vs initial vs latest physical projection grids.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, default=12500)
    parser.add_argument("--frame-start", type=int, default=100)
    parser.add_argument("--frame-stop", type=int, default=110)
    parser.add_argument("--max-rois", type=int, default=5)
    parser.add_argument("--target-emitters", type=int, default=5000)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir.parents[1] / ".local" / "tmp" / "standard" / f"resolved_standard_roi_gamma_batch_budget_{run_dir.name.split('_')[-1]}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = _config_with_current_physical_maps(load_config(config_path), run_dir=run_dir)
    train_cfg = _mapping(config.get("train"), "train")
    gamma_cfg = dict(_mapping(train_cfg.get("roi_bank_gamma"), "train.roi_bank_gamma"))
    gamma_cfg["roi_bank_frame_range"] = [int(args.frame_start), int(args.frame_stop)]
    gamma_cfg["roi_library_max_rois"] = max(int(args.max_rois) * 4, 32)
    gamma_cfg["roi_bank_max_rois"] = max(int(args.max_rois) * 4, 32)
    gamma_cfg["target_projected_emitters"] = int(args.target_emitters)
    config_base_dir = config_path.parent

    layout = ensure_run_layout(run_dir.parent, run_dir.name)
    runtime_config = build_localization_runtime_config(config, config_base_dir=config_base_dir, seed=0)
    runtime = build_trainer_runtime(
        runtime_config,
        layout=layout,
        model_registry=build_localization_model_registry(),
    )
    load_training_checkpoint(run_dir / "checkpoints" / "checkpoint_latest.pt", model=runtime.model, map_location=next(runtime.model.parameters()).device)

    roi_source = _resolve_roi_bank_source(gamma_cfg, train_cfg=train_cfg, config=config, config_base_dir=config_base_dir)
    if roi_source is None:
        raise ValueError("Could not resolve ROI source")
    bank = _auto_build_roi_bank(gamma_cfg, roi_source=roi_source, model=runtime.model, train_cfg=train_cfg)
    records = tuple(sorted(bank.records, key=lambda r: (-len(r.emitters), int(r.roi_id))))[: int(args.max_rois)]
    if not records:
        raise ValueError("No ROI records built")

    initial_objective = _build_vector_roi_gamma_objective(gamma_cfg, train_cfg=train_cfg, config=config, model=runtime.model)
    initial_gamma = initial_objective.initial_gamma().detach()

    latest_summary = json.loads(
        (
            run_dir
            / "artifacts"
            / "roi_bank_gamma"
            / f"step_{int(args.step):08d}"
            / "source_auto_built"
            / "domain_multi"
            / "gamma_alternation_summary.json"
        ).read_text(encoding="utf-8")
    )
    latest_feedback = latest_summary["feedback_coeff_maps"]
    latest_base_maps = tuple(
        (str(name), str((run_dir / Path(path_str)).resolve()) if not Path(path_str).is_absolute() else str(Path(path_str)))
        for name, path_str in sorted(latest_feedback.items())
    )
    latest_objective = _build_vector_roi_gamma_objective(
        {**gamma_cfg, "base_coeff_maps": latest_base_maps},
        train_cfg=train_cfg,
        config=config,
        model=runtime.model,
    )
    latest_gamma = latest_objective.initial_gamma().detach()

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        center_idx = int(record.raw_frames_photon.shape[0] // 2)
        raw_center = np.asarray(record.raw_frames_photon[center_idx], dtype=np.float32)
        initial_recon = (
            initial_objective.render_record_reconstruction(
                gamma=initial_gamma,
                raw_shape=tuple(raw_center.shape),
                emitters=record.emitters,
                background=record.background_smoothed,
                roi_origin_xy_px=record.roi_origin_xy_px,
                domain_name=str(record.domain_name),
            )
            .detach()
            .cpu()
            .numpy()
        )
        latest_recon = (
            latest_objective.render_record_reconstruction(
                gamma=latest_gamma,
                raw_shape=tuple(raw_center.shape),
                emitters=record.emitters,
                background=record.background_smoothed,
                roi_origin_xy_px=record.roi_origin_xy_px,
                domain_name=str(record.domain_name),
            )
            .detach()
            .cpu()
            .numpy()
        )
        by_domain.setdefault(str(record.domain_name), []).append(
            {
                "roi_id": int(record.roi_id),
                "frame_index": int(record.emitters[0].frame_index) if record.emitters else int(record.frame_window[0]),
                "emitter_count": len(record.emitters),
                "raw": raw_center,
                "photon": np.asarray(raw_center, dtype=np.float32),
                "initial": initial_recon,
                "latest": latest_recon,
                "raw_minus_latest": raw_center - latest_recon,
                "latest_minus_initial": latest_recon - initial_recon,
            }
        )

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else run_dir / "artifacts" / "roi_compare_grids" / f"step_{int(args.step):08d}_frames_{int(args.frame_start)}_{int(args.frame_stop)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for domain, items in sorted(by_domain.items()):
        png_path = out_dir / f"{domain}_roi128_raw_initial_vs_latest.png"
        _plot_compare_grid(png_path, domain=domain, items=items)
        written.append(str(png_path))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"files": written}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": written, "summary": str(summary_path)}, indent=2))
    return 0


def _plot_compare_grid(out_path: Path, *, domain: str, items: list[dict[str, Any]]) -> None:
    count = len(items)
    fig, axes = plt.subplots(6, count, figsize=(3.0 * max(count, 1), 15.5), dpi=180, squeeze=False, constrained_layout=True)
    for col, item in enumerate(items):
        raw = np.asarray(item["raw"], dtype=np.float32)
        photon = np.asarray(item["photon"], dtype=np.float32)
        initial = np.asarray(item["initial"], dtype=np.float32)
        latest = np.asarray(item["latest"], dtype=np.float32)
        raw_minus_latest = np.asarray(item["raw_minus_latest"], dtype=np.float32)
        latest_minus_initial = np.asarray(item["latest_minus_initial"], dtype=np.float32)
        photon_vmin, photon_vmax = _robust_limits(raw, photon, initial, latest)
        diff1 = max(float(np.percentile(np.abs(raw_minus_latest), 99.5)), 1e-6)
        diff2 = max(float(np.percentile(np.abs(latest_minus_initial), 99.5)), 1e-6)
        panels = (
            ("raw center frame", raw, "gray", photon_vmin, photon_vmax),
            ("photon ROI", photon, "magma", photon_vmin, photon_vmax),
            ("initial peak zmap recon", initial, "magma", photon_vmin, photon_vmax),
            ("latest physical zmap recon", latest, "magma", photon_vmin, photon_vmax),
            ("raw - latest", raw_minus_latest, "coolwarm", -diff1, diff1),
            ("latest - initial", latest_minus_initial, "coolwarm", -diff2, diff2),
        )
        for row, (label, image, cmap, lo, hi) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 0:
                ax.set_title(
                    f"roi {item['roi_id']} K={item['emitter_count']}\nframe {item['frame_index']}",
                    fontsize=8,
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{domain} same 128x128 ROI: raw vs photon ROI vs zmap recon", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _robust_limits(*arrays: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    valid = [np.asarray(arr, dtype=np.float32).reshape(-1) for arr in arrays if arr is not None and np.asarray(arr).size]
    if not valid:
        return 0.0, 1.0
    merged = np.concatenate(valid)
    return float(np.percentile(merged, low)), float(np.percentile(merged, high))


def _config_with_current_physical_maps(config: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    coeff_maps = _current_physical_coeff_maps(run_dir)
    if not coeff_maps:
        return config
    updated = dict(config)
    train_cfg = dict(_mapping(updated.get("train"), "train"))
    online_cfg = dict(_mapping(train_cfg.get("online_generation"), "train.online_generation"))
    online_cfg["dual_domain_coeff_maps"] = coeff_maps
    train_cfg["online_generation"] = online_cfg
    updated["train"] = train_cfg
    return updated


def _current_physical_coeff_maps(run_dir: Path):
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_latest.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        entries = checkpoint.get("physical_coeff_maps")
        if entries:
            return entries
    state_path = run_dir / "metadata" / "current_physical_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        entries = state.get("coeff_maps")
        if entries:
            return entries
    manifest_path = run_dir / "metadata" / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("current_physical_coeff_maps")
        if entries:
            return entries
    return None


if __name__ == "__main__":
    raise SystemExit(main())
