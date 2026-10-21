#!/usr/bin/env python3
"""Evaluate anchor-local-deblender merge ablations and oracle ceilings."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_pipeline_config
from phase_retrieval.api import run_phase_retrieval
from simulation.api import generate_dataset, normalize_dataset
from neptune.smlm_v2a.data.data_factory import BkgFrameRescalar, FrameRescalar, TargetProcess
from neptune.smlm_v2a.evaluation.matcher import GreedyHungarianMatching
from neptune.smlm_v2a.generic.localizations import FrameLocSet
from neptune.smlm_v2a.local_deblending import LocalSetPredictor, build_conflict_patches
from neptune.smlm_v2a.local_deblending.merge import MERGE_STRATEGIES, merge_patch_predictions_with_strategy
from neptune.smlm_v2a.residual_refinement import MeanPSFCenterRenderer
from neptune.smlm_v2a.training import train_kernels, train_setup
from neptune.smlm_v2a.training.train_local_deblender import (
    _build_window_loader,
    _eval_to_metrics,
    _freeze_module,
    _instantiate_global_model,
    _mine_patch_examples,
    _predict_patch_refinements,
    resolve_local_deblending_config,
)
from neptune.smlm_v2a.training.train_pipeline import (
    _build_epoch_cfg,
    _build_loc_meta,
    _build_param,
    _resolve_device,
    _setup_logger,
)
from neptune.smlm_v2a.utils.io import model_io


LOG = logging.getLogger("analyze_local_deblender_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze anchor-conditioned local-deblender ablations")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--phase-mode", type=str, default="single", choices=["single", "batch", "auto"])
    parser.add_argument("--global-prob-threshold", type=float, default=0.8)
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=[
            "replace_all",
            "keep_global_add_local_unmatched",
            "keep_high_conf_global_add_local_unmatched",
            "keep_anchor_global_local_fill",
        ],
    )
    parser.add_argument("--checkpoint-glob", type=str, default="model_*.pt")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--skip-oracles", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _discover_checkpoint_rows(run_dir: Path, glob_pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.glob(glob_pattern)):
        stem = path.stem
        if "_" not in stem:
            continue
        try:
            epoch = int(stem.split("_")[-1])
        except ValueError:
            continue
        rows.append({"epoch": epoch, "path": path})
    rows.sort(key=lambda item: int(item["epoch"]))
    return rows


def _patch_gt_mask(gt_locs: FrameLocSet, patch) -> torch.BoolTensor:
    return (
        (gt_locs.indf == int(patch.frame_index))
        & (gt_locs.xyz[:, 0] >= float(patch.x0))
        & (gt_locs.xyz[:, 0] < float(patch.x1))
        & (gt_locs.xyz[:, 1] >= float(patch.y0))
        & (gt_locs.xyz[:, 1] < float(patch.y1))
    )


def _template_aligned_locset(source_locs: FrameLocSet, *, template_locs: FrameLocSet) -> FrameLocSet:
    loc = FrameLocSet(**template_locs.meta)
    if len(source_locs) == 0:
        return loc

    n_loc = int(len(source_locs))
    active_attr = set(template_locs.get_activeattr())
    payload = {
        "id": torch.arange(n_loc, dtype=template_locs.i_type),
        "indf": source_locs.indf.to(dtype=template_locs.i_type),
        "xyz": source_locs.xyz.to(dtype=template_locs.f_type),
        "phot": source_locs.phot.to(dtype=template_locs.f_type),
    }
    if "prob" in active_attr:
        payload["prob"] = torch.ones((n_loc,), dtype=template_locs.f_type)
    if "fracN" in active_attr:
        payload["fracN"] = torch.zeros((n_loc, template_locs.nchannels), dtype=template_locs.f_type)
    if "xyz_sig" in active_attr:
        payload["xyz_sig"] = torch.full((n_loc, template_locs.ndim), 1.0, dtype=template_locs.f_type)
    if "phot_sig" in active_attr:
        payload["phot_sig"] = torch.full((n_loc,), 1.0, dtype=template_locs.f_type)
    if "fracN_sig" in active_attr:
        payload["fracN_sig"] = torch.full((n_loc, template_locs.nchannels), 1.0, dtype=template_locs.f_type)
    loc.init_load(**payload)
    return loc


def _gt_patch_locset(gt_locs: FrameLocSet, patch, *, template_locs: FrameLocSet) -> FrameLocSet:
    return _template_aligned_locset(
        gt_locs[_patch_gt_mask(gt_locs, patch)],
        template_locs=template_locs,
    )


def _gt_proxy_for_conflicts(gt_locs: FrameLocSet) -> FrameLocSet:
    gt_proxy = gt_locs.clone()
    if len(gt_proxy) == 0:
        return gt_proxy
    phot = gt_proxy.phot.to(dtype=gt_proxy.f_type)
    gt_proxy.prob = phot / max(float(phot.max().item()), 1e-6)
    return gt_proxy


def _reanchor_patch_to_global(patch, global_locs: FrameLocSet):
    mask = (
        (global_locs.indf == int(patch.frame_index))
        & (global_locs.xyz[:, 0] >= float(patch.x0))
        & (global_locs.xyz[:, 0] < float(patch.x1))
        & (global_locs.xyz[:, 1] >= float(patch.y0))
        & (global_locs.xyz[:, 1] < float(patch.y1))
    )
    if not bool(mask.any()):
        return replace(patch, anchor_index=-1)
    candidate_ix = torch.nonzero(mask, as_tuple=False).squeeze(1)
    best_ix = candidate_ix[torch.argmax(global_locs.prob[candidate_ix])].item()
    return replace(patch, anchor_index=int(best_ix))


def _build_gt_conflict_patches(gt_locs: FrameLocSet, global_locs: FrameLocSet, *, conflict_radius_px: float, patch_size: int, min_component_size: int) -> list:
    gt_proxy = _gt_proxy_for_conflicts(gt_locs)
    patches = build_conflict_patches(
        gt_proxy,
        conflict_radius_px=float(conflict_radius_px),
        patch_size=int(patch_size),
        min_component_size=int(min_component_size),
    )
    return [_reanchor_patch_to_global(patch, global_locs) for patch in patches]


def _patch_set_stats(patches: list, gt_locs: FrameLocSet, *, max_targets: int) -> dict[str, float | int]:
    if not patches:
        return {
            "num_patches": 0,
            "covered_gt_emitters": 0,
            "slot_overflow_patches": 0,
            "max_gt_emitters_per_patch": 0,
            "mean_gt_emitters_per_patch": 0.0,
        }

    covered_mask = torch.zeros((len(gt_locs),), dtype=torch.bool)
    patch_counts: list[int] = []
    overflow = 0
    for patch in patches:
        gt_mask = _patch_gt_mask(gt_locs, patch)
        count = int(gt_mask.sum().item())
        patch_counts.append(count)
        covered_mask |= gt_mask
        if count > int(max_targets):
            overflow += 1

    return {
        "num_patches": int(len(patches)),
        "covered_gt_emitters": int(covered_mask.sum().item()),
        "slot_overflow_patches": int(overflow),
        "max_gt_emitters_per_patch": int(max(patch_counts)),
        "mean_gt_emitters_per_patch": float(sum(patch_counts) / max(len(patch_counts), 1)),
    }


def _evaluate_patch_predictions(
    *,
    label: str,
    strategy: str,
    patch_predictions: list[tuple],
    global_locs: FrameLocSet,
    gt_locs: FrameLocSet,
    matcher: GreedyHungarianMatching,
    cam_pxszs: tuple[float, float, float],
    global_metrics: dict[str, float],
    global_prob_threshold: float,
) -> dict[str, float | int | str]:
    refined_locs = merge_patch_predictions_with_strategy(
        global_locs,
        patch_predictions,
        matcher=matcher,
        strategy=strategy,
        global_prob_threshold=float(global_prob_threshold),
    )
    _matching, refined_eval = train_kernels.eval_forward(
        cam_pxszs=cam_pxszs,
        matcher=matcher,
        loc_out=refined_locs,
        loc_tar=gt_locs,
    )
    metrics = _eval_to_metrics(refined_eval)
    metrics.update(
        {
            "label": label,
            "strategy": strategy,
            "num_patch_predictions": int(len(patch_predictions)),
            "num_global_preds": int(len(global_locs)),
            "num_refined_preds": int(len(refined_locs)),
            "global_precision": float(global_metrics["precision"]),
            "global_recall": float(global_metrics["recall"]),
            "global_jaccard": float(global_metrics["jaccard"]),
            "global_rmse_lat": float(global_metrics["rmse_lat"]),
            "global_rmse_ax": float(global_metrics["rmse_ax"]),
            "delta_precision": float(metrics["precision"] - global_metrics["precision"]),
            "delta_recall": float(metrics["recall"] - global_metrics["recall"]),
            "delta_jaccard": float(metrics["jaccard"] - global_metrics["jaccard"]),
        }
    )
    return metrics


def main() -> int:
    _setup_logger()
    args = parse_args()

    strategies = list(dict.fromkeys(args.strategies))
    unsupported = [strategy for strategy in strategies if strategy not in MERGE_STRATEGIES]
    if unsupported:
        raise ValueError(f"Unsupported strategies: {unsupported}")

    cfg = load_pipeline_config(args.config)
    local_cfg = resolve_local_deblending_config(cfg, repo_root=ROOT)

    device = _resolve_device(cfg, args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    LOG.info("Using device: %s", device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_jsonl_path = output_dir / "rows.jsonl"
    if rows_jsonl_path.exists():
        rows_jsonl_path.unlink()

    base_seed = int(((cfg.get("simulation") or {}).get("simulation") or {}).get("random_seed", 0))
    cfg_train_epoch0 = _build_epoch_cfg(cfg, base_seed, epoch=0)

    LOG.info("Running phase retrieval once with mode=%s ...", args.phase_mode)
    phase_result = run_phase_retrieval(cfg, mode=args.phase_mode, logger=LOG)
    phase_payload = phase_result.patches_payload

    train_sim_epoch0 = generate_dataset(cfg_train_epoch0, profile="train", phase_payload=phase_payload, logger=LOG)
    scaling_info = normalize_dataset(
        cfg_train_epoch0,
        profile="train",
        simulation_dataset=train_sim_epoch0,
        update_dataset=False,
    )
    eval_sim = generate_dataset(cfg, profile="eval", phase_payload=phase_payload, logger=LOG)

    param = _build_param(
        cfg,
        scaling_info,
        output_dir=output_dir,
        device=device,
        epochs=1,
        batch_size=int((cfg.get("train") or {}).get("batch_size", 64)),
        window_size=3,
    )
    loc_meta = _build_loc_meta(cfg, param)
    post_processor, matcher = train_setup.setup_postprocess(loc_meta, param)

    frame_proc = FrameRescalar(scale=param.Scaling.input_scale, offset=param.Scaling.input_offset)
    bkg_proc = BkgFrameRescalar(scale=param.Scaling.bkg_max)
    tar_proc = TargetProcess(
        disable_attr=param.DataFilter.disabled_attributes,
        phot_max=param.Scaling.phot_max,
        z_max=param.Scaling.z_max,
    )

    global_model = _instantiate_global_model(param)
    global_model = model_io.LoadSaveModel(
        global_model,
        output_file=None,
        input_file=local_cfg.global_checkpoint,
    ).load_init(str(device))
    global_model = _freeze_module(global_model)

    image_size = int(train_sim_epoch0.samples[0].frames.shape[-1])
    renderer = MeanPSFCenterRenderer.from_pipeline(
        cfg=cfg,
        param=param,
        phase_payload=phase_payload,
        device=device,
        image_size=image_size,
    )

    _eval_dataset, eval_loader = _build_window_loader(
        eval_sim,
        batch_size=int((cfg.get("train") or {}).get("batch_size", 64)),
        frame_proc=frame_proc,
        bkg_proc=bkg_proc,
        tar_proc=tar_proc,
        window_size=int(param.HyperParameter.channels_in),
        device=device,
    )
    eval_examples, eval_global_locs, eval_gt_locs = _mine_patch_examples(
        loader=eval_loader,
        global_model=global_model,
        post_processor=post_processor,
        renderer=renderer,
        loc_meta=loc_meta,
        device=device,
        local_cfg=local_cfg,
        phot_max=float(param.Scaling.phot_max),
        z_max=float(param.Scaling.z_max),
        patch_limit=local_cfg.eval_patch_limit,
        seed=base_seed,
    )

    _global_matching, global_eval = train_kernels.eval_forward(
        cam_pxszs=tuple(param.Camera.cam_pxszs),
        matcher=matcher,
        loc_out=eval_global_locs,
        loc_tar=eval_gt_locs,
    )
    global_metrics = _eval_to_metrics(global_eval)
    LOG.info(
        "Baseline global metrics P=%.3f R=%.3f J=%.3f RMSE_lat=%.3f RMSE_ax=%.3f eval_patches=%d",
        global_metrics["precision"],
        global_metrics["recall"],
        global_metrics["jaccard"],
        global_metrics["rmse_lat"],
        global_metrics["rmse_ax"],
        len(eval_examples),
    )

    results: list[dict] = []
    patch_sets: list[dict] = []

    mined_patches = [example.patch for example in eval_examples]
    mined_patch_predictions_oracle = [
        (patch, _gt_patch_locset(eval_gt_locs, patch, template_locs=eval_global_locs))
        for patch in mined_patches
    ]
    patch_sets.append(
        {
            "patch_source": "mined_from_global",
            **_patch_set_stats(mined_patches, eval_gt_locs, max_targets=int(local_cfg.max_targets)),
        }
    )

    gt_conflict_patches = _build_gt_conflict_patches(
        eval_gt_locs,
        eval_global_locs,
        conflict_radius_px=float(local_cfg.conflict_radius_px),
        patch_size=int(local_cfg.patch_size),
        min_component_size=int(local_cfg.min_component_size),
    )
    gt_conflict_patch_predictions_oracle = [
        (patch, _gt_patch_locset(eval_gt_locs, patch, template_locs=eval_global_locs))
        for patch in gt_conflict_patches
    ]
    patch_sets.append(
        {
            "patch_source": "gt_conflict",
            **_patch_set_stats(gt_conflict_patches, eval_gt_locs, max_targets=int(local_cfg.max_targets)),
        }
    )
    _write_json(output_dir / "patch_sets.json", patch_sets)

    checkpoint_rows = _discover_checkpoint_rows(args.local_run_dir.resolve(), args.checkpoint_glob)
    LOG.info("Discovered %d local checkpoints in %s", len(checkpoint_rows), args.local_run_dir)

    if not args.skip_checkpoints:
        for checkpoint_row in checkpoint_rows:
            checkpoint_path = checkpoint_row["path"]
            checkpoint_epoch = int(checkpoint_row["epoch"])
            LOG.info("Evaluating local checkpoint epoch=%d path=%s", checkpoint_epoch, checkpoint_path)

            local_model = LocalSetPredictor(
                in_channels=int(param.HyperParameter.channels_in + 1),
                hidden_channels=int(local_cfg.hidden_channels),
                num_slots=int(local_cfg.num_slots),
                patch_size=int(local_cfg.patch_size),
            )
            local_model = model_io.LoadSaveModel(
                local_model,
                output_file=None,
                input_file=checkpoint_path,
            ).load_init(str(device))
            local_model.eval()

            patch_predictions = _predict_patch_refinements(
                local_model,
                eval_examples,
                batch_size=int((cfg.get("train") or {}).get("batch_size", 64)),
                device=device,
                loc_meta=loc_meta,
                phot_max=float(param.Scaling.phot_max),
                z_max=float(param.Scaling.z_max),
                prob_threshold=float(local_cfg.prob_threshold),
            )

            for strategy in strategies:
                row = _evaluate_patch_predictions(
                    label=f"checkpoint_epoch_{checkpoint_epoch}",
                    strategy=strategy,
                    patch_predictions=patch_predictions,
                    global_locs=eval_global_locs,
                    gt_locs=eval_gt_locs,
                    matcher=matcher,
                    cam_pxszs=tuple(param.Camera.cam_pxszs),
                    global_metrics=global_metrics,
                    global_prob_threshold=float(args.global_prob_threshold),
                )
                row.update(
                    {
                        "row_type": "checkpoint",
                        "checkpoint_epoch": checkpoint_epoch,
                        "checkpoint_path": str(checkpoint_path),
                    }
                )
                results.append(row)
                _append_jsonl(rows_jsonl_path, row)

    oracle_sets = [
        ("oracle_mined_patches", mined_patch_predictions_oracle),
        ("oracle_gt_conflict_patches", gt_conflict_patch_predictions_oracle),
    ]
    if not args.skip_oracles:
        for label, patch_predictions in oracle_sets:
            LOG.info("Evaluating oracle set %s with %d patches", label, len(patch_predictions))
            for strategy in strategies:
                row = _evaluate_patch_predictions(
                    label=label,
                    strategy=strategy,
                    patch_predictions=patch_predictions,
                    global_locs=eval_global_locs,
                    gt_locs=eval_gt_locs,
                    matcher=matcher,
                    cam_pxszs=tuple(param.Camera.cam_pxszs),
                    global_metrics=global_metrics,
                    global_prob_threshold=float(args.global_prob_threshold),
                )
                row.update(
                    {
                        "row_type": "oracle",
                        "checkpoint_epoch": "",
                        "checkpoint_path": "",
                    }
                )
                results.append(row)
                _append_jsonl(rows_jsonl_path, row)

    best_by_label = {}
    for row in results:
        key = (row["label"], row["strategy"])
        prev = best_by_label.get(key)
        if prev is None or float(row["jaccard"]) > float(prev["jaccard"]):
            best_by_label[key] = row

    summary = {
        "config": str(args.config.resolve()),
        "local_run_dir": str(args.local_run_dir.resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "phase_mode": args.phase_mode,
        "global_prob_threshold": float(args.global_prob_threshold),
        "baseline_global": global_metrics,
        "num_eval_patches": int(len(eval_examples)),
        "num_eval_global_preds": int(len(eval_global_locs)),
        "num_eval_gt": int(len(eval_gt_locs)),
        "patch_sets": patch_sets,
        "rows": results,
        "best_by_label_and_strategy": list(best_by_label.values()),
    }

    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "rows.csv", results)

    LOG.info("Wrote summary to %s", output_dir / "summary.json")
    LOG.info("Wrote rows to %s", output_dir / "rows.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
