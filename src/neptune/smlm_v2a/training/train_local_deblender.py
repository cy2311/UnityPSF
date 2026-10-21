#!/usr/bin/env python3
"""Train anchor-conditioned local deblender on conflict patches mined from a frozen global SMLM model."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_pipeline_config
from phase_retrieval.api import run_phase_retrieval
from simulation.api import generate_dataset, normalize_dataset
from neptune.smlm_v2a.data.data_factory import BkgFrameRescalar, FrameRescalar, TargetProcess
from neptune.smlm_v2a.data.smlm_dataset import smlm_collate
from neptune.smlm_v2a.evaluation.matcher import GreedyHungarianMatching
from neptune.smlm_v2a.generic.localizations import FrameLocSet
from neptune.smlm_v2a.local_deblending import (
    LocalSetGMMLoss,
    LocalSetPredictor,
    PatchExample,
    PatchExampleDataset,
    build_anchor_render,
    build_conflict_patches,
    build_patch_examples,
    decode_patch_predictions,
    merge_patch_predictions,
    targets_to_locset,
)
from neptune.smlm_v2a.models.smlm_model import DoubleUNet, SingleUNet
from neptune.smlm_v2a.residual_refinement import MeanPSFCenterRenderer
from neptune.smlm_v2a.training import train_kernels, train_setup
from neptune.smlm_v2a.training.train_pipeline import (
    SMLMPipelineDataset,
    _build_epoch_cfg,
    _build_loc_meta,
    _build_param,
    _resolve_device,
    _setup_logger,
    _write_metrics,
    _write_seed_plan,
)
from neptune.smlm_v2a.utils.io import model_io


LOG = logging.getLogger("train_local_deblender")


@dataclass(frozen=True)
class LocalDeblendingConfig:
    global_checkpoint: Path
    conflict_radius_px: float = 3.0
    patch_size: int = 25
    min_component_size: int = 2
    max_targets: int = 4
    hidden_channels: int = 32
    num_slots: int = 4
    prob_threshold: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    train_patch_limit: int | None = None
    eval_patch_limit: int | None = None


def resolve_local_deblending_config(cfg: dict, *, repo_root: Path) -> LocalDeblendingConfig:
    local_cfg = (cfg.get("local_deblending") or cfg.get("anchor_local_deblending") or {}).copy()
    checkpoint_raw = local_cfg.get("global_checkpoint")
    if not checkpoint_raw:
        raise ValueError("local_deblending.global_checkpoint is required.")
    checkpoint = Path(checkpoint_raw)
    if not checkpoint.is_absolute():
        checkpoint = (repo_root / checkpoint).resolve()
    return LocalDeblendingConfig(
        global_checkpoint=checkpoint,
        conflict_radius_px=float(local_cfg.get("conflict_radius_px", 3.0)),
        patch_size=int(local_cfg.get("patch_size", 25)),
        min_component_size=int(local_cfg.get("min_component_size", 2)),
        max_targets=int(local_cfg.get("max_targets", 4)),
        hidden_channels=int(local_cfg.get("hidden_channels", 32)),
        num_slots=int(local_cfg.get("num_slots", 4)),
        prob_threshold=float(local_cfg.get("prob_threshold", 0.5)),
        learning_rate=float(local_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(local_cfg.get("weight_decay", 0.0)),
        train_patch_limit=(
            None if local_cfg.get("train_patch_limit") in (None, "") else int(local_cfg.get("train_patch_limit"))
        ),
        eval_patch_limit=(
            None if local_cfg.get("eval_patch_limit") in (None, "") else int(local_cfg.get("eval_patch_limit"))
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anchor-conditioned local deblender")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline_smlm_uni.json"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _instantiate_global_model(param) -> SingleUNet | DoubleUNet:
    models_available = {
        "SingleUNet": SingleUNet,
        "DoubleUNet": DoubleUNet,
    }
    architecture = str(param.HyperParameter.architecture)
    if architecture not in models_available:
        raise ValueError(f"Unsupported global architecture: {architecture!r}")
    return models_available[architecture].parse(param)


def _freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module


def _empty_locset(loc_meta: FrameLocSet) -> FrameLocSet:
    return FrameLocSet(**loc_meta.meta)


def _cat_locsets(parts: list[FrameLocSet], loc_meta: FrameLocSet) -> FrameLocSet:
    non_empty = [part for part in parts if len(part) > 0]
    if not non_empty:
        return _empty_locset(loc_meta)
    return FrameLocSet.loc_cat(non_empty)


def _subsample_examples(
    examples: list[PatchExample],
    *,
    limit: int | None,
    seed: int,
) -> list[PatchExample]:
    if limit is None or len(examples) <= int(limit):
        return examples
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(examples), generator=generator)[: int(limit)].tolist()
    return [examples[i] for i in indices]


def _build_window_loader(
    sim_dataset,
    *,
    batch_size: int,
    frame_proc: FrameRescalar,
    bkg_proc: BkgFrameRescalar,
    tar_proc: TargetProcess,
    window_size: int,
    device: torch.device,
) -> tuple[SMLMPipelineDataset, DataLoader]:
    dataset = SMLMPipelineDataset(
        sim_dataset,
        window_size=window_size,
        frame_proc=frame_proc,
        bkg_frame_proc=bkg_proc,
        tar_proc=tar_proc,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=smlm_collate,
    )
    return dataset, loader


def _mine_patch_examples(
    *,
    loader: DataLoader,
    global_model: torch.nn.Module,
    post_processor,
    renderer: MeanPSFCenterRenderer,
    loc_meta: FrameLocSet,
    device: torch.device,
    local_cfg: LocalDeblendingConfig,
    phot_max: float,
    z_max: float,
    patch_limit: int | None,
    seed: int,
) -> tuple[list[PatchExample], FrameLocSet, FrameLocSet]:
    examples: list[PatchExample] = []
    global_parts: list[FrameLocSet] = []
    gt_parts: list[FrameLocSet] = []
    frame_offset = 0

    global_model.eval()
    with torch.no_grad():
        for x, _detect_tar, _bkg_tar, pxyz_tar, mask_tar, _frame_loc in loader:
            batch_size, _channels, height, width = x.shape
            x_device = x.to(device)
            y_out = global_model(x_device)
            proposals_local = post_processor.forward(y_out.detach())
            patches_local = build_conflict_patches(
                proposals_local,
                conflict_radius_px=float(local_cfg.conflict_radius_px),
                patch_size=int(local_cfg.patch_size),
                min_component_size=int(local_cfg.min_component_size),
            )
            gt_local = targets_to_locset(
                pxyz_tar,
                mask_tar,
                loc_meta=loc_meta,
                phot_max=float(phot_max),
                z_max=float(z_max),
            )
            anchor_render = build_anchor_render(
                proposals_local,
                patches_local,
                renderer=renderer,
                batch_size=int(batch_size),
                height=int(height),
                width=int(width),
                device=device,
            ).cpu()
            examples.extend(
                build_patch_examples(
                    x.cpu(),
                    anchor_render,
                    gt_locs=gt_local,
                    patches=patches_local,
                    max_targets=int(local_cfg.max_targets),
                    phot_max=float(phot_max),
                    z_max=float(z_max),
                    frame_offset=int(frame_offset),
                )
            )
            global_parts.append(proposals_local.realign_frame(frame_offset))
            gt_parts.append(gt_local.realign_frame(frame_offset))
            frame_offset += int(batch_size)

    examples = _subsample_examples(examples, limit=patch_limit, seed=seed)
    return examples, _cat_locsets(global_parts, loc_meta), _cat_locsets(gt_parts, loc_meta)


def _patch_loader(
    examples: list[PatchExample],
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader | None:
    if not examples:
        return None
    return DataLoader(
        dataset=PatchExampleDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def _train_local_epoch(
    model: LocalSetPredictor,
    optimizer: torch.optim.Optimizer,
    criterion: LocalSetGMMLoss,
    loader: DataLoader | None,
    *,
    epoch: int,
    device: torch.device,
) -> float:
    if loader is None:
        raise ValueError("No local patch examples available for training.")
    model.train()
    losses: list[float] = []
    tqdm_enum = tqdm(loader, total=len(loader), smoothing=0.0)
    for x, target, mask in tqdm_enum:
        x = x.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.bool)

        optimizer.zero_grad()
        output = model(x)
        loss_vec = criterion(output, target, mask)
        loss = loss_vec.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        losses.append(float(loss.item()))
        tqdm_enum.set_description(f"(TrainLocal) E: {epoch} - L: {sum(losses) / max(len(losses), 1):.3f}")
    return float(sum(losses) / max(len(losses), 1))


def _eval_to_metrics(eval_set) -> dict[str, float]:
    eval_dict = eval_set._asdict()
    return {
        "precision": float(eval_dict.get("prec", 0.0)),
        "recall": float(eval_dict.get("rec", 0.0)),
        "jaccard": float(eval_dict.get("jac", 0.0)),
        "rmse_lat": float(eval_dict.get("rmse_lat", 0.0)),
        "rmse_ax": float(eval_dict.get("rmse_ax", 0.0)),
    }


def _predict_patch_refinements(
    model: LocalSetPredictor,
    examples: list[PatchExample],
    *,
    batch_size: int,
    device: torch.device,
    loc_meta: FrameLocSet,
    phot_max: float,
    z_max: float,
    prob_threshold: float,
) -> list[tuple]:
    if not examples:
        return []

    patch_predictions: list[tuple] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            x = torch.stack([example.x for example in chunk], dim=0).to(device=device, dtype=torch.float32)
            output = model(x)
            decoded = decode_patch_predictions(
                output,
                patches=[example.patch for example in chunk],
                loc_meta=loc_meta,
                phot_max=float(phot_max),
                z_max=float(z_max),
                prob_threshold=float(prob_threshold),
            )
            patch_predictions.extend(zip([example.patch for example in chunk], decoded))
    return patch_predictions


def _evaluate_refined_predictions(
    *,
    model: LocalSetPredictor,
    examples: list[PatchExample],
    global_locs: FrameLocSet,
    gt_locs: FrameLocSet,
    matcher: GreedyHungarianMatching,
    cam_pxszs: tuple[float, float, float],
    loc_meta: FrameLocSet,
    phot_max: float,
    z_max: float,
    prob_threshold: float,
    batch_size: int,
    device: torch.device,
    global_metrics: dict[str, float],
) -> tuple[dict[str, float], FrameLocSet]:
    patch_predictions = _predict_patch_refinements(
        model,
        examples,
        batch_size=batch_size,
        device=device,
        loc_meta=loc_meta,
        phot_max=float(phot_max),
        z_max=float(z_max),
        prob_threshold=float(prob_threshold),
    )
    refined_locs = merge_patch_predictions(global_locs, patch_predictions, matcher=matcher)
    _matching, refined_eval = train_kernels.eval_forward(
        cam_pxszs=cam_pxszs,
        matcher=matcher,
        loc_out=refined_locs,
        loc_tar=gt_locs,
    )
    metrics = _eval_to_metrics(refined_eval)
    metrics.update(
        {
            "global_precision": float(global_metrics["precision"]),
            "global_recall": float(global_metrics["recall"]),
            "global_jaccard": float(global_metrics["jaccard"]),
            "global_rmse_lat": float(global_metrics["rmse_lat"]),
            "global_rmse_ax": float(global_metrics["rmse_ax"]),
            "num_eval_patches": int(len(examples)),
            "num_global_preds": int(len(global_locs)),
            "num_refined_preds": int(len(refined_locs)),
        }
    )
    return metrics, refined_locs


def main() -> int:
    _setup_logger()
    args = parse_args()

    cfg = load_pipeline_config(args.config)
    local_cfg = resolve_local_deblending_config(cfg, repo_root=ROOT)
    if not local_cfg.global_checkpoint.is_file():
        raise FileNotFoundError(f"Global checkpoint not found: {local_cfg.global_checkpoint}")
    LOG.info("Frozen global checkpoint: %s", local_cfg.global_checkpoint)

    train_cfg = cfg.get("train") or {}
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 100))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64))
    regen_train = bool(train_cfg.get("regen_train_dataset_each_epoch", True))

    device = _resolve_device(cfg, args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    device_str = str(device)
    LOG.info("Using device: %s", device_str)

    output_dir = args.output_dir if args.output_dir is not None else ROOT / "output" / "smlm_local_deblender"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(((cfg.get("simulation") or {}).get("simulation") or {}).get("random_seed", 0)))

    LOG.info("Running phase retrieval once...")
    phase_result = run_phase_retrieval(cfg, mode="auto", logger=LOG)
    phase_payload = phase_result.patches_payload

    base_seed = int(((cfg.get("simulation") or {}).get("simulation") or {}).get("random_seed", 0))
    _write_seed_plan(cfg, base_seed, repo_root=ROOT)

    cfg_train_epoch0 = _build_epoch_cfg(cfg, base_seed, epoch=0)
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
        epochs=epochs,
        batch_size=batch_size,
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
    ).load_init(device_str)
    global_model = _freeze_module(global_model)

    image_size = int(train_sim_epoch0.samples[0].frames.shape[-1])
    renderer = MeanPSFCenterRenderer.from_pipeline(
        cfg=cfg,
        param=param,
        phase_payload=phase_payload,
        device=device,
        image_size=image_size,
    )

    local_model = LocalSetPredictor(
        in_channels=int(param.HyperParameter.channels_in + 1),
        hidden_channels=int(local_cfg.hidden_channels),
        num_slots=int(local_cfg.num_slots),
        patch_size=int(local_cfg.patch_size),
    ).to(device)
    model_saver = model_io.LoadSaveModel(local_model, output_file=output_dir / "model.pt")
    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=float(local_cfg.learning_rate),
        weight_decay=float(local_cfg.weight_decay),
    )
    criterion = LocalSetGMMLoss()

    _eval_dataset, eval_loader = _build_window_loader(
        eval_sim,
        batch_size=batch_size,
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
        cam_pxszs=param.Camera.cam_pxszs,
        matcher=matcher,
        loc_out=eval_global_locs,
        loc_tar=eval_gt_locs,
    )
    global_metrics = _eval_to_metrics(global_eval)
    LOG.info(
        "Frozen global metrics P=%.3f R=%.3f J=%.3f RMSE_lat=%.3f RMSE_ax=%.3f eval_patches=%d",
        global_metrics["precision"],
        global_metrics["recall"],
        global_metrics["jaccard"],
        global_metrics["rmse_lat"],
        global_metrics["rmse_ax"],
        len(eval_examples),
    )

    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(epochs):
        if epoch == 0 or regen_train:
            cfg_train = _build_epoch_cfg(cfg, base_seed, epoch=epoch)
            train_sim = train_sim_epoch0 if epoch == 0 else generate_dataset(
                cfg_train,
                profile="train",
                phase_payload=phase_payload,
                logger=LOG,
            )
            _train_dataset, train_loader = _build_window_loader(
                train_sim,
                batch_size=batch_size,
                frame_proc=frame_proc,
                bkg_proc=bkg_proc,
                tar_proc=tar_proc,
                window_size=int(param.HyperParameter.channels_in),
                device=device,
            )
            train_examples, _train_global_locs, _train_gt_locs = _mine_patch_examples(
                loader=train_loader,
                global_model=global_model,
                post_processor=post_processor,
                renderer=renderer,
                loc_meta=loc_meta,
                device=device,
                local_cfg=local_cfg,
                phot_max=float(param.Scaling.phot_max),
                z_max=float(param.Scaling.z_max),
                patch_limit=local_cfg.train_patch_limit,
                seed=base_seed + epoch,
            )
            if not train_examples:
                raise RuntimeError("No conflict patches were mined for training.")

        train_patch_loader = _patch_loader(
            train_examples,
            batch_size=batch_size,
            shuffle=True,
            device=device,
        )
        train_loss = _train_local_epoch(
            local_model,
            optimizer,
            criterion,
            train_patch_loader,
            epoch=epoch,
            device=device,
        )
        eval_metrics, _refined_locs = _evaluate_refined_predictions(
            model=local_model,
            examples=eval_examples,
            global_locs=eval_global_locs,
            gt_locs=eval_gt_locs,
            matcher=matcher,
            cam_pxszs=tuple(param.Camera.cam_pxszs),
            loc_meta=loc_meta,
            phot_max=float(param.Scaling.phot_max),
            z_max=float(param.Scaling.z_max),
            prob_threshold=float(local_cfg.prob_threshold),
            batch_size=batch_size,
            device=device,
            global_metrics=global_metrics,
        )

        LOG.info(
            "epoch=%d local_loss=%.6f global_J=%.3f refined_J=%.3f refined_P=%.3f refined_R=%.3f train_patches=%d eval_patches=%d",
            epoch,
            train_loss,
            global_metrics["jaccard"],
            eval_metrics["jaccard"],
            eval_metrics["precision"],
            eval_metrics["recall"],
            len(train_examples),
            len(eval_examples),
        )
        model_saver.save(local_model, metric_val=None)
        _write_metrics(
            metrics_path,
            {
                "epoch": int(epoch),
                "loss": float(train_loss),
                "num_train_patches": int(len(train_examples)),
                **eval_metrics,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
