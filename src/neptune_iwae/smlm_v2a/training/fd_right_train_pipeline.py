#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys

import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_pipeline_config
from simulation.api import SimulationDataset, generate_dataset, normalize_dataset
from neptune.smlm_v2a.data.data_factory import BkgFrameRescalar, FrameRescalar, TargetProcess
from neptune.smlm_v2a.data.smlm_dataset import smlm_collate
from neptune.smlm_v2a.fd_conditioning.assets import FDChannelAsset, load_fd_channel_asset
from neptune.smlm_v2a.fd_conditioning.coords import append_coord_channels, build_coord_channels
from neptune.smlm_v2a.generic.localizations import FrameLocSet
from neptune.smlm_v2a.training import train_kernels, train_setup
from neptune.smlm_v2a.training.train_pipeline import _build_loc_meta, _build_param, _resolve_device, _setup_logger


LOG = logging.getLogger("fd_right_smlm")
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train right-ROI FD-conditioned SMLM-v2a.")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline_fd_right_smlm_v2a.json"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _epoch_seed_plan(base_seed: int, epoch: int) -> dict[str, int]:
    epoch_offset = int(epoch) * 1000
    return {
        "train": int(base_seed) + epoch_offset + 100,
        "eval": int(base_seed) + epoch_offset + 200,
    }


def _resolve_patch_size(cfg: dict) -> tuple[int, int]:
    emitter = ((cfg.get("simulation") or {}).get("emitter") or {})
    extent = emitter.get("emitter_extent") or [[-0.5, 39.5], [-0.5, 39.5]]
    patch_w = int(round(float(extent[0][1]) - float(extent[0][0])))
    patch_h = int(round(float(extent[1][1]) - float(extent[1][0])))
    return patch_h, patch_w


def _field_span_in_cells(asset: FDChannelAsset, *, patch_size: tuple[int, int]) -> tuple[float, float]:
    patch_h, patch_w = (int(patch_size[0]), int(patch_size[1]))
    span_y = float(patch_h) / float(max(1, asset.cell_height_px))
    span_x = float(patch_w) / float(max(1, asset.cell_width_px))
    return span_y, span_x


def _sample_field_cell(
    asset: FDChannelAsset,
    rng: np.random.Generator,
    *,
    patch_size: tuple[int, int],
) -> tuple[int, int]:
    height, width = asset.field_shape
    span_y, span_x = _field_span_in_cells(asset, patch_size=patch_size)
    max_x = max(0, int(math.floor(width - span_x)))
    max_y = max(0, int(math.floor(height - span_y)))
    return int(rng.integers(0, max_x + 1)), int(rng.integers(0, max_y + 1))


def _right_cfg(
    cfg: dict,
    *,
    field_cell: tuple[int, int],
    profile: str,
    seed: int,
) -> dict:
    cfg_i = copy.deepcopy(cfg)
    sim_cfg = cfg_i.setdefault("simulation", {})
    sim_runtime = sim_cfg.setdefault("simulation", {})
    sim_runtime["random_seed"] = int(seed)
    emitter = sim_cfg.setdefault("emitter", {})
    patch_h, patch_w = _resolve_patch_size(cfg_i)
    asset = load_fd_channel_asset((cfg_i.get("fd_right_channel") or {})["summary_path"])
    span_y, span_x = _field_span_in_cells(asset, patch_size=(patch_h, patch_w))
    emitter["field_extent"] = [
        [float(field_cell[0]), float(field_cell[0] + span_x)],
        [float(field_cell[1]), float(field_cell[1] + span_y)],
    ]
    profiles = sim_cfg.setdefault("profiles", {})
    prof = profiles.setdefault(profile, {})
    prof["num_samples"] = 1
    prof["use_gpu"] = bool(prof.get("use_gpu", sim_cfg.get("use_gpu", True)))
    prof["gpu_id"] = int(prof.get("gpu_id", sim_cfg.get("gpu_id", 0)))
    return cfg_i


def _generate_right_dataset(
    cfg: dict,
    *,
    asset: FDChannelAsset,
    profile: str,
    num_field_samples: int,
    seed_base: int,
) -> SimulationDataset:
    patch_size = _resolve_patch_size(cfg)
    samples = []
    rng = np.random.default_rng(seed_base)
    for sample_idx in range(int(num_field_samples)):
        seed = int(seed_base + sample_idx)
        field_cell = _sample_field_cell(asset, rng, patch_size=patch_size)
        cfg_i = _right_cfg(cfg, field_cell=field_cell, profile=profile, seed=seed)
        dataset_i = generate_dataset(cfg_i, profile=profile, phase_payload=asset.build_phase_payload(), logger=LOG)
        samples.extend(dataset_i.samples)
    return SimulationDataset(
        frames_per_sample=int((((cfg.get("simulation") or {}).get("profiles") or {}).get(profile, {}) or {}).get("frames_per_sample", 256)),
        num_samples=len(samples),
        profile=profile,
        samples=samples,
    )


class RightFDDataset(Dataset):
    def __init__(
        self,
        sim_dataset: SimulationDataset,
        *,
        asset: FDChannelAsset,
        frame_proc: FrameRescalar | None,
        bkg_frame_proc: BkgFrameRescalar | None,
        tar_proc: TargetProcess | None,
        window_size: int,
        coord_size: tuple[int, int],
    ) -> None:
        self.samples = sim_dataset.samples
        self.asset = asset
        self.frame_proc = frame_proc
        self.bkg_frame_proc = bkg_frame_proc
        self.tar_proc = tar_proc
        self.window_size = int(window_size)
        self.coord_size = tuple(int(v) for v in coord_size)
        self.index: list[tuple[int, int]] = []
        for sample_idx, sample in enumerate(self.samples):
            sample.frames = self._as_numpy(sample.frames, dtype=np.float32)
            sample.param_tar = self._as_numpy(sample.param_tar, dtype=np.float32)
            sample.mask_tar = self._as_numpy(sample.mask_tar, dtype=bool)
            sample.field_tar = None if sample.field_tar is None else self._as_numpy(sample.field_tar, dtype=np.float32)
            sample.bg_tar = None if sample.bg_tar is None else self._as_numpy(sample.bg_tar, dtype=np.float32)
            frame_count = sample.frames.shape[0]
            if frame_count < self.window_size:
                continue
            for start in range(frame_count - self.window_size + 1):
                self.index.append((sample_idx, start))
        if not self.index:
            raise ValueError("No frame windows available for right FD dataset.")

    @staticmethod
    def _as_numpy(value, *, dtype):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _build_detect_map(param: np.ndarray, mask: np.ndarray, height: int, width: int) -> np.ndarray:
        detect = np.zeros((height, width), dtype=np.float32)
        if mask is None or not np.any(mask):
            return detect
        coords = param[mask]
        if coords.size == 0:
            return detect
        col = np.clip(np.floor(coords[:, 1] + 0.5).astype(np.int64), 0, width - 1)
        row = np.clip(np.floor(coords[:, 2] + 0.5).astype(np.int64), 0, height - 1)
        detect[row, col] = 1.0
        return detect

    def _field_center(self, sample_idx: int, target_idx: int) -> torch.Tensor:
        sample = self.samples[sample_idx]
        if sample.field_tar is None:
            height, width = self.asset.field_shape
            return torch.tensor([(width - 1) / 2.0, (height - 1) / 2.0], dtype=torch.float32)
        field_tar = sample.field_tar[target_idx]
        mask = sample.mask_tar[target_idx]
        if np.any(mask):
            center = np.asarray(field_tar[mask], dtype=np.float32).mean(axis=0)
        else:
            center = np.asarray(field_tar[0], dtype=np.float32)
        return torch.from_numpy(center.astype(np.float32, copy=False))

    def __getitem__(self, idx: int):
        sample_idx, start = self.index[idx]
        sample = self.samples[sample_idx]
        frames = sample.frames[start : start + self.window_size]
        target_idx = start + self.window_size // 2
        param = sample.param_tar[target_idx]
        mask = sample.mask_tar[target_idx]
        bkg = sample.bg_tar[target_idx] if sample.bg_tar is not None else np.zeros_like(frames[0], dtype=np.float32)

        height, width = frames.shape[-2:]
        detect = self._build_detect_map(param, mask, height, width)

        frames_t = torch.from_numpy(np.ascontiguousarray(frames))
        bkg_t = torch.from_numpy(np.ascontiguousarray(bkg))
        param_t = torch.from_numpy(np.ascontiguousarray(param))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))
        detect_t = torch.from_numpy(np.ascontiguousarray(detect))

        if self.frame_proc is not None:
            frames_t = self.frame_proc.forward(frames_t)
        if self.bkg_frame_proc is not None:
            bkg_t = self.bkg_frame_proc.forward(bkg_t)
        if self.tar_proc is not None:
            param_t = self.tar_proc.forward(param_t)

        field_xy = self._field_center(sample_idx, target_idx)
        coords = build_coord_channels(self.asset, field_xy=field_xy, out_size=self.coord_size)
        frames_t = append_coord_channels(frames_t, coords)
        return frames_t, detect_t, bkg_t, param_t, mask_t, field_xy

    def build_locset(self, loc_meta: FrameLocSet) -> FrameLocSet:
        ids, indf, xyz, phot = [], [], [], []
        next_id = 0
        for dataset_idx, (sample_idx, start) in enumerate(self.index):
            sample = self.samples[sample_idx]
            target_idx = start + self.window_size // 2
            param = sample.param_tar[target_idx]
            mask = sample.mask_tar[target_idx]
            if not np.any(mask):
                continue
            valid = param[mask]
            n_loc = valid.shape[0]
            ids.append(np.arange(next_id, next_id + n_loc, dtype=np.int32))
            indf.append(np.full((n_loc,), dataset_idx, dtype=np.int32))
            xyz.append(valid[:, 1:4].astype(np.float32, copy=False))
            phot.append(valid[:, 0].astype(np.float32, copy=False))
            next_id += n_loc
        if not ids:
            return FrameLocSet(**loc_meta.meta)
        loc = FrameLocSet(**loc_meta.meta)
        loc.init_load(
            id=torch.from_numpy(np.concatenate(ids)).to(dtype=loc_meta.i_type),
            indf=torch.from_numpy(np.concatenate(indf)).to(dtype=loc_meta.i_type),
            xyz=torch.from_numpy(np.concatenate(xyz)).to(dtype=loc_meta.f_type),
            phot=torch.from_numpy(np.concatenate(phot)).to(dtype=loc_meta.f_type),
        )
        return loc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_metric(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> int:
    _setup_logger()
    args = _parse_args()
    cfg = load_pipeline_config(args.config)
    train_cfg = cfg.get("train") or {}
    fd_cfg = cfg.get("fd_right_channel") or {}
    if not fd_cfg.get("enabled", False):
        raise ValueError("fd_right_channel.enabled must be true for fd_right_train_pipeline.")

    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 300))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32))
    device = _resolve_device(cfg, args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    asset = load_fd_channel_asset(fd_cfg["summary_path"])
    condition_channels = 2
    window_size = int(fd_cfg.get("window_size", 3))
    coord_size = tuple(int(v) for v in fd_cfg.get("coord_size", [40, 40]))
    base_seed = int(((cfg.get("simulation") or {}).get("simulation") or {}).get("random_seed", 0))
    regen_train = bool(train_cfg.get("regen_train_dataset_each_epoch", False))

    epoch0_seeds = _epoch_seed_plan(base_seed, epoch=0)
    train_sim = _generate_right_dataset(
        cfg,
        asset=asset,
        profile="train",
        num_field_samples=int(fd_cfg.get("train_field_samples", 2)),
        seed_base=epoch0_seeds["train"],
    )
    scaling_info = normalize_dataset(cfg, profile="train", simulation_dataset=train_sim, update_dataset=True)

    eval_sim = None
    if int(fd_cfg.get("eval_field_samples", 1)) > 0:
        eval_sim = _generate_right_dataset(
            cfg,
            asset=asset,
            profile="eval",
            num_field_samples=int(fd_cfg.get("eval_field_samples", 1)),
            seed_base=epoch0_seeds["eval"],
        )
        normalize_dataset(cfg, profile="train", simulation_dataset=eval_sim, update_dataset=True)

    param = _build_param(
        cfg,
        scaling_info,
        output_dir=output_dir,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        window_size=window_size + condition_channels,
    )
    overrides = cfg.get("smlm_overrides") or {}
    opt_param = param.HyperParameter.opt_param
    if overrides.get("optimizer_lr") is not None:
        opt_param.lr = float(overrides["optimizer_lr"])
    if overrides.get("optimizer_weight_decay") is not None:
        opt_param.weight_decay = float(overrides["optimizer_weight_decay"])
    loc_meta = _build_loc_meta(cfg, param)

    frame_proc = FrameRescalar(scale=param.Scaling.input_scale, offset=param.Scaling.input_offset)
    bkg_proc = BkgFrameRescalar(scale=param.Scaling.bkg_max)
    tar_proc = TargetProcess(
        disable_attr=param.DataFilter.disabled_attributes,
        phot_max=param.Scaling.phot_max,
        z_max=param.Scaling.z_max,
    )
    train_dataset = RightFDDataset(
        train_sim,
        asset=asset,
        frame_proc=frame_proc,
        bkg_frame_proc=bkg_proc,
        tar_proc=tar_proc,
        window_size=window_size,
        coord_size=coord_size,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=smlm_collate,
    )

    eval_loader = None
    eval_loc_tar = None
    if eval_sim is not None:
        eval_dataset = RightFDDataset(
            eval_sim,
            asset=asset,
            frame_proc=frame_proc,
            bkg_frame_proc=bkg_proc,
            tar_proc=tar_proc,
            window_size=window_size,
            coord_size=coord_size,
        )
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=smlm_collate,
        )
        eval_loc_tar = eval_dataset.build_locset(loc_meta)

    model, model_ls, optimizer, lr_scheduler = train_setup.setup_fitter(param, output_dir / "model.pt", device)
    criterion = train_setup.setup_criterion(loc_meta, param)
    post_processor, matcher = train_setup.setup_postprocess(loc_meta, param)

    runtime_state = {
        "window_size": window_size,
        "conditioning_channels": condition_channels,
        "coord_size": list(coord_size),
        "input_offset": float(param.Scaling.input_offset),
        "input_scale": float(param.Scaling.input_scale),
        "phot_max": float(param.Scaling.phot_max),
        "z_max": float(param.Scaling.z_max),
        "bg_max": float(param.Scaling.bkg_max),
        "camera_pixel_nm_x": float((cfg.get("optical") or {}).get("pixel_size_nm_x", 108.0)),
        "camera_pixel_nm_y": float((cfg.get("optical") or {}).get("pixel_size_nm_y", 108.0)),
        "crop_left": int(asset.crop_left),
        "crop_top": int(asset.crop_top),
        "crop_width": int(asset.crop_width),
        "crop_height": int(asset.crop_height),
        "asset": {
            "summary_path": str(asset.summary_path),
            "field_shape": list(asset.field_shape),
            "block_span_frames": int(asset.block_span_frames),
        },
    }
    _write_json(output_dir / "runtime_state.json", runtime_state)
    _write_json(output_dir / "resolved_config.json", cfg)

    metrics_path = output_dir / "metrics.jsonl"
    best_jaccard = float("-inf")
    best_epoch = -1
    for epoch in range(epochs):
        if regen_train and epoch > 0:
            train_sim = _generate_right_dataset(
                cfg,
                asset=asset,
                profile="train",
                num_field_samples=int(fd_cfg.get("train_field_samples", 2)),
                seed_base=_epoch_seed_plan(base_seed, epoch=epoch)["train"],
            )
            normalize_dataset(cfg, profile="train", simulation_dataset=train_sim, update_dataset=True)
            train_dataset = RightFDDataset(
                train_sim,
                asset=asset,
                frame_proc=frame_proc,
                bkg_frame_proc=bkg_proc,
                tar_proc=tar_proc,
                window_size=window_size,
                coord_size=coord_size,
            )
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                collate_fn=smlm_collate,
            )
        loss_cmp_train = train_kernels.train_forward(
            model=model,
            optimizer=optimizer,
            loss=criterion,
            dataloader=train_loader,
            grad_mod=param.HyperParameter.grad_mod,
            epoch=epoch,
            device=str(device),
        )
        train_loss = float(loss_cmp_train.sum(1).mean().item())
        eval_loss = None
        eval_metrics: dict[str, float] = {}
        if eval_loader is not None and eval_loc_tar is not None:
            loss_cmp_test, test_data = train_kernels.test_forward(
                model=model,
                loss=criterion,
                dataloader=eval_loader,
                epoch=epoch,
                device=str(device),
            )
            loc_out = post_processor.forward(test_data.y_out)
            _, eval_sets = train_kernels.eval_forward(
                cam_pxszs=param.Camera.cam_pxszs,
                matcher=matcher,
                loc_out=loc_out,
                loc_tar=eval_loc_tar,
            )
            eval_dict = eval_sets._asdict()
            eval_loss = float(loss_cmp_test.sum(1).mean().item())
            eval_metrics = {
                "precision": float(eval_dict.get("prec", 0.0)),
                "recall": float(eval_dict.get("rec", 0.0)),
                "jaccard": float(eval_dict.get("jac", 0.0)),
                "rmse_lat": float(eval_dict.get("rmse_lat", 0.0)),
                "rmse_ax": float(eval_dict.get("rmse_ax", 0.0)),
            }
            if eval_metrics["jaccard"] > best_jaccard:
                best_jaccard = eval_metrics["jaccard"]
                best_epoch = epoch
                torch.save(model.state_dict(), output_dir / "checkpoint_best.pt")
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(eval_loss if eval_loss is not None else train_loss)
        else:
            lr_scheduler.step()
        torch.save(model.state_dict(), output_dir / "checkpoint_latest.pt")
        model_ls.save(model, metric_val=None)
        _write_metric(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **eval_metrics,
            },
        )
        LOG.info("epoch=%d train_loss=%.6f %s", epoch, train_loss, eval_metrics)

    torch.save(model.state_dict(), output_dir / "checkpoint_final.pt")
    _write_json(
        output_dir / "summary.json",
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "best_jaccard": best_jaccard,
            "runtime_state": runtime_state,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
