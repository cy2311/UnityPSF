#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_pipeline_config
from simulation.api import SimulationDataset, generate_dataset, normalize_dataset
from neptune.smlm_v2a.data.data_factory import BkgFrameRescalar, FrameRescalar, TargetProcess
from neptune.smlm_v2a.data.smlm_dataset import smlm_collate
from neptune.smlm_v2a.fd_conditioning.film_patch import patch_model_for_fd_film_conditioning
from neptune.smlm_v2a.fd_conditioning.fullres import FullResZernikeConditioning
from neptune.smlm_v2a.generic.localizations import FrameLocSet
from neptune.smlm_v2a.models.smlm_model import DoubleUNet, SingleUNet
from neptune.smlm_v2a.training import train_kernels, train_setup
from neptune.smlm_v2a.training.train_pipeline import _build_loc_meta, _build_param, _resolve_device, _setup_logger
from neptune.smlm_v2a.utils.io import model_io


LOG = logging.getLogger("fd_left_fullres_smlm")


@dataclass(frozen=True)
class FDTwoStageSchedule:
    warmup_epochs: int = 30
    warmup_lr: float = 3e-5
    finetune_lr: float = 1e-5
    condition_hidden_dim: int = 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train left-half full-res zernike conditioned SMLM-v2a.")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline_fd_left_fullres_smlm_v2a.json"))
    parser.add_argument("--checkpoint-init", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _resolve_patch_size(cfg: dict) -> tuple[int, int]:
    emitter = ((cfg.get("simulation") or {}).get("emitter") or {})
    extent = emitter.get("emitter_extent") or [[-0.5, 39.5], [-0.5, 39.5]]
    patch_w = int(round(float(extent[0][1]) - float(extent[0][0])))
    patch_h = int(round(float(extent[1][1]) - float(extent[1][0])))
    return patch_h, patch_w


def _epoch_seed(base_seed: int, epoch: int, profile: str) -> int:
    return int(base_seed) + int(epoch) * 1000 + (100 if profile == "train" else 200)


def _sample_patch_origin(
    provider: FullResZernikeConditioning,
    rng: np.random.Generator,
    *,
    patch_size: tuple[int, int],
    fixed_origin_xy: tuple[int, int] | None = None,
) -> tuple[int, int]:
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    height, width = provider.image_shape_hw
    max_x = max(0, width - patch_w)
    max_y = max(0, height - patch_h)
    if fixed_origin_xy is not None:
        x0, y0 = int(fixed_origin_xy[0]), int(fixed_origin_xy[1])
        if x0 < 0 or y0 < 0 or x0 > max_x or y0 > max_y:
            raise ValueError(
                f"fixed_origin_xy={(x0, y0)!r} exceeds map size {provider.image_shape_hw} "
                f"for patch_size={patch_size!r}"
            )
        return x0, y0
    return int(rng.integers(0, max_x + 1)), int(rng.integers(0, max_y + 1))


def _left_cfg(cfg: dict, *, origin_xy: tuple[int, int], profile: str, seed: int) -> dict:
    cfg_i = copy.deepcopy(cfg)
    sim_cfg = cfg_i.setdefault("simulation", {})
    sim_runtime = sim_cfg.setdefault("simulation", {})
    sim_runtime["random_seed"] = int(seed)
    patch_h, patch_w = _resolve_patch_size(cfg_i)
    emitter = sim_cfg.setdefault("emitter", {})
    x0, y0 = int(origin_xy[0]), int(origin_xy[1])
    emitter["field_extent"] = [[float(x0), float(x0 + patch_w)], [float(y0), float(y0 + patch_h)]]
    profiles = sim_cfg.setdefault("profiles", {})
    prof = profiles.setdefault(profile, {})
    prof["num_samples"] = 1
    prof["use_gpu"] = bool(prof.get("use_gpu", sim_cfg.get("use_gpu", True)))
    prof["gpu_id"] = int(prof.get("gpu_id", sim_cfg.get("gpu_id", 0)))
    return cfg_i


def _generate_left_dataset(
    cfg: dict,
    *,
    provider: FullResZernikeConditioning,
    profile: str,
    num_field_samples: int,
    seed_base: int,
) -> tuple[SimulationDataset, list[tuple[int, int]]]:
    patch_size = _resolve_patch_size(cfg)
    rng = np.random.default_rng(seed_base)
    samples = []
    origins: list[tuple[int, int]] = []
    optical_cfg = cfg.get("optical") or {}
    zernike_cfg = (((cfg.get("phase_retrieval") or {}).get("simulator_diff") or {}).get("zernike") or {})
    phase_payload = provider.build_phase_payload(
        wavelength_nm=float(optical_cfg.get("wavelength_nm", 660.0)),
        noll_start=int(zernike_cfg.get("noll_start", 4)),
        noll_end=int(zernike_cfg.get("noll_end", 12)),
    )
    fd_cfg = cfg.get("fd_left_fullres_channel") or {}
    raw_fixed_origin = fd_cfg.get("fixed_origin_xy")
    fixed_origin_xy = None
    if raw_fixed_origin is not None:
        if not isinstance(raw_fixed_origin, (list, tuple)) or len(raw_fixed_origin) != 2:
            raise ValueError("fd_left_fullres_channel.fixed_origin_xy must be [x0, y0].")
        fixed_origin_xy = (int(raw_fixed_origin[0]), int(raw_fixed_origin[1]))
    for sample_idx in range(int(num_field_samples)):
        origin_xy = _sample_patch_origin(
            provider,
            rng,
            patch_size=patch_size,
            fixed_origin_xy=fixed_origin_xy,
        )
        cfg_i = _left_cfg(cfg, origin_xy=origin_xy, profile=profile, seed=int(seed_base + sample_idx))
        dataset_i = generate_dataset(cfg_i, profile=profile, phase_payload=phase_payload, logger=LOG)
        samples.extend(dataset_i.samples)
        origins.extend([origin_xy] * len(dataset_i.samples))
    combined = SimulationDataset(
        frames_per_sample=int((((cfg.get("simulation") or {}).get("profiles") or {}).get(profile, {}) or {}).get("frames_per_sample", 256)),
        num_samples=len(samples),
        profile=profile,
        samples=samples,
    )
    return combined, origins


class FullResConditionedDataset(Dataset):
    def __init__(
        self,
        sim_dataset: SimulationDataset,
        *,
        origins_xy: list[tuple[int, int]],
        provider: FullResZernikeConditioning,
        frame_proc: FrameRescalar | None,
        bkg_frame_proc: BkgFrameRescalar | None,
        tar_proc: TargetProcess | None,
        window_size: int,
    ) -> None:
        self.samples = sim_dataset.samples
        self.origins_xy = origins_xy
        self.provider = provider
        self.frame_proc = frame_proc
        self.bkg_frame_proc = bkg_frame_proc
        self.tar_proc = tar_proc
        self.window_size = int(window_size)
        self.index: list[tuple[int, int]] = []
        for sample_idx, sample in enumerate(self.samples):
            sample.frames = self._as_numpy(sample.frames, dtype=np.float32)
            sample.param_tar = self._as_numpy(sample.param_tar, dtype=np.float32)
            sample.mask_tar = self._as_numpy(sample.mask_tar, dtype=bool)
            sample.bg_tar = None if sample.bg_tar is None else self._as_numpy(sample.bg_tar, dtype=np.float32)
            frame_count = sample.frames.shape[0]
            if frame_count < self.window_size:
                continue
            for start in range(frame_count - self.window_size + 1):
                self.index.append((sample_idx, start))
        if not self.index:
            raise ValueError("No frame windows available for full-res conditioned dataset.")

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
        col = np.clip(np.floor(coords[:, 1] + 0.5).astype(np.int64), 0, width - 1)
        row = np.clip(np.floor(coords[:, 2] + 0.5).astype(np.int64), 0, height - 1)
        detect[row, col] = 1.0
        return detect

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
        x0, y0 = self.origins_xy[sample_idx]
        cond_vec = self.provider.condition_vector_from_xy(x0=x0, y0=y0, height=height, width=width)
        return [frames_t, cond_vec], detect_t, bkg_t, param_t, mask_t, torch.tensor([x0, y0])

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


def _resolve_two_stage_schedule(cfg: dict) -> FDTwoStageSchedule:
    schedule_cfg = cfg.get("fd_training_schedule") or {}
    return FDTwoStageSchedule(
        warmup_epochs=max(int(schedule_cfg.get("warmup_epochs", 30)), 0),
        warmup_lr=float(schedule_cfg.get("warmup_lr", 3e-5)),
        finetune_lr=float(schedule_cfg.get("finetune_lr", 1e-5)),
        condition_hidden_dim=int((cfg.get("fd_left_fullres_channel") or {}).get("condition_hidden_dim", 64)),
    )


def _get_conditioned_stem(model: torch.nn.Module) -> nn.Module:
    raise TypeError(f"Stem freezing is not used for fd film conditioning: {type(model)!r}")


def _warmup_modules(model: torch.nn.Module) -> list[nn.Module]:
    if isinstance(model, DoubleUNet):
        return [
            model.unet_shared.fd_encoder_film,
            model.unet_shared.fd_base_film,
            model.unet_shared.fd_decoder_film,
            model.unet_union.fd_encoder_film,
            model.unet_union.fd_base_film,
            model.unet_union.fd_decoder_film,
        ]
    if isinstance(model, SingleUNet):
        return [model.unet.fd_encoder_film, model.unet.fd_base_film, model.unet.fd_decoder_film]
    raise TypeError(f"Unsupported model type for conditioned warmup: {type(model)!r}")


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(enabled)


def _configure_warmup_trainable(model: torch.nn.Module) -> None:
    _set_requires_grad(model, False)
    for module in _warmup_modules(model):
        _set_requires_grad(module, True)


def _configure_full_finetune_trainable(model: torch.nn.Module) -> None:
    _set_requires_grad(model, True)


def _register_old_stem_freeze_hook(model: torch.nn.Module, *, freeze_old_stem_channels: bool) -> list[torch.utils.hooks.RemovableHandle]:
    return []


def _build_optimizer(model: torch.nn.Module, param, *, lr: float) -> torch.optim.Optimizer:
    optimizer_cls = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[param.HyperParameter.optimizer]
    opt_kwargs = dict(param.HyperParameter.opt_param.__dict__)
    opt_kwargs["lr"] = float(lr)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found for optimizer.")
    return optimizer_cls(trainable, **opt_kwargs)


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, param):
    scheduler_cls = {
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "StepLR": torch.optim.lr_scheduler.StepLR,
    }[param.HyperParameter.learning_rate_scheduler]
    scheduler_kwargs = dict(param.HyperParameter.learning_rate_scheduler_param.__dict__)
    return scheduler_cls(optimizer, **scheduler_kwargs)


def _setup_conditioned_fitter(param, *, output_dir: Path, device: torch.device, checkpoint_init: Path, condition_dim: int, condition_hidden_dim: int):
    models_available = {"SingleUNet": SingleUNet, "DoubleUNet": DoubleUNet}
    model = models_available[param.HyperParameter.architecture].parse(param)
    model_ls = model_io.LoadSaveModel(model, output_file=output_dir / "model.pt", input_file=checkpoint_init)
    model = model_ls.load_init(device)
    model = patch_model_for_fd_film_conditioning(
        model,
        condition_dim=int(condition_dim),
        hidden_dim=int(condition_hidden_dim),
    )
    model.to(device)
    return model, model_ls


def main() -> int:
    _setup_logger()
    args = _parse_args()
    cfg = load_pipeline_config(args.config)
    train_cfg = cfg.get("train") or {}
    fd_cfg = cfg.get("fd_left_fullres_channel") or {}
    if not fd_cfg.get("enabled", False):
        raise ValueError("fd_left_fullres_channel.enabled must be true.")

    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 300))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32))
    device = _resolve_device(cfg, args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = FullResZernikeConditioning.from_npz(fd_cfg["coeff_maps_npz"])
    provider.save_stats_json(output_dir / "field_conditioning_stats.json")
    condition_dim = int(provider.condition_vector_dim)
    window_size = int(fd_cfg.get("window_size", 3))
    base_seed = int(((cfg.get("simulation") or {}).get("simulation") or {}).get("random_seed", 0))
    regen_train = bool(train_cfg.get("regen_train_dataset_each_epoch", False))

    train_sim, train_origins = _generate_left_dataset(
        cfg,
        provider=provider,
        profile="train",
        num_field_samples=int(fd_cfg.get("train_field_samples", 2)),
        seed_base=_epoch_seed(base_seed, 0, "train"),
    )
    scaling_info = normalize_dataset(cfg, profile="train", simulation_dataset=train_sim, update_dataset=False)
    eval_sim = None
    eval_origins: list[tuple[int, int]] = []
    if int(fd_cfg.get("eval_field_samples", 1)) > 0:
        eval_sim, eval_origins = _generate_left_dataset(
            cfg,
            provider=provider,
            profile="eval",
            num_field_samples=int(fd_cfg.get("eval_field_samples", 1)),
            seed_base=_epoch_seed(base_seed, 0, "eval"),
        )
        normalize_dataset(cfg, profile="train", simulation_dataset=eval_sim, update_dataset=False)

    param = _build_param(cfg, scaling_info, output_dir=output_dir, device=device, epochs=epochs, batch_size=batch_size, window_size=window_size)
    overrides = cfg.get("smlm_overrides") or {}
    if overrides.get("optimizer_lr") is not None:
        param.HyperParameter.opt_param.lr = float(overrides["optimizer_lr"])
    if overrides.get("optimizer_weight_decay") is not None:
        param.HyperParameter.opt_param.weight_decay = float(overrides["optimizer_weight_decay"])
    loc_meta = _build_loc_meta(cfg, param)
    frame_proc = FrameRescalar(scale=param.Scaling.input_scale, offset=param.Scaling.input_offset)
    bkg_proc = BkgFrameRescalar(scale=param.Scaling.bkg_max)
    tar_proc = TargetProcess(
        disable_attr=param.DataFilter.disabled_attributes,
        phot_max=param.Scaling.phot_max,
        z_max=param.Scaling.z_max,
    )

    def _make_dataset(sim, origins):
        return FullResConditionedDataset(
            sim,
            origins_xy=origins,
            provider=provider,
            frame_proc=frame_proc,
            bkg_frame_proc=bkg_proc,
            tar_proc=tar_proc,
            window_size=window_size,
        )

    train_dataset = _make_dataset(train_sim, train_origins)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"), collate_fn=smlm_collate)
    eval_loader = None
    eval_loc_tar = None
    if eval_sim is not None:
        eval_dataset = _make_dataset(eval_sim, eval_origins)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"), collate_fn=smlm_collate)
        eval_loc_tar = eval_dataset.build_locset(loc_meta)

    schedule = _resolve_two_stage_schedule(cfg)

    model, model_ls = _setup_conditioned_fitter(
        param,
        output_dir=output_dir,
        device=device,
        checkpoint_init=args.checkpoint_init,
        condition_dim=condition_dim,
        condition_hidden_dim=int(schedule.condition_hidden_dim),
    )
    criterion = train_setup.setup_criterion(loc_meta, param)
    post_processor, matcher = train_setup.setup_postprocess(loc_meta, param)
    runtime_state = {
        "window_size": window_size,
        "conditioning_vector_dim": condition_dim,
        "conditioning_hidden_dim": int(schedule.condition_hidden_dim),
        "condition_mode": "film_tile_vector_v1",
        "coeff_maps_npz": str(Path(fd_cfg["coeff_maps_npz"]).resolve()),
        "input_offset": float(param.Scaling.input_offset),
        "input_scale": float(param.Scaling.input_scale),
        "phot_max": float(param.Scaling.phot_max),
        "z_max": float(param.Scaling.z_max),
        "bg_max": float(param.Scaling.bkg_max),
        "camera_pixel_nm_x": float((cfg.get("optical") or {}).get("pixel_size_nm_x", 108.0)),
        "camera_pixel_nm_y": float((cfg.get("optical") or {}).get("pixel_size_nm_y", 108.0)),
        "crop_left": int(fd_cfg.get("crop_left", 0)),
        "crop_top": int(fd_cfg.get("crop_top", 0)),
        "crop_width": int(fd_cfg.get("crop_width", provider.image_shape_hw[1])),
        "crop_height": int(fd_cfg.get("crop_height", provider.image_shape_hw[0])),
        "fixed_origin_xy": fd_cfg.get("fixed_origin_xy"),
        "checkpoint_init": str(args.checkpoint_init.resolve()),
        "warmup_epochs": int(schedule.warmup_epochs),
        "warmup_lr": float(schedule.warmup_lr),
        "finetune_lr": float(schedule.finetune_lr),
        "preserve_backbone_at_init": True,
    }
    _write_json(output_dir / "runtime_state.json", runtime_state)
    _write_json(output_dir / "resolved_config.json", cfg)

    metrics_path = output_dir / "metrics.jsonl"
    best_jaccard = float("-inf")
    best_epoch = -1
    warmup_handles: list[torch.utils.hooks.RemovableHandle] = []
    in_warmup = int(schedule.warmup_epochs) > 0
    if in_warmup:
        _configure_warmup_trainable(model)
        warmup_handles = _register_old_stem_freeze_hook(
            model,
            freeze_old_stem_channels=False,
        )
        optimizer = _build_optimizer(model, param, lr=float(schedule.warmup_lr))
        lr_scheduler = None
    else:
        _configure_full_finetune_trainable(model)
        optimizer = _build_optimizer(model, param, lr=float(schedule.finetune_lr))
        lr_scheduler = _build_lr_scheduler(optimizer, param)

    for epoch in range(epochs):
        if in_warmup and epoch == int(schedule.warmup_epochs):
            for handle in warmup_handles:
                handle.remove()
            warmup_handles = []
            _configure_full_finetune_trainable(model)
            optimizer = _build_optimizer(model, param, lr=float(schedule.finetune_lr))
            lr_scheduler = _build_lr_scheduler(optimizer, param)
            in_warmup = False

        if regen_train and epoch > 0:
            train_sim, train_origins = _generate_left_dataset(
                cfg,
                provider=provider,
                profile="train",
                num_field_samples=int(fd_cfg.get("train_field_samples", 2)),
                seed_base=_epoch_seed(base_seed, epoch, "train"),
            )
            normalize_dataset(cfg, profile="train", simulation_dataset=train_sim, update_dataset=False)
            train_dataset = _make_dataset(train_sim, train_origins)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"), collate_fn=smlm_collate)
        loss_cmp_train = train_kernels.train_forward(model=model, optimizer=optimizer, loss=criterion, dataloader=train_loader, grad_mod=param.HyperParameter.grad_mod, epoch=epoch, device=str(device))
        train_loss = float(loss_cmp_train.sum(1).mean().item())
        eval_metrics: dict[str, float] = {}
        eval_loss = None
        if eval_loader is not None and eval_loc_tar is not None:
            loss_cmp_test, test_data = train_kernels.test_forward(model=model, loss=criterion, dataloader=eval_loader, epoch=epoch, device=str(device))
            loc_out = post_processor.forward(test_data.y_out)
            _, eval_sets = train_kernels.eval_forward(cam_pxszs=param.Camera.cam_pxszs, matcher=matcher, loc_out=loc_out, loc_tar=eval_loc_tar)
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
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(eval_loss if eval_loss is not None else train_loss)
            else:
                lr_scheduler.step()
        torch.save(model.state_dict(), output_dir / "checkpoint_latest.pt")
        model_ls.save(model, metric_val=None)
        row = {
            "epoch": epoch,
            "stage": "warmup" if in_warmup else "finetune",
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            **eval_metrics,
        }
        _write_metric(metrics_path, row)
        LOG.info("epoch=%d stage=%s lr=%.6g train_loss=%.6f %s", epoch, row["stage"], row["lr"], train_loss, eval_metrics)

    torch.save(model.state_dict(), output_dir / "checkpoint_final.pt")
    for handle in warmup_handles:
        handle.remove()
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
