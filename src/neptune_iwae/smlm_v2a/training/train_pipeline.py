#!/usr/bin/env python3
"""SMLM training entry wired to the phase-retrieval + simulation pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pipeline imports
from config import load_pipeline_config
from phase_retrieval.api import run_phase_retrieval
from simulation.api import generate_dataset, normalize_dataset, SimulationDataset

# SMLM imports
from neptune.smlm_v2a.data.data_factory import FrameRescalar, BkgFrameRescalar, TargetProcess
from neptune.smlm_v2a.data.smlm_dataset import smlm_collate
from neptune.smlm_v2a.generic.localizations import FrameLocSet
from neptune.smlm_v2a.postprocess.bin_count_render import write_rows_csv
from neptune.smlm_v2a.residual_refinement import (
    MeanPSFCenterRenderer,
    ResidualRefinementRuntime,
    test_forward_residual,
    train_forward_residual,
)
from neptune.smlm_v2a.training import train_setup, train_kernels
from neptune.smlm_v2a.utils.param_types import RecursiveNamespace


LOG = logging.getLogger("smlm_pipeline")


@dataclass(frozen=True)
class ResidualRefinementConfig:
    enabled: bool = False
    rounds: int = 3
    acceptance_threshold: float = 0.8
    training_render_source: str = "matched_gt"
    inference_render_source: str = "prediction"


def _setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _resolve_residual_refinement_config(cfg: dict) -> ResidualRefinementConfig:
    rr_cfg = cfg.get("residual_refinement") or {}
    return ResidualRefinementConfig(
        enabled=bool(rr_cfg.get("enabled", False)),
        rounds=int(rr_cfg.get("rounds", 3)),
        acceptance_threshold=float(rr_cfg.get("acceptance_threshold", 0.8)),
        training_render_source=str(rr_cfg.get("training_render_source", "matched_gt")),
        inference_render_source=str(rr_cfg.get("inference_render_source", "prediction")),
    )


def _as_numpy_cpu(val, dtype=None) -> np.ndarray:
    if torch.is_tensor(val):
        val = val.detach().cpu().numpy()
    else:
        val = np.asarray(val)
    if dtype is not None:
        val = val.astype(dtype, copy=False)
    return val


def _build_epoch_cfg(cfg: dict, base_seed: int, epoch: int) -> dict:
    cfg_copy = copy.deepcopy(cfg)
    sim_cfg = cfg_copy.setdefault("simulation", {})
    sim_runtime = sim_cfg.setdefault("simulation", {})
    sim_runtime["random_seed"] = int(base_seed + epoch)
    return cfg_copy


def _locset_to_rows(locset) -> list[dict[str, float | int | str]]:
    if locset is None or len(locset) == 0:
        return []
    xyz = _as_numpy_cpu(getattr(locset, "xyz"))
    indf = _as_numpy_cpu(getattr(locset, "indf"), dtype=np.int64).reshape(-1)
    phot = _as_numpy_cpu(getattr(locset, "phot", np.zeros((len(xyz),), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if hasattr(locset, "prob"):
        prob = _as_numpy_cpu(getattr(locset, "prob"), dtype=np.float32).reshape(-1)
    else:
        prob = np.full((len(xyz),), np.nan, dtype=np.float32)
    if prob.shape[0] != len(xyz):
        prob = np.full((len(xyz),), np.nan, dtype=np.float32)
    rows: list[dict[str, float | int | str]] = []
    for idx in range(len(xyz)):
        rows.append(
            {
                "frame": int(indf[idx]),
                "x_px": float(xyz[idx, 0]),
                "y_px": float(xyz[idx, 1]),
                "z": float(xyz[idx, 2]) if xyz.shape[1] > 2 else 0.0,
                "photon": float(phot[idx]),
                "prob": "" if np.isnan(prob[idx]) else float(prob[idx]),
            }
        )
    return rows


def _write_seed_plan(cfg: dict, base_seed: int, *, repo_root: Path) -> None:
    paths_cfg = cfg.get("paths") or {}
    seeds_dir = paths_cfg.get("seeds_dir")
    if not seeds_dir:
        return
    seed_dir = Path(seeds_dir)
    if not seed_dir.is_absolute():
        seed_dir = (repo_root / seed_dir).resolve()
    seed_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = cfg.get("simulation") or {}
    profiles_cfg = sim_cfg.get("profiles") or {}
    train_cfg = profiles_cfg.get("train", {}) or {}
    eval_cfg = profiles_cfg.get("eval", {}) or {}
    train_samples = int(train_cfg.get("num_samples", sim_cfg.get("num_samples", 1)))
    eval_samples = int(eval_cfg.get("num_samples", sim_cfg.get("num_samples", 1)))
    payload = {
        "base_seed": int(base_seed),
        "train": {
            "num_samples": int(train_samples),
            "seed_stride": 1,
            "seed_formula": "base_seed + epoch",
        },
        "eval": {
            "num_samples": int(eval_samples),
            "seed": int(base_seed),
        },
    }
    (seed_dir / "simulation_seed_plan.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _resolve_device(cfg: dict, override: str | None) -> torch.device:
    if override:
        dev_str = override
    else:
        dev_str = str((cfg.get("train") or {}).get("device", "cuda"))
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device(dev_str)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    return device


def _resolve_input_scaling(cfg: dict, photon_max: float) -> tuple[float, float]:
    camera_cfg = cfg.get("camera") or {}
    baseline = float(camera_cfg.get("baseline", 0.0))
    qe = float(camera_cfg.get("qe", 1.0))
    spurious = float(camera_cfg.get("spurious_charge", 0.0))
    e_per_adu = float(camera_cfg.get("e_per_adu", 1.0))

    sim_cfg = cfg.get("simulation") or {}
    bkg_uniform = sim_cfg.get("background_uniform") or [0.0, 0.0]
    bg_mean = 0.0
    if isinstance(bkg_uniform, (list, tuple)) and len(bkg_uniform) == 2:
        bg_mean = 0.5 * (float(bkg_uniform[0]) + float(bkg_uniform[1]))

    input_offset = baseline + (bg_mean * qe + spurious) / max(e_per_adu, 1e-6)
    input_scale = photon_max * qe / max(e_per_adu, 1e-6)
    input_scale = max(float(input_scale), 1e-6)
    return float(input_offset), float(input_scale)


def _resolve_fixed_scaling(cfg: dict) -> dict[str, float | None]:
    train_scaling = (cfg.get("train") or {}).get("scaling", {}) or {}

    def _resolve(name: str) -> float | None:
        value = os.getenv(f"SMLM_{name.upper()}", train_scaling.get(name))
        if value is None or value == "":
            return None
        return float(value)

    return {
        "input_offset": _resolve("input_offset"),
        "input_scale": _resolve("input_scale"),
        "photon_max": _resolve("photon_max"),
        "z_max": _resolve("z_max"),
        "bg_max": _resolve("bg_max"),
    }


def _torch_dtype(name: str, default: torch.dtype) -> torch.dtype:
    lookup = {
        "int16": torch.int16,
        "short": torch.int16,
        "int32": torch.int32,
        "int": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if not name:
        return default
    return lookup.get(str(name).lower(), default)


def _build_param(
    cfg: dict,
    scaling_info,
    *,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    window_size: int,
) -> RecursiveNamespace:
    ref_path = ROOT / "smlm_v2a" / "utils" / "reference_files" / "param_reference.yaml"
    with ref_path.open("r", encoding="utf-8") as handle:
        template = yaml_safe_load(handle)

    template.setdefault("ExpInOut", {})["experiment_out"] = str(output_dir)
    template.setdefault("ExpInOut", {})["checkpoint_init"] = None

    optical = cfg.get("optical") or {}
    px_x = float(optical.get("pixel_size_nm_x", 100.0))
    px_y = float(optical.get("pixel_size_nm_y", 100.0))
    cam_pxszs = [px_x, px_y, 1000.0]
    template.setdefault("Camera", {})["cam_pxszs"] = cam_pxszs

    template.setdefault("Hardware", {})["device"] = str(device)
    template.setdefault("Hardware", {})["device_simulation"] = str(device)

    template.setdefault("HyperParameter", {})["epochs"] = int(epochs)
    template.setdefault("HyperParameter", {})["batch_size"] = int(batch_size)
    template.setdefault("HyperParameter", {})["channels_in"] = int(window_size)

    # Optional run-time overrides to keep SMLM experiments aligned with external baselines.
    # Supported config key: cfg["smlm_overrides"] = {
    #   "lr_scheduler": "StepLR",
    #   "lr_step_size": 1000,
    #   "lr_gamma": 0.9,
    #   "loss_weight": [1.0, 0.0],
    #   "raw_th": 0.3,
    #   "split_th": 0.6,
    # }
    # Environment variables take precedence when present.
    overrides = cfg.get("smlm_overrides") or {}
    hp_cfg = template.setdefault("HyperParameter", {})
    lr_sched_cfg = hp_cfg.setdefault("learning_rate_scheduler_param", {})
    pp_cfg = template.setdefault("PostProcessingParam", {})

    lr_scheduler = os.getenv("SMLM_LR_SCHEDULER", overrides.get("lr_scheduler"))
    if lr_scheduler:
        hp_cfg["learning_rate_scheduler"] = str(lr_scheduler)

    lr_step_size = os.getenv("SMLM_LR_STEP_SIZE", overrides.get("lr_step_size"))
    if lr_step_size is not None:
        lr_sched_cfg["step_size"] = int(lr_step_size)

    lr_gamma = os.getenv("SMLM_LR_GAMMA", overrides.get("lr_gamma"))
    if lr_gamma is not None:
        lr_sched_cfg["gamma"] = float(lr_gamma)

    loss_weight_override = os.getenv("SMLM_LOSS_WEIGHT", None)
    if loss_weight_override is not None:
        parts = [p.strip() for p in str(loss_weight_override).split(",") if p.strip() != ""]
        if len(parts) != 2:
            raise ValueError(f"SMLM_LOSS_WEIGHT must have exactly 2 comma-separated values, got: {loss_weight_override!r}")
        hp_cfg["loss_weight"] = [float(parts[0]), float(parts[1])]
    elif overrides.get("loss_weight") is not None:
        lw = list(overrides.get("loss_weight"))
        if len(lw) != 2:
            raise ValueError(f"smlm_overrides.loss_weight must have length 2, got: {lw!r}")
        hp_cfg["loss_weight"] = [float(lw[0]), float(lw[1])]

    raw_th = os.getenv("SMLM_RAW_TH", overrides.get("raw_th"))
    if raw_th is not None:
        pp_cfg["raw_th"] = float(raw_th)

    split_th = os.getenv("SMLM_SPLIT_TH", overrides.get("split_th"))
    if split_th is not None:
        pp_cfg["split_th"] = float(split_th)

    depthwise = os.getenv("SMLM_DEPTHWISE", overrides.get("depthwise"))
    if depthwise is not None:
        arch_cfg = hp_cfg.setdefault("arch_param", {})
        if isinstance(depthwise, str):
            arch_cfg["depthwise"] = depthwise.strip().lower() not in {"0", "false", "no", "off"}
        else:
            arch_cfg["depthwise"] = bool(depthwise)

    sim_cfg = cfg.get("simulation") or {}
    emitter_cfg = sim_cfg.get("emitter") or {}
    emitter_extent = emitter_cfg.get("emitter_extent") or [[-0.5, 127.5], [-0.5, 127.5]]
    z_range = emitter_cfg.get("z_range")
    if z_range is None and len(emitter_extent) >= 3:
        z_range = emitter_extent[2]
    if z_range is None:
        z_range = [-0.5, 0.5]
    template.setdefault("Simulation", {})["ndim"] = 3
    template["Simulation"]["emitter_extent"] = [emitter_extent[0], emitter_extent[1], z_range]
    template["Simulation"]["nchannels"] = 1

    n_max = sim_cfg.get("n_max")
    if n_max is not None:
        template.setdefault("DataFilter", {})["max_number_targets"] = int(n_max)

    fixed_scaling = _resolve_fixed_scaling(cfg)

    photon_max = float(
        fixed_scaling["photon_max"] if fixed_scaling["photon_max"] is not None else scaling_info.photon_max
    )
    z_max = float(
        fixed_scaling["z_max"] if fixed_scaling["z_max"] is not None else scaling_info.z_max
    )
    bg_max = float(
        fixed_scaling["bg_max"] if fixed_scaling["bg_max"] is not None else scaling_info.bg_max
    )
    input_offset_default, input_scale_default = _resolve_input_scaling(cfg, photon_max)
    input_offset = float(
        fixed_scaling["input_offset"] if fixed_scaling["input_offset"] is not None else input_offset_default
    )
    input_scale = float(
        fixed_scaling["input_scale"] if fixed_scaling["input_scale"] is not None else input_scale_default
    )

    scaling = template.setdefault("Scaling", {})
    scaling["input_scale"] = float(input_scale)
    scaling["input_offset"] = float(input_offset)
    scaling["bkg_max"] = float(bg_max)
    scaling["phot_max"] = float(photon_max)
    scaling["z_max"] = float(z_max)

    dtype_cfg = template.setdefault("DataType", {})
    dtype_cfg["i_type"] = "int32" if dtype_cfg.get("i_type") is None else dtype_cfg.get("i_type")
    dtype_cfg["f_type"] = "float32" if dtype_cfg.get("f_type") is None else dtype_cfg.get("f_type")

    param = RecursiveNamespace(**template)
    # Replace dtype strings with torch dtypes for runtime.
    param.DataType.i_type = _torch_dtype(param.DataType.i_type, torch.int32)
    param.DataType.f_type = _torch_dtype(param.DataType.f_type, torch.float32)

    return param


def yaml_safe_load(handle):
    import yaml

    return yaml.load(handle, Loader=yaml.SafeLoader)


def _build_loc_meta(cfg: dict, param: RecursiveNamespace) -> FrameLocSet:
    sim_cfg = cfg.get("simulation") or {}
    emitter_cfg = sim_cfg.get("emitter") or {}
    emitter_extent = emitter_cfg.get("emitter_extent") or [[-0.5, 127.5], [-0.5, 127.5]]
    z_range = emitter_cfg.get("z_range")
    if z_range is None and len(emitter_extent) >= 3:
        z_range = emitter_extent[2]
    if z_range is None:
        z_range = [-0.5, 0.5]

    return FrameLocSet(
        ndim=3,
        xextent=tuple(emitter_extent[0]),
        yextent=tuple(emitter_extent[1]),
        zextent=tuple(z_range),
        nchannels=1,
        R2Ts=None,
        Ts2R=None,
        i_type=param.DataType.i_type,
        f_type=param.DataType.f_type,
    )


class SMLMPipelineDataset(Dataset):
    def __init__(
        self,
        sim_dataset: SimulationDataset,
        *,
        window_size: int,
        frame_proc: FrameRescalar | None,
        bkg_frame_proc: BkgFrameRescalar | None,
        tar_proc: TargetProcess | None,
    ) -> None:
        self.samples = sim_dataset.samples
        self.window_size = int(window_size)
        self.frame_proc = frame_proc
        self.bkg_frame_proc = bkg_frame_proc
        self.tar_proc = tar_proc
        self.index: list[tuple[int, int]] = []

        for idx, sample in enumerate(self.samples):
            sample.frames = _as_numpy_cpu(sample.frames, dtype=np.float32)
            sample.param_tar = _as_numpy_cpu(sample.param_tar, dtype=np.float32)
            sample.mask_tar = _as_numpy_cpu(sample.mask_tar, dtype=bool)
            if sample.bg_tar is not None:
                sample.bg_tar = _as_numpy_cpu(sample.bg_tar, dtype=np.float32)

            frame_count = sample.frames.shape[0]
            if frame_count < self.window_size:
                continue
            for start in range(frame_count - self.window_size + 1):
                self.index.append((idx, start))

        if not self.index:
            raise ValueError("SimulationDataset 样本帧数不足，无法构建训练窗口。")

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
        x = coords[:, 1]
        y = coords[:, 2]
        col = np.floor(x + 0.5).astype(np.int64)
        row = np.floor(y + 0.5).astype(np.int64)
        col = np.clip(col, 0, width - 1)
        row = np.clip(row, 0, height - 1)
        detect[row, col] = 1.0
        return detect

    def __getitem__(self, idx: int):
        sample_idx, start = self.index[idx]
        sample = self.samples[sample_idx]

        frames = sample.frames[start : start + self.window_size]
        target_idx = start + self.window_size // 2
        param = sample.param_tar[target_idx]
        mask = sample.mask_tar[target_idx]
        if sample.bg_tar is not None:
            bkg = sample.bg_tar[target_idx]
        else:
            bkg = np.zeros_like(frames[0], dtype=np.float32)

        height, width = frames.shape[1], frames.shape[2]
        detect = self._build_detect_map(param, mask, height, width)

        frames_t = torch.from_numpy(np.ascontiguousarray(frames))
        detect_t = torch.from_numpy(np.ascontiguousarray(detect))
        bkg_t = torch.from_numpy(np.ascontiguousarray(bkg))
        param_t = torch.from_numpy(np.ascontiguousarray(param))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))

        if self.frame_proc is not None:
            frames_t = self.frame_proc.forward(frames_t)
        if self.bkg_frame_proc is not None:
            bkg_t = self.bkg_frame_proc.forward(bkg_t)
        if self.tar_proc is not None:
            param_t = self.tar_proc.forward(param_t)

        return frames_t, detect_t, bkg_t, param_t, mask_t, None

    def build_locset(self, loc_meta: FrameLocSet) -> FrameLocSet:
        ids: list[np.ndarray] = []
        indf: list[np.ndarray] = []
        xyz: list[np.ndarray] = []
        phot: list[np.ndarray] = []
        next_id = 0

        for dataset_idx, (sample_idx, start) in enumerate(self.index):
            sample = self.samples[sample_idx]
            target_idx = start + self.window_size // 2
            param = sample.param_tar[target_idx]
            mask = sample.mask_tar[target_idx]
            if mask is None or not np.any(mask):
                continue
            valid = param[mask]
            if valid.size == 0:
                continue
            n_loc = valid.shape[0]
            ids.append(np.arange(next_id, next_id + n_loc, dtype=np.int32))
            indf.append(np.full((n_loc,), dataset_idx, dtype=np.int32))
            xyz.append(valid[:, 1:4].astype(np.float32, copy=False))
            phot.append(valid[:, 0].astype(np.float32, copy=False))
            next_id += n_loc

        if not ids:
            loc = FrameLocSet(**loc_meta.meta)
            return loc

        id_t = torch.from_numpy(np.concatenate(ids)).to(dtype=loc_meta.i_type)
        indf_t = torch.from_numpy(np.concatenate(indf)).to(dtype=loc_meta.i_type)
        xyz_t = torch.from_numpy(np.concatenate(xyz)).to(dtype=loc_meta.f_type)
        phot_t = torch.from_numpy(np.concatenate(phot)).to(dtype=loc_meta.f_type)

        loc = FrameLocSet(**loc_meta.meta)
        loc.init_load(id=id_t, indf=indf_t, xyz=xyz_t, phot=phot_t)
        return loc


def _write_metrics(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMLM training via pipeline simulation")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline_smlm.json"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-init", type=Path, default=None)
    parser.add_argument("--epoch-offset", type=int, default=0)
    return parser.parse_args()


def _resolve_phase_payload_for_training(
    cfg: dict,
    *,
    phase_payload,
    skip_phase_retrieval: bool,
):
    if phase_payload is not None:
        return phase_payload
    if skip_phase_retrieval:
        raise ValueError("phase_payload must be provided when skip_phase_retrieval=True")
    LOG.info("Running phase retrieval once...")
    phase_result = run_phase_retrieval(cfg, mode="auto", logger=LOG)
    return phase_result.patches_payload


def run_training(
    cfg: dict,
    *,
    output_dir: Path,
    device_override: str | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    phase_payload=None,
    skip_phase_retrieval: bool = False,
    checkpoint_init: Path | None = None,
    epoch_offset: int = 0,
) -> dict[str, object]:
    train_cfg = cfg.get("train") or {}
    epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs", 1000))
    batch_size = int(batch_size_override if batch_size_override is not None else train_cfg.get("batch_size", 8))
    regen_train = bool(train_cfg.get("regen_train_dataset_each_epoch", False))
    rr_cfg = _resolve_residual_refinement_config(cfg)

    device = _resolve_device(cfg, device_override)
    device_str = str(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_payload = _resolve_phase_payload_for_training(
        cfg,
        phase_payload=phase_payload,
        skip_phase_retrieval=skip_phase_retrieval,
    )

    base_seed = int(((cfg.get("simulation") or {}).get("simulation") or {}).get("random_seed", 0))
    _write_seed_plan(cfg, base_seed, repo_root=ROOT)

    cfg_train = _build_epoch_cfg(cfg, base_seed, epoch=0)
    train_sim = generate_dataset(cfg_train, profile="train", phase_payload=phase_payload, logger=LOG)
    scaling_info = normalize_dataset(cfg_train, profile="train", simulation_dataset=train_sim, update_dataset=False)

    eval_sim = None
    sim_profiles = (cfg.get("simulation") or {}).get("profiles") or {}
    if "eval" in sim_profiles:
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
    if checkpoint_init is not None:
        param.ExpInOut.checkpoint_init = str(checkpoint_init)
    loc_meta = _build_loc_meta(cfg, param)

    model_out = output_dir / "model.pt"
    model, model_ls, optimizer, lr_scheduler = train_setup.setup_fitter(param, model_out, device)
    criterion = train_setup.setup_criterion(loc_meta, param)
    post_processor, matcher = train_setup.setup_postprocess(loc_meta, param)

    frame_proc = FrameRescalar(scale=param.Scaling.input_scale, offset=param.Scaling.input_offset)
    bkg_proc = BkgFrameRescalar(scale=param.Scaling.bkg_max)
    tar_proc = TargetProcess(
        disable_attr=param.DataFilter.disabled_attributes,
        phot_max=param.Scaling.phot_max,
        z_max=param.Scaling.z_max,
    )

    train_dataset = SMLMPipelineDataset(
        train_sim,
        window_size=param.HyperParameter.channels_in,
        frame_proc=frame_proc,
        bkg_frame_proc=bkg_proc,
        tar_proc=tar_proc,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=smlm_collate,
    )

    eval_dataset = None
    eval_loader = None
    eval_loc_tar = None
    if eval_sim is not None:
        eval_dataset = SMLMPipelineDataset(
            eval_sim,
            window_size=param.HyperParameter.channels_in,
            frame_proc=frame_proc,
            bkg_frame_proc=bkg_proc,
            tar_proc=tar_proc,
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
        LOG.info("Building eval locset for metrics...")
        eval_loc_tar = eval_dataset.build_locset(loc_meta)

    rr_runtime = None
    if rr_cfg.enabled:
        image_size = int(train_sim.samples[0].frames.shape[-1])
        rr_renderer = MeanPSFCenterRenderer.from_pipeline(
            cfg=cfg,
            param=param,
            phase_payload=phase_payload,
            device=device,
            image_size=image_size,
        )
        rr_runtime = ResidualRefinementRuntime(
            rounds=int(rr_cfg.rounds),
            acceptance_threshold=float(rr_cfg.acceptance_threshold),
            training_render_source=str(rr_cfg.training_render_source),
            inference_render_source=str(rr_cfg.inference_render_source),
            post_processor=post_processor,
            matcher=matcher,
            renderer=rr_renderer,
            loc_meta=loc_meta,
            phot_max=float(param.Scaling.phot_max),
            z_max=float(param.Scaling.z_max),
        )

    metrics_path = output_dir / "metrics.jsonl"
    best_model_path = output_dir / "checkpoint_best_jaccard.pt"
    best_meta_path = output_dir / "best_jaccard.json"
    best_jaccard = float("-inf")
    best_epoch: int | None = None
    last_eval_loc_out = None
    last_eval_loc_tar = None
    LOG.info("Output dir: %s", output_dir)
    LOG.info("Epochs: %d, batch_size: %d", epochs, batch_size)
    LOG.info(
        "SMLM scaling input_offset=%.6f input_scale=%.6f phot_max=%.6f z_max=%.6f bg_max=%.6f",
        float(param.Scaling.input_offset),
        float(param.Scaling.input_scale),
        float(param.Scaling.phot_max),
        float(param.Scaling.z_max),
        float(param.Scaling.bkg_max),
    )
    if rr_runtime is not None:
        LOG.info(
            "Residual refinement enabled rounds=%d p_thresh=%.3f train_render=%s infer_render=%s",
            rr_runtime.rounds,
            rr_runtime.acceptance_threshold,
            rr_runtime.training_render_source,
            rr_runtime.inference_render_source,
        )

    for epoch in range(epochs):
        effective_epoch = int(epoch_offset) + int(epoch)
        if regen_train and effective_epoch > 0:
            cfg_train = _build_epoch_cfg(cfg, base_seed, epoch=effective_epoch)
            train_sim = generate_dataset(cfg_train, profile="train", phase_payload=phase_payload, logger=LOG)
            train_dataset = SMLMPipelineDataset(
                train_sim,
                window_size=param.HyperParameter.channels_in,
                frame_proc=frame_proc,
                bkg_frame_proc=bkg_proc,
                tar_proc=tar_proc,
            )
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                collate_fn=smlm_collate,
            )

        if rr_runtime is None:
            loss_cmp_train = train_kernels.train_forward(
                model=model,
                optimizer=optimizer,
                loss=criterion,
                dataloader=train_loader,
                grad_mod=param.HyperParameter.grad_mod,
                epoch=effective_epoch,
                device=device_str,
            )
        else:
            loss_cmp_train = train_forward_residual(
                model=model,
                optimizer=optimizer,
                loss=criterion,
                dataloader=train_loader,
                grad_mod=param.HyperParameter.grad_mod,
                epoch=effective_epoch,
                device=device_str,
                runtime=rr_runtime,
            )
        train_loss = float(loss_cmp_train.sum(1).mean().item())

        eval_metrics = {}
        eval_loss = None
        if eval_loader is not None:
            if rr_runtime is None:
                loss_cmp_test, test_data = train_kernels.test_forward(
                    model=model,
                    loss=criterion,
                    dataloader=eval_loader,
                    epoch=effective_epoch,
                    device=device_str,
                )
                loc_out = post_processor.forward(test_data.y_out)
            else:
                loss_cmp_test, loc_out = test_forward_residual(
                    model=model,
                    loss=criterion,
                    dataloader=eval_loader,
                    epoch=effective_epoch,
                    device=device_str,
                    runtime=rr_runtime,
                )
            eval_loss = float(loss_cmp_test.sum(1).mean().item())
            matching, eval_sets = train_kernels.eval_forward(
                cam_pxszs=param.Camera.cam_pxszs,
                matcher=matcher,
                loc_out=loc_out,
                loc_tar=eval_loc_tar,
            )
            eval_dict = eval_sets._asdict()
            eval_metrics = {
                "precision": float(eval_dict.get("prec", 0.0)),
                "recall": float(eval_dict.get("rec", 0.0)),
                "jaccard": float(eval_dict.get("jac", 0.0)),
                "rmse_lat": float(eval_dict.get("rmse_lat", 0.0)),
                "rmse_ax": float(eval_dict.get("rmse_ax", 0.0)),
            }
            if eval_metrics["jaccard"] > best_jaccard:
                best_jaccard = float(eval_metrics["jaccard"])
                best_epoch = int(effective_epoch)
                if hasattr(model, "state_dict"):
                    torch.save(model.state_dict(), best_model_path)
                best_meta_path.write_text(
                    json.dumps({"epoch": int(effective_epoch), "jaccard": float(best_jaccard)}, indent=2) + "\n",
                    encoding="utf-8",
                )
            last_eval_loc_out = loc_out
            last_eval_loc_tar = eval_loc_tar
            LOG.info(
                "epoch=%d train_loss=%.6f eval_loss=%.6f P=%.3f R=%.3f J=%.3f RMSE_lat=%.3f RMSE_ax=%.3f",
                effective_epoch,
                train_loss,
                eval_loss,
                eval_metrics["precision"],
                eval_metrics["recall"],
                eval_metrics["jaccard"],
                eval_metrics["rmse_lat"],
                eval_metrics["rmse_ax"],
            )
        else:
            LOG.info("epoch=%d train_loss=%.6f (no eval profile)", effective_epoch, train_loss)

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(eval_loss if eval_loss is not None else train_loss)
        else:
            lr_scheduler.step()

        model_ls.save(model, metric_val=None)

        _write_metrics(
            metrics_path,
            {
                "epoch": int(effective_epoch),
                "loss": float(train_loss),
                **eval_metrics,
            },
        )

    if last_eval_loc_out is not None:
        write_rows_csv(_locset_to_rows(last_eval_loc_out), output_dir / "predictions.csv")
    if last_eval_loc_tar is not None:
        write_rows_csv(_locset_to_rows(last_eval_loc_tar), output_dir / "targets.csv")

    return {
        "output_dir": output_dir,
        "metrics_path": metrics_path,
        "model_path": model_out,
        "best_model_path": best_model_path if best_model_path.is_file() else None,
        "device": device_str,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "eval_enabled": bool(eval_loader is not None),
        "best_epoch": best_epoch,
        "best_jaccard": None if best_epoch is None else float(best_jaccard),
    }


def main() -> int:
    _setup_logger()
    args = parse_args()

    cfg = load_pipeline_config(args.config)
    output_dir = ROOT / "output" / "smlm_pipeline" if args.output_dir is None else args.output_dir
    run_training(
        cfg,
        output_dir=output_dir,
        device_override=args.device,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        checkpoint_init=args.checkpoint_init,
        epoch_offset=args.epoch_offset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
