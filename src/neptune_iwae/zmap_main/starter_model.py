from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
import yaml

from .dependencies import DoubleUNet, FrameLocSet, LoadSaveModel, RecursiveNamespace, SingleUNet


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STARTER_DIR = HERE / "starters" / "density_1"
RUNTIME_ROOT = ROOT / "smlm_v2a"
RUNTIME_TEMPLATE_PATH = RUNTIME_ROOT / "utils" / "reference_files" / "param_reference.yaml"
STARTER_PIPELINE_CONFIG = STARTER_DIR / "pipeline_density_1_real_sample_default.json"
LEGACY_STARTER_CHECKPOINT = STARTER_DIR / "checkpoint_best_jaccard_stdconv_density1.pt"
STARTER_CHECKPOINT = LEGACY_STARTER_CHECKPOINT if LEGACY_STARTER_CHECKPOINT.exists() else ROOT / "starter" / "checkpoint_best_jaccard.pt"
STARTER_BEST_META = STARTER_DIR / "best_jaccard_stdconv_density1.json"


@dataclass(frozen=True)
class StarterCheckpointSpec:
    name: str
    pipeline_config_path: Path
    checkpoint_path: Path
    best_meta_path: Path


def density1_starter_spec() -> StarterCheckpointSpec:
    return StarterCheckpointSpec(
        name="density_1_mu1_starter",
        pipeline_config_path=STARTER_PIPELINE_CONFIG,
        checkpoint_path=STARTER_CHECKPOINT,
        best_meta_path=STARTER_BEST_META,
    )


def load_density1_pipeline_config() -> dict:
    with STARTER_PIPELINE_CONFIG.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_density1_best_meta() -> dict:
    with STARTER_BEST_META.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    return lookup.get(str(name).lower(), default)


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _build_density1_param(
    cfg: dict,
    *,
    checkpoint_path: Path | None = None,
    device: str | torch.device = "cpu",
) -> RecursiveNamespace:
    with RUNTIME_TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        template = yaml.load(handle, Loader=yaml.SafeLoader)

    optical = cfg.get("optical") or {}
    emitter_cfg = ((cfg.get("simulation") or {}).get("emitter") or {})
    train_cfg = cfg.get("train") or {}
    scaling_cfg = train_cfg.get("scaling") or {}
    localization_overrides = cfg.get("localization_overrides") or {}

    template.setdefault("ExpInOut", {})["experiment_out"] = str(STARTER_DIR)
    template.setdefault("ExpInOut", {})["checkpoint_init"] = str(
        (STARTER_CHECKPOINT if checkpoint_path is None else checkpoint_path).resolve()
    )

    template.setdefault("Camera", {})["cam_pxszs"] = [
        float(optical.get("pixel_size_nm_x", 100.0)),
        float(optical.get("pixel_size_nm_y", 100.0)),
        1000.0,
    ]

    template.setdefault("Hardware", {})["device"] = str(device)
    template.setdefault("Hardware", {})["device_simulation"] = str(device)

    hp_cfg = template.setdefault("HyperParameter", {})
    hp_cfg["channels_in"] = 3
    hp_cfg["epochs"] = int(train_cfg.get("epochs", hp_cfg.get("epochs", 500)))
    hp_cfg["batch_size"] = int(train_cfg.get("batch_size", hp_cfg.get("batch_size", 64)))
    arch_param_cfg = hp_cfg.setdefault("arch_param", {})
    if localization_overrides.get("depthwise") is not None:
        arch_param_cfg["depthwise"] = bool(localization_overrides["depthwise"])
    if localization_overrides.get("conv_block") is not None:
        arch_param_cfg["conv_block"] = str(localization_overrides["conv_block"])
    overrides = cfg.get("smlm_overrides") or {}
    lr_scheduler = overrides.get("lr_scheduler")
    if lr_scheduler:
        hp_cfg["learning_rate_scheduler"] = str(lr_scheduler)
    lr_sched_cfg = hp_cfg.setdefault("learning_rate_scheduler_param", {})
    if overrides.get("lr_step_size") is not None:
        lr_sched_cfg["step_size"] = int(overrides["lr_step_size"])
    if overrides.get("lr_gamma") is not None:
        lr_sched_cfg["gamma"] = float(overrides["lr_gamma"])

    sim_cfg = template.setdefault("Simulation", {})
    sim_cfg["ndim"] = 3
    sim_cfg["nchannels"] = 1
    emitter_extent = emitter_cfg.get("emitter_extent", [[-0.5, 63.5], [-0.5, 63.5], [-0.6, 0.6]])
    sim_cfg["emitter_extent"] = [
        list(emitter_extent[0]),
        list(emitter_extent[1]),
        list(emitter_cfg.get("z_range", emitter_extent[2])),
    ]

    data_filter_cfg = template.setdefault("DataFilter", {})
    if ((cfg.get("simulation") or {}).get("n_max")) is not None:
        data_filter_cfg["max_number_targets"] = int((cfg.get("simulation") or {}).get("n_max"))

    scaling = template.setdefault("Scaling", {})
    background_uniform = (cfg.get("simulation") or {}).get("background_uniform", [0.0, 200.0])
    intensity_mu_sig = emitter_cfg.get("intensity_mu_sig", [20_000.0, 1_000.0])
    z_range = emitter_cfg.get("z_range", sim_cfg["emitter_extent"][2])
    scaling["input_offset"] = float(
        _first_non_none(
            scaling_cfg.get("input_offset"),
            scaling.get("input_offset"),
            sum(background_uniform) / 2.0,
        )
    )
    scaling["input_scale"] = float(
        _first_non_none(
            scaling_cfg.get("input_scale"),
            scaling.get("input_scale"),
            max(float(intensity_mu_sig[0]) / 50.0, 1.0),
        )
    )
    scaling["phot_max"] = float(
        _first_non_none(
            scaling_cfg.get("photon_max"),
            scaling.get("phot_max"),
            float(intensity_mu_sig[0]) + 8.0 * float(intensity_mu_sig[1]),
        )
    )
    scaling["z_max"] = float(
        _first_non_none(
            scaling_cfg.get("z_max"),
            scaling.get("z_max"),
            1.2 * max(abs(float(z_range[0])), abs(float(z_range[1]))),
        )
    )
    scaling["bkg_max"] = float(
        _first_non_none(
            scaling_cfg.get("bg_max"),
            scaling.get("bkg_max"),
            1.2 * float(background_uniform[1]),
        )
    )

    param = RecursiveNamespace(**template)
    param.DataType.i_type = _torch_dtype(param.DataType.i_type, torch.int32)
    param.DataType.f_type = _torch_dtype(param.DataType.f_type, torch.float32)
    return param


def build_density1_training_param(
    cfg: dict,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> RecursiveNamespace:
    ckpt_path = None if checkpoint_path is None else Path(checkpoint_path)
    return _build_density1_param(cfg, checkpoint_path=ckpt_path, device=device)


def build_density1_loc_meta(cfg: dict, param: RecursiveNamespace) -> FrameLocSet:
    emitter_cfg = ((cfg.get("simulation") or {}).get("emitter") or {})
    emitter_extent = emitter_cfg.get("emitter_extent", param.Simulation.emitter_extent)
    z_extent = emitter_cfg.get("z_range", param.Simulation.emitter_extent[2])
    return FrameLocSet(
        ndim=int(param.Simulation.ndim),
        xextent=tuple(emitter_extent[0]),
        yextent=tuple(emitter_extent[1]),
        zextent=tuple(z_extent) if int(param.Simulation.ndim) == 3 else None,
        nchannels=int(param.Simulation.nchannels),
        R2Ts=None,
        Ts2R=None,
        i_type=param.DataType.i_type,
        f_type=param.DataType.f_type,
    )


def build_density1_starter_model(
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    cfg: dict | None = None,
) -> torch.nn.Module:
    cfg = load_density1_pipeline_config() if cfg is None else cfg
    ckpt_path = STARTER_CHECKPOINT if checkpoint_path is None else Path(checkpoint_path)
    param = _build_density1_param(cfg, checkpoint_path=ckpt_path, device=device)
    model_cls = {
        "SingleUNet": SingleUNet,
        "DoubleUNet": DoubleUNet,
    }[param.HyperParameter.architecture]
    model = model_cls.parse(param)
    model_loader = LoadSaveModel(model, output_file=None, input_file=ckpt_path)
    return model_loader.load_init(device)


__all__ = [
    "RUNTIME_TEMPLATE_PATH",
    "STARTER_BEST_META",
    "STARTER_CHECKPOINT",
    "STARTER_DIR",
    "STARTER_PIPELINE_CONFIG",
    "StarterCheckpointSpec",
    "build_density1_loc_meta",
    "build_density1_starter_model",
    "build_density1_training_param",
    "density1_starter_spec",
    "load_density1_best_meta",
    "load_density1_pipeline_config",
]
