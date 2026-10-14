"""Runtime batch-provider configuration resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from unity_psf.localization.scale_resolver import PhysicalScaleSpec, resolve_physical_scales
from .conditioning_config import (
    append_domain_onehot_enabled,
    astigmatism_condition_dim,
    astigmatism_condition_fields,
    condition_dim as resolve_condition_dim,
    condition_feature_dim as resolve_condition_feature_dim,
    emitter_2d_condition_fields,
    has_explicit_astigmatism_condition_contract,
    is_astigmatism_expert,
    is_emitter_2d_expert,
    single_channel_online_config,
    validate_condition_dimensions,
)
from .paths import (
    grid_size,
    mapping,
    optional_mapping,
    resolve_path,
)
from .physical_config import (
    dual_domain_coeff_maps as resolve_dual_domain_coeff_maps,
    pupil_carrier_complex,
    select_single_channel_coeff_maps,
)
from .types import RuntimeProviderConfig
from .modality_config import input_frame_spec
from .simulator_range_config import online_simulator_range_params


def online_provider_config(
    config: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    config_base_dir: str | Path | None,
    seed: int,
    single_channel: bool = False,
    model_params: Mapping[str, Any] | None = None,
    scales: PhysicalScaleSpec | None = None,
    pupil_carrier_resolver: Callable[..., Any] = pupil_carrier_complex,
) -> RuntimeProviderConfig:
    astigmatism_single_channel = single_channel and is_astigmatism_expert(train_cfg)
    emitter_2d_single_channel = single_channel and is_emitter_2d_expert(train_cfg)
    ppg_single_channel = single_channel and str(online_cfg.get("psf_type", "")).strip().lower() == "ppg_direct"
    film_expert_single_channel = astigmatism_single_channel or emitter_2d_single_channel or ppg_single_channel
    if film_expert_single_channel:
        if astigmatism_single_channel:
            expert_condition_fields = astigmatism_condition_fields(train_cfg, online_cfg, model_params=model_params)
            expert_condition_dim = astigmatism_condition_dim(train_cfg, online_cfg, model_params=model_params)
        elif emitter_2d_single_channel:
            expert_condition_fields = emitter_2d_condition_fields(train_cfg, online_cfg, model_params=model_params)
            expert_condition_dim = int((model_params or {}).get("condition_dim", len(expert_condition_fields)))
        else:
            expert_condition_fields = tuple(
                (model_params or {}).get(
                    "condition_fields",
                    ("field_x", "field_y", "ppg_pca_0", "ppg_pca_1", "ppg_pca_2", "ppg_pca_3"),
                )
            )
            expert_condition_dim = int((model_params or {}).get("condition_dim", len(expert_condition_fields)))
        if has_explicit_astigmatism_condition_contract(train_cfg) or ppg_single_channel:
            if "condition_feature_dim" in online_cfg and int(online_cfg["condition_feature_dim"]) != expert_condition_dim:
                raise ValueError("expert condition_feature_dim must match model condition_fields")
            if "condition_dim" in online_cfg and int(online_cfg["condition_dim"]) != expert_condition_dim:
                raise ValueError("expert condition_dim must match model condition_fields")
        effective_online_cfg = dict(online_cfg)
        effective_online_cfg.update(
            {
                "domain_count": 1,
                "append_domain_onehot": False,
                "condition_feature_dim": expert_condition_dim,
                "condition_dim": expert_condition_dim,
            }
        )
    elif single_channel:
        expert_condition_fields = ()
        expert_condition_dim = 0
        effective_online_cfg = single_channel_online_config(online_cfg)
    else:
        expert_condition_fields = ()
        expert_condition_dim = 0
        effective_online_cfg = dict(online_cfg)
    online_cfg = effective_online_cfg
    if "lut_simulation" in online_cfg:
        raise ValueError("train.online_generation.lut_simulation is removed; use cuda_gemm settings")
    if str(online_cfg.get("simulation_backend", "cuda_gemm")).strip().lower() == "lut":
        raise ValueError("simulation_backend='lut' is removed; use 'cuda_gemm'")
    batch_size = int(train_cfg.get("batch_size", online_cfg.get("batch_size", 1)))
    soft_moe = str(online_cfg.get("conditioning_mode", "channels")) == "film" and str(online_cfg.get("expert_mode", "")) == "soft_moe"
    validate_condition_dimensions(online_cfg, soft_moe=soft_moe)
    dual_domain_coeff_maps = resolve_dual_domain_coeff_maps(
        online_cfg.get("dual_domain_coeff_maps", ()),
        base_dir=None if config_base_dir is None else Path(config_base_dir),
    )
    if single_channel:
        dual_domain_coeff_maps = select_single_channel_coeff_maps(dual_domain_coeff_maps, train_cfg)
    pupil_carrier_complex = pupil_carrier_resolver(
        online_cfg.get("pupil_carrier_complex_npz"),
        base_dir=None if config_base_dir is None else Path(config_base_dir),
    )
    domain_count = 1 if single_channel else int(online_cfg.get("domain_count", 2))
    if dual_domain_coeff_maps and domain_count != len(dual_domain_coeff_maps):
        raise ValueError("domain_count must match dual_domain_coeff_maps length")
    simulation_cfg = optional_mapping(config.get("simulation"), "simulation") or {}
    psf_cfg = optional_mapping(simulation_cfg.get("psf"), "simulation.psf") or {}
    vector_cfg = optional_mapping(psf_cfg.get("vector"), "simulation.psf.vector") or {}
    optical_cfg = optional_mapping(config.get("optical"), "optical") or {}
    default_conditioning_profile = (
        "circular_2d_50nm" if is_emitter_2d_expert(train_cfg) else "default_nat"
    )
    height = int(online_cfg.get("height", 128))
    width = int(online_cfg.get("width", 128))
    pixel_size_nm_x = float(online_cfg.get("pixel_size_nm_x", optical_cfg.get("pixel_size_nm_x", 101.11)))
    pixel_size_nm_y = float(online_cfg.get("pixel_size_nm_y", optical_cfg.get("pixel_size_nm_y", 98.83)))
    if scales is None:
        scales = resolve_physical_scales(train_cfg)
    expert_cfg = optional_mapping(train_cfg.get("expert"), "train.expert") or {}
    bound_channel_id = expert_cfg.get("channel_id", expert_cfg.get("instance_id"))
    empirical_psf_path = online_cfg.get("empirical_psf_path")
    if empirical_psf_path is not None:
        empirical_psf_path = resolve_path(
            str(empirical_psf_path),
            base_dir=None if config_base_dir is None else Path(config_base_dir),
        )
    if film_expert_single_channel:
        if expert_condition_dim != len(expert_condition_fields):
            raise ValueError("expert model condition_dim must equal condition_fields length")
        if "condition_feature_dim" in online_cfg and int(online_cfg["condition_feature_dim"]) != expert_condition_dim:
            raise ValueError("expert condition_feature_dim must match model condition_fields")
        if "condition_dim" in online_cfg and int(online_cfg["condition_dim"]) != expert_condition_dim:
            raise ValueError("expert condition_dim must match model condition_fields")
    condition_feature_dim = (
        int(online_cfg.get("condition_feature_dim", expert_condition_dim))
        if film_expert_single_channel
        else resolve_condition_feature_dim(online_cfg)
    )
    condition_dim = (
        int(online_cfg.get("condition_dim", condition_feature_dim))
        if film_expert_single_channel
        else resolve_condition_dim(online_cfg)
    )
    sim_ranges = online_simulator_range_params(
        config,
        online_cfg,
        train_cfg=train_cfg,
        height=height,
        width=width,
        pixel_size_nm_x=pixel_size_nm_x,
        pixel_size_nm_y=pixel_size_nm_y,
        scales=scales,
    )
    return {
        "name": "online_train_batch",
        "params": {
            "batch_size": batch_size,
            "channels": input_frame_spec(train_cfg, online_cfg).input_frame_channels if single_channel else int(online_cfg.get("channels", 3)),
            "height": height,
            "width": width,
            "emitters_per_sample": int(online_cfg.get("emitters_per_sample", 8)),
            "seed": int(seed),
            "steps_per_epoch": int(online_cfg.get("steps_per_epoch", 1)),
            "background": float(online_cfg.get("background", 0.0)),
            "signal": float(online_cfg.get("signal", 1.0)),
            "simulation_backend": str(online_cfg.get("simulation_backend", "cuda_gemm")),
            "simulation_output_device": str(online_cfg.get("simulation_output_device", "cpu")),
            "cached_window_order": str(online_cfg.get("cached_window_order", "auto")),
            "cached_window_max_gpu_sequences": int(online_cfg.get("cached_window_max_gpu_sequences", 0)),
            "cached_window_cache_epochs": int(online_cfg.get("cached_window_cache_epochs", 8)),
            "cached_window_cache_frames": int(online_cfg.get("cached_window_cache_frames", 0)),
            "cached_window_cache_storage": str(online_cfg.get("cached_window_cache_storage", "cpu")),
            "cached_window_prefetch_batches": int(online_cfg.get("cached_window_prefetch_batches", 1)),
            "psf_type": str(online_cfg.get("psf_type", psf_cfg.get("psf_type", "vector"))),
            "empirical_psf_path": empirical_psf_path,
            "empirical_psf_channel": str(online_cfg.get("empirical_psf_channel", bound_channel_id or "")) or None,
            "empirical_psf_focus_index": (
                None
                if online_cfg.get("empirical_psf_focus_index") is None
                else int(online_cfg["empirical_psf_focus_index"])
            ),
            "pixel_size_nm_x": pixel_size_nm_x,
            "pixel_size_nm_y": pixel_size_nm_y,
            "wavelength_nm": float(online_cfg.get("wavelength_nm", optical_cfg.get("wavelength_nm", 660.0))),
            "na": float(online_cfg.get("NA", online_cfg.get("na", optical_cfg.get("NA", 1.4)))),
            "refmed": float(online_cfg.get("refmed", vector_cfg.get("refmed", optical_cfg.get("n_medium", 1.518)))),
            "refcov": float(online_cfg.get("refcov", vector_cfg.get("refcov", optical_cfg.get("n_medium", 1.518)))),
            "refimm": float(online_cfg.get("refimm", vector_cfg.get("refimm", optical_cfg.get("n_medium", 1.518)))),
            "objstage0": float(online_cfg.get("objstage0", vector_cfg.get("objstage0", 0.0))),
            "otf_rescale_xy": tuple(float(value) for value in online_cfg.get("otf_rescale_xy", vector_cfg.get("otf_rescale_xy", (0.0, 0.0)))),
            "npupil": int(online_cfg.get("npupil", vector_cfg.get("npupil", 128))),
            "vector_psf_size": int(online_cfg.get("vector_psf_size", vector_cfg.get("psf_size", 51))),
            "vector_batch_size": int(online_cfg.get("vector_batch_size", vector_cfg.get("batch_size", 96))),
            "zemit0": None if online_cfg.get("zemit0", vector_cfg.get("zemit0")) is None else float(online_cfg.get("zemit0", vector_cfg.get("zemit0"))),
            "ppg_map_path": (
                None
                if online_cfg.get("ppg_map_path") is None
                else resolve_path(str(online_cfg["ppg_map_path"]), base_dir=config_base_dir)
            ),
            "field_origin_sampling_mode": str(online_cfg.get("field_origin_sampling_mode", "grid")),
            "field_origin_stride_px": int(online_cfg.get("field_origin_stride_px", online_cfg.get("field_origin_stride", 40))),
            **sim_ranges,
            "conditioning_mode": str(online_cfg.get("conditioning_mode", "channels")),
            "nat_simulation_mode": str(online_cfg.get("nat_simulation_mode", "tile_center")),
            "nat_grid_size": grid_size(online_cfg.get("nat_grid_size", 32)),
            "nat_grid_z_steps": int(online_cfg.get("nat_grid_z_steps", 41)),
            "append_domain_onehot": False if film_expert_single_channel else append_domain_onehot_enabled(online_cfg, soft_moe=soft_moe),
            "condition_feature_dim": condition_feature_dim,
            "condition_dim": condition_dim,
            "condition_fields": expert_condition_fields,
            "conditioning_profile": str(online_cfg.get("conditioning_profile", default_conditioning_profile)),
            "domain_count": domain_count,
            "domain_balance_mode": "fixed" if single_channel else str(online_cfg.get("domain_balance_mode", "fixed")),
            "dual_domain_coeff_maps": dual_domain_coeff_maps,
            "fixed_astig_anchor_nm": (
                None
                if online_cfg.get("fixed_astig_anchor_nm") is None
                else float(online_cfg["fixed_astig_anchor_nm"])
            ),
            "circular_psf_radius_nm": (
                None
                if online_cfg.get("circular_psf_radius_nm") is None
                else float(online_cfg["circular_psf_radius_nm"])
            ),
            "pupil_carrier_complex": pupil_carrier_complex,
            "batch_strategy": str(online_cfg.get("batch_strategy", "triplet")),
            "sequence_window_chunks": int(online_cfg.get("sequence_window_chunks", 1)),
            "sequence_count": int(online_cfg.get("sequence_count", 64)),
            "camera_qe": float((config.get("camera") or {}).get("qe", online_cfg.get("camera_qe", 0.9))),
            "camera_spurious_charge": float(
                (config.get("camera") or {}).get("spurious_charge", online_cfg.get("camera_spurious_charge", 0.002))
            ),
            "camera_readout": bool(online_cfg.get("camera_readout", False)),
            "camera_baseline": float((config.get("camera") or {}).get("baseline", online_cfg.get("camera_baseline", 398.6))),
            "camera_e_per_adu": float((config.get("camera") or {}).get("e_per_adu", online_cfg.get("camera_e_per_adu", 1.020784562122306))),
            "camera_read_sigma": float(
                (config.get("camera") or {}).get("read_sigma", online_cfg.get("camera_read_sigma", 0.0))
            ),
            "pxyz_target_order": str(online_cfg.get("pxyz_target_order", "legacy_iwae")),
            "photon_scale": scales.photon_scale,
            "z_scale": scales.z_scale,
            "z_scale_unit": scales.z_unit,
        },
    }

def microtube_tiff_provider_config(
    train_cfg: Mapping[str, Any],
    microtube_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> RuntimeProviderConfig:
    return {
        "name": "microtube_tiff_train_batch",
        "params": {
            "tiff_path": str(microtube_cfg["tiff_path"]),
            "batch_size": int(train_cfg.get("batch_size", microtube_cfg.get("batch_size", 1))),
            "channels": int(microtube_cfg.get("channels", 3)),
            "height": None if microtube_cfg.get("height") is None else int(microtube_cfg["height"]),
            "width": None if microtube_cfg.get("width") is None else int(microtube_cfg["width"]),
            "steps_per_epoch": int(microtube_cfg.get("steps_per_epoch", 1)),
            "frame_start": int(microtube_cfg.get("frame_start", 0)),
            "frame_stop": None if microtube_cfg.get("frame_stop") is None else int(microtube_cfg["frame_stop"]),
            "crop_top": int(microtube_cfg.get("crop_top", 0)),
            "crop_left": int(microtube_cfg.get("crop_left", 0)),
            "seed": int(seed),
            "calibration": dict(mapping(microtube_cfg.get("calibration", {}), "train.microtube_tiff.calibration")),
            "normalization": dict(mapping(microtube_cfg.get("normalization", {}), "train.microtube_tiff.normalization")),
        },
    }
