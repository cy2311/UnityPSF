from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .conditioning_config import append_domain_onehot_enabled as _append_domain_onehot, condition_dim as _condition_dim, condition_feature_dim as _condition_feature_dim, has_explicit_astigmatism_condition_contract as _has_explicit_astigmatism_condition_contract, is_astigmatism_expert as _is_astigmatism_expert, is_emitter_2d_expert as _is_emitter_2d_expert, single_channel_online_config as _single_channel_online_config, validate_condition_dimensions as _validate_condition_dimensions
from .contracts import _astigmatism_condition_dim, _astigmatism_condition_fields, _emitter_2d_condition_fields, _input_frame_spec
from .paths import grid_size as _grid_size, mapping as _mapping, optional_mapping as _optional_mapping, pair_from_config as _pair_from_config, range_from_config as _range_from_config, resolve_path as _resolve_path


def _online_provider_config(
    config: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    config_base_dir: str | Path | None,
    seed: int,
    single_channel: bool = False,
    model_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    astigmatism_single_channel = single_channel and _is_astigmatism_expert(train_cfg)
    emitter_2d_single_channel = single_channel and _is_emitter_2d_expert(train_cfg)
    film_expert_single_channel = astigmatism_single_channel or emitter_2d_single_channel
    if film_expert_single_channel:
        expert_condition_fields = (
            _astigmatism_condition_fields(train_cfg, online_cfg, model_params=model_params)
            if astigmatism_single_channel
            else _emitter_2d_condition_fields(train_cfg, online_cfg, model_params=model_params)
        )
        expert_condition_dim = (
            _astigmatism_condition_dim(train_cfg, online_cfg, model_params=model_params)
            if astigmatism_single_channel
            else int((model_params or {}).get("condition_dim", len(expert_condition_fields)))
        )
        if _has_explicit_astigmatism_condition_contract(train_cfg):
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
        effective_online_cfg = _single_channel_online_config(online_cfg)
    else:
        expert_condition_fields = ()
        expert_condition_dim = 0
        effective_online_cfg = dict(online_cfg)
    online_cfg = effective_online_cfg
    batch_size = int(train_cfg.get("batch_size", online_cfg.get("batch_size", 1)))
    soft_moe = str(online_cfg.get("conditioning_mode", "channels")) == "film" and str(online_cfg.get("expert_mode", "")) == "soft_moe"
    _validate_condition_dimensions(online_cfg, soft_moe=soft_moe)
    dual_domain_coeff_maps = _dual_domain_coeff_maps(
        online_cfg.get("dual_domain_coeff_maps", ()),
        base_dir=None if config_base_dir is None else Path(config_base_dir),
    )
    if single_channel:
        dual_domain_coeff_maps = _select_single_channel_coeff_maps(dual_domain_coeff_maps, train_cfg)
    pupil_carrier_complex = _pupil_carrier_complex(
        online_cfg.get("pupil_carrier_complex_npz"),
        base_dir=None if config_base_dir is None else Path(config_base_dir),
    )
    domain_count = 1 if single_channel else int(online_cfg.get("domain_count", 2))
    if dual_domain_coeff_maps and domain_count != len(dual_domain_coeff_maps):
        raise ValueError("domain_count must match dual_domain_coeff_maps length")
    simulation_cfg = _optional_mapping(config.get("simulation"), "simulation") or {}
    psf_cfg = _optional_mapping(simulation_cfg.get("psf"), "simulation.psf") or {}
    vector_cfg = _optional_mapping(psf_cfg.get("vector"), "simulation.psf.vector") or {}
    optical_cfg = _optional_mapping(config.get("optical"), "optical") or {}
    height = int(online_cfg.get("height", 128))
    width = int(online_cfg.get("width", 128))
    pixel_size_nm_x = float(online_cfg.get("pixel_size_nm_x", optical_cfg.get("pixel_size_nm_x", 101.11)))
    pixel_size_nm_y = float(online_cfg.get("pixel_size_nm_y", optical_cfg.get("pixel_size_nm_y", 98.83)))
    scaling_cfg = _optional_mapping(train_cfg.get("scaling"), "train.scaling") or {}
    expert_cfg = _optional_mapping(train_cfg.get("expert"), "train.expert") or {}
    bound_channel_id = expert_cfg.get("channel_id", expert_cfg.get("instance_id"))
    empirical_psf_path = online_cfg.get("empirical_psf_path")
    if empirical_psf_path is not None:
        empirical_psf_path = _resolve_path(
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
        else _condition_feature_dim(online_cfg)
    )
    condition_dim = (
        int(online_cfg.get("condition_dim", condition_feature_dim))
        if film_expert_single_channel
        else _condition_dim(online_cfg)
    )
    sim_ranges = _online_simulator_range_params(
        config,
        online_cfg,
        train_cfg=train_cfg,
        height=height,
        width=width,
        pixel_size_nm_x=pixel_size_nm_x,
        pixel_size_nm_y=pixel_size_nm_y,
    )
    return {
        "name": "online_train_batch",
        "params": {
            "batch_size": batch_size,
            "channels": _input_frame_spec(train_cfg, online_cfg).input_frame_channels if single_channel else int(online_cfg.get("channels", 3)),
            "height": height,
            "width": width,
            "emitters_per_sample": int(online_cfg.get("emitters_per_sample", 8)),
            "seed": int(seed),
            "steps_per_epoch": int(online_cfg.get("steps_per_epoch", 1)),
            "background": float(online_cfg.get("background", 0.0)),
            "signal": float(online_cfg.get("signal", 1.0)),
            "simulation_backend": str(online_cfg.get("simulation_backend", "native")),
            "simulation_output_device": str(online_cfg.get("simulation_output_device", "cpu")),
            "cached_window_order": str(online_cfg.get("cached_window_order", "auto")),
            "cached_window_max_gpu_sequences": int(online_cfg.get("cached_window_max_gpu_sequences", 0)),
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
            "vector_psf_size": int(online_cfg.get("vector_psf_size", (online_cfg.get("lut_simulation") or {}).get("psf_size", vector_cfg.get("psf_size", 51)))),
            "vector_batch_size": int(online_cfg.get("vector_batch_size", vector_cfg.get("batch_size", 96))),
            "zemit0": None if online_cfg.get("zemit0", vector_cfg.get("zemit0")) is None else float(online_cfg.get("zemit0", vector_cfg.get("zemit0"))),
            "lut_field_stride": int((online_cfg.get("lut_simulation") or {}).get("field_stride", online_cfg.get("lut_field_stride", 16))),
            "lut_z_steps": int(online_cfg.get("nat_grid_z_steps", (online_cfg.get("lut_simulation") or {}).get("z_steps", online_cfg.get("lut_z_steps", 41)))),
            "lut_subpixel_bins": int((online_cfg.get("lut_simulation") or {}).get("subpixel_bins", online_cfg.get("lut_subpixel_bins", 1))),
            "lut_field_mode": str((online_cfg.get("lut_simulation") or {}).get("field_mode", online_cfg.get("lut_field_mode", "roi_origin"))),
            "lut_storage_dtype": str((online_cfg.get("lut_simulation") or {}).get("storage_dtype", online_cfg.get("lut_storage_dtype", "fp32"))),
            "field_origin_sampling_mode": str(online_cfg.get("field_origin_sampling_mode", "grid")),
            "field_origin_stride_px": int(online_cfg.get("field_origin_stride_px", online_cfg.get("field_origin_stride", 40))),
            **sim_ranges,
            "conditioning_mode": str(online_cfg.get("conditioning_mode", "channels")),
            "nat_simulation_mode": str(online_cfg.get("nat_simulation_mode", "tile_center")),
            "nat_grid_size": _grid_size(online_cfg.get("nat_grid_size", 32)),
            "nat_grid_z_steps": int(online_cfg.get("nat_grid_z_steps", 41)),
            "append_domain_onehot": False if film_expert_single_channel else _append_domain_onehot(online_cfg, soft_moe=soft_moe),
            "condition_feature_dim": condition_feature_dim,
            "condition_dim": condition_dim,
            "condition_fields": expert_condition_fields,
            "conditioning_profile": str(online_cfg.get("conditioning_profile", "default_nat")),
            "domain_count": domain_count,
            "domain_balance_mode": "fixed" if single_channel else str(online_cfg.get("domain_balance_mode", "fixed")),
            "dual_domain_coeff_maps": dual_domain_coeff_maps,
            "pupil_carrier_complex": pupil_carrier_complex,
            "batch_strategy": str(online_cfg.get("batch_strategy", "triplet")),
            "sequence_window_chunks": int(online_cfg.get("sequence_window_chunks", 1)),
            "sequence_count": int(online_cfg.get("sequence_count", 64)),
            "camera_qe": float((config.get("camera") or {}).get("qe", online_cfg.get("camera_qe", 0.9))),
            "camera_spurious_charge": float(
                (config.get("camera") or {}).get("spurious_charge", online_cfg.get("camera_spurious_charge", 0.002))
            ),
            "camera_baseline": float((config.get("camera") or {}).get("baseline", online_cfg.get("camera_baseline", 398.6))),
            "camera_e_per_adu": float((config.get("camera") or {}).get("e_per_adu", online_cfg.get("camera_e_per_adu", 1.020784562122306))),
            "camera_read_sigma": float(
                (config.get("camera") or {}).get("read_sigma", online_cfg.get("camera_read_sigma", 0.0))
            ),
            "pxyz_target_order": str(online_cfg.get("pxyz_target_order", "legacy_iwae")),
            "photon_scale": (
                None
                if "photon_max" not in scaling_cfg
                else float(scaling_cfg["photon_max"])
            ),
            "z_scale": (
                None
                if "z_max" not in scaling_cfg
                else float(scaling_cfg["z_max"])
            ),
        },
    }


def _microtube_tiff_provider_config(
    train_cfg: Mapping[str, Any],
    microtube_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
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
            "calibration": dict(_mapping(microtube_cfg.get("calibration", {}), "train.microtube_tiff.calibration")),
            "normalization": dict(_mapping(microtube_cfg.get("normalization", {}), "train.microtube_tiff.normalization")),
        },
    }


def _online_simulator_range_params(
    config: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    train_cfg: Mapping[str, Any],
    height: int,
    width: int,
    pixel_size_nm_x: float,
    pixel_size_nm_y: float,
) -> dict[str, Any]:
    simulation_cfg = _optional_mapping(config.get("simulation"), "simulation") or {}
    emitter_cfg = _optional_mapping(simulation_cfg.get("emitter"), "simulation.emitter") or {}
    scaling_cfg = _optional_mapping(train_cfg.get("scaling"), "train.scaling") or {}
    online_lut_cfg = _optional_mapping(online_cfg.get("lut_simulation"), "train.online_generation.lut_simulation") or {}

    params: dict[str, Any] = {}
    background_range = _range_from_config(
        online_cfg.get("background_range", simulation_cfg.get("background_uniform")),
        label="background_range",
    )
    if background_range is not None:
        params["background_range"] = background_range
    if "background_scale" in online_cfg:
        params["background_scale"] = float(online_cfg["background_scale"])
    elif "bg_max" in scaling_cfg:
        params["background_scale"] = float(scaling_cfg["bg_max"])

    z_range = _range_from_config(online_cfg.get("z_range", emitter_cfg.get("z_range")), label="z_range")
    if z_range is None and "z_max" in scaling_cfg:
        z_max = float(scaling_cfg["z_max"])
        z_range = (-z_max, z_max)
    if z_range is not None:
        params["z_range"] = z_range

    photon_range = _range_from_config(
        online_cfg.get("photon_range", emitter_cfg.get("intensity_clip")),
        label="photon_range",
    )
    if photon_range is None and "photon_max" in scaling_cfg:
        photon_range = (0.0, float(scaling_cfg["photon_max"]))
    if photon_range is not None:
        params["photon_range"] = photon_range

    intensity_mu_sig = _pair_from_config(
        online_cfg.get("photon_mean_sigma", emitter_cfg.get("intensity_mu_sig")),
        label="photon_mean_sigma",
    )
    if intensity_mu_sig is not None:
        params["photon_mean"] = float(intensity_mu_sig[0])
        params["photon_sigma"] = float(intensity_mu_sig[1])
    elif "photon_mean" in online_cfg and "photon_sigma" in online_cfg:
        params["photon_mean"] = float(online_cfg["photon_mean"])
        params["photon_sigma"] = float(online_cfg["photon_sigma"])

    density = _online_density_um2(
        online_cfg,
        simulation_cfg,
        emitter_cfg,
        frames_per_sample=int(simulation_cfg.get("frames_per_sample", online_cfg.get("channels", 3))),
        height=int(height),
        width=int(width),
        pixel_size_nm_x=float(pixel_size_nm_x),
        pixel_size_nm_y=float(pixel_size_nm_y),
    )
    if density is not None:
        params["emitter_density_um2"] = float(density)
    if "lifetime_avg" in online_cfg:
        params["lifetime_avg"] = float(online_cfg["lifetime_avg"])
    elif "lifetime_avg" in emitter_cfg:
        params["lifetime_avg"] = float(emitter_cfg["lifetime_avg"])
    if "warmup_frames" in online_cfg:
        params["warmup_frames"] = float(online_cfg["warmup_frames"])
    elif "warmup_frames" in online_lut_cfg:
        params["warmup_frames"] = float(online_lut_cfg["warmup_frames"])

    return params


def _online_density_um2(
    online_cfg: Mapping[str, Any],
    simulation_cfg: Mapping[str, Any],
    emitter_cfg: Mapping[str, Any],
    *,
    frames_per_sample: int,
    height: int,
    width: int,
    pixel_size_nm_x: float,
    pixel_size_nm_y: float,
) -> float | None:
    for source in (online_cfg, emitter_cfg, simulation_cfg):
        if "emitter_density_um2" in source:
            return float(source["emitter_density_um2"])
        if "density_um2" in source:
            return float(source["density_um2"])
        if "density" in source:
            return float(source["density"])
    num_emitters = simulation_cfg.get("num_emitters")
    if num_emitters is not None:
        lifetime_avg = float(emitter_cfg.get("lifetime_avg", 1.0))
        active = float(num_emitters) * (lifetime_avg + 1.0) / (float(frames_per_sample) + 6.0 * lifetime_avg)
        area_um2 = (
            int(width)
            * float(pixel_size_nm_x)
            / 1000.0
            * int(height)
            * float(pixel_size_nm_y)
            / 1000.0
        )
        return active / max(area_um2, 1e-9)
    return None


def _dual_domain_coeff_maps(value: Any, *, base_dir: Path | None = None) -> tuple[dict[str, str], ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("dual_domain_coeff_maps must be a list")
    maps: list[dict[str, str]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("dual_domain_coeff_maps entries must be mappings")
        name = str(item.get("name", f"domain{idx}"))
        path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
        if path is None:
            raise ValueError("dual_domain_coeff_maps entries must include coeff_maps_npz, alternating_coeff_maps_npz, or path")
        maps.append({"name": name, "coeff_maps_npz": _resolve_path(str(path), base_dir=base_dir)})
    return tuple(maps)


def _select_single_channel_coeff_maps(
    entries: tuple[dict[str, str], ...],
    train_cfg: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Bind a single-channel runtime to its own physical coefficient map."""

    if not entries:
        return ()
    expert_cfg = train_cfg.get("expert")
    channel_id = "main"
    if isinstance(expert_cfg, Mapping) and expert_cfg.get("channel_id") is not None:
        channel_id = str(expert_cfg["channel_id"])
    matching = tuple(item for item in entries if str(item.get("name", "")) == channel_id)
    if len(entries) > 1:
        if len(matching) != 1:
            names = sorted(str(item.get("name", "")) for item in entries)
            raise ValueError(
                "single-channel runtime has no unique coefficient map for "
                f"channel={channel_id!r}; available domains={names!r}"
            )
        selected = matching[0]
    else:
        selected = entries[0]
    return ({**selected, "name": channel_id},)


def _pupil_carrier_complex(value: Any, *, base_dir: Path | None) -> torch.Tensor | None:
    if value in (None, ""):
        return None
    path = Path(_resolve_path(str(value), base_dir=base_dir))
    with np.load(path, allow_pickle=False) as payload:
        key = "carrier_complex" if "carrier_complex" in payload else "complex_pupil"
        carrier = np.asarray(payload[key], dtype=np.complex64)
    if carrier.ndim != 2 or carrier.shape[0] != carrier.shape[1]:
        raise ValueError("pupil carrier must be a square complex pupil array")
    return torch.from_numpy(carrier)
