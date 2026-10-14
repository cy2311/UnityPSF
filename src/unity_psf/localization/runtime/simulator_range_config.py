"""Online simulator range and emitter-density configuration resolution."""

from __future__ import annotations

from typing import Any, Mapping

from unity_psf.localization.scale_resolver import PhysicalScaleSpec, resolve_physical_scales
from .paths import optional_mapping, pair_from_config, range_from_config


def online_simulator_range_params(
    config: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    train_cfg: Mapping[str, Any],
    height: int,
    width: int,
    pixel_size_nm_x: float,
    pixel_size_nm_y: float,
    scales: PhysicalScaleSpec | None = None,
) -> dict[str, Any]:
    simulation_cfg = optional_mapping(config.get("simulation"), "simulation") or {}
    emitter_cfg = optional_mapping(simulation_cfg.get("emitter"), "simulation.emitter") or {}
    scaling_cfg = optional_mapping(train_cfg.get("scaling"), "train.scaling") or {}
    if scales is None:
        scales = resolve_physical_scales(train_cfg)
    params: dict[str, Any] = {}
    background_range = range_from_config(
        online_cfg.get("background_range", simulation_cfg.get("background_uniform")),
        label="background_range",
    )
    if background_range is not None:
        params["background_range"] = background_range
    if "background_scale" in online_cfg:
        params["background_scale"] = float(online_cfg["background_scale"])
    elif "bg_max" in scaling_cfg:
        params["background_scale"] = float(scaling_cfg["bg_max"])

    z_range = range_from_config(online_cfg.get("z_range", emitter_cfg.get("z_range")), label="z_range")
    if z_range is None and scales.z_scale is not None:
        z_range = (-float(scales.z_scale), float(scales.z_scale))
    if z_range is not None:
        params["z_range"] = z_range

    photon_range = range_from_config(
        online_cfg.get("photon_range", emitter_cfg.get("intensity_clip")),
        label="photon_range",
    )
    if photon_range is None and scales.photon_scale is not None:
        photon_range = (0.0, float(scales.photon_scale))
    if photon_range is not None:
        params["photon_range"] = photon_range

    intensity_mu_sig = pair_from_config(
        online_cfg.get("photon_mean_sigma", emitter_cfg.get("intensity_mu_sig")),
        label="photon_mean_sigma",
    )
    if intensity_mu_sig is not None:
        params["photon_mean"] = float(intensity_mu_sig[0])
        params["photon_sigma"] = float(intensity_mu_sig[1])
    elif "photon_mean" in online_cfg and "photon_sigma" in online_cfg:
        params["photon_mean"] = float(online_cfg["photon_mean"])
        params["photon_sigma"] = float(online_cfg["photon_sigma"])

    density = online_density_um2(
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

    return params


def online_density_um2(
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

