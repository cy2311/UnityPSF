"""ROI source resolution and ROI-bank construction for high-fidelity training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.roi_library import (
    InferredEmitter,
    ROIBank,
    ROIBankBuildConfig,
    ROIBankDomain,
    RawInferenceResult,
    build_roi_bank_from_inference,
)
from unity_psf.roi_library.loc_harvest import LocHarvestConfig, build_roi_bank_from_loc_harvest
from unity_psf.training.high_fidelity.raw_tiff_inference import build_model_raw_tiff_infer_fn


@dataclass(frozen=True)
class ROIBankSource:
    mode: str
    raw_path: str
    candidate_mode: str
    frame_range: tuple[int, int] | None
    domains: tuple[Mapping[str, Any], ...] | None
    alias: str | None = None


def auto_build_roi_bank(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: ROIBankSource,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
) -> ROIBank:
    roi_size = int(gamma_cfg.get("roi_size_px", 8))
    camera_backward = camera_backward_from_training_config(train_cfg)
    config = ROIBankBuildConfig(
        roi_size_px=roi_size,
        window_size=int(gamma_cfg.get("roi_bank_window_size", 3)),
        frame_range=roi_source.frame_range,
        grid_shape=tuple(int(v) for v in gamma_cfg.get("roi_bank_grid_shape", (2, 2))),
        max_rois=int(gamma_cfg.get("roi_bank_max_rois", gamma_cfg.get("roi_library_max_rois", gamma_cfg.get("target_projected_emitters", 2)))),
        target_emitters=int(gamma_cfg.get("target_projected_emitters", 2)),
        candidate_probability_threshold=float(gamma_cfg.get("roi_bank_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))),
        probability_threshold=float(gamma_cfg.get("roi_bank_probability_threshold", gamma_cfg.get("probability_threshold", 0.5))),
        max_overlap_fraction=float(gamma_cfg.get("roi_bank_max_overlap_fraction", 0.95)),
        seed=int(gamma_cfg.get("seed", 0)),
        background_smoothing_kernel=int(gamma_cfg.get("background_smoothing_kernel", gamma_cfg.get("roi_bank_background_smoothing_kernel", 3))),
        camera_backward=camera_backward,
        over_cut_px=int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0))),
        origin_mode=str(gamma_cfg.get("roi_bank_origin_mode", gamma_cfg.get("origin_mode", "emitter_centered"))),
        origin_stride_px=None if gamma_cfg.get("roi_bank_origin_stride_px", gamma_cfg.get("origin_stride_px")) is None else int(gamma_cfg.get("roi_bank_origin_stride_px", gamma_cfg.get("origin_stride_px"))),
        valid_core_size_px=None if gamma_cfg.get("roi_bank_valid_core_size_px", gamma_cfg.get("valid_core_size_px")) is None else int(gamma_cfg.get("roi_bank_valid_core_size_px", gamma_cfg.get("valid_core_size_px"))),
    )
    if uses_model_infer(roi_source):
        return _auto_build_roi_bank_from_loc_harvest(gamma_cfg, roi_source=roi_source, model=model, train_cfg=train_cfg, config=config, roi_size=roi_size)
    bank = build_roi_bank_from_inference(
        raw_frames_photon=roi_source.raw_path,
        domains=auto_build_domains(gamma_cfg, roi_size=roi_size, roi_source=roi_source),
        infer_fn=roi_bank_infer_fn(gamma_cfg, roi_source=roi_source, model=model, window_size=int(config.window_size), train_cfg=train_cfg),
        config=config,
    )
    return ROIBank(
        records=bank.records,
        config=bank.config,
        metadata={
            **bank.metadata,
            "roi_bank_source_alias": roi_source.alias,
            "roi_bank_candidate_mode": roi_source.candidate_mode,
            "roi_bank_infer_source": "bright_pixel",
            "roi_bank_observed_units": "camera_corrected_photons" if camera_backward is not None else "raw_input",
            **roi_bank_camera_backward_metadata(bank.metadata, camera_backward),
        },
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    )


def _auto_build_roi_bank_from_loc_harvest(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: ROIBankSource,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
    config: ROIBankBuildConfig,
    roi_size: int,
) -> ROIBank:
    harvest_config = LocHarvestConfig(
        raw_path=roi_source.raw_path,
        domains=auto_build_domains(gamma_cfg, roi_size=roi_size, roi_source=roi_source),
        roi_bank_config=config,
        input_offset=posterior_input_offset(train_cfg),
        input_scale=posterior_input_scale(train_cfg),
        photon_scale=posterior_photon_scale(train_cfg) or 1.0,
        z_scale=posterior_z_scale(train_cfg) or 1.0,
        bg_scale=posterior_bg_scale(train_cfg),
        normalization_config={"normalization": dict(train_cfg.get("normalization", {}))} if isinstance(train_cfg.get("normalization"), Mapping) else None,
        candidate_probability_threshold=float(gamma_cfg.get("posterior_candidate_probability_threshold", gamma_cfg.get("roi_bank_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3)))),
        probability_threshold=float(gamma_cfg.get("roi_bank_probability_threshold", gamma_cfg.get("probability_threshold", 0.5))),
        split_threshold=float(gamma_cfg.get("posterior_adjacent_probability_threshold", gamma_cfg.get("split_threshold", 0.6))),
        phot_min=float(gamma_cfg.get("roi_bank_phot_min", gamma_cfg.get("phot_min", 100.0))),
        sigma_max_px=float(gamma_cfg.get("roi_bank_sigma_max_px", gamma_cfg.get("sigma_max_px", 2.5))),
        tile_size_px=int(gamma_cfg.get("roi_bank_spatial_tile_size", 128)),
        overlap_px=int(gamma_cfg.get("roi_bank_spatial_overlap_px", 16)),
        max_emitters_per_window=None if gamma_cfg.get("roi_bank_infer_max_emitters") is None else int(gamma_cfg.get("roi_bank_infer_max_emitters")),
    )
    bank = build_roi_bank_from_loc_harvest(model=model, config=harvest_config, condition_context=roi_conditioning_context(train_cfg))
    return ROIBank(
        records=bank.records,
        config=bank.config,
        metadata={
            **bank.metadata,
            "roi_bank_source_alias": roi_source.alias,
            "roi_bank_candidate_mode": roi_source.candidate_mode,
            "roi_bank_infer_source": "loc_harvest_raw_tiff",
            "roi_bank_harvest_channel_order": "p,photons,x,y,z,photons_sigma,x_sigma,y_sigma,z_sigma,bg",
            "roi_bank_observed_units": "camera_corrected_photons" if config.camera_backward is not None else "raw_input",
            **roi_bank_camera_backward_metadata(bank.metadata, config.camera_backward),
        },
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    )


def roi_bank_infer_fn(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: ROIBankSource,
    model: torch.nn.Module,
    window_size: int,
    train_cfg: Mapping[str, Any],
):
    if not uses_model_infer(roi_source):
        return bright_pixel_infer_fn
    return build_model_raw_tiff_infer_fn(
        model=model,
        threshold=float(gamma_cfg.get("roi_bank_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))),
        max_emitters=int(gamma_cfg.get("roi_bank_infer_max_emitters", gamma_cfg.get("num_posterior_samples", gamma_cfg.get("target_projected_emitters", 2)))),
        expected_channels=int(gamma_cfg.get("roi_bank_infer_channels", window_size)),
        photon_scale=posterior_photon_scale(train_cfg),
        z_scale=posterior_z_scale(train_cfg),
        condition_context=roi_conditioning_context(train_cfg),
    )


def resolve_roi_bank_source(
    gamma_cfg: Mapping[str, Any],
    *,
    train_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
    config_base_dir: Path,
) -> ROIBankSource | None:
    source_cfg = gamma_cfg.get("roi_bank_source")
    if isinstance(source_cfg, Mapping):
        mode = str(source_cfg.get("mode", "auto_build"))
        if mode not in {"auto_build", "loc_infer_raw_tiff"}:
            return None
        raw_path = source_cfg.get("raw_path", source_cfg.get("tiff_path"))
        if raw_path is None:
            raise ValueError("train.roi_bank_gamma.roi_bank_source requires raw_path for auto_build mode")
        return ROIBankSource(
            mode="auto_build",
            raw_path=resolve_config_path(str(raw_path), base_dir=config_base_dir),
            candidate_mode=str(source_cfg.get("candidate_mode", gamma_cfg.get("roi_bank_candidate_mode", "bright_pixel"))),
            frame_range=frame_range_tuple(source_cfg.get("frame_range", gamma_cfg.get("roi_bank_frame_range"))),
            domains=domains_tuple(source_cfg.get("domains")),
            alias=None if mode == "auto_build" else mode,
        )
    if gamma_cfg.get("auto_build_roi_bank") is True and "auto_build_source_path" in gamma_cfg:
        return ROIBankSource(
            mode="auto_build",
            raw_path=resolve_config_path(str(gamma_cfg["auto_build_source_path"]), base_dir=config_base_dir),
            candidate_mode=str(gamma_cfg.get("roi_bank_candidate_mode", "bright_pixel")),
            frame_range=frame_range_tuple(gamma_cfg.get("roi_bank_frame_range")),
            domains=domains_tuple(gamma_cfg.get("auto_build_domains")),
        )
    if gamma_cfg.get("auto_build_roi_bank") is True and source_cfg == "loc_infer_raw_tiff":
        real_tiff_cfg = mapping(train_cfg.get("real_tiff_wake"), "train.real_tiff_wake")
        raw_path = real_tiff_cfg.get("tiff_path") or phase_retrieval_tiff_path(config)
        if raw_path is None:
            raise ValueError("loc_infer_raw_tiff ROI-bank source requires train.real_tiff_wake.tiff_path")
        return ROIBankSource(
            mode="auto_build",
            raw_path=resolve_config_path(str(raw_path), base_dir=config_base_dir),
            candidate_mode=str(gamma_cfg.get("roi_bank_candidate_mode", "bright_pixel")),
            frame_range=frame_range_tuple(gamma_cfg.get("roi_bank_frame_range")),
            domains=domains_tuple(real_tiff_cfg.get("domains")),
            alias="loc_infer_raw_tiff",
        )
    return None


def auto_build_domains(gamma_cfg: Mapping[str, Any], *, roi_size: int, roi_source: ROIBankSource) -> tuple[ROIBankDomain, ...]:
    raw_domains = roi_source.domains if roi_source.domains is not None else gamma_cfg.get("auto_build_domains")
    if raw_domains is None:
        size = int(gamma_cfg.get("auto_build_domain_size_px", max(roi_size + 4, roi_size)))
        return (ROIBankDomain("auto", crop_left=0, crop_top=0, crop_width=size, crop_height=size),)
    return tuple(
        ROIBankDomain(
            str(item.get("name", "auto")),
            crop_left=int(item.get("crop_left", 0)),
            crop_top=int(item.get("crop_top", 0)),
            crop_width=int(item["crop_width"]),
            crop_height=int(item["crop_height"]),
        )
        for item in (mapping(raw, "auto_build_domains[]") for raw in raw_domains)
    )


def bright_pixel_infer_fn(*, domain: ROIBankDomain, frame_window: tuple[int, int], raw_domain_frames_photon: np.ndarray) -> RawInferenceResult:
    projection = np.asarray(raw_domain_frames_photon, dtype=np.float32).mean(axis=0)
    flat_indices = np.argsort(projection.reshape(-1))[::-1][:2]
    height, width = projection.shape
    emitters = tuple(
        InferredEmitter(
            probability=0.9,
            mu_xy_px=(float(x), float(y)),
            sigma_xy_px=(0.3, 0.3),
            mu_z_nm=0.0,
            sigma_z_nm=10.0,
            mu_photons=float(max(projection[y, x], 1.0)),
            sigma_photons=1.0,
            cell_xy_px=(float(x), float(y)),
        )
        for flat_index in flat_indices
        for y, x in (divmod(int(flat_index), int(width)),)
    )
    return RawInferenceResult(
        emitters=emitters,
        background_mu=np.full((height, width), float(np.median(projection)), dtype=np.float32),
        metadata={"domain": domain.name, "frame_window": frame_window},
    )


def uses_model_infer(roi_source: ROIBankSource) -> bool:
    return roi_source.alias == "loc_infer_raw_tiff" or roi_source.candidate_mode == "dense_tile_temporal"


def roi_bank_source_metrics(source: ROIBankSource) -> dict[str, object]:
    return {
        "roi_bank_source_mode": source.mode,
        "roi_bank_raw_path": source.raw_path,
        "roi_bank_candidate_mode": source.candidate_mode,
        **({"roi_bank_source_alias": source.alias} if source.alias is not None else {}),
        **({"roi_bank_frame_range": list(source.frame_range)} if source.frame_range is not None else {}),
    }


def roi_conditioning_context(train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    online_cfg = train_cfg.get("online_generation") if isinstance(train_cfg.get("online_generation"), Mapping) else {}
    providers: dict[str, FullResZernikeConditioning] = {}
    entries = online_cfg.get("dual_domain_coeff_maps", ())
    if isinstance(entries, (list, tuple)):
        for item in entries:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name", f"domain{len(providers)}"))
            path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
            if path:
                providers[name] = FullResZernikeConditioning.from_npz(str(path))
    return {
        "providers": providers or None,
        "append_domain_onehot": bool(online_cfg.get("append_domain_onehot", False)),
        "domain_names": tuple(providers.keys()),
        "condition_feature_dim": online_cfg.get("condition_feature_dim"),
        "condition_dim": online_cfg.get("condition_dim"),
    }


def camera_backward_from_training_config(train_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    camera_cfg = train_cfg.get("camera") if isinstance(train_cfg.get("camera"), Mapping) else {}
    normalization = train_cfg.get("normalization") if isinstance(train_cfg.get("normalization"), Mapping) else {}
    params: dict[str, Any] = {}
    for key in ("qe", "e_per_adu", "em_gain", "spurious_charge"):
        if key in camera_cfg:
            params[key] = float(camera_cfg[key])
        elif key in normalization:
            params[key] = float(normalization[key])
    if str(camera_cfg.get("baseline_mode", "")).strip():
        params["baseline_mode"] = str(camera_cfg["baseline_mode"])
    for key in ("baseline_percentile", "baseline_frame_range", "baseline_by_domain"):
        if key in camera_cfg:
            params[key] = camera_cfg[key]
    if "baseline" in camera_cfg:
        params["baseline"] = float(camera_cfg["baseline"])
    elif "baseline_adu" in camera_cfg:
        params["baseline"] = float(camera_cfg["baseline_adu"])
    elif "baseline" in normalization:
        params["baseline"] = float(normalization["baseline"])
    elif "baseline_adu" in normalization:
        params["baseline"] = float(normalization["baseline_adu"])
    if not params:
        return None
    resolved = {
        "baseline": float(params.get("baseline", 0.0)),
        "e_per_adu": float(params.get("e_per_adu", 1.0)),
        "em_gain": float(params.get("em_gain", 1.0)),
        "qe": float(params.get("qe", 1.0)),
        "spurious_charge": float(params.get("spurious_charge", 0.0)),
    }
    for key in ("baseline_mode", "baseline_percentile", "baseline_frame_range", "baseline_by_domain"):
        if key in params:
            resolved[key] = params[key]
    return resolved


def roi_bank_camera_backward_metadata(bank_metadata: Mapping[str, Any], configured: Mapping[str, Any] | None) -> dict[str, object]:
    if not configured:
        return {}
    resolved = bank_metadata.get("camera_backward") if isinstance(bank_metadata, Mapping) else None
    return {"camera_backward": resolved if isinstance(resolved, Mapping) else dict(configured)}


def posterior_photon_scale(train_cfg: Mapping[str, Any]) -> float | None:
    normalization = train_cfg.get("normalization")
    if isinstance(normalization, Mapping) and "photon_scale" in normalization:
        return float(normalization["photon_scale"])
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping):
        for key in ("photon_max", "phot_max", "ph_scale"):
            if key in scaling:
                return float(scaling[key])
    return None


def posterior_z_scale(train_cfg: Mapping[str, Any]) -> float | None:
    scaling = train_cfg.get("scaling")
    return None if not isinstance(scaling, Mapping) or "z_max" not in scaling else float(scaling["z_max"])


def posterior_bg_scale(train_cfg: Mapping[str, Any]) -> float:
    scaling = train_cfg.get("scaling")
    if isinstance(scaling, Mapping) and "bg_max" in scaling:
        return float(scaling["bg_max"])
    normalization = train_cfg.get("normalization")
    return float(normalization["bg_scale"]) if isinstance(normalization, Mapping) and "bg_scale" in normalization else 1.0


def posterior_input_offset(train_cfg: Mapping[str, Any]) -> float:
    for section in (train_cfg.get("scaling"), train_cfg.get("normalization")):
        if isinstance(section, Mapping) and "input_offset" in section:
            return float(section["input_offset"])
    return 0.0


def posterior_input_scale(train_cfg: Mapping[str, Any]) -> float:
    for section in (train_cfg.get("scaling"), train_cfg.get("normalization")):
        if isinstance(section, Mapping) and "input_scale" in section:
            return float(section["input_scale"])
    return 1.0


def phase_retrieval_tiff_path(config: Mapping[str, Any]) -> str | None:
    phase_cfg = config.get("phase_retrieval")
    if not isinstance(phase_cfg, Mapping):
        return None
    input_cfg = phase_cfg.get("input")
    return str(input_cfg["tiff_path"]) if isinstance(input_cfg, Mapping) and input_cfg.get("tiff_path") is not None else None


def resolve_config_path(value: str, *, base_dir: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def frame_range_tuple(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("ROI-bank frame_range must contain exactly two values")
    return int(value[0]), int(value[1])


def domains_tuple(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if value is None:
        return None
    return tuple(mapping(item, "roi_bank_source.domains[]") for item in value)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value
