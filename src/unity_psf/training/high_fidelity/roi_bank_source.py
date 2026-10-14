"""ROI-bank source resolution and conditioning metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.roi_library import ROIBankDomain


@dataclass(frozen=True)
class ROIBankSource:
    mode: str
    raw_path: str
    candidate_mode: str
    frame_range: tuple[int, int] | None
    domains: tuple[Mapping[str, Any], ...] | None
    alias: str | None = None


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
        if "baseline_percentile" in camera_cfg:
            params["baseline_percentile"] = float(camera_cfg["baseline_percentile"])
        if "baseline_frame_range" in camera_cfg:
            frame_range = camera_cfg["baseline_frame_range"]
            if isinstance(frame_range, (list, tuple)) and len(frame_range) == 2:
                params["baseline_frame_range"] = [int(frame_range[0]), int(frame_range[1])]
        if "baseline_by_domain" in camera_cfg and isinstance(camera_cfg["baseline_by_domain"], Mapping):
            params["baseline_by_domain"] = {
                str(key): float(value) for key, value in camera_cfg["baseline_by_domain"].items()
            }
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


def roi_conditioning_context(train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    online_cfg = (
        train_cfg.get("online_generation")
        if isinstance(train_cfg.get("online_generation"), Mapping)
        else {}
    )
    providers: dict[str, FullResZernikeConditioning] = {}
    entries = online_cfg.get("dual_domain_coeff_maps", ())
    if isinstance(entries, list):
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


def auto_build_domains(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_size: int,
    roi_source: ROIBankSource,
) -> tuple[ROIBankDomain, ...]:
    raw_domains = roi_source.domains if roi_source.domains is not None else gamma_cfg.get("auto_build_domains")
    if raw_domains is None:
        size = int(gamma_cfg.get("auto_build_domain_size_px", max(roi_size + 4, roi_size)))
        return (ROIBankDomain("auto", crop_left=0, crop_top=0, crop_width=size, crop_height=size),)
    domains = []
    for item in raw_domains:
        mapping = _mapping(item, "auto_build_domains[]")
        domains.append(
            ROIBankDomain(
                str(mapping.get("name", "auto")),
                crop_left=int(mapping.get("crop_left", 0)),
                crop_top=int(mapping.get("crop_top", 0)),
                crop_width=int(mapping["crop_width"]),
                crop_height=int(mapping["crop_height"]),
            )
        )
    return tuple(domains)


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
        frame_range = source_cfg.get("frame_range", gamma_cfg.get("roi_bank_frame_range"))
        return ROIBankSource(
            mode="auto_build",
            raw_path=resolve_config_path(str(raw_path), base_dir=config_base_dir),
            candidate_mode=str(source_cfg.get("candidate_mode", gamma_cfg.get("roi_bank_candidate_mode", "bright_pixel"))),
            frame_range=frame_range_tuple(frame_range),
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
        real_tiff_cfg = _mapping(train_cfg.get("real_tiff_wake"), "train.real_tiff_wake")
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


def phase_retrieval_tiff_path(config: Mapping[str, Any]) -> str | None:
    phase_cfg = config.get("phase_retrieval")
    if not isinstance(phase_cfg, Mapping):
        return None
    input_cfg = phase_cfg.get("input")
    if isinstance(input_cfg, Mapping) and input_cfg.get("tiff_path") is not None:
        return str(input_cfg["tiff_path"])
    return None


def frame_range_tuple(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("ROI-bank frame_range must contain exactly two values")
    return int(value[0]), int(value[1])


def domains_tuple(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if value is None:
        return None
    return tuple(_mapping(item, "roi_bank_source.domains[]") for item in value)


def resolve_config_path(value: str, *, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def roi_bank_source_metrics(source: ROIBankSource) -> dict[str, object]:
    return {
        "roi_bank_source_mode": source.mode,
        "roi_bank_raw_path": source.raw_path,
        "roi_bank_candidate_mode": source.candidate_mode,
        **({"roi_bank_source_alias": source.alias} if source.alias is not None else {}),
        **({"roi_bank_frame_range": list(source.frame_range)} if source.frame_range is not None else {}),
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value

