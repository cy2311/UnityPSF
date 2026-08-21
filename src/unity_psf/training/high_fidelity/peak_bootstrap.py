from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from unity_psf.peak import PeakBootstrapConfig, run_peak_bootstrap_pipeline
from unity_psf.training.channel_context import ASTIGMATISM_660NM_ANCHOR_PROFILE
from unity_psf.training.high_fidelity.diagnostics import _path_token
from unity_psf.training.high_fidelity.roi_bank_source import phase_retrieval_tiff_path


@dataclass(frozen=True)
class _DomainPeakLayout:
    base: Any
    domain_name: str

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def stage_dir(self, stage: str) -> Path:
        if str(stage) == "peak":
            return self.base.stage_dir("peak") / _path_token(self.domain_name)
        return self.base.stage_dir(stage)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def is_single_astigmatism_runtime(train_cfg: Mapping[str, Any]) -> bool:
    model_cfg = train_cfg.get("model")
    if isinstance(model_cfg, Mapping) and str(model_cfg.get("name", "")).strip().lower() == "astigmatism_expert":
        return True
    expert_cfg = train_cfg.get("expert")
    if not isinstance(expert_cfg, Mapping):
        return False
    return str(expert_cfg.get("name", expert_cfg.get("expert_type", ""))).strip().lower() in {"astigmatism", "astigmatism_expert"}


def is_single_channel_runtime(train_cfg: Mapping[str, Any]) -> bool:
    layout_cfg = train_cfg.get("channel_layout")
    channels = layout_cfg.get("channels", layout_cfg.get("measurement_channels")) if isinstance(layout_cfg, Mapping) else None
    if not isinstance(channels, (list, tuple)) or len(channels) != 1:
        return False
    channel = channels[0]
    channel_id = channel.get("id", channel.get("channel_id")) if isinstance(channel, Mapping) else channel
    expert_cfg = train_cfg.get("expert")
    if not isinstance(expert_cfg, Mapping):
        return False
    expert_channel = expert_cfg.get("channel_id", expert_cfg.get("instance_id"))
    return expert_channel is not None and str(expert_channel) == str(channel_id)


def single_channel_peak_domain(train_cfg: Mapping[str, Any], domains: list[object]) -> dict[str, Any]:
    expert = train_cfg.get("expert") if isinstance(train_cfg.get("expert"), Mapping) else {}
    channel_id = str(expert.get("channel_id", "main"))
    layout_cfg = train_cfg.get("channel_layout")
    channels = layout_cfg.get("channels", layout_cfg.get("measurement_channels")) if isinstance(layout_cfg, Mapping) else None
    channel_cfg: Mapping[str, Any] = {}
    if isinstance(channels, (list, tuple)):
        for item in channels:
            if isinstance(item, Mapping) and str(item.get("id", item.get("channel_id", ""))) == channel_id:
                channel_cfg = item
                break
    crop = channel_cfg.get("crop")
    if isinstance(crop, (list, tuple)) and len(crop) == 4:
        x, y, width, height = (int(value) for value in crop)
        return {"name": channel_id, "crop_left": x, "crop_top": y, "crop_width": width, "crop_height": height}
    for item in domains:
        if isinstance(item, Mapping) and str(item.get("name", "")) == channel_id:
            return {**dict(item), "name": channel_id}
    if len(domains) != 1:
        raise ValueError(f"single-channel peak bootstrap has no matching domain for channel={channel_id!r}")
    if not isinstance(domains[0], Mapping):
        raise ValueError("train.real_tiff_wake.domains entries must be mappings")
    return {**dict(domains[0]), "name": channel_id}


def peak_bootstrap_config(bootstrap_cfg: Mapping[str, Any], *, domain: Mapping[str, Any], name: str, tiff_path: Path, anchor_profile=None) -> PeakBootstrapConfig:
    return PeakBootstrapConfig(
        sample=str(bootstrap_cfg.get("sample", "microtube")), side=name,
        frame_range=(int(bootstrap_cfg.get("frame_start", 0)), int(bootstrap_cfg.get("frame_stop", 100))),
        tiff_path=tiff_path, crop_x0=int(domain.get("crop_left", 0)), crop_x1=int(domain.get("crop_left", 0)) + int(domain["crop_width"]),
        crop_y0=int(domain.get("crop_top", 0)), crop_y1=int(domain.get("crop_top", 0)) + int(domain["crop_height"]),
        max_emitters=int(bootstrap_cfg.get("max_emitters", 1000)), target_selected_emitters=int(bootstrap_cfg.get("target_selected_emitters", 0)),
        min_distance_px=float(bootstrap_cfg.get("min_distance_px", 15.0)), gaussian_sigma_px=float(bootstrap_cfg.get("gaussian_sigma_px", 1.0)), threshold_sigma=float(bootstrap_cfg.get("threshold_sigma", 5.0)), patch_size_px=int(bootstrap_cfg.get("patch_size_px", 15)), nat_config_kind=str(bootstrap_cfg.get("nat_config_kind", "order1")), alternating_rounds=int(bootstrap_cfg.get("alternating_rounds", 3)), alternating_local_steps=int(bootstrap_cfg.get("alternating_local_steps", 2)), alternating_global_steps=int(bootstrap_cfg.get("alternating_global_steps", 2)), alternating_local_warmup_rounds=int(bootstrap_cfg.get("alternating_local_warmup_rounds", 0)), alternating_local_warmup_steps=int(bootstrap_cfg.get("alternating_local_warmup_steps", 0)), alternating_optimizer_kind=str(bootstrap_cfg.get("alternating_optimizer_kind", "lm")), formal_export_stage=str(bootstrap_cfg.get("formal_export_stage", "alternating")), global_projected_min_distance_px=float(bootstrap_cfg.get("global_projected_min_distance_px", 10.0)), spatial_balance_grid_px=int(bootstrap_cfg.get("spatial_balance_grid_px", 100)), spatial_balance_max_per_cell=int(bootstrap_cfg.get("spatial_balance_max_per_cell", 0)), max_patch_peak_distance_px=float(bootstrap_cfg.get("max_patch_peak_distance_px", 2.5)), max_secondary_peak_fraction=float(bootstrap_cfg.get("max_secondary_peak_fraction", 0.45)), min_center_peak_norm=float(bootstrap_cfg.get("min_center_peak_norm", 0.0)), min_signal_sum_norm=float(bootstrap_cfg.get("min_signal_sum_norm", 0.0)), ncc_threshold=float(bootstrap_cfg.get("ncc_threshold", 0.7)), freeze_initial_astig_standard=bool(bootstrap_cfg.get("freeze_initial_astig_standard", False)), freeze_defocus_zero_gauge=bool(bootstrap_cfg.get("freeze_defocus_zero_gauge", True)), vectorfit_astig_gauge=bool(bootstrap_cfg.get("vectorfit_astig_gauge", True)), vectorfit_astig_anchor_nm=bootstrap_cfg.get("vectorfit_astig_anchor_nm", None if anchor_profile is None else float(anchor_profile.anchor_nm)), vectorfit_astig_anchor_mode=str(bootstrap_cfg.get("vectorfit_astig_anchor_mode", "init_only")), vectorfit_phasor_z_init=bool(bootstrap_cfg.get("vectorfit_phasor_z_init", True)), include_fixed_astig_baseline=bool(bootstrap_cfg.get("include_fixed_astig_baseline", False)),
    )


def run_peak_zmap_bootstrap_if_enabled(config: Mapping[str, Any], *, config_base_dir: Path, layout) -> dict[str, Any]:
    train_cfg = _mapping(config.get("train"), "train")
    bootstrap_cfg = train_cfg.get("peak_zmap_bootstrap")
    if not isinstance(bootstrap_cfg, Mapping) or bootstrap_cfg.get("enabled") is not True:
        return dict(config)
    real_tiff_cfg = _mapping(train_cfg.get("real_tiff_wake"), "train.real_tiff_wake")
    raw_path = real_tiff_cfg.get("tiff_path") or phase_retrieval_tiff_path(config)
    if raw_path is None:
        raise ValueError("train.peak_zmap_bootstrap.enabled=True requires train.real_tiff_wake.tiff_path")
    tiff_path = _resolve_path(str(raw_path), base_dir=config_base_dir)
    domains = real_tiff_cfg.get("domains")
    if not isinstance(domains, list) or not domains:
        generation = _mapping(train_cfg.get("online_generation"), "train.online_generation")
        domains = [{"name": "left", "crop_left": 0, "crop_top": 0, "crop_width": int(generation.get("width", 128)), "crop_height": int(generation.get("height", 128))}]
    if is_single_channel_runtime(train_cfg):
        domains = [single_channel_peak_domain(train_cfg, domains)]
    coeff_maps, summaries = [], {}
    for idx, raw_domain in enumerate(domains):
        domain = _mapping(raw_domain, "train.real_tiff_wake.domains[]")
        name = str(domain.get("name", f"domain{idx}"))
        result = run_peak_bootstrap_pipeline(layout=_DomainPeakLayout(layout, name), config=peak_bootstrap_config(bootstrap_cfg, domain=domain, name=name, tiff_path=tiff_path, anchor_profile=ASTIGMATISM_660NM_ANCHOR_PROFILE if is_single_astigmatism_runtime(train_cfg) else None))
        coeff_path = Path(result.export["coeff_map_path"]).resolve()
        coeff_maps.append({"name": name, "coeff_maps_npz": str(coeff_path)})
        summaries[name] = {"summary_path": str(result.artifacts.summary_path), "coeff_map_path": str(coeff_path), "zmap_path": str(Path(result.export["zmap_path"]).resolve()), "selected_emitters": int(result.summary.selected_emitters), "kept_count": int(result.summary.kept_count)}
    updated = dict(config); updated_train = dict(train_cfg); online_cfg = dict(_mapping(updated_train.get("online_generation"), "train.online_generation")); online_cfg["dual_domain_coeff_maps"] = coeff_maps; updated_train["online_generation"] = online_cfg; updated["train"] = updated_train
    updated["metadata"] = {**dict(updated.get("metadata", {})), "peak_zmap_bootstrap": {"source": "raw_tiff_peak_bootstrap", "tiff_path": str(tiff_path), "domains": summaries}}
    return updated
