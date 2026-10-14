from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from unity_psf.peak import PeakBootstrapConfig, run_peak_bootstrap_pipeline
from unity_psf.training.channel_context import ASTIGMATISM_660NM_ANCHOR_PROFILE
from unity_psf.training.high_fidelity.roi_bank_source import (
    phase_retrieval_tiff_path,
    resolve_config_path,
)
from unity_psf.training.high_fidelity.workflow_context import DomainPeakLayout


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def run_peak_zmap_bootstrap_if_enabled(
    config: Mapping[str, Any],
    *,
    config_base_dir: Path,
    layout,
) -> dict[str, Any]:
    train_cfg = _mapping(config.get("train"), "train")
    bootstrap_cfg = train_cfg.get("peak_zmap_bootstrap")
    if not isinstance(bootstrap_cfg, Mapping) or bootstrap_cfg.get("enabled") is not True:
        return dict(config)

    real_tiff_cfg = _mapping(train_cfg.get("real_tiff_wake"), "train.real_tiff_wake")
    raw_path = real_tiff_cfg.get("tiff_path")
    if raw_path is None:
        raw_path = phase_retrieval_tiff_path(config)
    if raw_path is None:
        raise ValueError("train.peak_zmap_bootstrap.enabled=True requires train.real_tiff_wake.tiff_path")
    tiff_path = Path(resolve_config_path(str(raw_path), base_dir=config_base_dir))

    domains = real_tiff_cfg.get("domains")
    if not isinstance(domains, list) or not domains:
        height = int(_mapping(train_cfg.get("online_generation"), "train.online_generation").get("height", 128))
        width = int(_mapping(train_cfg.get("online_generation"), "train.online_generation").get("width", 128))
        domains = [{"name": "left", "crop_left": 0, "crop_top": 0, "crop_width": width, "crop_height": height}]
    if is_single_channel_runtime(train_cfg):
        domains = [single_channel_peak_domain(train_cfg, domains)]

    coeff_maps: list[dict[str, str]] = []
    peak_summaries: dict[str, Any] = {}
    for idx, domain_raw in enumerate(domains):
        domain = _mapping(domain_raw, "train.real_tiff_wake.domains[]")
        name = str(domain.get("name", f"domain{idx}"))
        peak_layout = DomainPeakLayout(layout, name)
        result = run_peak_bootstrap_pipeline(
            layout=peak_layout,
            config=peak_bootstrap_config(
                bootstrap_cfg,
                domain=domain,
                name=name,
                tiff_path=tiff_path,
                anchor_profile=(
                    ASTIGMATISM_660NM_ANCHOR_PROFILE
                    if is_single_astigmatism_runtime(train_cfg)
                    else None
                ),
            ),
        )
        coeff_path = Path(result.export["coeff_map_path"]).resolve()
        zmap_path = Path(result.export["zmap_path"]).resolve()
        coeff_maps.append({"name": name, "coeff_maps_npz": str(coeff_path)})
        peak_summaries[name] = {
            "summary_path": str(result.artifacts.summary_path),
            "coeff_map_path": str(coeff_path),
            "zmap_path": str(zmap_path),
            "selected_emitters": int(result.summary.selected_emitters),
            "kept_count": int(result.summary.kept_count),
        }

    updated = dict(config)
    updated_train = dict(train_cfg)
    online_cfg = dict(_mapping(updated_train.get("online_generation"), "train.online_generation"))
    online_cfg["dual_domain_coeff_maps"] = coeff_maps
    updated_train["online_generation"] = online_cfg
    updated["train"] = updated_train
    updated.setdefault("metadata", {})
    updated["metadata"] = {
        **dict(updated.get("metadata", {})),
        "peak_zmap_bootstrap": {
            "source": "raw_tiff_peak_bootstrap",
            "tiff_path": str(tiff_path),
            "domains": peak_summaries,
        },
    }
    return updated


def is_single_astigmatism_runtime(train_cfg: Mapping[str, Any]) -> bool:
    model_cfg = train_cfg.get("model")
    if isinstance(model_cfg, Mapping) and str(model_cfg.get("name", "")).strip().lower() == "astigmatism_expert":
        return True
    expert_cfg = train_cfg.get("expert")
    if not isinstance(expert_cfg, Mapping):
        return False
    return str(expert_cfg.get("name", expert_cfg.get("expert_type", ""))).strip().lower() in {
        "astigmatism",
        "astigmatism_expert",
    }


def is_single_channel_runtime(train_cfg: Mapping[str, Any]) -> bool:
    layout_cfg = train_cfg.get("channel_layout")
    if not isinstance(layout_cfg, Mapping):
        return False
    channels = layout_cfg.get("channels", layout_cfg.get("measurement_channels"))
    if not isinstance(channels, (list, tuple)) or len(channels) != 1:
        return False
    channel = channels[0]
    if isinstance(channel, Mapping):
        channel_id = channel.get("id", channel.get("channel_id"))
    else:
        channel_id = channel
    expert_cfg = train_cfg.get("expert")
    if not isinstance(expert_cfg, Mapping):
        return False
    expert_channel = expert_cfg.get("channel_id", expert_cfg.get("instance_id"))
    return expert_channel is not None and str(expert_channel) == str(channel_id)


def single_channel_peak_domain(
    train_cfg: Mapping[str, Any],
    domains: list[object],
) -> dict[str, Any]:
    expert_cfg = train_cfg.get("expert")
    expert = expert_cfg if isinstance(expert_cfg, Mapping) else {}
    channel_id = str(expert.get("channel_id", "main"))
    layout_cfg = train_cfg.get("channel_layout")
    channels = (
        layout_cfg.get("channels", layout_cfg.get("measurement_channels"))
        if isinstance(layout_cfg, Mapping)
        else None
    )
    channel_cfg: Mapping[str, Any] = {}
    if isinstance(channels, (list, tuple)):
        for item in channels:
            if isinstance(item, Mapping) and str(item.get("id", item.get("channel_id", ""))) == channel_id:
                channel_cfg = item
                break
    crop = channel_cfg.get("crop")
    if isinstance(crop, (list, tuple)) and len(crop) == 4:
        x, y, width, height = (int(value) for value in crop)
        return {
            "name": channel_id,
            "crop_left": x,
            "crop_top": y,
            "crop_width": width,
            "crop_height": height,
        }
    for item in domains:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("name", "")) == channel_id:
            return {**dict(item), "name": channel_id}
    if len(domains) != 1:
        raise ValueError(
            "single-channel peak bootstrap has no matching domain for "
            f"channel={channel_id!r}; available domains={[str(item.get('name', '')) for item in domains if isinstance(item, Mapping)]!r}"
        )
    selected = domains[0]
    if not isinstance(selected, Mapping):
        raise ValueError("train.real_tiff_wake.domains entries must be mappings")
    return {**dict(selected), "name": channel_id}


def peak_bootstrap_config(
    bootstrap_cfg: Mapping[str, Any],
    *,
    domain: Mapping[str, Any],
    name: str,
    tiff_path: Path,
    anchor_profile=None,
) -> PeakBootstrapConfig:
    frame_start = int(bootstrap_cfg.get("frame_start", 0))
    frame_stop = int(bootstrap_cfg.get("frame_stop", 100))
    crop_left = int(domain.get("crop_left", 0))
    crop_top = int(domain.get("crop_top", 0))
    return PeakBootstrapConfig(
        sample=str(bootstrap_cfg.get("sample", "microtube")),
        side=name,
        frame_range=(frame_start, frame_stop),
        tiff_path=tiff_path,
        crop_x0=crop_left,
        crop_x1=crop_left + int(domain["crop_width"]),
        crop_y0=crop_top,
        crop_y1=crop_top + int(domain["crop_height"]),
        max_emitters=int(bootstrap_cfg.get("max_emitters", 1000)),
        target_selected_emitters=int(bootstrap_cfg.get("target_selected_emitters", 0)),
        min_distance_px=float(bootstrap_cfg.get("min_distance_px", 15.0)),
        gaussian_sigma_px=float(bootstrap_cfg.get("gaussian_sigma_px", 1.0)),
        threshold_sigma=float(bootstrap_cfg.get("threshold_sigma", 5.0)),
        patch_size_px=int(bootstrap_cfg.get("patch_size_px", 15)),
        nat_config_kind=str(bootstrap_cfg.get("nat_config_kind", "order1")),
        alternating_rounds=int(bootstrap_cfg.get("alternating_rounds", 3)),
        alternating_local_steps=int(bootstrap_cfg.get("alternating_local_steps", 2)),
        alternating_global_steps=int(bootstrap_cfg.get("alternating_global_steps", 2)),
        alternating_local_warmup_rounds=int(bootstrap_cfg.get("alternating_local_warmup_rounds", 0)),
        alternating_local_warmup_steps=int(bootstrap_cfg.get("alternating_local_warmup_steps", 0)),
        alternating_optimizer_kind=str(bootstrap_cfg.get("alternating_optimizer_kind", "lm")),
        formal_export_stage=str(bootstrap_cfg.get("formal_export_stage", "alternating")),
        global_projected_min_distance_px=float(bootstrap_cfg.get("global_projected_min_distance_px", 10.0)),
        spatial_balance_grid_px=int(bootstrap_cfg.get("spatial_balance_grid_px", 100)),
        spatial_balance_max_per_cell=int(bootstrap_cfg.get("spatial_balance_max_per_cell", 0)),
        max_patch_peak_distance_px=float(bootstrap_cfg.get("max_patch_peak_distance_px", 2.5)),
        max_secondary_peak_fraction=float(bootstrap_cfg.get("max_secondary_peak_fraction", 0.45)),
        min_center_peak_norm=float(bootstrap_cfg.get("min_center_peak_norm", 0.0)),
        min_signal_sum_norm=float(bootstrap_cfg.get("min_signal_sum_norm", 0.0)),
        ncc_threshold=float(bootstrap_cfg.get("ncc_threshold", 0.7)),
        freeze_initial_astig_standard=bool(bootstrap_cfg.get("freeze_initial_astig_standard", False)),
        freeze_defocus_zero_gauge=bool(bootstrap_cfg.get("freeze_defocus_zero_gauge", True)),
        vectorfit_astig_gauge=bool(bootstrap_cfg.get("vectorfit_astig_gauge", True)),
        vectorfit_astig_anchor_nm=bootstrap_cfg.get(
            "vectorfit_astig_anchor_nm",
            None if anchor_profile is None else float(anchor_profile.anchor_nm),
        ),
        vectorfit_astig_anchor_mode=str(bootstrap_cfg.get("vectorfit_astig_anchor_mode", "init_only")),
        vectorfit_phasor_z_init=bool(bootstrap_cfg.get("vectorfit_phasor_z_init", True)),
        include_fixed_astig_baseline=bool(bootstrap_cfg.get("include_fixed_astig_baseline", False)),
    )
