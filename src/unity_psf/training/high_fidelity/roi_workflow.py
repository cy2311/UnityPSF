from __future__ import annotations

from typing import Any, Mapping

import torch

from unity_psf.roi_library import ROIBank, ROIBankBuildConfig
from unity_psf.roi_library.loc_harvest import LocHarvestConfig, build_roi_bank_from_loc_harvest
from unity_psf.training.high_fidelity.roi_bank_source import ROIBankSource, auto_build_domains
from unity_psf.training.high_fidelity.roi_bank_builder import roi_bank_camera_backward_metadata


def build_roi_bank_from_loc_harvest_source(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: ROIBankSource,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
    config: ROIBankBuildConfig,
    roi_size: int,
    input_offset: float,
    input_scale: float,
    photon_scale: float | None,
    z_scale: float | None,
    bg_scale: float | None,
    roi_conditioning_context,
) -> ROIBank:
    harvest_cfg = _build_loc_harvest_config(
        gamma_cfg,
        roi_source=roi_source,
        train_cfg=train_cfg,
        config=config,
        roi_size=roi_size,
        input_offset=input_offset,
        input_scale=input_scale,
        photon_scale=photon_scale,
        z_scale=z_scale,
        bg_scale=bg_scale,
    )
    bank = build_roi_bank_from_loc_harvest(
        model=model,
        config=harvest_cfg,
        condition_context=roi_conditioning_context(train_cfg),
    )
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


def _build_loc_harvest_config(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: ROIBankSource,
    train_cfg: Mapping[str, Any],
    config: ROIBankBuildConfig,
    roi_size: int,
    input_offset: float,
    input_scale: float,
    photon_scale: float | None,
    z_scale: float | None,
    bg_scale: float | None,
):
    return LocHarvestConfig(
        raw_path=roi_source.raw_path,
        domains=auto_build_domains(gamma_cfg, roi_size=roi_size, roi_source=roi_source),
        roi_bank_config=config,
        input_offset=input_offset,
        input_scale=input_scale,
        photon_scale=photon_scale or 1.0,
        z_scale=z_scale or 1.0,
        bg_scale=bg_scale,
        normalization_config={"normalization": dict(train_cfg.get("normalization", {}))}
        if isinstance(train_cfg.get("normalization"), Mapping)
        else None,
        candidate_probability_threshold=float(
            gamma_cfg.get(
                "posterior_candidate_probability_threshold",
                gamma_cfg.get("roi_bank_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3)),
            )
        ),
        probability_threshold=float(gamma_cfg.get("roi_bank_probability_threshold", gamma_cfg.get("probability_threshold", 0.5))),
        split_threshold=float(gamma_cfg.get("posterior_adjacent_probability_threshold", gamma_cfg.get("split_threshold", 0.6))),
        phot_min=float(gamma_cfg.get("roi_bank_phot_min", gamma_cfg.get("phot_min", 100.0))),
        sigma_max_px=float(gamma_cfg.get("roi_bank_sigma_max_px", gamma_cfg.get("sigma_max_px", 2.5))),
        tile_size_px=int(
            gamma_cfg.get(
                "roi_bank_harvest_tile_size_px",
                gamma_cfg.get("roi_bank_tile_size_px", gamma_cfg.get("roi_bank_spatial_tile_size", 64)),
            )
        ),
        overlap_px=int(
            gamma_cfg.get(
                "roi_bank_harvest_overlap_px",
                gamma_cfg.get("roi_bank_tile_overlap_px", gamma_cfg.get("roi_bank_spatial_overlap_px", 32)),
            )
        ),
        max_emitters_per_window=(
            None
            if gamma_cfg.get("roi_bank_infer_max_emitters") is None
            else int(gamma_cfg.get("roi_bank_infer_max_emitters"))
        ),
    )
