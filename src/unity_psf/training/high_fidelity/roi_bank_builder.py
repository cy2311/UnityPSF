"""ROI-bank construction orchestration for high-fidelity training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

import torch

from unity_psf.roi_library import (
    ROIBank,
    ROIBankBuildConfig,
    build_roi_bank_from_inference,
)
from unity_psf.training.high_fidelity.roi_bank_source import ROIBankSource, auto_build_domains
from unity_psf.training.high_fidelity.roi_inference import (
    bright_pixel_infer_fn,
    model_raw_tiff_infer_fn,
    uses_model_infer,
)


def roi_bank_camera_backward_metadata(
    bank_metadata: Mapping[str, Any],
    configured: Mapping[str, Any] | None,
) -> dict[str, object]:
    if not configured:
        return {}
    resolved = bank_metadata.get("camera_backward") if isinstance(bank_metadata, Mapping) else None
    return {"camera_backward": resolved if isinstance(resolved, Mapping) else dict(configured)}


def roi_bank_infer_fn(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: ROIBankSource,
    model: torch.nn.Module,
    window_size: int,
    train_cfg: Mapping[str, Any],
    posterior_photon_scale: Callable[[Mapping[str, Any]], float | None],
    posterior_z_scale_nm: Callable[[Mapping[str, Any]], float | None],
    roi_conditioning_context: Callable[[Mapping[str, Any]], dict[str, Any]],
):
    if not uses_model_infer(roi_source):
        return bright_pixel_infer_fn
    return model_raw_tiff_infer_fn(
        model=model,
        threshold=float(
            gamma_cfg.get("roi_bank_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))
        ),
        max_emitters=int(
            gamma_cfg.get(
                "roi_bank_infer_max_emitters",
                gamma_cfg.get("num_posterior_samples", gamma_cfg.get("target_projected_emitters", 2)),
            )
        ),
        seed=int(gamma_cfg.get("seed", 0)),
        expected_channels=int(gamma_cfg.get("roi_bank_infer_channels", window_size)),
        photon_scale=posterior_photon_scale(train_cfg),
        z_scale_nm=posterior_z_scale_nm(train_cfg),
        condition_context=roi_conditioning_context(train_cfg),
    )


def auto_build_roi_bank(
    gamma_cfg: Mapping[str, Any],
    *,
    roi_source: ROIBankSource,
    model: torch.nn.Module,
    train_cfg: Mapping[str, Any],
    camera_backward_from_training_config: Callable[[Mapping[str, Any]], dict[str, Any] | None],
    posterior_photon_scale: Callable[[Mapping[str, Any]], float | None],
    posterior_z_scale_nm: Callable[[Mapping[str, Any]], float | None],
    roi_conditioning_context: Callable[[Mapping[str, Any]], dict[str, Any]],
    loc_harvest_builder: Callable[..., ROIBank],
) -> ROIBank:
    roi_size = int(gamma_cfg.get("roi_size_px", 8))
    camera_backward = camera_backward_from_training_config(train_cfg)
    config = ROIBankBuildConfig(
        roi_size_px=roi_size,
        window_size=int(gamma_cfg.get("roi_bank_window_size", 3)),
        frame_range=roi_source.frame_range,
        grid_shape=_grid_shape(gamma_cfg),
        max_rois=int(
            gamma_cfg.get(
                "roi_library_max_rois",
                gamma_cfg.get("roi_bank_max_rois", gamma_cfg.get("target_projected_emitters", 2)),
            )
        ),
        target_emitters=int(gamma_cfg.get("target_projected_emitters", 2)),
        candidate_probability_threshold=float(
            gamma_cfg.get("roi_bank_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))
        ),
        probability_threshold=float(gamma_cfg.get("roi_bank_probability_threshold", gamma_cfg.get("probability_threshold", 0.5))),
        max_overlap_fraction=float(gamma_cfg.get("roi_bank_max_overlap_fraction", 0.95)),
        seed=int(gamma_cfg.get("seed", 0)),
        background_smoothing_kernel=int(
            gamma_cfg.get("background_smoothing_kernel", gamma_cfg.get("roi_bank_background_smoothing_kernel", 3))
        ),
        camera_backward=camera_backward,
        over_cut_px=int(gamma_cfg.get("roi_bank_over_cut_px", gamma_cfg.get("over_cut_px", 0))),
        origin_mode=str(gamma_cfg.get("roi_bank_origin_mode", gamma_cfg.get("origin_mode", "emitter_centered"))),
        origin_stride_px=_optional_int(gamma_cfg, "roi_bank_origin_stride_px", "origin_stride_px"),
        valid_core_size_px=_optional_int(gamma_cfg, "roi_bank_valid_core_size_px", "valid_core_size_px"),
    )
    if uses_model_infer(roi_source):
        return loc_harvest_builder(
            gamma_cfg,
            roi_source=roi_source,
            model=model,
            train_cfg=train_cfg,
            config=config,
            roi_size=roi_size,
        )
    bank = build_roi_bank_from_inference(
        raw_frames_photon=roi_source.raw_path,
        domains=auto_build_domains(gamma_cfg, roi_size=roi_size, roi_source=roi_source),
        infer_fn=roi_bank_infer_fn(
            gamma_cfg,
            roi_source=roi_source,
            model=model,
            window_size=int(config.window_size),
            train_cfg=train_cfg,
            posterior_photon_scale=posterior_photon_scale,
            posterior_z_scale_nm=posterior_z_scale_nm,
            roi_conditioning_context=roi_conditioning_context,
        ),
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


def _optional_int(gamma_cfg: Mapping[str, Any], primary: str, legacy: str) -> int | None:
    value = gamma_cfg.get(primary, gamma_cfg.get(legacy))
    return None if value is None else int(value)


def _grid_shape(gamma_cfg: Mapping[str, Any]) -> tuple[int, int]:
    values = tuple(int(v) for v in gamma_cfg.get("roi_bank_grid_shape", (2, 2)))
    if len(values) != 2:
        raise ValueError("roi_bank_grid_shape must contain exactly two values")
    return values


__all__ = ["auto_build_roi_bank", "roi_bank_camera_backward_metadata", "roi_bank_infer_fn"]
