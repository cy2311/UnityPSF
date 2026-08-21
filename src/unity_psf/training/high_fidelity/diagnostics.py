from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import tifffile

from unity_psf.gamma_update import GammaProjectionObjective
from unity_psf.localization.posterior import DetectionPosteriorSamples
from unity_psf.optics.vector_psf import render_vector_psf_bank
from unity_psf.roi_library import ROIBank
from unity_psf.training.high_fidelity.diagnostic_rendering import (
    background_anchored_display_scale as _background_anchored_display_scale,
    ncc_value as _ncc_value,
    poisson_nll_value as _poisson_nll_value,
    render_gamma_monitor_markdown as _render_gamma_monitor_markdown,
    tile_frames_uint8 as _tile_frames_uint8,
    to_uint8 as _to_uint8,
    to_uint8_background_anchored as _to_uint8_background_anchored,
    write_grayscale_png as _write_grayscale_png,
    write_raw_vs_reconstruction_png as _write_raw_vs_reconstruction_png,
)
from unity_psf.training.loop import TrainingRunEpochResult, TrainingStepResult


def _write_gamma_monitor_report(
    layout,
    metrics: Mapping[str, object],
    *,
    result: TrainingRunEpochResult | TrainingStepResult,
    raw_frames: torch.Tensor,
    background: torch.Tensor,
    samples: DetectionPosteriorSamples,
    gamma_before: torch.Tensor | None = None,
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    roi_origin_xy_px: torch.Tensor | None = None,
    domain_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    report_dir = (
        layout.artifacts_dir
        / "roi_bank_gamma"
        / f"step_{int(result.global_step):08d}"
        / f"source_{_path_token(metrics.get('artifact_source_group', 'unknown'))}"
        / f"domain_{_path_token(metrics.get('artifact_domain_group', 'unknown'))}"
    )
    summary_path = report_dir / "gamma_alternation_summary.json"
    report_path = report_dir / "gamma_update_monitor.md"
    raw_tiff_png_path = report_dir / "raw_tiff_adu_vs_recon.png"
    observed_png_path = report_dir / "observed_photons_vs_recon.png"
    diagnostics_manifest_path = report_dir / "diagnostics" / "diagnostics_manifest.json"
    resolved_checkpoint_path = result.checkpoint_path
    if resolved_checkpoint_path is None:
        resolved_checkpoint_path = layout.checkpoints_dir / "checkpoint_latest.pt"
    checkpoint_path = None if resolved_checkpoint_path is None else str(resolved_checkpoint_path)
    reconstruction = objective.render_reconstruction(
        background=background[0],
        samples=samples,
        batch_index=0,
        gamma=gamma,
        roi_origin_xy_px=roi_origin_xy_px,
        domain_names=domain_names,
    )
    raw_tiff_frame = _raw_tiff_adu_frame_for_diagnostic(
        metrics,
        samples=samples,
        roi_origin_xy_px=roi_origin_xy_px,
        domain_names=domain_names,
        fallback_shape=raw_frames[0].shape,
    )
    diagnostic_path = raw_tiff_png_path if raw_tiff_frame is not None else observed_png_path
    diagnostic_units = "raw_tiff_adu" if raw_tiff_frame is not None else "camera_corrected_photons"
    diagnostic_frame = raw_tiff_frame if raw_tiff_frame is not None else raw_frames[0]
    _write_raw_vs_reconstruction_png(
        diagnostic_path,
        raw_frame=diagnostic_frame,
        reconstruction=reconstruction,
        background=background[0],
        raw_is_photon=raw_tiff_frame is None,
    )
    if raw_tiff_frame is not None:
        _write_raw_vs_reconstruction_png(
            observed_png_path,
            raw_frame=raw_frames[0],
            reconstruction=reconstruction,
            background=background[0],
            raw_is_photon=True,
        )
    payload = dict(metrics)
    diagnostic_rel = str(diagnostic_path.relative_to(layout.run_dir))
    diagnostics_manifest_rel = str(diagnostics_manifest_path.relative_to(layout.run_dir))
    payload.update(
        {
            "epoch": int(result.epoch),
            "steps_completed": int(metrics["steps"]) if "steps" in metrics else None,
            "checkpoint_path": checkpoint_path,
            "diagnostic_png_path": diagnostic_rel,
            "diagnostic_observed_units": diagnostic_units,
            "diagnostics_manifest_path": diagnostics_manifest_rel,
        }
    )
    _write_diagnostics_manifest(
        diagnostics_manifest_path,
        payload,
        summary_path=summary_path,
        report_path=report_path,
        raw_tiff_vs_recon_path=raw_tiff_png_path if raw_tiff_frame is not None else None,
        observed_vs_recon_path=observed_png_path,
        raw_frame=diagnostic_frame,
        observed_photons_frame=raw_frames[0],
        observed_photons_frames=raw_frames,
        reconstruction=reconstruction,
        gamma_before=gamma_before,
        gamma=gamma,
        objective=objective,
        samples=samples,
        background=background,
        roi_origin_xy_px=roi_origin_xy_px,
        domain_names=domain_names,
        layout=layout,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_gamma_monitor_markdown(payload), encoding="utf-8")
    return {
        "summary_path": str(summary_path.relative_to(layout.run_dir)),
        "report_path": str(report_path.relative_to(layout.run_dir)),
        "checkpoint_path": checkpoint_path,
        "diagnostic_png_path": diagnostic_rel,
        "diagnostic_observed_units": diagnostic_units,
        "diagnostics_manifest_path": diagnostics_manifest_rel,
    }


def _write_diagnostics_manifest(
    path: Path,
    payload: Mapping[str, object],
    *,
    summary_path: Path,
    report_path: Path,
    raw_tiff_vs_recon_path: Path | None,
    observed_vs_recon_path: Path,
    raw_frame: torch.Tensor,
    observed_photons_frame: torch.Tensor,
    observed_photons_frames: torch.Tensor | None = None,
    reconstruction: torch.Tensor,
    gamma_before: torch.Tensor | None = None,
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    samples: DetectionPosteriorSamples | None = None,
    background: torch.Tensor | None = None,
    roi_origin_xy_px: torch.Tensor | None = None,
    domain_names: list[str] | tuple[str, ...] | None = None,
    layout,
) -> None:
    def rel(item: Path) -> str:
        return str(item.relative_to(layout.run_dir))

    diagnostics_dir = path.parent
    zmap = _write_zmap_delta_summary(
        diagnostics_dir / "zmap_before_after",
        payload=payload,
        gamma=gamma,
        objective=objective,
        layout=layout,
    )
    fixed_roi = _write_fixed_roi_recon_smoke(
        diagnostics_dir / "fixed_roi_recon",
        payload=payload,
        raw_frame=observed_photons_frame,
        reconstruction=reconstruction,
        background=background[0] if background is not None and torch.as_tensor(background).ndim == 3 else background,
        layout=layout,
    )
    raw_patch = _write_raw_tiff_patch_recon_smoke(
        diagnostics_dir / "raw_tiff_patch_recon",
        payload=payload,
        raw_frame=raw_frame,
        reconstruction=reconstruction,
        background=background[0] if background is not None and torch.as_tensor(background).ndim == 3 else background,
        layout=layout,
    )
    raw_patch_montage = None
    if samples is not None and background is not None:
        raw_patch_montage = _write_raw_tiff_patch_recon_montage(
            diagnostics_dir / "raw_tiff_patch_recon_montage",
            payload=payload,
            samples=samples,
            background=background,
            gamma=gamma,
            objective=objective,
            roi_origin_xy_px=roi_origin_xy_px,
            domain_names=domain_names,
            observed_photons_frames=observed_photons_frames,
            layout=layout,
        )
    observed_patch = _write_observed_photons_patch_recon_smoke(
        diagnostics_dir / "observed_photons_patch_recon",
        payload=payload,
        observed_frame=observed_photons_frame,
        reconstruction=reconstruction,
        background=background[0] if background is not None and torch.as_tensor(background).ndim == 3 else background,
        layout=layout,
    )
    model_triplet = None
    if gamma_before is not None and samples is not None and background is not None:
        initial_reconstruction = objective.render_reconstruction(
            background=background[0],
            samples=samples,
            batch_index=0,
            gamma=gamma_before,
            roi_origin_xy_px=roi_origin_xy_px,
            domain_names=domain_names,
        )
        model_triplet = _write_raw_initial_latest_triplet(
            diagnostics_dir / "raw_initial_latest_triplet",
            payload=payload,
            raw_frame=raw_frame,
            initial_reconstruction=initial_reconstruction,
            latest_reconstruction=reconstruction,
            layout=layout,
        )
    psf_grid = _write_vector_psf_shape_grid(
        diagnostics_dir / "psf_shape_grid",
        payload=payload,
        gamma=gamma,
        objective=objective,
        layout=layout,
    )
    manifest = {
        "schema_version": "roi_gamma_diagnostics_manifest.v1",
        "epoch": int(payload.get("epoch", 0)),
        "artifact_source_group": payload.get("artifact_source_group"),
        "artifact_domain_group": payload.get("artifact_domain_group"),
        "selected_domain_names": payload.get("selected_domain_names", []),
        "diagnostics": {
            "compact_monitor": {
                "status": "available",
                "summary_path": rel(summary_path),
                "report_path": rel(report_path),
            },
            **(
                {
                    "raw_tiff_adu_vs_recon": {
                        "status": "available",
                        "png_path": rel(raw_tiff_vs_recon_path),
                    }
                }
                if raw_tiff_vs_recon_path is not None
                else {}
            ),
            "observed_photons_vs_recon": {
                "status": "available",
                "png_path": rel(observed_vs_recon_path),
            },
            "zmap_before_after": zmap,
            "fixed_roi_recon": fixed_roi,
            "raw_tiff_patch_recon": raw_patch,
            **({"raw_tiff_patch_recon_montage": raw_patch_montage} if raw_patch_montage is not None else {}),
            "observed_photons_patch_recon": observed_patch,
            **({"raw_initial_latest_triplet": model_triplet} if model_triplet is not None else {}),
            "psf_shape_grid": psf_grid,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _raw_tiff_adu_frame_for_diagnostic(
    payload: Mapping[str, object],
    *,
    samples: DetectionPosteriorSamples,
    roi_origin_xy_px: torch.Tensor | None,
    domain_names: list[str] | tuple[str, ...] | None,
    fallback_shape: torch.Size | tuple[int, ...],
    batch_index: int = 0,
) -> torch.Tensor | None:
    raw_path = payload.get("roi_bank_raw_path")
    if raw_path is None:
        return None
    frame_indices = samples.metadata.get("frame_index") if isinstance(samples.metadata, Mapping) else None
    if not isinstance(frame_indices, (list, tuple)) or not frame_indices:
        return None
    if roi_origin_xy_px is None or int(roi_origin_xy_px.numel()) < 2:
        return None
    shape = tuple(int(v) for v in fallback_shape)
    if len(shape) != 2:
        return None
    height, width = shape
    try:
        sample_index = int(batch_index)
        if sample_index < 0 or sample_index >= len(frame_indices):
            return None
        frame_index = int(frame_indices[sample_index])
        origin = roi_origin_xy_px.detach().cpu().to(dtype=torch.float32)
        if origin.ndim == 2:
            if sample_index >= int(origin.shape[0]):
                return None
            x0 = int(round(float(origin[sample_index, 0].item())))
            y0 = int(round(float(origin[sample_index, 1].item())))
        else:
            x0 = int(round(float(origin[0].item())))
            y0 = int(round(float(origin[1].item())))
        domain = str(domain_names[sample_index]).strip().lower() if domain_names and sample_index < len(domain_names) else ""
        with tifffile.TiffFile(str(raw_path)) as tif:
            frame = np.asarray(tif.series[0].asarray(key=frame_index), dtype=np.float32)
        if frame.ndim != 2:
            frame = np.squeeze(frame)
        if frame.ndim != 2:
            return None
        x_offset = _diagnostic_domain_x_offset(domain, frame_width=int(frame.shape[1]), x0=x0, crop_width=width)
        crop = frame[y0 : y0 + height, x_offset + x0 : x_offset + x0 + width]
        if crop.shape != (height, width):
            return None
        return torch.as_tensor(np.ascontiguousarray(crop), dtype=torch.float32)
    except Exception:
        return None


def _diagnostic_domain_x_offset(domain: str, *, frame_width: int, x0: int, crop_width: int) -> int:
    if domain in {"right", "r", "domain_right"}:
        half_width = int(frame_width) // 2
        if int(x0) >= half_width:
            return 0
        if int(x0) + int(crop_width) <= half_width:
            return half_width
    return 0


def _write_zmap_delta_summary(
    path: Path,
    *,
    payload: Mapping[str, object],
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    layout,
) -> dict[str, object]:
    stack = objective.nat_config
    zmap = torch.zeros((len(stack.aberrations), 8, 8), dtype=torch.float32)
    xs = torch.linspace(0.5, float(objective.config.image_size_x) - 0.5, 8)
    ys = torch.linspace(0.5, float(objective.config.image_size_y) - 0.5, 8)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    roixy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    domain_name = _diagnostic_domain_name(payload, objective)
    coeffs = objective.coefficients_at(gamma=gamma, full_xy_px=roixy, domain_name=domain_name).detach().cpu()
    zmap = coeffs.reshape(8, 8, len(stack.aberrations)).permute(2, 0, 1).contiguous()
    mode_delta = zmap.abs().mean(dim=(1, 2))
    dominant = int(torch.argmax(mode_delta).item()) if mode_delta.numel() else 0
    png_path = path / "zmap_delta_vector_nat.png"
    summary_path = path / "delta_gamma_physical_zmap_before_after_summary.json"
    _write_grayscale_png(png_path, _tile_frames_uint8([zmap[index] for index in range(min(3, int(zmap.shape[0])))]))
    summary = {
        "schema_version": "roi_gamma_vector_nat_zmap_delta.v1",
        "epoch": int(payload.get("epoch", 0)),
        "artifact_domain_group": payload.get("artifact_domain_group"),
        "delta_abs_mean": float(zmap.abs().mean().item()),
        "delta_abs_max": float(zmap.abs().max().item()),
        "dominant_delta_mode": dominant,
        "mode_order": [list(mode) for mode in stack.aberrations],
        "gamma_size": int(gamma.numel()),
        "coeff_domain": domain_name,
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_fixed_roi_recon_smoke(
    path: Path,
    *,
    payload: Mapping[str, object],
    raw_frame: torch.Tensor,
    reconstruction: torch.Tensor,
    background: torch.Tensor | None = None,
    layout,
) -> dict[str, object]:
    residual = (raw_frame.detach().cpu() - reconstruction.detach().cpu()).abs()
    png_path = path / "fixed_roi_recon_smoke.png"
    summary_path = path / "fixed_roi_recon_summary.json"
    display_scale = None
    if background is not None:
        display_scale = _background_anchored_display_scale([raw_frame, reconstruction], [background, background])
        canvas = np.concatenate(
            [
                _to_uint8_background_anchored(raw_frame.detach().cpu(), background, display_scale),
                _to_uint8_background_anchored(reconstruction.detach().cpu(), background, display_scale),
                _to_uint8(residual.detach().cpu()),
            ],
            axis=1,
        )
    else:
        canvas = _tile_frames_uint8([raw_frame, reconstruction, residual])
    _write_grayscale_png(png_path, canvas)
    rms = float(torch.sqrt((residual.to(dtype=torch.float32).square()).mean()).item())
    summary = {
        "schema_version": "roi_gamma_fixed_roi_recon_smoke.v1",
        "epoch": int(payload.get("epoch", 0)),
        "selected_roi_count": int(payload.get("roi_count", 0)),
        "rendered_count": 1,
        "poisson_nll": _poisson_nll_value(raw_frame, reconstruction),
        "rms": rms,
        "display_scale": display_scale,
        "panels": ["corrected_camera_photon", "reconstruction_with_background", "absolute_residual"],
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_raw_tiff_patch_recon_smoke(
    path: Path,
    *,
    payload: Mapping[str, object],
    raw_frame: torch.Tensor,
    reconstruction: torch.Tensor,
    background: torch.Tensor | None = None,
    layout,
) -> dict[str, object]:
    residual = raw_frame.detach().cpu().to(dtype=torch.float32) - reconstruction.detach().cpu().to(dtype=torch.float32)
    png_path = path / "raw_tiff_patch_recon_smoke.png"
    summary_path = path / "raw_tiff_patch_recon_summary.json"
    display_scale = None
    if background is not None:
        display_scale = _background_anchored_display_scale([reconstruction], [background])
        canvas = np.concatenate(
            [
                _to_uint8(raw_frame.detach().cpu()),
                _to_uint8_background_anchored(reconstruction.detach().cpu(), background, display_scale),
                _to_uint8(residual.abs().detach().cpu()),
            ],
            axis=1,
        )
    else:
        canvas = _tile_frames_uint8([raw_frame, reconstruction, residual.abs()])
    _write_grayscale_png(png_path, canvas)
    summary = {
        "schema_version": "roi_gamma_raw_tiff_patch_recon_smoke.v1",
        "epoch": int(payload.get("epoch", 0)),
        "selected_patch_count": 1,
        "observed_units": "raw_tiff_adu",
        "note": "Raw TIFF ADU and reconstruction are not in the same physical units; this diagnostic is visual only. Reconstruction uses background-anchored display when background is available.",
        "display_scale": display_scale,
        "panels": ["raw_tiff_adu", "reconstruction_with_background", "absolute_residual_unit_mixed"],
        "ncc": _ncc_value(raw_frame, reconstruction),
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_raw_tiff_patch_recon_montage(
    path: Path,
    *,
    payload: Mapping[str, object],
    samples: DetectionPosteriorSamples,
    background: torch.Tensor,
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    roi_origin_xy_px: torch.Tensor | None,
    domain_names: list[str] | tuple[str, ...] | None,
    observed_photons_frames: torch.Tensor | None,
    layout,
) -> dict[str, object] | None:
    batch_count = int(samples.xyzph.shape[0])
    if batch_count <= 0:
        return None
    requested = int(payload.get("diagnostic_raw_tiff_patch_recon_montage_count", 5) or 5)
    rendered_count = max(1, min(requested, batch_count))
    if rendered_count == 1:
        indices = [0]
    else:
        indices = sorted({int(round(v)) for v in np.linspace(0, batch_count - 1, num=rendered_count)})

    row_payloads: list[dict[str, torch.Tensor]] = []
    rendered_indices: list[int] = []
    ncc_values: list[float] = []
    roi_ids: list[object] = []
    frame_indices: list[object] = []
    for index in indices:
        bkg = torch.as_tensor(background, dtype=torch.float32)
        frame_shape = tuple(int(v) for v in bkg[index].shape[-2:]) if bkg.ndim == 3 else tuple(int(v) for v in bkg.shape[-2:])
        raw_frame = _raw_tiff_adu_frame_for_diagnostic(
            payload,
            samples=samples,
            roi_origin_xy_px=roi_origin_xy_px,
            domain_names=domain_names,
            fallback_shape=frame_shape,
            batch_index=index,
        )
        if raw_frame is None:
            continue
        reconstruction = objective.render_reconstruction(
            background=background[index] if torch.as_tensor(background).ndim == 3 else background,
            samples=samples,
            batch_index=index,
            gamma=gamma,
            roi_origin_xy_px=roi_origin_xy_px,
            domain_names=domain_names,
        )
        corrected = None
        if observed_photons_frames is not None:
            observed_all = torch.as_tensor(observed_photons_frames, dtype=torch.float32)
            if observed_all.ndim == 3 and index < int(observed_all.shape[0]):
                corrected = observed_all[index]
        if corrected is None:
            corrected = raw_frame
        residual = corrected.detach().cpu().to(dtype=torch.float32) - reconstruction.detach().cpu().to(dtype=torch.float32)
        bkg_frame = background[index] if torch.as_tensor(background).ndim == 3 else background
        row_payloads.append(
            {
                "raw_frame": raw_frame.detach().cpu(),
                "corrected": corrected.detach().cpu(),
                "reconstruction": reconstruction.detach().cpu(),
                "residual_abs": residual.abs().detach().cpu(),
                "background": torch.as_tensor(bkg_frame).detach().cpu(),
            }
        )
        rendered_indices.append(index)
        ncc_values.append(_ncc_value(corrected, reconstruction))
        roi_ids.append(_metadata_item(samples.metadata.get("roi_id") if isinstance(samples.metadata, Mapping) else None, index))
        frame_indices.append(_metadata_item(samples.metadata.get("frame_index") if isinstance(samples.metadata, Mapping) else None, index))

    if not row_payloads:
        return None
    photon_frames = [item["corrected"] for item in row_payloads] + [item["reconstruction"] for item in row_payloads]
    photon_backgrounds = [item["background"] for item in row_payloads] + [item["background"] for item in row_payloads]
    display_scale = _background_anchored_display_scale(
        photon_frames,
        photon_backgrounds,
    )
    rows: list[np.ndarray] = []
    for item in row_payloads:
        background_frame = item["background"]
        rows.append(
            np.concatenate(
                [
                    _to_uint8(torch.as_tensor(item["raw_frame"])),
                    _to_uint8_background_anchored(torch.as_tensor(item["corrected"]), torch.as_tensor(background_frame), display_scale),
                    _to_uint8_background_anchored(
                        torch.as_tensor(item["reconstruction"]),
                        torch.as_tensor(background_frame),
                        display_scale,
                    ),
                    _to_uint8(torch.as_tensor(item["residual_abs"])),
                ],
                axis=1,
            )
        )
    spacer = np.full((4, int(rows[0].shape[1])), 255, dtype=np.uint8)
    canvas_parts: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        if row_index:
            canvas_parts.append(spacer)
        canvas_parts.append(row)
    canvas = np.concatenate(canvas_parts, axis=0)
    png_path = path / "raw_tiff_patch_recon_montage.png"
    summary_path = path / "raw_tiff_patch_recon_montage_summary.json"
    _write_grayscale_png(png_path, canvas)
    summary = {
        "schema_version": "roi_gamma_raw_tiff_patch_recon_montage.v1",
        "epoch": int(payload.get("epoch", 0)),
        "selected_patch_count": int(len(rendered_indices)),
        "observed_units": "raw_tiff_adu",
        "note": "Each row is raw TIFF ADU | corrected camera photon | reconstruction with background | photon-domain absolute residual. Corrected/reconstruction panels use a shared background-anchored display scale with background mapped to a fixed gray level.",
        "batch_indices": rendered_indices,
        "roi_ids": roi_ids,
        "frame_indices": frame_indices,
        "ncc": ncc_values,
        "display_scale": display_scale,
        "panels": ["raw_tiff_adu", "corrected_camera_photon", "reconstruction_with_background", "absolute_residual_photon_domain"],
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _metadata_item(value: object, index: int) -> object:
    if isinstance(value, torch.Tensor):
        if index < int(value.numel()):
            item = value.reshape(-1)[index].item()
            if isinstance(item, (bool, np.bool_)):
                return bool(item)
            if isinstance(item, (int, np.integer)):
                return int(item)
            if isinstance(item, (float, np.floating)):
                return int(item) if float(item).is_integer() else float(item)
            return item
        return None
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if index < int(flat.size):
            item = flat[index].item()
            if isinstance(item, (bool, np.bool_)):
                return bool(item)
            if isinstance(item, (int, np.integer)):
                return int(item)
            if isinstance(item, (float, np.floating)):
                return int(item) if float(item).is_integer() else float(item)
            return item
        return None
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return None


def _write_observed_photons_patch_recon_smoke(
    path: Path,
    *,
    payload: Mapping[str, object],
    observed_frame: torch.Tensor,
    reconstruction: torch.Tensor,
    background: torch.Tensor | None = None,
    layout,
) -> dict[str, object]:
    residual = observed_frame.detach().cpu().to(dtype=torch.float32) - reconstruction.detach().cpu().to(dtype=torch.float32)
    png_path = path / "observed_photons_patch_recon_smoke.png"
    summary_path = path / "observed_photons_patch_recon_summary.json"
    display_scale = None
    if background is not None:
        display_scale = _background_anchored_display_scale([observed_frame, reconstruction], [background, background])
        canvas = np.concatenate(
            [
                _to_uint8_background_anchored(observed_frame.detach().cpu(), background, display_scale),
                _to_uint8_background_anchored(reconstruction.detach().cpu(), background, display_scale),
                _to_uint8(residual.abs().detach().cpu()),
            ],
            axis=1,
        )
    else:
        canvas = _tile_frames_uint8([observed_frame, reconstruction, residual.abs()])
    _write_grayscale_png(png_path, canvas)
    summary = {
        "schema_version": "roi_gamma_observed_photons_patch_recon_smoke.v1",
        "epoch": int(payload.get("epoch", 0)),
        "selected_patch_count": 1,
        "observed_units": "camera_corrected_photons",
        "poisson_nll": _poisson_nll_value(observed_frame, reconstruction),
        "mse": float(residual.square().mean().item()),
        "ncc": _ncc_value(observed_frame, reconstruction),
        "display_scale": display_scale,
        "panels": ["corrected_camera_photon", "reconstruction_with_background", "absolute_residual"],
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_raw_initial_latest_triplet(
    path: Path,
    *,
    payload: Mapping[str, object],
    raw_frame: torch.Tensor,
    initial_reconstruction: torch.Tensor,
    latest_reconstruction: torch.Tensor,
    layout,
) -> dict[str, object]:
    delta = latest_reconstruction.detach().cpu().to(dtype=torch.float32) - initial_reconstruction.detach().cpu().to(dtype=torch.float32)
    png_path = path / "raw_initial_latest_triplet.png"
    summary_path = path / "raw_initial_latest_triplet_summary.json"
    _write_grayscale_png(png_path, _tile_frames_uint8([raw_frame, initial_reconstruction, latest_reconstruction]))
    summary = {
        "schema_version": "roi_gamma_raw_initial_latest_triplet.v1",
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "selected_roi_count": int(payload.get("roi_count", 0)),
        "rendered_count": 1,
        "initial_poisson_nll": _poisson_nll_value(raw_frame, initial_reconstruction),
        "latest_poisson_nll": _poisson_nll_value(raw_frame, latest_reconstruction),
        "latest_minus_initial_rms": float(torch.sqrt(delta.square().mean()).item()),
        "png_path": str(png_path.relative_to(layout.run_dir)),
        "panels": ["raw_frame", "initial_physical_model_projection", "latest_updated_physical_model_projection"],
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _write_vector_psf_shape_grid(
    path: Path,
    *,
    payload: Mapping[str, object],
    gamma: torch.Tensor,
    objective: GammaProjectionObjective,
    layout,
) -> dict[str, object]:
    z_values = torch.tensor([-600.0, 0.0, 600.0], dtype=torch.float32)
    center_xy = torch.tensor(
        [[float(objective.config.image_size_x) * 0.5, float(objective.config.image_size_y) * 0.5]],
        dtype=torch.float32,
        device=objective.device,
    )
    domain_name = _diagnostic_domain_name(payload, objective)
    center_coeffs = objective.coefficients_at(gamma=gamma, full_xy_px=center_xy, domain_name=domain_name)
    coeffs = center_coeffs.expand(3, -1).contiguous()
    coeffs_rad = coeffs * (2.0 * np.pi / max(float(objective.config.wavelength_nm), 1e-6)) * objective.ctx.normfac[None, :]
    psf = render_vector_psf_bank(
        objective.ctx,
        coeffs_rad,
        z_values.to(device=objective.device) * 1e-9,
        out_size=int(objective.config.patch_size_px),
        batch_size=int(objective.config.renderer_batch_size),
        return_torch=True,
    ).detach().cpu()
    png_path = path / "vector_psf_zstack.png"
    summary_path = path / "psf_shape_grid_summary.json"
    _write_grayscale_png(png_path, _tile_frames_uint8([psf[index] for index in range(int(psf.shape[0]))]))
    summary = {
        "schema_version": "roi_gamma_vector_psf_zstack.v1",
        "epoch": int(payload.get("epoch", 0)),
        "psf_sum": [float(value) for value in psf.sum(dim=(1, 2)).tolist()],
        "z_nm": [float(value) for value in z_values.tolist()],
        "center_coeffs_nm": [float(value) for value in center_coeffs.detach().cpu().reshape(-1).tolist()],
        "coeff_domain": domain_name,
        "renderer": "vector_psf",
        "png_path": str(png_path.relative_to(layout.run_dir)),
    }
    path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "available", "summary_path": str(summary_path.relative_to(layout.run_dir)), "png_path": str(png_path.relative_to(layout.run_dir))}


def _diagnostic_domain_name(payload: Mapping[str, object], objective: GammaProjectionObjective) -> str | None:
    domain = payload.get("artifact_domain_group")
    if isinstance(domain, str) and domain not in {"multi", "unknown"}:
        return domain
    names = payload.get("selected_domain_names")
    if isinstance(names, list) and names:
        return str(names[0])
    if objective.base_maps_by_domain:
        return next(iter(objective.base_maps_by_domain))
    return None


def _artifact_group_metrics(
    bank: ROIBank,
    *,
    roi_library_source: str | None,
    objective_source: str,
) -> dict[str, object]:
    domains = sorted({str(record.domain_name) for record in bank.records})
    domain_group = domains[0] if len(domains) == 1 else "multi"
    source_group = roi_library_source or objective_source
    return {
        "artifact_source_group": source_group,
        "artifact_domain_group": domain_group,
        "selected_domain_names": domains,
    }


def _path_token(value: object) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    token = "_".join(part for part in "".join(chars).split("_") if part)
    return token or "unknown"
