from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from unity_psf.config import load_config
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.localization.model import LocalizationModelOutput
from unity_psf.localization.posterior import sample_detection_posterior
from unity_psf.localization.roi_batches import build_roi_batch_provider
from unity_psf.localization.smlm_output import SMLMOutputChannels, decode_smlm_output
from unity_psf.roi_library import ROIBank, save_roi_bank
from unity_psf.runtime import ensure_run_layout
from unity_psf.training import build_trainer_runtime
from unity_psf.training.high_fidelity.engine import _mapping
from unity_psf.training.high_fidelity.peak_bootstrap import run_peak_zmap_bootstrap_if_enabled
from unity_psf.training.high_fidelity.gamma_runtime import build_vector_roi_gamma_objective
from unity_psf.training.high_fidelity.roi_bank_source import (
    auto_build_roi_bank,
    posterior_photon_scale,
    posterior_z_scale,
    resolve_roi_bank_source,
    roi_conditioning_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build only the standard ROI library and write 128x128 smoke diagnostics.")
    parser.add_argument("--config", required=True, help="Resolved YAML config.")
    parser.add_argument("--run-root", default="output", help="Directory containing the smoke run directory.")
    parser.add_argument("--run-name", default="roi_library_smoke", help="Relative run directory name.")
    parser.add_argument("--seed", type=int, default=0, help="Runtime seed.")
    parser.add_argument("--checkpoint", default=None, help="Optional localizer checkpoint to load before ROI harvest.")
    parser.add_argument("--diagnostic-rois", type=int, default=8, help="Number of selected ROI records to render.")
    parser.add_argument("--max-rois", type=int, default=None, help="Optional smoke override for roi_library_max_rois.")
    parser.add_argument("--target-emitters", type=int, default=None, help="Optional smoke override for target_projected_emitters.")
    parser.add_argument("--frame-start", type=int, default=None, help="Optional smoke override for ROI-bank frame start.")
    parser.add_argument("--frame-stop", type=int, default=None, help="Optional smoke override for ROI-bank frame stop.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config_base_dir = config_path.parent
    layout = ensure_run_layout(Path(args.run_root), args.run_name, stage_names=("peak", "roi_library_smoke"))
    config = run_peak_zmap_bootstrap_if_enabled(config, config_base_dir=config_base_dir, layout=layout)
    train_cfg = _mapping(config.get("train"), "train")
    gamma_cfg = _smoke_gamma_cfg(
        _mapping(train_cfg.get("roi_bank_gamma"), "train.roi_bank_gamma"),
        max_rois=args.max_rois,
        target_emitters=args.target_emitters,
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
    )
    runtime_config = build_localization_runtime_config(config, config_base_dir=config_base_dir, seed=int(args.seed))
    runtime = build_trainer_runtime(
        runtime_config,
        layout=layout,
        model_registry=build_localization_model_registry(),
    )
    if args.checkpoint is not None:
        _load_model_checkpoint(runtime.model, Path(args.checkpoint))
    roi_source = resolve_roi_bank_source(gamma_cfg, train_cfg=train_cfg, config=config, config_base_dir=config_base_dir)
    if roi_source is None:
        raise ValueError("ROI library smoke requires train.roi_bank_gamma.roi_bank_source / auto_build_roi_bank.")
    bank = auto_build_roi_bank(gamma_cfg, roi_source=roi_source, model=runtime.model, train_cfg=train_cfg)
    output_dir = layout.stage_dir("roi_library_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_bank_path = output_dir / "roi_bank.h5"
    save_roi_bank(bank, roi_bank_path)
    manifest = write_roi_library_smoke_diagnostics(
        model=runtime.model,
        bank=bank,
        config=config,
        gamma_cfg=gamma_cfg,
        train_cfg=train_cfg,
        output_dir=output_dir / "diagnostics",
        max_rois=int(args.diagnostic_rois),
        threshold=float(gamma_cfg.get("probability_threshold", gamma_cfg.get("roi_bank_probability_threshold", 0.5))),
    )
    summary = {
        "schema_version": "roi_library_only_smoke.v1",
        "config_path": str(config_path),
        "checkpoint_path": None if args.checkpoint is None else str(Path(args.checkpoint).resolve()),
        "roi_bank_path": str(roi_bank_path.relative_to(layout.run_dir)),
        "roi_count": len(bank.records),
        "domains": sorted({str(record.domain_name) for record in bank.records}),
        "diagnostics_manifest_path": str((output_dir / "diagnostics" / "roi_library_smoke_manifest.json").relative_to(layout.run_dir)),
        "diagnostic_png_count": len(manifest["diagnostics"]),
        "roi_bank_source": {
            "mode": roi_source.mode,
            "alias": roi_source.alias,
            "candidate_mode": roi_source.candidate_mode,
            "raw_path": roi_source.raw_path,
            "frame_range": None if roi_source.frame_range is None else list(roi_source.frame_range),
        },
    }
    summary_path = output_dir / "roi_library_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def write_roi_library_smoke_diagnostics(
    *,
    model: torch.nn.Module,
    bank: ROIBank,
    config: Mapping[str, Any] | None = None,
    gamma_cfg: Mapping[str, Any] | None = None,
    train_cfg: Mapping[str, Any],
    output_dir: Path,
    max_rois: int = 8,
    threshold: float = 0.5,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = tuple(bank.records[: max(0, int(max_rois))])
    manifest: dict[str, Any] = {
        "schema_version": "roi_library_smoke_diagnostics.v1",
        "column_order": ["raw_center_frame", "model_input_center_frame", "p_map", "raw_with_p_threshold_circles"],
        "physical_projection_column_order": [
            "raw_center_frame",
            "raw_with_loc_circles",
            "initial_vector_psf_projection",
            "abs_residual",
        ],
        "threshold": float(threshold),
        "roi_count": len(bank.records),
        "rendered_roi_count": len(records),
        "diagnostics": [],
    }
    if not records:
        (output_dir / "roi_library_smoke_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    subbank = ROIBank(records=records, config=bank.config, metadata=bank.metadata, empty_grid_cell_ids=bank.empty_grid_cell_ids)
    conditioning = roi_conditioning_context(train_cfg)
    loc_batch = build_roi_batch_provider(
        subbank,
        batch_size=len(records),
        seed=0,
        condition_providers_by_domain=conditioning["providers"],
        append_domain_onehot=conditioning["append_domain_onehot"],
        domain_names=conditioning["domain_names"],
    )(epoch=1)[0].inputs
    p_maps = _predict_p_maps(model, loc_batch.model_input)
    samples = sample_detection_posterior(
        model=model,
        batch=loc_batch,
        threshold=float(threshold),
        max_emitters=100,
        seed=0,
        photon_scale=posterior_photon_scale(train_cfg),
        z_scale=posterior_z_scale(train_cfg),
        candidate_threshold=float(
            gamma_cfg.get("posterior_candidate_probability_threshold", gamma_cfg.get("candidate_probability_threshold", 0.3))
        )
        if gamma_cfg is not None
        else 0.3,
        split_threshold=float(gamma_cfg.get("posterior_adjacent_probability_threshold", gamma_cfg.get("split_threshold", 0.6)))
        if gamma_cfg is not None
        else 0.6,
    )
    objective = None
    gamma = None
    if config is not None and gamma_cfg is not None:
        objective = build_vector_roi_gamma_objective(gamma_cfg, train_cfg=train_cfg, config=config, model=model)
        gamma = objective.initial_gamma().detach()
    image_input = loc_batch.model_input[0] if isinstance(loc_batch.model_input, tuple) else loc_batch.model_input
    roi_origin_xy_px = torch.as_tensor(loc_batch.metadata["roi_origin_xy_px"], dtype=torch.float32)
    domain_names = list(loc_batch.metadata["domain_names"])
    for idx, record in enumerate(records):
        raw = torch.as_tensor(record.raw_frames_photon, dtype=torch.float32)
        center_idx = int(raw.shape[0] // 2)
        raw_center = raw[center_idx].detach().cpu().numpy()
        input_center = image_input[idx, center_idx].detach().cpu().numpy()
        p_map = p_maps[idx].detach().cpu().numpy()
        points = [
            (float(samples.xyzph[idx, sample_idx, 0].item()), float(samples.xyzph[idx, sample_idx, 1].item()))
            for sample_idx in torch.nonzero(samples.mask[idx], as_tuple=False).flatten().tolist()
        ]
        png_name = f"roi_{int(record.roi_id):04d}_{_path_token(record.domain_name)}.png"
        png_path = output_dir / png_name
        _write_diagnostic_png(
            png_path,
            raw_center=raw_center,
            input_center=input_center,
            p_map=p_map,
            circle_points=points,
        )
        physical_png_name = None
        if objective is not None and gamma is not None:
            reconstruction = objective.render_record_reconstruction(
                gamma=gamma,
                raw_shape=tuple(raw_center.shape),
                emitters=record.emitters,
                background=record.background_smoothed,
                roi_origin_xy_px=record.roi_origin_xy_px,
                domain_name=str(record.domain_name),
            ).detach().cpu().numpy()
            physical_png_name = f"roi_{int(record.roi_id):04d}_{_path_token(record.domain_name)}_initial_physical_projection.png"
            _write_physical_projection_png(
                output_dir / physical_png_name,
                raw_center=raw_center,
                reconstruction=reconstruction,
                circle_points=points,
            )
        manifest["diagnostics"].append(
            {
                "roi_id": int(record.roi_id),
                "domain_name": str(record.domain_name),
                "frame_window": [int(record.frame_window[0]), int(record.frame_window[1])],
                "roi_origin_xy_px": [float(v) for v in record.roi_origin_xy_px],
                "domain_local_roi_origin_xy_px": list(record.summary.get("domain_local_roi_origin_xy_px", [])),
                "full_fov_roi_origin_xy_px": list(record.summary.get("full_fov_roi_origin_xy_px", [])),
                "raw_shape": list(raw.shape),
                "active_emitters_at_threshold": len(points),
                "record_emitter_count": len(record.emitters),
                "png_path": png_name,
                "initial_physical_projection_png_path": physical_png_name,
            }
        )
    manifest_path = output_dir / "roi_library_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _smoke_gamma_cfg(
    gamma_cfg: Mapping[str, Any],
    *,
    max_rois: int | None,
    target_emitters: int | None,
    frame_start: int | None,
    frame_stop: int | None,
) -> dict[str, Any]:
    cfg = dict(gamma_cfg)
    if max_rois is not None:
        cfg["roi_library_max_rois"] = int(max_rois)
        cfg["roi_bank_max_rois"] = int(max_rois)
    if target_emitters is not None:
        cfg["target_projected_emitters"] = int(target_emitters)
    if frame_start is not None or frame_stop is not None:
        if frame_start is None or frame_stop is None:
            raise ValueError("--frame-start and --frame-stop must be provided together")
        cfg["roi_bank_frame_range"] = [int(frame_start), int(frame_stop)]
    return cfg


def _load_model_checkpoint(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location=next(model.parameters()).device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)


def _predict_p_maps(model: torch.nn.Module, model_input) -> torch.Tensor:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model_input = _model_input_to_device(model_input, device)
    with torch.no_grad():
        output = model(model_input)
    if was_training:
        model.train()
    if isinstance(output, LocalizationModelOutput):
        return output.probability.detach().cpu()
    if isinstance(output, torch.Tensor) and output.ndim == 4 and int(output.shape[1]) == SMLMOutputChannels.count:
        return decode_smlm_output(output.detach()).p.detach().cpu()
    if isinstance(output, torch.Tensor) and output.ndim == 4 and int(output.shape[1]) == 1:
        return torch.sigmoid(output[:, 0]).detach().cpu()
    if isinstance(output, torch.Tensor) and output.ndim == 3:
        return torch.sigmoid(output).detach().cpu()
    raise ValueError(f"Unsupported localizer output for ROI smoke diagnostics: {type(output)!r}")


def _model_input_to_device(model_input, device: torch.device):
    if isinstance(model_input, tuple):
        return tuple(item.to(device=device) for item in model_input)
    return model_input.to(device=device)


def _write_diagnostic_png(
    path: Path,
    *,
    raw_center: np.ndarray,
    input_center: np.ndarray,
    p_map: np.ndarray,
    circle_points: list[tuple[float, float]],
) -> None:
    raw = _gray_to_rgb(_to_uint8(raw_center))
    model_input = _gray_to_rgb(_to_uint8(input_center))
    p_panel = _gray_to_rgb(_to_uint8(p_map, low=0.0, high=max(1.0, float(np.nanmax(p_map)) if np.size(p_map) else 1.0)))
    overlay = raw.copy()
    for x, y in circle_points:
        _draw_circle(overlay, x=x, y=y, radius=4)
    canvas = np.concatenate([raw, model_input, p_panel, overlay], axis=1)
    _write_rgb_png(path, canvas)


def _write_physical_projection_png(
    path: Path,
    *,
    raw_center: np.ndarray,
    reconstruction: np.ndarray,
    circle_points: list[tuple[float, float]],
) -> None:
    raw_u8 = _to_uint8(raw_center)
    recon_u8 = _to_uint8(reconstruction)
    residual_u8 = _to_uint8(np.abs(np.asarray(raw_center, dtype=np.float32) - np.asarray(reconstruction, dtype=np.float32)))
    raw = _gray_to_rgb(raw_u8)
    overlay = raw.copy()
    for x, y in circle_points:
        _draw_circle(overlay, x=x, y=y, radius=4)
    canvas = np.concatenate([raw, overlay, _gray_to_rgb(recon_u8), _gray_to_rgb(residual_u8)], axis=1)
    _write_rgb_png(path, canvas)


def _to_uint8(frame: np.ndarray, *, low: float | None = None, high: float | None = None) -> np.ndarray:
    array = np.asarray(frame, dtype=np.float32)
    lo = float(np.nanmin(array)) if low is None else float(low)
    hi = float(np.nanmax(array)) if high is None else float(high)
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(gray, dtype=np.uint8)[..., None], 3, axis=2)


def _draw_circle(image: np.ndarray, *, x: float, y: float, radius: int) -> None:
    height, width = int(image.shape[0]), int(image.shape[1])
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    r = int(radius)
    for yy in range(max(0, cy - r), min(height, cy + r + 1)):
        for xx in range(max(0, cx - r), min(width, cx + r + 1)):
            dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
            if r * r - r <= dist2 <= r * r + r:
                image[yy, xx] = np.asarray([255, 32, 32], dtype=np.uint8)


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or int(image.shape[2]) != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {tuple(image.shape)}")
    height, width = int(image.shape[0]), int(image.shape[1])
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw_rows)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _path_token(value: object) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    token = "_".join(part for part in "".join(chars).split("_") if part)
    return token or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
