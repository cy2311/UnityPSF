#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
import torch

ROOT = Path(__file__).resolve().parents[3]
NEPTUNE_DIR = Path(__file__).resolve().parents[2]
SRC_ROOT = NEPTUNE_DIR / "src"
NEPTUNE_IWAE_ROOT = ROOT / "neptune_iwae"
for path in (ROOT, SRC_ROOT, NEPTUNE_IWAE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Normalization import build_inference_frame_normalizer
from neptune_v03.config import load_config
from neptune_v03.infer_recon.degrid import default_reconstruction_predictions, degrid_predictions_h5
from neptune_v03.infer_recon.predictions_io import H5PredictionWriter
from neptune_v03.infer_recon.tiling import (
    build_liteloc_subfov_tiles,
    emitter_in_valid_core,
    tile_local_to_field_coordinates,
)
from neptune_v03.localization import build_localization_model_registry, build_localization_runtime_config
from neptune_v03.localization.conditioning import FullResZernikeConditioning
from neptune_v03.localization.legacy_decode import decode_liteloc_formal_infer_emitters


RAW_TIFF = ROOT / "neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif"
RUN_3371 = (
    NEPTUNE_DIR
    / "output/standard_3367_fast_route_roi96_psf25_fresh_hqzmap_emit500_round20_p1baseline_3052lr_start30_interval5_emit5000_epoch300_bs24_steps417_3371"
    / "standard_3367_fast_route_roi96_psf25_fresh_hqzmap_emit500_round20_p1baseline_3052lr_start30_interval5_emit5000_epoch300_bs24_steps417_3371_3371"
)
CONFIG = NEPTUNE_DIR / "configs/microtube_base.yaml"

FIELDNAMES = [
    "frame",
    "x_px",
    "y_px",
    "x_px_full",
    "y_px_full",
    "x_nm",
    "y_nm",
    "x_nm_full",
    "y_nm_full",
    "z",
    "z_nm",
    "photon",
    "prob",
    "x_sig",
    "y_sig",
    "x_sig_px",
    "y_sig_px",
    "x_sig_nm",
    "y_sig_nm",
    "z_sig",
    "z_sig_nm",
    "photon_sig",
    "x_offset_px",
    "y_offset_px",
    "x_offset_nm",
    "y_offset_nm",
    "logLikelihood",
    "log_likelihood",
    "negative_log_likelihood",
    "LLrel",
    "llrel",
    "PSFxpix",
    "PSFypix",
    "PSFxnm",
    "PSFynm",
    "psf_x_nm",
    "psf_y_nm",
    "psf_xy_nm",
    "postfit_status",
    "tile_index",
    "sample_tiff",
]


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_float_list(value: str | None) -> list[float]:
    if value is None:
        return []
    out: list[float] = []
    text_value = str(value).replace(";", ",")
    for item in text_value.split(","):
        text = item.strip()
        if not text:
            continue
        out.append(float(text))
    return out


def unique_prob_values(values: Iterable[float]) -> list[float]:
    seen: set[int] = set()
    out: list[float] = []
    for value in values:
        key = int(round(float(value) * 1000.0))
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3371 v0.3 full 8000-frame infer -> filter/recon, ROI96 keep80.")
    parser.add_argument("--run-dir", type=Path, default=RUN_3371)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--right-coeff-map", type=Path, default=None)
    parser.add_argument("--domain-count", type=int, default=2)
    parser.add_argument("--right-domain-index", type=int, default=1)
    parser.add_argument("--sample-tiff", type=Path, default=RAW_TIFF)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--frame-block", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=8000)
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--valid-roi-size", type=int, default=80)
    parser.add_argument("--decode-accept-threshold", type=float, default=0.70)
    parser.add_argument("--filter-prob-min", type=float, default=0.70)
    parser.add_argument("--filter-prob-sweep", type=str, default=None, help="Comma-separated prob_min values, e.g. 0.7,0.8,0.9.")
    parser.add_argument("--locprec-xy-nm-max", type=float, default=None)
    parser.add_argument("--x-sig-px-max", type=float, default=None)
    parser.add_argument("--y-sig-px-max", type=float, default=None)
    parser.add_argument("--llrel-min", type=float, default=None)
    parser.add_argument("--psf-xy-nm-max", type=float, default=None)
    parser.add_argument("--require-fit-status", action="store_true", default=False)
    parser.add_argument("--quality-metrics", type=parse_bool, default=False)
    parser.add_argument("--quality-metric-mode", choices=("moment", "optimize"), default="moment")
    parser.add_argument("--quality-roi-radius-px", type=int, default=3)
    parser.add_argument("--degrid", type=parse_bool, default=True)
    parser.add_argument("--degrid-rescale-bins", type=int, default=20)
    parser.add_argument("--degrid-threshold", type=float, default=0.01)
    parser.add_argument("--degrid-min-bin-count", type=int, default=32)
    parser.add_argument("--degrid-spatial-bins-x", type=int, default=6)
    parser.add_argument("--degrid-spatial-bins-y", type=int, default=12)
    parser.add_argument("--rcc-drift", type=parse_bool, default=True)
    parser.add_argument("--rcc-frame-block-size", type=int, default=500)
    parser.add_argument("--rcc-pixel-nm", type=float, default=50.0)
    parser.add_argument("--rcc-sigma-px", type=float, default=1.0)
    parser.add_argument("--rcc-upsample-factor", type=int, default=10)
    parser.add_argument("--rcc-max-pair-gap", type=int, default=8)
    parser.add_argument("--rcc-max-shift-nm", type=float, default=1000.0)
    parser.add_argument("--rcc-min-pair-correlation", type=float, default=0.02)
    parser.add_argument("--render-pixel-nm", type=float, default=20.0)
    parser.add_argument("--spot-radius-nm", type=float, default=28.0)
    parser.add_argument("--radius-mode", choices=("fixed", "xy_uncertainty_mean"), default="fixed")
    parser.add_argument("--display-mode", choices=("quantile", "fixed_imax"), default="quantile")
    parser.add_argument("--display-imax", type=float, default=None)
    parser.add_argument("--display-imax-min", type=float, default=-2.5228787452803374)
    parser.add_argument("--normalization-fov", type=str, default=None)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--left-crop-left", type=int, default=0)
    parser.add_argument("--left-crop-top", type=int, default=0)
    parser.add_argument("--left-crop-width", type=int, default=600)
    parser.add_argument("--left-crop-height", type=int, default=1200)
    parser.add_argument("--right-crop-left", type=int, default=600)
    parser.add_argument("--right-crop-top", type=int, default=0)
    parser.add_argument("--right-crop-width", type=int, default=600)
    parser.add_argument("--right-crop-height", type=int, default=1200)
    parser.add_argument("--infer-recon-root", type=Path, default=SRC_ROOT / "neptune_v03" / "infer_recon")
    parser.add_argument("--input-preprocess", choices=("fd_deeploc_recenter", "raw_adu"), default="fd_deeploc_recenter")
    parser.add_argument("--overwrite-output", action="store_true", default=False)
    parser.add_argument("--infer-amp", action="store_true", default=True)
    parser.add_argument("--no-infer-amp", dest="infer_amp", action="store_false")
    return parser.parse_args()


def assert_output_dir_available(output_dir: Path, *, overwrite: bool) -> None:
    if bool(overwrite):
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty infer/recon output directory: {output_dir}. "
            "Use --overwrite-output only for an intentional rerun. "
            "Phase-0 baseline outputs must be kept immutable and new postprocess experiments "
            "should use a new output directory."
        )


def iter_blocks(*, n_frames: int, frame_block: int, window: int) -> Iterable[tuple[int, int]]:
    step = int(frame_block) - (int(window) - 1)
    if step <= 0:
        raise ValueError("frame_block must exceed window_size - 1")
    start = 0
    while start + window <= n_frames:
        stop = min(start + int(frame_block), n_frames)
        yield start, stop
        if stop >= n_frames:
            break
        start += step


def tiff_shape(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as tif:
        shape = tuple(int(v) for v in tif.series[0].shape)
    if len(shape) != 3:
        raise ValueError(f"Expected TIFF shape (frames,height,width), got {shape}")
    return shape


def load_tiff_block(path: Path, start: int, stop: int) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        return np.asarray(tif.series[0].asarray(key=slice(int(start), int(stop))), dtype=np.float32)


def final_coeff_maps(run_dir: Path) -> dict[str, Path]:
    state_path = run_dir / "metadata/current_physical_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for item in payload["coeff_maps"]:
        out[str(item["name"])] = Path(item["coeff_maps_npz"])
    return out


def load_model(config_path: Path, checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    cfg = load_config(config_path)
    runtime = build_localization_runtime_config(cfg, config_base_dir=config_path.parent, seed=0)
    model_spec = runtime["model"]
    model = build_localization_model_registry()[str(model_spec["name"])](dict(model_spec["params"]))
    payload = torch.load(checkpoint, map_location=device)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, runtime


def domain_condition(
    provider: FullResZernikeConditioning,
    *,
    x0: int,
    y0: int,
    height: int,
    width: int,
    domain_index: int,
    feature_dim: int,
    domain_count: int = 2,
) -> torch.Tensor:
    base = provider.condition_vector_from_xy(x0=x0, y0=y0, height=height, width=width)
    if int(base.shape[0]) < int(feature_dim):
        padded = torch.zeros((int(feature_dim),), dtype=base.dtype, device=base.device)
        padded[: int(base.shape[0])] = base
        base = padded
    elif int(base.shape[0]) > int(feature_dim):
        base = base[: int(feature_dim)].contiguous()
    onehot = torch.zeros(int(domain_count), dtype=base.dtype)
    onehot[int(domain_index)] = 1.0
    return torch.cat([base, onehot], dim=0).contiguous()


def condition_feature_dim(*, condition_dim: int, domain_count: int, domain_index: int) -> int:
    if int(domain_count) <= 0:
        raise ValueError("domain_count must be positive")
    if not 0 <= int(domain_index) < int(domain_count):
        raise ValueError(f"domain_index={int(domain_index)} must be in [0, {int(domain_count)})")
    feature_dim = int(condition_dim) - int(domain_count)
    if feature_dim <= 0:
        raise ValueError("condition_dim must be greater than domain_count")
    return feature_dim


def rows_from_emitters(
    emitters,
    *,
    metas: list[dict[str, int]],
    crop_left: int,
    crop_top: int,
    sample_name: str,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(int(emitters.probability.numel())):
        batch_idx = int(emitters.batch_index[idx].item())
        meta = metas[batch_idx]
        x_patch = float(emitters.xyz_px_nm[idx, 0].item())
        y_patch = float(emitters.xyz_px_nm[idx, 1].item())
        if not emitter_in_valid_core(x_patch=x_patch, y_patch=y_patch, tile=meta):
            continue
        coordinates = tile_local_to_field_coordinates(
            x_patch=x_patch,
            y_patch=y_patch,
            tile=meta,
            crop_left=int(crop_left),
            crop_top=int(crop_top),
        )
        x_roi = coordinates["x_crop"]
        y_roi = coordinates["y_crop"]
        x_full = coordinates["x_full"]
        y_full = coordinates["y_full"]
        x_sig_px = float(emitters.sigma_xy_px[idx, 0].item())
        y_sig_px = float(emitters.sigma_xy_px[idx, 1].item())
        z_sig_nm = float(emitters.sigma_z_nm[idx].item()) if emitters.sigma_z_nm is not None else None
        photon_sig = float(emitters.sigma_photons[idx].item()) if emitters.sigma_photons is not None else None
        x_offset_px = x_roi - np.floor(x_roi) - 0.5
        y_offset_px = y_roi - np.floor(y_roi) - 0.5
        z_nm = float(emitters.xyz_px_nm[idx, 2].item())
        rows.append(
            {
                "frame": int(meta["frame_id"]),
                "x_px": x_roi,
                "y_px": y_roi,
                "x_px_full": x_full,
                "y_px_full": y_full,
                "x_nm": x_roi * float(camera_pixel_nm_x),
                "y_nm": y_roi * float(camera_pixel_nm_y),
                "x_nm_full": x_full * float(camera_pixel_nm_x),
                "y_nm_full": y_full * float(camera_pixel_nm_y),
                "z": z_nm,
                "z_nm": z_nm,
                "photon": float(emitters.photons[idx].item()),
                "prob": float(emitters.probability[idx].item()),
                "x_sig": x_sig_px,
                "y_sig": y_sig_px,
                "x_sig_px": x_sig_px,
                "y_sig_px": y_sig_px,
                "x_sig_nm": x_sig_px * float(camera_pixel_nm_x),
                "y_sig_nm": y_sig_px * float(camera_pixel_nm_y),
                "z_sig": z_sig_nm,
                "z_sig_nm": z_sig_nm,
                "photon_sig": photon_sig,
                "x_offset_px": x_offset_px,
                "y_offset_px": y_offset_px,
                "x_offset_nm": x_offset_px * float(camera_pixel_nm_x),
                "y_offset_nm": y_offset_px * float(camera_pixel_nm_y),
                "logLikelihood": None,
                "log_likelihood": None,
                "negative_log_likelihood": None,
                "LLrel": None,
                "llrel": None,
                "PSFxpix": None,
                "PSFypix": None,
                "PSFxnm": None,
                "PSFynm": None,
                "psf_x_nm": None,
                "psf_y_nm": None,
                "psf_xy_nm": None,
                "postfit_status": None,
                "tile_index": int(meta["tile_index"]),
                "sample_tiff": sample_name,
            }
        )
    return rows


def flush_bucket(
    *,
    patches: list[torch.Tensor],
    conds: list[torch.Tensor],
    metas: list[dict[str, int]],
    writer: H5PredictionWriter,
    model: torch.nn.Module,
    device: torch.device,
    infer_amp: bool,
    decode_accept_threshold: float,
    photon_scale: float,
    z_scale: float,
    crop_left: int,
    crop_top: int,
    sample_name: str,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
) -> int:
    if not patches:
        return 0
    batch = torch.stack(patches, dim=0).to(device=device, dtype=torch.float32)
    cond = torch.stack(conds, dim=0).to(device=device, dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(infer_amp) and device.type == "cuda"):
        output = model([batch, cond])
    emitters = decode_liteloc_formal_infer_emitters(
        output.float(),
        accept_threshold=float(decode_accept_threshold),
        photon_scale=float(photon_scale),
        z_scale=float(z_scale),
    )
    rows = rows_from_emitters(
        emitters,
        metas=metas,
        crop_left=crop_left,
        crop_top=crop_top,
        sample_name=sample_name,
        camera_pixel_nm_x=float(camera_pixel_nm_x),
        camera_pixel_nm_y=float(camera_pixel_nm_y),
    )
    writer.append_rows(rows)
    count = len(rows)
    patches.clear()
    conds.clear()
    metas.clear()
    return count


def run_side(
    *,
    side: str,
    args: argparse.Namespace,
    model: torch.nn.Module,
    runtime: dict[str, object],
    device: torch.device,
    coeff_map: Path,
    crop_left: int,
    crop_top: int,
    crop_width: int,
    crop_height: int,
    domain_index: int,
    frame_proc,
    domain_count: int = 2,
) -> dict[str, object]:
    side_dir = args.output_dir / side
    infer_dir = side_dir / "infer"
    infer_dir.mkdir(parents=True, exist_ok=True)
    provider = FullResZernikeConditioning.from_npz(coeff_map)
    condition_dim = int(getattr(model, "condition_dim"))
    feature_dim = condition_feature_dim(
        condition_dim=condition_dim,
        domain_count=int(domain_count),
        domain_index=int(domain_index),
    )
    tiles = build_liteloc_subfov_tiles(
        field_height=int(crop_height),
        field_width=int(crop_width),
        context_size=int(args.roi_size),
        valid_core_size=int(args.valid_roi_size),
    )
    runtime_state = {
        "window_size": 3,
        "conditioning_vector_dim": condition_dim,
        "condition_feature_dim": feature_dim,
        "conditioning_mode": "film",
        "condition_mode": "film",
        "coeff_maps_npz": str(coeff_map),
        "nat_coeff_maps_path": str(coeff_map),
        "append_domain_onehot": True,
        "domain_count": int(domain_count),
        "domain_index": int(domain_index),
        "crop_left": int(crop_left),
        "crop_top": int(crop_top),
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
        "roi_size": int(args.roi_size),
        "valid_roi_size": int(args.valid_roi_size),
        "cut_edge_px": int((int(args.roi_size) - int(args.valid_roi_size)) // 2),
        "spatial_stitching_contract": "liteloc_subfov_overcut_v1",
        "spatial_boundary_rule": "lower_inclusive_upper_exclusive",
        "tiling_mode": "edgecover",
        "decode_contract": "liteloc_formal_infer_nms_v1",
        "decode_candidate_threshold": 0.3,
        "decode_adjacent_threshold": 0.6,
        "decode_accept_threshold": float(args.decode_accept_threshold),
        "decode_aggregation": "sum",
        "decode_accept_rule": ">",
        "camera_pixel_nm_x": 101.11,
        "camera_pixel_nm_y": 98.83,
        "localization_schema": "neptune_v03_localization_v0.2",
        "localization_units": {
            "x_px": "camera_pixel",
            "y_px": "camera_pixel",
            "x_nm": "nm",
            "y_nm": "nm",
            "z": "nm",
            "z_nm": "nm",
            "x_sig": "camera_pixel",
            "y_sig": "camera_pixel",
            "x_sig_px": "camera_pixel",
            "y_sig_px": "camera_pixel",
            "z_sig_nm": "nm",
            "photon": "photon",
            "photon_sig": "photon",
        },
        "source_train_dir": str(args.run_dir),
        "input_preprocess": str(args.input_preprocess),
        "model_input": "recentered_raw_adu" if frame_proc is not None else "raw_adu",
    }
    (side_dir / "derived_runtime_state.json").write_text(json.dumps(runtime_state, indent=2) + "\n", encoding="utf-8")
    (infer_dir / "derived_runtime_state.json").write_text(json.dumps(runtime_state, indent=2) + "\n", encoding="utf-8")

    n_frames, full_h, full_w = tiff_shape(args.sample_tiff)
    n_frames = min(int(n_frames), int(args.max_frames))
    photon_scale = float(runtime["loss"]["params"].get("photon_scale", 31000.0))
    z_scale = float(runtime["loss"]["params"].get("z_scale", 0.6))
    sample_name = args.sample_tiff.name

    pred_path = infer_dir / "predictions_merged.h5"
    total_rows = 0
    started = time.time()
    with H5PredictionWriter(
        pred_path,
        fieldnames=FIELDNAMES,
        schema="infer_recon_predictions_h5_v0.2",
        attributes={
            "localization_schema": "neptune_v03_localization_v0.2",
            "units": runtime_state["localization_units"],
        },
    ) as writer:
        for block_idx, (start, stop) in enumerate(iter_blocks(n_frames=n_frames, frame_block=int(args.frame_block), window=3), start=1):
            raw_block = load_tiff_block(args.sample_tiff, start, stop)
            crop = np.ascontiguousarray(
                raw_block[:, int(crop_top) : int(crop_top) + int(crop_height), int(crop_left) : int(crop_left) + int(crop_width)],
                dtype=np.float32,
            )
            patches: list[torch.Tensor] = []
            conds: list[torch.Tensor] = []
            metas: list[dict[str, int]] = []
            n_windows = int(crop.shape[0]) - 2
            for win_idx in range(n_windows):
                frame_id = int(start) + int(win_idx) + 1
                for tile in tiles:
                    y0 = int(tile["patch_y0"])
                    x0 = int(tile["patch_x0"])
                    patch = np.ascontiguousarray(crop[win_idx : win_idx + 3, y0 : y0 + int(args.roi_size), x0 : x0 + int(args.roi_size)], dtype=np.float32)
                    if frame_proc is not None:
                        patch = np.ascontiguousarray(frame_proc.normalize_numpy(patch), dtype=np.float32)
                    patches.append(torch.from_numpy(patch))
                    conds.append(
                        domain_condition(
                            provider,
                            x0=x0,
                            y0=y0,
                            height=int(args.roi_size),
                            width=int(args.roi_size),
                            domain_index=int(domain_index),
                            feature_dim=feature_dim,
                            domain_count=int(domain_count),
                        )
                    )
                    metas.append(
                        {
                            "frame_id": frame_id,
                            "patch_y0": y0,
                            "patch_x0": x0,
                            "keep_y0": int(tile["keep_y0"]),
                            "keep_x0": int(tile["keep_x0"]),
                            "keep_h": int(tile["keep_h"]),
                            "keep_w": int(tile["keep_w"]),
                            "tile_index": int(tile["tile_index"]),
                        }
                    )
                    if len(patches) >= int(args.batch_size):
                        total_rows += flush_bucket(
                            patches=patches,
                            conds=conds,
                            metas=metas,
                            writer=writer,
                            model=model,
                            device=device,
                            infer_amp=bool(args.infer_amp),
                            decode_accept_threshold=float(args.decode_accept_threshold),
                            photon_scale=photon_scale,
                            z_scale=z_scale,
                            crop_left=int(crop_left),
                            crop_top=int(crop_top),
                            sample_name=sample_name,
                            camera_pixel_nm_x=float(runtime_state["camera_pixel_nm_x"]),
                            camera_pixel_nm_y=float(runtime_state["camera_pixel_nm_y"]),
                        )
            total_rows += flush_bucket(
                patches=patches,
                conds=conds,
                metas=metas,
                writer=writer,
                model=model,
                device=device,
                infer_amp=bool(args.infer_amp),
                decode_accept_threshold=float(args.decode_accept_threshold),
                photon_scale=photon_scale,
                z_scale=z_scale,
                crop_left=int(crop_left),
                crop_top=int(crop_top),
                sample_name=sample_name,
                camera_pixel_nm_x=float(runtime_state["camera_pixel_nm_x"]),
                camera_pixel_nm_y=float(runtime_state["camera_pixel_nm_y"]),
            )
            elapsed = time.time() - started
            print(
                json.dumps(
                    {
                        "side": side,
                        "block": block_idx,
                        "frames": [start, stop],
                        "rows": total_rows,
                        "elapsed_sec": round(elapsed, 2),
                    }
                ),
                flush=True,
            )

    degrid_predictions = None
    degrid_summary = None
    degrid_payload = None
    if bool(args.degrid):
        degrid_predictions = infer_dir / "predictions_degrid.h5"
        degrid_summary = infer_dir / "degrid_summary.json"
        degrid_payload = degrid_predictions_h5(
            predictions=pred_path,
            output=degrid_predictions,
            summary_json=degrid_summary,
            histogram_png=infer_dir / "degrid_offset_histograms.png",
            pixel_size_nm_x=float(runtime_state["camera_pixel_nm_x"]),
            pixel_size_nm_y=float(runtime_state["camera_pixel_nm_y"]),
            rescale_bins=int(args.degrid_rescale_bins),
            threshold=float(args.degrid_threshold),
            min_bin_count=int(args.degrid_min_bin_count),
            spatial_bins_x=int(args.degrid_spatial_bins_x),
            spatial_bins_y=int(args.degrid_spatial_bins_y),
            field_width_px=float(crop_width),
            field_height_px=float(crop_height),
        )

    pred_for_filter = default_reconstruction_predictions(
        raw=pred_path,
        degrid=infer_dir / "predictions_degrid.h5",
        degrid_enabled=bool(args.degrid),
    )
    spatial_degrid = bool(args.degrid) and (
        int(args.degrid_spatial_bins_x) > 1 or int(args.degrid_spatial_bins_y) > 1
    )
    reconstruction_coordinate_source = "spatial_degrid" if spatial_degrid else ("degrid" if bool(args.degrid) else "raw")
    rcc_summary = None
    if bool(args.rcc_drift):
        if not bool(args.degrid):
            raise ValueError("standard RCC drift correction requires degrid to be enabled")
        rcc_script = NEPTUNE_DIR / "scripts" / "analysis" / "run_rcc_drift_diagnostic.py"
        if not rcc_script.is_file():
            raise FileNotFoundError(rcc_script)
        rcc_cmd = [
            sys.executable,
            str(rcc_script),
            "--predictions",
            str(pred_for_filter),
            "--output-dir",
            str(infer_dir),
            "--frame-block-size",
            str(args.rcc_frame_block_size),
            "--camera-pixel-nm-x",
            str(runtime_state["camera_pixel_nm_x"]),
            "--camera-pixel-nm-y",
            str(runtime_state["camera_pixel_nm_y"]),
            "--width-px",
            str(crop_width),
            "--height-px",
            str(crop_height),
            "--rcc-pixel-nm",
            str(args.rcc_pixel_nm),
            "--rcc-sigma-px",
            str(args.rcc_sigma_px),
            "--upsample-factor",
            str(args.rcc_upsample_factor),
            "--max-pair-gap",
            str(args.rcc_max_pair_gap),
            "--max-shift-nm",
            str(args.rcc_max_shift_nm),
            "--min-pair-correlation",
            str(args.rcc_min_pair_correlation),
        ]
        subprocess.run(rcc_cmd, check=True)
        pred_for_filter = infer_dir / "predictions_degrid_rcc_corrected.h5"
        rcc_summary = infer_dir / "rcc_summary.json"
        reconstruction_coordinate_source = "spatial_degrid_rcc_corrected" if spatial_degrid else "degrid_rcc_corrected"
    quality_summary = None
    if bool(args.quality_metrics):
        quality_script = args.infer_recon_root / "quality_enrich_h5.py"
        if not quality_script.is_file():
            raise FileNotFoundError(quality_script)
        pred_for_filter = infer_dir / "predictions_quality_enriched.h5"
        quality_summary = infer_dir / "quality_summary.json"
        quality_cmd = [
            sys.executable,
            str(quality_script),
            "--predictions",
            str(pred_for_filter),
            "--sample-tiff",
            str(args.sample_tiff),
            "--runtime-state",
            str(side_dir / "derived_runtime_state.json"),
            "--output",
            str(pred_for_filter),
            "--summary-json",
            str(quality_summary),
            "--quality-metric-mode",
            str(args.quality_metric_mode),
            "--roi-radius-px",
            str(args.quality_roi_radius_px),
        ]
        subprocess.run(quality_cmd, check=True)

    filter_script = args.infer_recon_root / "filter" / "apply_filter_recon.py"
    if not filter_script.is_file():
        raise FileNotFoundError(filter_script)
    filter_runs: list[dict[str, object]] = []
    sweep_values = parse_float_list(args.filter_prob_sweep)
    prob_values = unique_prob_values(sweep_values if sweep_values else [float(args.filter_prob_min)])
    for prob_min in prob_values:
        prob_tag = f"prob{int(round(float(prob_min) * 100)):03d}"
        locprec_tag = "no_locprec" if args.locprec_xy_nm_max is None else f"locprec{int(round(float(args.locprec_xy_nm_max)))}"
        filter_name = f"filter_recon_{prob_tag}_{locprec_tag}"
        if bool(args.quality_metrics):
            filter_name += "_quality"
        filter_dir = side_dir / filter_name
        cmd = [
            sys.executable,
            str(filter_script),
            "--infer-dir",
            str(infer_dir),
            "--output-dir",
            str(filter_dir),
            "--runtime-state",
            str(side_dir / "derived_runtime_state.json"),
            "--predictions",
            str(pred_for_filter),
            "--width-px",
            str(crop_width),
            "--height-px",
            str(crop_height),
            "--filter-profile",
            "basic" if args.locprec_xy_nm_max is None else "strict",
            "--prob-min",
            str(prob_min),
            "--render-pixel-nm",
            str(args.render_pixel_nm),
            "--spot-radius-nm",
            str(args.spot_radius_nm),
            "--renderer",
            "integrated_gaussian",
            "--gamma",
            "1.0",
            "--display-mode",
            str(args.display_mode),
            "--display-imax-min",
            str(args.display_imax_min),
            "--radius-mode",
            str(args.radius_mode),
            "--filtered-format",
            "h5",
            "--keep-filtered-predictions",
        ]
        if args.display_imax is not None:
            cmd.extend(["--display-imax", str(args.display_imax)])
        if args.normalization_fov:
            cmd.extend(["--normalization-fov", str(args.normalization_fov)])
        if args.locprec_xy_nm_max is not None:
            cmd.extend(["--locprec-xy-nm-max", str(args.locprec_xy_nm_max)])
        if args.x_sig_px_max is not None:
            cmd.extend(["--x-sig-px-max", str(args.x_sig_px_max)])
        if args.y_sig_px_max is not None:
            cmd.extend(["--y-sig-px-max", str(args.y_sig_px_max)])
        if args.llrel_min is not None:
            cmd.extend(["--llrel-min", str(args.llrel_min)])
        if args.psf_xy_nm_max is not None:
            cmd.extend(["--psf-xy-nm-max", str(args.psf_xy_nm_max)])
        if bool(args.require_fit_status):
            cmd.append("--require-fit-status")
        subprocess.run(cmd, check=True)
        filter_runs.append(
            {
                "prob_min": float(prob_min),
                "filter_recon_dir": str(filter_dir),
                "filter_summary": str(filter_dir / "filter_summary.json"),
            }
        )
    summary = {
        "side": side,
        "predictions": str(pred_path),
        "decode_contract": "liteloc_formal_infer_nms_v1",
        "decode_effective": {
            "candidate_threshold": 0.3,
            "adjacent_threshold": 0.6,
            "accept_threshold": float(args.decode_accept_threshold),
            "aggregation": "sum",
            "accept_rule": ">",
        },
        "quality_metrics_enabled": bool(args.quality_metrics),
        "quality_metric_mode": str(args.quality_metric_mode),
        "quality_predictions": str(pred_for_filter) if bool(args.quality_metrics) else None,
        "quality_summary": str(quality_summary) if quality_summary is not None else None,
        "degrid_enabled": bool(args.degrid),
        "degrid_spatial_bins_x": int(args.degrid_spatial_bins_x),
        "degrid_spatial_bins_y": int(args.degrid_spatial_bins_y),
        "degrid_predictions": str(degrid_predictions) if degrid_predictions is not None else None,
        "degrid_summary": str(degrid_summary) if degrid_summary is not None else None,
        "degrid": degrid_payload,
        "rcc_drift_enabled": bool(args.rcc_drift),
        "rcc_predictions": str(infer_dir / "predictions_degrid_rcc_corrected.h5") if bool(args.rcc_drift) else None,
        "rcc_summary": str(rcc_summary) if rcc_summary is not None else None,
        "rcc_frame_block_size": int(args.rcc_frame_block_size),
        "reconstruction_coordinate_source": reconstruction_coordinate_source,
        "reconstruction_predictions": str(pred_for_filter),
        "filter_recon_dir": filter_runs[-1]["filter_recon_dir"] if filter_runs else None,
        "filter_runs": filter_runs,
        "raw_rows": int(total_rows),
        "elapsed_sec": round(time.time() - started, 2),
        "tiles": len(tiles),
        "tile_geometry": {
            "contract": "liteloc_subfov_overcut_v1",
            "roi_size": int(args.roi_size),
            "valid_roi_size": int(args.valid_roi_size),
            "cut_edge_px": int((int(args.roi_size) - int(args.valid_roi_size)) // 2),
            "tiling_mode": "edgecover",
            "boundary_rule": "lower_inclusive_upper_exclusive",
        },
    }
    (side_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    assert_output_dir_available(args.output_dir, overwrite=bool(args.overwrite_output))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or (args.run_dir / "checkpoints/checkpoint_latest.pt")
    coeff_maps = final_coeff_maps(args.run_dir)
    if args.right_coeff_map is not None:
        coeff_maps["right"] = args.right_coeff_map.resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "cuda_available": torch.cuda.is_available(),
                "device": str(device),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "checkpoint": str(checkpoint),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    if device.type != "cuda":
        raise RuntimeError("Formal 3371 infer requires CUDA; refusing CPU fallback.")
    model, runtime = load_model(args.config, checkpoint, device)
    frame_proc = None
    if str(args.input_preprocess) == "fd_deeploc_recenter":
        frame_proc = build_inference_frame_normalizer(load_config(args.config))
    sides = ("left", "right") if args.side == "both" else (args.side,)
    summaries = []
    for side in sides:
        if side == "left":
            summaries.append(
                run_side(
                    side="left",
                    args=args,
                    model=model,
                    runtime=runtime,
                    device=device,
                    coeff_map=coeff_maps["left"],
                    crop_left=int(args.left_crop_left),
                    crop_top=int(args.left_crop_top),
                    crop_width=int(args.left_crop_width),
                    crop_height=int(args.left_crop_height),
                    domain_index=0,
                    domain_count=int(args.domain_count),
                    frame_proc=frame_proc,
                )
            )
        else:
            summaries.append(
                run_side(
                    side="right",
                    args=args,
                    model=model,
                    runtime=runtime,
                    device=device,
                    coeff_map=coeff_maps["right"],
                    crop_left=int(args.right_crop_left),
                    crop_top=int(args.right_crop_top),
                    crop_width=int(args.right_crop_width),
                    crop_height=int(args.right_crop_height),
                    domain_index=int(args.right_domain_index),
                    domain_count=int(args.domain_count),
                    frame_proc=frame_proc,
                )
            )
    manifest = {
        "standard": "neptune_v03_3371_full8000_infer_filter_recon_roi96_valid80_v0.1",
        "checkpoint": str(checkpoint),
        "run_dir": str(args.run_dir),
        "sample_tiff": str(args.sample_tiff),
        "sides": summaries,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
