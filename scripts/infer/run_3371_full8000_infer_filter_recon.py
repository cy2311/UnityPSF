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
from neptune_v03.infer_recon.predictions_io import H5PredictionWriter
from neptune_v03.localization import build_localization_model_registry, build_localization_runtime_config
from neptune_v03.localization.conditioning import FullResZernikeConditioning
from neptune_v03.localization.legacy_decode import decode_legacy_smlm_emitters


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
    "z",
    "photon",
    "prob",
    "x_sig",
    "y_sig",
    "z_sig",
    "photon_sig",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3371 v0.3 full 8000-frame infer -> filter/recon, ROI96 keep80.")
    parser.add_argument("--run-dir", type=Path, default=RUN_3371)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--sample-tiff", type=Path, default=RAW_TIFF)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--frame-block", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=8000)
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--valid-roi-size", type=int, default=80)
    parser.add_argument("--prob-threshold", type=float, default=0.70)
    parser.add_argument("--raw-th", type=float, default=0.5)
    parser.add_argument("--split-th", type=float, default=0.6)
    parser.add_argument("--filter-prob-min", type=float, default=0.90)
    parser.add_argument("--locprec-xy-nm-max", type=float, default=None)
    parser.add_argument("--render-pixel-nm", type=float, default=10.0)
    parser.add_argument("--spot-radius-nm", type=float, default=45.0)
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
    parser.add_argument("--infer-amp", action="store_true", default=True)
    parser.add_argument("--no-infer-amp", dest="infer_amp", action="store_false")
    return parser.parse_args()


def build_edgecover_tiles(*, crop_h: int, crop_w: int, context: int, valid: int) -> list[dict[str, int]]:
    if context < valid:
        raise ValueError("context must be >= valid")
    pad = (context - valid) // 2
    max_patch_y0 = max(0, crop_h - context)
    max_patch_x0 = max(0, crop_w - context)

    def bounds(length: int) -> list[int]:
        values = {0, int(length)}
        values.update(range(valid, int(length), valid))
        return sorted(values)

    tiles: list[dict[str, int]] = []
    tile_index = 0
    for y0, y1 in zip(bounds(crop_h)[:-1], bounds(crop_h)[1:]):
        for x0, x1 in zip(bounds(crop_w)[:-1], bounds(crop_w)[1:]):
            patch_y0 = min(max(y0 - pad, 0), max_patch_y0)
            patch_x0 = min(max(x0 - pad, 0), max_patch_x0)
            tiles.append(
                {
                    "tile_index": tile_index,
                    "patch_y0": int(patch_y0),
                    "patch_x0": int(patch_x0),
                    "keep_y0": int(y0 - patch_y0),
                    "keep_x0": int(x0 - patch_x0),
                    "keep_h": int(y1 - y0),
                    "keep_w": int(x1 - x0),
                    "valid_y0": int(y0),
                    "valid_x0": int(x0),
                    "valid_y1": int(y1),
                    "valid_x1": int(x1),
                }
            )
            tile_index += 1
    return tiles


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
    domain_count: int = 2,
) -> torch.Tensor:
    base = provider.condition_vector_from_xy(x0=x0, y0=y0, height=height, width=width)
    onehot = torch.zeros(int(domain_count), dtype=base.dtype)
    onehot[int(domain_index)] = 1.0
    return torch.cat([base, onehot], dim=0).contiguous()


def rows_from_emitters(
    emitters,
    *,
    metas: list[dict[str, int]],
    crop_left: int,
    crop_top: int,
    sample_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(int(emitters.probability.numel())):
        batch_idx = int(emitters.batch_index[idx].item())
        meta = metas[batch_idx]
        x_patch = float(emitters.xyz_px_nm[idx, 0].item())
        y_patch = float(emitters.xyz_px_nm[idx, 1].item())
        if not (
            int(meta["keep_x0"]) <= x_patch < int(meta["keep_x0"]) + int(meta["keep_w"])
            and int(meta["keep_y0"]) <= y_patch < int(meta["keep_y0"]) + int(meta["keep_h"])
        ):
            continue
        x_roi = float(meta["patch_x0"]) + x_patch
        y_roi = float(meta["patch_y0"]) + y_patch
        rows.append(
            {
                "frame": int(meta["frame_id"]),
                "x_px": x_roi,
                "y_px": y_roi,
                "x_px_full": x_roi + float(crop_left),
                "y_px_full": y_roi + float(crop_top),
                "z": float(emitters.xyz_px_nm[idx, 2].item()),
                "photon": float(emitters.photons[idx].item()),
                "prob": float(emitters.probability[idx].item()),
                "x_sig": float(emitters.sigma_xy_px[idx, 0].item()),
                "y_sig": float(emitters.sigma_xy_px[idx, 1].item()),
                "z_sig": None,
                "photon_sig": None,
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
    raw_th: float,
    split_th: float,
    prob_threshold: float,
    photon_scale: float,
    z_scale: float,
    crop_left: int,
    crop_top: int,
    sample_name: str,
) -> int:
    if not patches:
        return 0
    batch = torch.stack(patches, dim=0).to(device=device, dtype=torch.float32)
    cond = torch.stack(conds, dim=0).to(device=device, dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(infer_amp) and device.type == "cuda"):
        output = model([batch, cond])
    emitters = decode_legacy_smlm_emitters(
        output.float(),
        raw_th=float(raw_th),
        split_th=float(split_th),
        accept_th=float(prob_threshold),
        photon_scale=float(photon_scale),
        z_scale=float(z_scale),
    )
    rows = rows_from_emitters(emitters, metas=metas, crop_left=crop_left, crop_top=crop_top, sample_name=sample_name)
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
) -> dict[str, object]:
    side_dir = args.output_dir / side
    infer_dir = side_dir / "infer"
    infer_dir.mkdir(parents=True, exist_ok=True)
    provider = FullResZernikeConditioning.from_npz(coeff_map)
    tiles = build_edgecover_tiles(
        crop_h=int(crop_height),
        crop_w=int(crop_width),
        context=int(args.roi_size),
        valid=int(args.valid_roi_size),
    )
    runtime_state = {
        "window_size": 3,
        "conditioning_vector_dim": 10,
        "conditioning_mode": "film",
        "condition_mode": "film",
        "coeff_maps_npz": str(coeff_map),
        "nat_coeff_maps_path": str(coeff_map),
        "append_domain_onehot": True,
        "domain_count": 2,
        "domain_index": int(domain_index),
        "crop_left": int(crop_left),
        "crop_top": int(crop_top),
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
        "roi_size": int(args.roi_size),
        "valid_roi_size": int(args.valid_roi_size),
        "cut_edge_px": int((int(args.roi_size) - int(args.valid_roi_size)) // 2),
        "camera_pixel_nm_x": 101.11,
        "camera_pixel_nm_y": 98.83,
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
    with H5PredictionWriter(pred_path, fieldnames=FIELDNAMES) as writer:
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
                            domain_count=2,
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
                            raw_th=float(args.raw_th),
                            split_th=float(args.split_th),
                            prob_threshold=float(args.prob_threshold),
                            photon_scale=photon_scale,
                            z_scale=z_scale,
                            crop_left=int(crop_left),
                            crop_top=int(crop_top),
                            sample_name=sample_name,
                        )
            total_rows += flush_bucket(
                patches=patches,
                conds=conds,
                metas=metas,
                writer=writer,
                model=model,
                device=device,
                infer_amp=bool(args.infer_amp),
                raw_th=float(args.raw_th),
                split_th=float(args.split_th),
                prob_threshold=float(args.prob_threshold),
                photon_scale=photon_scale,
                z_scale=z_scale,
                crop_left=int(crop_left),
                crop_top=int(crop_top),
                sample_name=sample_name,
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

    prob_tag = f"prob{int(round(float(args.filter_prob_min) * 100)):03d}"
    locprec_tag = "no_locprec" if args.locprec_xy_nm_max is None else f"locprec{int(round(float(args.locprec_xy_nm_max)))}"
    filter_dir = side_dir / f"filter_recon_{prob_tag}_{locprec_tag}"
    filter_script = args.infer_recon_root / "filter" / "apply_filter_recon.py"
    if not filter_script.is_file():
        raise FileNotFoundError(filter_script)
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
        str(pred_path),
        "--width-px",
        str(crop_width),
        "--height-px",
        str(crop_height),
        "--filter-profile",
        "basic" if args.locprec_xy_nm_max is None else "strict",
        "--prob-min",
        str(args.filter_prob_min),
        "--render-pixel-nm",
        str(args.render_pixel_nm),
        "--spot-radius-nm",
        str(args.spot_radius_nm),
        "--renderer",
        "integrated_gaussian",
        "--gamma",
        "1.0",
        "--scale-percentile",
        "99.7",
        "--radius-mode",
        "xy_uncertainty_mean",
        "--filtered-format",
        "h5",
        "--keep-filtered-predictions",
    ]
    if args.locprec_xy_nm_max is not None:
        cmd.extend(["--locprec-xy-nm-max", str(args.locprec_xy_nm_max)])
    subprocess.run(cmd, check=True)
    summary = {
        "side": side,
        "predictions": str(pred_path),
        "filter_recon_dir": str(filter_dir),
        "raw_rows": int(total_rows),
        "elapsed_sec": round(time.time() - started, 2),
        "tiles": len(tiles),
        "tile_geometry": {
            "roi_size": int(args.roi_size),
            "valid_roi_size": int(args.valid_roi_size),
            "cut_edge_px": int((int(args.roi_size) - int(args.valid_roi_size)) // 2),
            "tiling_mode": "edgecover",
        },
    }
    (side_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or (args.run_dir / "checkpoints/checkpoint_latest.pt")
    coeff_maps = final_coeff_maps(args.run_dir)
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
                    domain_index=1,
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
