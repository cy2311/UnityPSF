#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unity_psf.infer_recon.predictions_io import default_predictions_path
from unity_psf.infer_recon.standard import camera_pixels_from_runtime, read_json, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard subpixel Gaussian reconstruction from standard infer output.")
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--runtime-state", type=Path, default=None)
    parser.add_argument("--width-px", type=int, default=None)
    parser.add_argument("--height-px", type=int, default=None)
    parser.add_argument("--render-pixel-nm", type=float, default=20.0)
    parser.add_argument("--spot-radius-nm", type=float, default=28.0)
    parser.add_argument("--prob-threshold", type=float, default=0.70)
    parser.add_argument("--z-min", "--z-min-nm", dest="z_min_nm", type=float, default=-600.0)
    parser.add_argument("--z-max", "--z-max-nm", dest="z_max_nm", type=float, default=600.0)
    parser.add_argument("--render-weight", choices=["count", "photon", "probability"], default="count")
    parser.add_argument("--suffix", type=str, default="standard_fixed128_keep108_full8000")
    parser.add_argument("--renderer", choices=["subpixel", "integrated_gaussian", "liteloc_style"], default="subpixel")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--display-mode", choices=["quantile", "fixed_imax"], default="quantile")
    parser.add_argument("--display-imax", type=float, default=None)
    parser.add_argument("--display-imax-min", type=float, default=-2.5228787452803374)
    parser.add_argument("--normalization-fov", type=str, default=None)
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--radius-mode", choices=["fixed", "xy_uncertainty_mean"], default="fixed")
    parser.add_argument("--uncertainty-cap-mode", choices=["fixed", "median10"], default="fixed")
    parser.add_argument("--uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--uncertainty-min-sigma-px", type=float, default=0.75)
    parser.add_argument("--uncertainty-max-sigma-px", type=float, default=6.0)
    parser.add_argument("--uncertainty-bin-size-px", type=float, default=0.5)
    return parser.parse_args()


def _resolve_render_dimensions(args: argparse.Namespace, runtime: dict[str, object]) -> tuple[int, int]:
    width_px = args.width_px if args.width_px is not None else runtime.get("crop_width")
    height_px = args.height_px if args.height_px is not None else runtime.get("crop_height")
    if width_px is None or height_px is None:
        raise KeyError("render dimensions require --width-px/--height-px or runtime crop_width/crop_height")
    width = int(width_px)
    height = int(height_px)
    if width <= 0 or height <= 0:
        raise ValueError(f"render dimensions must be positive, got width_px={width}, height_px={height}")
    return width, height


def main() -> int:
    args = _parse_args()
    infer_dir = args.infer_dir
    predictions = args.predictions or default_predictions_path(infer_dir)
    runtime_state = args.runtime_state or (infer_dir / "derived_runtime_state.json")
    if not runtime_state.is_file():
        parent_runtime = infer_dir.parent / "derived_runtime_state.json"
        if parent_runtime.is_file():
            runtime_state = parent_runtime
    for path in [predictions, runtime_state]:
        if not path.is_file():
            raise FileNotFoundError(path)
    # The delegated renderer runs from PROJECT_ROOT. Resolve all external paths
    # before that cwd change so CLI use is independent of the caller's cwd.
    infer_dir = infer_dir.resolve()
    predictions = predictions.resolve()
    runtime_state = runtime_state.resolve()
    args.output_dir = args.output_dir.resolve()
    runtime = read_json(runtime_state)
    camera_x, camera_y = camera_pixels_from_runtime(runtime)
    width_px, height_px = _resolve_render_dimensions(args, runtime)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.suffix}_prob{args.prob_threshold:.2f}_r{args.spot_radius_nm:g}nm"
    manifest = {
        "standard": "infer_recon_reconstruction_v0.2",
        "renderer": str(args.renderer),
        "infer_dir": str(infer_dir),
        "predictions": str(predictions),
        "runtime_state": str(runtime_state),
        "width_px": int(width_px),
        "height_px": int(height_px),
        "camera_pixel_nm_x": float(camera_x),
        "camera_pixel_nm_y": float(camera_y),
        "render_pixel_nm": float(args.render_pixel_nm),
        "spot_radius_nm": float(args.spot_radius_nm),
        "prob_threshold": float(args.prob_threshold),
        "z_unit": "nm",
        "z_min_nm": float(args.z_min_nm),
        "z_max_nm": float(args.z_max_nm),
        "render_weight": str(args.render_weight),
        "gamma": float(args.gamma),
        "brightness": float(args.brightness),
        "display_mode": str(args.display_mode),
        "display_imax": float(args.display_imax) if args.display_imax is not None else None,
        "display_imax_min": float(args.display_imax_min),
        "normalization_fov": args.normalization_fov,
        "radius_mode": str(args.radius_mode),
        "uncertainty_cap_mode": str(args.uncertainty_cap_mode),
        "uncertainty_scale": float(args.uncertainty_scale),
        "uncertainty_min_sigma_px": float(args.uncertainty_min_sigma_px),
        "uncertainty_max_sigma_px": float(args.uncertainty_max_sigma_px),
        "uncertainty_bin_size_px": float(args.uncertainty_bin_size_px),
        "suffix": suffix,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    renderer_script = (
        Path(__file__).resolve().with_name("render_subpixel.py")
        if str(args.renderer) in {"subpixel", "integrated_gaussian"}
        else PROJECT_ROOT / "scripts" / "analysis" / "render_liteloc_style_reconstruction.py"
    )
    cmd = [
        sys.executable,
        str(renderer_script),
        "--predictions",
        str(predictions),
        "--output-dir",
        str(args.output_dir),
        "--width-px",
        str(width_px),
        "--height-px",
        str(height_px),
        "--camera-pixel-nm-x",
        str(camera_x),
        "--camera-pixel-nm-y",
        str(camera_y),
        "--render-pixel-nm",
        str(args.render_pixel_nm),
        "--spot-radius-nm",
        str(args.spot_radius_nm),
        "--prob-threshold",
        str(args.prob_threshold),
        "--z-min",
        str(args.z_min_nm),
        "--z-max",
        str(args.z_max_nm),
        "--suffix",
        suffix,
    ]
    if str(args.renderer) in {"subpixel", "integrated_gaussian"}:
        cmd.extend(
            [
                "--gamma",
                str(args.gamma),
                "--render-weight",
                str(args.render_weight),
                "--brightness",
                str(args.brightness),
                "--display-mode",
                str(args.display_mode),
                "--display-imax-min",
                str(args.display_imax_min),
                "--radius-mode",
                str(args.radius_mode),
                "--uncertainty-cap-mode",
                str(args.uncertainty_cap_mode),
                "--uncertainty-scale",
                str(args.uncertainty_scale),
                "--uncertainty-min-sigma-px",
                str(args.uncertainty_min_sigma_px),
                "--uncertainty-max-sigma-px",
                str(args.uncertainty_max_sigma_px),
                "--uncertainty-bin-size-px",
                str(args.uncertainty_bin_size_px),
            ]
        )
        if args.display_imax is not None:
            cmd.extend(["--display-imax", str(args.display_imax)])
        if args.normalization_fov:
            cmd.extend(["--normalization-fov", str(args.normalization_fov)])
    print(json.dumps({"manifest": manifest, "cmd": cmd}, indent=2), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_ROOT}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(SRC_ROOT)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
