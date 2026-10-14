#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unity_psf.infer_recon.predictions_io import (
    H5PredictionWriter,
    default_predictions_path,
    is_h5_path,
    iter_prediction_rows,
    prediction_attributes,
    prediction_fieldnames,
)
from unity_psf.infer_recon.filter.filter import FilterConfig, compute_locprec_xy_nm
from unity_psf.infer_recon.standard import camera_pixels_from_runtime, read_json, write_json


COMPACT_RENDER_COLUMNS = (
    "frame",
    "x_px",
    "y_px",
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
    "locprec_xy_nm",
    "fit_status",
    "postfit_status",
    "LLrel",
    "llrel",
    "logLikelihood",
    "log_likelihood",
    "negative_log_likelihood",
    "PSFxpix",
    "PSFypix",
    "PSFxnm",
    "PSFynm",
    "psf_x_nm",
    "psf_y_nm",
    "psf_xy_nm",
)


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _first_optional_float(row: dict[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _psf_xy_nm_from_row(row: dict[str, object]) -> float | None:
    direct = _first_optional_float(row, ("psf_xy_nm", "PSFxy_nm", "PSFxynm"))
    if direct is not None:
        return direct
    psf_x = _first_optional_float(row, ("psf_x_nm", "PSFxnm"))
    psf_y = _first_optional_float(row, ("psf_y_nm", "PSFynm"))
    if psf_x is None:
        return None
    if psf_y is None:
        psf_y = psf_x
    return ((psf_x * psf_x + psf_y * psf_y) / 2.0) ** 0.5


def _compact_fieldnames(source_fieldnames: list[str]) -> list[str]:
    source = set(source_fieldnames)
    return [key for key in COMPACT_RENDER_COLUMNS if key == "locprec_xy_nm" or key in source]


def _write_filtered_rows_streaming(
    *,
    predictions: Path,
    output_predictions: Path,
    config: FilterConfig,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
) -> tuple[dict[str, int], list[str]]:
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    source_fieldnames = prediction_fieldnames(predictions)
    rows_iter = iter_prediction_rows(predictions)
    fieldnames = _compact_fieldnames(source_fieldnames)
    summary = {
        "total_in": 0,
        "quality_total_rows": 0,
        "quality_locprec_xy_nm_rows": 0,
        "quality_llrel_rows": 0,
        "quality_psf_xy_nm_rows": 0,
        "after_prob": 0,
        "after_frame": 0,
        "after_emitter_z": 0,
        "after_locprec_xy_nm": 0,
        "after_llrel": 0,
        "after_psf_xy_nm": 0,
        "total_out": 0,
    }
    if config.locprec_xy_nm_max is not None:
        summary["missing_locprec_xy_nm_for_requested_gate"] = 0
    if config.llrel_min is not None:
        summary["missing_llrel_for_requested_gate"] = 0
    if config.psf_xy_nm_max is not None:
        summary["missing_psf_xy_nm_for_requested_gate"] = 0
    if config.x_sig_px_max is not None:
        summary["missing_x_sig_for_requested_gate"] = 0
    if config.y_sig_px_max is not None:
        summary["missing_y_sig_for_requested_gate"] = 0

    def _iter_filtered_rows():
        for row in rows_iter:
            row_out = dict(row)
            locprec = compute_locprec_xy_nm(
                x_sig_px=_optional_float(row_out.get("x_sig")),
                y_sig_px=_optional_float(row_out.get("y_sig")),
                camera_pixel_nm_x=float(camera_pixel_nm_x),
                camera_pixel_nm_y=float(camera_pixel_nm_y),
            )
            row_out["locprec_xy_nm"] = locprec

            summary["total_in"] += 1
            summary["quality_total_rows"] += 1
            if locprec is not None:
                summary["quality_locprec_xy_nm_rows"] += 1
            if _first_optional_float(row_out, ("llrel", "LLrel")) is not None:
                summary["quality_llrel_rows"] += 1
            if _psf_xy_nm_from_row(row_out) is not None:
                summary["quality_psf_xy_nm_rows"] += 1

            if config.prob_min is not None and float(row_out["prob"]) < float(config.prob_min):
                continue
            summary["after_prob"] += 1

            if config.frame_min is not None or config.frame_max is not None:
                frame = int(row_out["frame"])
                if config.frame_min is not None and frame < int(config.frame_min):
                    continue
                if config.frame_max is not None and frame > int(config.frame_max):
                    continue
            summary["after_frame"] += 1

            z_nm = _optional_float(row_out.get("z"))
            if config.emitter_z_min_nm is not None or config.emitter_z_max_nm is not None:
                if z_nm is None:
                    continue
                if config.emitter_z_min_nm is not None and z_nm <= float(config.emitter_z_min_nm):
                    continue
                if config.emitter_z_max_nm is not None and z_nm >= float(config.emitter_z_max_nm):
                    continue
            summary["after_emitter_z"] += 1

            llrel = _first_optional_float(row_out, ("llrel", "LLrel"))
            psf_xy_nm = _psf_xy_nm_from_row(row_out)
            if config.locprec_xy_nm_max is not None and locprec is None:
                summary["missing_locprec_xy_nm_for_requested_gate"] += 1
            if config.llrel_min is not None and llrel is None:
                summary["missing_llrel_for_requested_gate"] += 1
            if config.psf_xy_nm_max is not None and psf_xy_nm is None:
                summary["missing_psf_xy_nm_for_requested_gate"] += 1
            if config.x_sig_px_max is not None and _optional_float(row_out.get("x_sig")) is None:
                summary["missing_x_sig_for_requested_gate"] += 1
            if config.y_sig_px_max is not None and _optional_float(row_out.get("y_sig")) is None:
                summary["missing_y_sig_for_requested_gate"] += 1

            if config.locprec_xy_nm_max is not None and (locprec is None or locprec > float(config.locprec_xy_nm_max)):
                continue
            summary["after_locprec_xy_nm"] += 1

            if config.llrel_min is not None and (llrel is None or llrel < float(config.llrel_min)):
                continue
            summary["after_llrel"] += 1

            if config.psf_xy_nm_max is not None and (psf_xy_nm is None or psf_xy_nm > float(config.psf_xy_nm_max)):
                continue
            summary["after_psf_xy_nm"] += 1

            if config.x_sig_px_max is not None:
                x_sig = _optional_float(row_out.get("x_sig"))
                if x_sig is None or abs(x_sig) > float(config.x_sig_px_max):
                    continue
            summary["after_x_sig_px"] = summary.get("after_x_sig_px", 0) + 1

            if config.y_sig_px_max is not None:
                y_sig = _optional_float(row_out.get("y_sig"))
                if y_sig is None or abs(y_sig) > float(config.y_sig_px_max):
                    continue
            summary["after_y_sig_px"] = summary.get("after_y_sig_px", 0) + 1

            summary["total_out"] += 1
            yield {key: row_out.get(key, "") for key in fieldnames}

    if is_h5_path(output_predictions):
        buffered_rows: list[dict[str, object]] = []
        source_attributes = prediction_attributes(predictions)
        source_schema = str(source_attributes.pop("schema", "infer_recon_predictions_h5_v0.1"))
        source_attributes["derived_kind"] = "probability_filtered_localizations"
        source_attributes["source_predictions"] = str(predictions)
        with H5PredictionWriter(
            output_predictions,
            fieldnames=fieldnames,
            schema=source_schema,
            attributes=source_attributes,
        ) as writer:
            for filtered_row in _iter_filtered_rows():
                buffered_rows.append(filtered_row)
                if len(buffered_rows) >= 65536:
                    writer.append_rows(buffered_rows)
                    buffered_rows.clear()
            if buffered_rows:
                writer.append_rows(buffered_rows)
    else:
        with output_predictions.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for filtered_row in _iter_filtered_rows():
                writer.writerow(filtered_row)
    return summary, fieldnames


def _filter_config_from_args(args: argparse.Namespace) -> FilterConfig:
    locprec_min = getattr(args, "locprec_xy_nm_min", None)
    if str(getattr(args, "filter_profile", "basic")) == "strict" and locprec_min is None:
        locprec_min = 0.0
    return FilterConfig(
        prob_min=args.prob_min,
        frame_min=args.frame_min,
        frame_max=args.frame_max,
        emitter_z_min_nm=args.emitter_z_min_nm,
        emitter_z_max_nm=args.emitter_z_max_nm,
        locprec_xy_nm_min=locprec_min,
        locprec_xy_nm_max=args.locprec_xy_nm_max,
        photon_min=getattr(args, "photon_min", None),
        photon_max=getattr(args, "photon_max", None),
        x_sig_px_max=getattr(args, "x_sig_px_max", None),
        y_sig_px_max=getattr(args, "y_sig_px_max", None),
        llrel_min=args.llrel_min,
        psf_xy_nm_max=args.psf_xy_nm_max,
        require_fit_status=bool(getattr(args, "require_fit_status", False)),
    )


def _build_render_command(
    *,
    args: argparse.Namespace,
    infer_dir: Path,
    runtime_state: Path,
    filtered_csv: Path,
    recon_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "recon" / "render_standard.py"),
        "--infer-dir",
        str(infer_dir),
        "--runtime-state",
        str(runtime_state),
        "--predictions",
        str(filtered_csv),
        "--output-dir",
        str(recon_dir),
        "--prob-threshold",
        "0.0",
        "--render-pixel-nm",
        str(args.render_pixel_nm),
        "--spot-radius-nm",
        str(args.spot_radius_nm),
        "--renderer",
        str(args.renderer),
        "--render-weight",
        str(args.render_weight),
        "--gamma",
        str(args.gamma),
        "--display-mode",
        str(args.display_mode),
        "--display-imax-min",
        str(args.display_imax_min),
        "--brightness",
        str(args.brightness),
        "--radius-mode",
        str(args.radius_mode),
        "--uncertainty-cap-mode",
        str(getattr(args, "uncertainty_cap_mode", "median10")),
        "--uncertainty-scale",
        str(args.uncertainty_scale),
        "--uncertainty-min-sigma-px",
        str(args.uncertainty_min_sigma_px),
        "--uncertainty-max-sigma-px",
        str(args.uncertainty_max_sigma_px),
        "--uncertainty-bin-size-px",
        str(args.uncertainty_bin_size_px),
        "--z-min",
        str(args.z_min_nm),
        "--z-max",
        str(args.z_max_nm),
        "--suffix",
        str(args.suffix),
    ]
    if args.display_imax is not None:
        cmd.extend(["--display-imax", str(args.display_imax)])
    if args.normalization_fov:
        cmd.extend(["--normalization-fov", str(args.normalization_fov)])
    if args.width_px is not None:
        cmd.extend(["--width-px", str(args.width_px)])
    if args.height_px is not None:
        cmd.extend(["--height-px", str(args.height_px)])
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SMAP-like infer filters before standard reconstruction.")
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-state", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--width-px", type=int, default=None)
    parser.add_argument("--height-px", type=int, default=None)
    parser.add_argument("--filter-profile", choices=["basic", "strict"], default="basic")
    parser.add_argument("--prob-min", type=float, default=0.70)
    parser.add_argument("--frame-min", type=int, default=None)
    parser.add_argument("--frame-max", type=int, default=None)
    parser.add_argument("--emitter-z-min", "--emitter-z-min-nm", dest="emitter_z_min_nm", type=float, default=None)
    parser.add_argument("--emitter-z-max", "--emitter-z-max-nm", dest="emitter_z_max_nm", type=float, default=None)
    parser.add_argument("--locprec-xy-nm-min", type=float, default=None)
    parser.add_argument("--locprec-xy-nm-max", type=float, default=None)
    parser.add_argument("--photon-min", type=float, default=None)
    parser.add_argument("--photon-max", type=float, default=None)
    parser.add_argument("--x-sig-px-max", type=float, default=None)
    parser.add_argument("--y-sig-px-max", type=float, default=None)
    parser.add_argument("--llrel-min", type=float, default=None)
    parser.add_argument("--psf-xy-nm-max", type=float, default=None)
    parser.add_argument("--require-fit-status", action="store_true")
    parser.add_argument("--render-pixel-nm", type=float, default=20.0)
    parser.add_argument("--spot-radius-nm", type=float, default=28.0)
    parser.add_argument("--renderer", choices=["subpixel", "integrated_gaussian", "liteloc_style"], default="integrated_gaussian")
    parser.add_argument("--render-weight", choices=["count", "photon", "probability"], default="count")
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
    parser.add_argument("--z-min", "--z-min-nm", dest="z_min_nm", type=float, default=-600.0)
    parser.add_argument("--z-max", "--z-max-nm", dest="z_max_nm", type=float, default=600.0)
    parser.add_argument("--suffix", type=str, default="filtered_recon")
    parser.add_argument("--filtered-format", choices=["h5", "csv"], default="h5")
    parser.add_argument("--keep-filtered-csv", action="store_true")
    parser.add_argument("--keep-filtered-predictions", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    infer_dir = args.infer_dir
    predictions = args.predictions or default_predictions_path(infer_dir)
    runtime_state = args.runtime_state or (infer_dir / "derived_runtime_state.json")
    if not runtime_state.is_file():
        parent_runtime = infer_dir.parent / "derived_runtime_state.json"
        if parent_runtime.is_file():
            runtime_state = parent_runtime
    if not predictions.is_file():
        raise FileNotFoundError(predictions)
    if not runtime_state.is_file():
        raise FileNotFoundError(runtime_state)

    runtime = read_json(runtime_state)
    camera_pixel_nm_x, camera_pixel_nm_y = camera_pixels_from_runtime(runtime)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_format = str(getattr(args, "filtered_format", "h5"))
    filtered_predictions = args.output_dir / f"filtered_predictions.{filtered_format}"
    summary_json = args.output_dir / "filter_summary.json"
    recon_dir = args.output_dir / "recon"
    filter_config = _filter_config_from_args(args)
    filter_summary, filtered_columns = _write_filtered_rows_streaming(
        predictions=predictions,
        output_predictions=filtered_predictions,
        config=filter_config,
        camera_pixel_nm_x=float(camera_pixel_nm_x),
        camera_pixel_nm_y=float(camera_pixel_nm_y),
    )
    keep_filtered_csv = bool(getattr(args, "keep_filtered_csv", False))
    keep_filtered_predictions = bool(getattr(args, "keep_filtered_predictions", False) or keep_filtered_csv)
    filter_payload = {
        "predictions": str(predictions),
        "filtered_predictions": str(filtered_predictions),
        "filtered_predictions_columns": filtered_columns,
        "filtered_predictions_format": filtered_format,
        "filtered_predictions_kept": keep_filtered_predictions,
        "runtime_state": str(runtime_state),
        "camera_pixel_nm_x": float(camera_pixel_nm_x),
        "camera_pixel_nm_y": float(camera_pixel_nm_y),
        "filters": {
            "filter_profile": str(getattr(args, "filter_profile", "basic")),
            "prob_min": args.prob_min,
            "frame_min": args.frame_min,
            "frame_max": args.frame_max,
            "emitter_z_min_nm": args.emitter_z_min_nm,
            "emitter_z_max_nm": args.emitter_z_max_nm,
            "locprec_xy_nm_min": filter_config.locprec_xy_nm_min,
            "locprec_xy_nm_max": args.locprec_xy_nm_max,
            "photon_min": getattr(args, "photon_min", None),
            "photon_max": getattr(args, "photon_max", None),
            "x_sig_px_max": getattr(args, "x_sig_px_max", None),
            "y_sig_px_max": getattr(args, "y_sig_px_max", None),
            "llrel_min": args.llrel_min,
            "psf_xy_nm_max": args.psf_xy_nm_max,
            "require_fit_status": bool(getattr(args, "require_fit_status", False)),
        },
        **filter_summary,
    }
    write_json(summary_json, filter_payload)

    render_cmd = _build_render_command(
        args=args,
        infer_dir=infer_dir,
        runtime_state=runtime_state,
        filtered_csv=filtered_predictions,
        recon_dir=recon_dir,
    )
    manifest = {
        "standard": "infer_recon_filter_then_recon_v0.1",
        "infer_dir": str(infer_dir),
        "predictions": str(predictions),
        "filtered_predictions": str(filtered_predictions),
        "runtime_state": str(runtime_state),
        "output_dir": str(args.output_dir),
        "recon_dir": str(recon_dir),
        "filter_summary_json": str(summary_json),
        "filtered_predictions_format": filtered_format,
        "filtered_predictions_kept": keep_filtered_predictions,
        "filter_summary": filter_payload,
        "render_cmd": render_cmd,
    }
    subprocess.run(render_cmd, check=True)
    if not keep_filtered_predictions and filtered_predictions.exists():
        filtered_predictions.unlink()
        manifest["filtered_predictions_removed_after_render"] = True
    else:
        manifest["filtered_predictions_removed_after_render"] = False
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    print(json.dumps(run(_parse_args()), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
