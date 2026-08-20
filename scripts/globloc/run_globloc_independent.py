#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile

from globloc_core import build_dT_all, match_frame_candidates, roi_start_xy
from globloc_ctypes import GlobLocFit
from globloc_raw_detector import detect_channel
from run_globloc_multichannel import (
    DEFAULT_GLOBLOC_SOURCE,
    DEFAULT_LEFT_PSF,
    DEFAULT_LIBRARY,
    DEFAULT_RAW,
    DEFAULT_RIGHT_PSF,
    ROOT,
    _H5Appender,
    _extract_rois,
    _fields,
    _gpu_snapshot,
    _load_coefficients,
    _rows_from_fit,
    _sha256,
    _write_manifest,
)


def _detected_fields() -> dict[str, np.dtype]:
    return {
        "frame": np.dtype("i4"),
        "channel": np.dtype("i1"),
        "x_px": np.dtype("f4"),
        "y_px": np.dtype("f4"),
        "score": np.dtype("f4"),
    }


def _append_detections(
    appender: _H5Appender,
    frame_id: int,
    channel_id: int,
    detections: dict[str, np.ndarray],
) -> None:
    count = len(detections["x_px"])
    appender.append(
        {
            "frame": np.full(count, frame_id, dtype=np.int32),
            "channel": np.full(count, channel_id, dtype=np.int8),
            "x_px": detections["x_px"],
            "y_px": detections["y_px"],
            "score": detections["score"],
        }
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    raw_tiff = args.raw_tiff.resolve()
    left_psf_path = args.left_psf.resolve()
    right_psf_path = args.right_psf.resolve()
    library_path = args.library.resolve()
    globloc_source = args.globloc_source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_globloc.h5"
    filtered_path = output_dir / "filtered_predictions.h5"
    detected_path = output_dir / "detected_candidates.h5"
    manifest_path = output_dir / "manifest.json"
    diagnostics_path = output_dir / "diagnostics" / "fit_summary.json"
    if predictions_path.exists() or filtered_path.exists() or detected_path.exists():
        raise FileExistsError(f"refusing to overwrite existing independent outputs in {output_dir}")
    for path in (raw_tiff, left_psf_path, right_psf_path, library_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    left_coeff, left_psf_meta = _load_coefficients(left_psf_path, globloc_source)
    right_coeff, right_psf_meta = _load_coefficients(right_psf_path, globloc_source)
    coefficients = np.ascontiguousarray(np.stack([left_coeff, right_coeff]).astype(np.float32))
    fitter = GlobLocFit(library_path)
    fields = _fields()
    detected_fields = _detected_fields()
    attrs = {
        "schema": "globloc_v1.0_independent_raw_detector_v1",
        "candidate_source": "raw TIFF DoG local-max detector",
        "unitypsf_candidate_paths_used": False,
        "fit_method": "official GlobLoc v1.0 GPU spline EMCCD kernel",
    }
    manifest: dict[str, object] = {
        "schema_version": "globloc_microtube_independent_formal_v1",
        "status": "running",
        "formal": True,
        "comparison_valid": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_snapshot": _gpu_snapshot(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "raw_tiff": str(raw_tiff),
        "raw_tiff_sha256": _sha256(raw_tiff),
        "raw_contract": {"shape_tyx": [8000, 1200, 1200], "frame_id_to_raw_index": "frame_id - 1"},
        "candidate_source": "raw TIFF only; no UnityPSF predictions are read",
        "unitypsf_candidate_paths_used": False,
        "detector": {
            "algorithm": "difference_of_gaussians_plus_local_maximum",
            "sigma_signal_px": float(args.sigma_signal),
            "sigma_background_px": float(args.sigma_background),
            "threshold": "median(response) + threshold_sigma * 1.4826 * MAD(response)",
            "threshold_sigma": float(args.threshold_sigma),
            "min_distance_px": int(args.min_distance),
            "exclude_border_px": int(args.exclude_border),
        },
        "matching": {
            "algorithm": "frame-wise greedy one-to-one nearest-distance matching of raw detections",
            "max_distance_px": float(args.match_distance),
        },
        "roi": {
            "size_px": int(args.roi_size),
            "left_crop_x": 0,
            "right_crop_x": int(args.right_crop_x),
            "crop_width_px": 600,
            "crop_height_px": 1200,
        },
        "globloc": {
            "source": str(globloc_source),
            "version": "v1.0",
            "library": str(library_path),
            "library_sha256": _sha256(library_path),
            "kernel": "globloc_fit_multichannel_emccd_spline",
            "iterations": int(args.iterations),
            "initial_z_index": float(args.initial_z_index),
            "shared": [1, 1, 1, 0, 0],
            "parameter_order": [
                "x_shared",
                "y_shared",
                "z_shared",
                "photon_left",
                "photon_right",
                "background_left",
                "background_right",
                "iteration_count",
            ],
            "coefficient_layout": list(coefficients.shape),
        },
        "psf": {
            "left": left_psf_meta,
            "right": right_psf_meta,
            "z_min_nm": float(args.z_min_nm),
            "z_step_nm": float(args.z_step_nm),
        },
        "camera": {
            "pixel_size_nm_x": float(args.pixel_nm_x),
            "pixel_size_nm_y": float(args.pixel_nm_y),
            "fit_noise_model": "EMCCD kernel used on raw uint16 camera values",
        },
        "outputs": {
            "predictions": str(predictions_path),
            "filtered_predictions": str(filtered_path),
            "detected_candidates": str(detected_path),
            "diagnostics": str(diagnostics_path),
        },
    }
    _write_manifest(manifest_path, manifest)

    all_appender = _H5Appender(predictions_path, fields, {**attrs, "columns_json": json.dumps(list(fields))})
    filtered_appender = _H5Appender(filtered_path, fields, {**attrs, "columns_json": json.dumps(list(fields))})
    detected_appender = _H5Appender(detected_path, detected_fields, {**attrs, "columns_json": json.dumps(list(detected_fields))})
    frame_count = 8000 if args.max_frames is None else min(8000, int(args.max_frames))
    counts = {
        "frames_seen": 0,
        "frames_with_left_detections": 0,
        "frames_with_right_detections": 0,
        "frames_with_matches": 0,
        "left_detections": 0,
        "right_detections": 0,
        "matched_candidates": 0,
        "boundary_rejected": 0,
        "fits": 0,
        "valid_fits": 0,
    }
    try:
        with tifffile.TiffFile(raw_tiff) as raw_handle:
            series = raw_handle.series[0]
            raw_shape = tuple(int(value) for value in series.shape)
            if raw_shape != (8000, 1200, 1200):
                raise ValueError(f"expected raw shape (8000, 1200, 1200), got {raw_shape}")
            for frame_id in range(1, frame_count + 1):
                counts["frames_seen"] += 1
                image = np.asarray(series.asarray(key=frame_id - 1), dtype=np.float32)
                left_detection = detect_channel(
                    image[:, :600],
                    sigma_signal=args.sigma_signal,
                    sigma_background=args.sigma_background,
                    threshold_sigma=args.threshold_sigma,
                    min_distance=args.min_distance,
                    exclude_border=args.exclude_border,
                )
                right_detection = detect_channel(
                    image[:, 600:],
                    sigma_signal=args.sigma_signal,
                    sigma_background=args.sigma_background,
                    threshold_sigma=args.threshold_sigma,
                    min_distance=args.min_distance,
                    exclude_border=args.exclude_border,
                )
                _append_detections(detected_appender, frame_id, 0, left_detection)
                _append_detections(detected_appender, frame_id, 1, right_detection)
                left_count = len(left_detection["x_px"])
                right_count = len(right_detection["x_px"])
                counts["left_detections"] += left_count
                counts["right_detections"] += right_count
                counts["frames_with_left_detections"] += int(left_count > 0)
                counts["frames_with_right_detections"] += int(right_count > 0)
                left_xy = np.column_stack([left_detection["x_px"], left_detection["y_px"]])
                right_xy = np.column_stack([right_detection["x_px"], right_detection["y_px"]])
                pairs, distances = match_frame_candidates(left_xy, right_xy, args.match_distance)
                if not len(pairs):
                    continue
                counts["frames_with_matches"] += 1
                left_indices = pairs[:, 0]
                right_indices = pairs[:, 1]
                matched_left_xy = left_xy[left_indices]
                matched_right_xy = right_xy[right_indices]
                left_starts = np.asarray([roi_start_xy(x, y, args.roi_size) for x, y in matched_left_xy], dtype=np.int32)
                right_starts = np.asarray([roi_start_xy(x, y, args.roi_size) for x, y in matched_right_xy], dtype=np.int32)
                boundary_ok = (
                    (left_starts[:, 0] >= 0)
                    & (left_starts[:, 1] >= 0)
                    & (left_starts[:, 0] + args.roi_size <= args.right_crop_x)
                    & (left_starts[:, 1] + args.roi_size <= 1200)
                    & (right_starts[:, 0] >= 0)
                    & (right_starts[:, 1] >= 0)
                    & (right_starts[:, 0] + args.roi_size <= 600)
                    & (right_starts[:, 1] + args.roi_size <= 1200)
                )
                counts["boundary_rejected"] += int(np.count_nonzero(~boundary_ok))
                left_indices = left_indices[boundary_ok]
                right_indices = right_indices[boundary_ok]
                matched_left_xy = matched_left_xy[boundary_ok]
                matched_right_xy = matched_right_xy[boundary_ok]
                distances = distances[boundary_ok]
                if args.max_fits is not None:
                    remaining = int(args.max_fits) - int(counts["fits"])
                    if remaining <= 0:
                        break
                    left_indices = left_indices[:remaining]
                    right_indices = right_indices[:remaining]
                    matched_left_xy = matched_left_xy[:remaining]
                    matched_right_xy = matched_right_xy[:remaining]
                    distances = distances[:remaining]
                for batch_start in range(0, len(left_indices), int(args.batch_size)):
                    batch_stop = min(batch_start + int(args.batch_size), len(left_indices))
                    batch_left_xy = matched_left_xy[batch_start:batch_stop]
                    batch_right_xy = matched_right_xy[batch_start:batch_stop]
                    left_rois, right_rois, batch_left_starts, batch_right_starts = _extract_rois(
                        image, batch_left_xy, batch_right_xy, 0, int(args.right_crop_x), int(args.roi_size)
                    )
                    dts = build_dT_all(batch_left_xy, batch_right_xy)
                    shared = np.repeat(np.asarray([[1, 1, 1, 0, 0]], dtype=np.int32), len(batch_left_xy), axis=0)
                    initial_z = np.full(len(batch_left_xy), float(args.initial_z_index), dtype=np.float32)
                    parameters, crlbs, log_likelihood = fitter.fit(
                        np.ascontiguousarray(np.stack([left_rois, right_rois])),
                        shared,
                        int(args.iterations),
                        coefficients,
                        dts,
                        initial_z,
                    )
                    rows, valid = _rows_from_fit(
                        frame_id,
                        left_indices[batch_start:batch_stop],
                        right_indices[batch_start:batch_stop],
                        batch_left_starts,
                        batch_right_starts,
                        distances[batch_start:batch_stop],
                        dts,
                        parameters,
                        crlbs,
                        log_likelihood,
                        np.zeros(len(batch_left_xy), dtype=np.float32),
                        np.zeros(len(batch_right_xy), dtype=np.float32),
                        float(args.pixel_nm_x),
                        float(args.pixel_nm_y),
                        float(args.z_min_nm),
                        float(args.z_step_nm),
                        int(args.right_crop_x),
                    )
                    all_appender.append(rows)
                    filtered_appender.append({name: values[valid] for name, values in rows.items()})
                    counts["fits"] += int(len(valid))
                    counts["valid_fits"] += int(np.count_nonzero(valid))
                counts["matched_candidates"] += int(len(left_indices))
                if frame_id == 1 or frame_id % 100 == 0:
                    print(json.dumps({"frame": frame_id, **counts}), flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        manifest["counts"] = counts
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        all_appender.close()
        filtered_appender.close()
        detected_appender.close()

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        **counts,
        "output_count": int(all_appender.count),
        "filtered_output_count": int(filtered_appender.count),
        "detected_candidate_count": int(detected_appender.count),
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    manifest["status"] = "completed"
    manifest["counts"] = diagnostics
    manifest["frames_processed"] = frame_count
    _write_manifest(manifest_path, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw-TIFF-only detection followed by official GlobLoc v1.0 fitting.")
    parser.add_argument("--raw-tiff", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--left-psf", type=Path, default=DEFAULT_LEFT_PSF)
    parser.add_argument("--right-psf", type=Path, default=DEFAULT_RIGHT_PSF)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--globloc-source", type=Path, default=DEFAULT_GLOBLOC_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/microtube_globloc/globloc_independent_dual")
    parser.add_argument("--sigma-signal", type=float, default=1.0)
    parser.add_argument("--sigma-background", type=float, default=3.0)
    parser.add_argument("--threshold-sigma", type=float, default=6.0)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--exclude-border", type=int, default=10)
    parser.add_argument("--match-distance", type=float, default=2.0)
    parser.add_argument("--roi-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--initial-z-index", type=float, default=6.0)
    parser.add_argument("--right-crop-x", type=int, default=600)
    parser.add_argument("--pixel-nm-x", type=float, default=101.11)
    parser.add_argument("--pixel-nm-y", type=float, default=98.83)
    parser.add_argument("--z-min-nm", type=float, default=-600.0)
    parser.add_argument("--z-step-nm", type=float, default=100.0)
    parser.add_argument("--max-frames", type=int, default=None, help="Probe-only frame limit")
    parser.add_argument("--max-fits", type=int, default=None, help="Probe-only total-fit limit")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
