#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import tifffile

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neptune_v03.infer_recon.filter.filter import (
    QUALITY_METRIC_KEYS,
    compute_locprec_xy_nm,
    estimate_gaussian_roi_metrics,
    estimate_gaussian_roi_metrics_moment,
)
from neptune_v03.infer_recon.predictions_io import H5PredictionWriter, prediction_fieldnames
from neptune_v03.infer_recon.standard import camera_pixels_from_runtime, read_json, write_json


warnings.filterwarnings("ignore", message=r".*reading array from closed file.*", category=UserWarning, module=r"tifffile.*")


EXTRA_QUALITY_COLUMNS = (
    "postfit_status",
    "locprec_xy_nm",
    *QUALITY_METRIC_KEYS,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich Neptune v0.3 predictions H5 with raw-TIFF post-fit quality metrics.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample-tiff", type=Path, required=True)
    parser.add_argument("--runtime-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--roi-radius-px", type=int, default=3)
    parser.add_argument("--quality-metric-mode", choices=("moment", "optimize"), default="moment")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--em-factor", type=float, default=1.0)
    parser.add_argument("--num-channels", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def _finite_optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _with_unique_columns(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column not in seen:
            out.append(column)
            seen.add(column)
    return out


def _percentiles(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if finite.size == 0:
        return {key: None for key in ("p01", "p05", "p25", "p50", "p75", "p95", "p99")}
    qs = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    return {key: float(value) for key, value in zip(("p01", "p05", "p25", "p50", "p75", "p95", "p99"), qs)}


def _read_chunk(group: h5py.Group, columns: list[str], start: int, stop: int) -> dict[str, np.ndarray]:
    return {column: np.asarray(group[column][start:stop]) for column in columns if column in group}


def _value_from_arrays(arrays: dict[str, np.ndarray], column: str, ix: int) -> object:
    if column not in arrays:
        return None
    value = arrays[column][ix]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _load_frame_crop(tif: tifffile.TiffFile, frame_index: int, *, crop_left: int, crop_top: int, crop_width: int, crop_height: int) -> np.ndarray:
    frame = np.asarray(tif.series[0].asarray(key=int(frame_index)), dtype=np.float32)
    return np.ascontiguousarray(
        frame[int(crop_top) : int(crop_top) + int(crop_height), int(crop_left) : int(crop_left) + int(crop_width)],
        dtype=np.float32,
    )


def enrich_h5(
    *,
    predictions: Path,
    sample_tiff: Path,
    runtime_state: Path,
    output: Path,
    summary_json: Path,
    roi_radius_px: int,
    quality_metric_mode: str,
    chunk_size: int,
    em_factor: float,
    num_channels: int,
    max_rows: int | None = None,
) -> dict[str, object]:
    runtime = read_json(runtime_state)
    camera_x, camera_y = camera_pixels_from_runtime(runtime)
    crop_left = int(runtime.get("crop_left", 0))
    crop_top = int(runtime.get("crop_top", 0))
    crop_width = int(runtime["crop_width"])
    crop_height = int(runtime["crop_height"])
    source_columns = prediction_fieldnames(predictions)
    output_columns = _with_unique_columns([*source_columns, *EXTRA_QUALITY_COLUMNS])
    estimate_fn = estimate_gaussian_roi_metrics_moment if str(quality_metric_mode) == "moment" else estimate_gaussian_roi_metrics

    summary: dict[str, object] = {
        "predictions": str(predictions),
        "sample_tiff": str(sample_tiff),
        "runtime_state": str(runtime_state),
        "output": str(output),
        "quality_metric_mode": str(quality_metric_mode),
        "roi_radius_px": int(roi_radius_px),
        "camera_pixel_nm_x": float(camera_x),
        "camera_pixel_nm_y": float(camera_y),
        "crop_left": int(crop_left),
        "crop_top": int(crop_top),
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
        "total_in": 0,
        "total_out": 0,
        "fit_status_good": 0,
        "postfit_status_good": 0,
        "quality_llrel_rows": 0,
        "quality_psf_xy_nm_rows": 0,
        "quality_locprec_xy_nm_rows": 0,
    }
    llrel_values: list[float] = []
    psf_xy_values: list[float] = []
    locprec_values: list[float] = []
    x_sig_values: list[float] = []
    y_sig_values: list[float] = []

    with h5py.File(predictions, "r") as src, tifffile.TiffFile(sample_tiff) as tif, H5PredictionWriter(output, fieldnames=output_columns) as writer:
        group = src["locs"]
        if not source_columns:
            raise ValueError(f"No columns found in predictions: {predictions}")
        count = int(group[source_columns[0]].shape[0])
        if max_rows is not None:
            count = min(count, int(max_rows))
        cached_frame_index: int | None = None
        cached_frame: np.ndarray | None = None
        for start in range(0, count, int(chunk_size)):
            stop = min(count, start + int(chunk_size))
            arrays = _read_chunk(group, source_columns, start, stop)
            rows: list[dict[str, object]] = []
            for local_ix in range(stop - start):
                row = {column: _value_from_arrays(arrays, column, local_ix) for column in source_columns}
                frame_index = int(float(row["frame"]))
                x_px = _finite_optional_float(row.get("x_px"))
                y_px = _finite_optional_float(row.get("y_px"))
                x_sig = _finite_optional_float(row.get("x_sig"))
                y_sig = _finite_optional_float(row.get("y_sig"))
                locprec = compute_locprec_xy_nm(
                    x_sig_px=x_sig,
                    y_sig_px=y_sig,
                    camera_pixel_nm_x=float(camera_x),
                    camera_pixel_nm_y=float(camera_y),
                )
                if cached_frame_index != frame_index or cached_frame is None:
                    cached_frame = _load_frame_crop(
                        tif,
                        frame_index,
                        crop_left=crop_left,
                        crop_top=crop_top,
                        crop_width=crop_width,
                        crop_height=crop_height,
                    )
                    cached_frame_index = frame_index
                if x_px is None or y_px is None:
                    metrics = {
                        "fit_status": 0.0,
                        "postfit_status": 0.0,
                        "LLrel": float("nan"),
                        "llrel": float("nan"),
                        "psf_xy_nm": float("nan"),
                    }
                else:
                    metrics = estimate_fn(
                        cached_frame,
                        x_px=float(x_px),
                        y_px=float(y_px),
                        camera_pixel_nm_x=float(camera_x),
                        camera_pixel_nm_y=float(camera_y),
                        roi_radius_px=int(roi_radius_px),
                        em_factor=float(em_factor),
                        num_channels=int(num_channels),
                    )
                    metrics["postfit_status"] = float(metrics.get("fit_status", 0.0))
                row.update(metrics)
                row["locprec_xy_nm"] = locprec

                fit_status = _finite_optional_float(row.get("fit_status"))
                postfit_status = _finite_optional_float(row.get("postfit_status"))
                llrel = _finite_optional_float(row.get("llrel"))
                psf_xy = _finite_optional_float(row.get("psf_xy_nm"))
                if fit_status is not None and fit_status > 0:
                    summary["fit_status_good"] = int(summary["fit_status_good"]) + 1
                if postfit_status is not None and postfit_status > 0:
                    summary["postfit_status_good"] = int(summary["postfit_status_good"]) + 1
                if llrel is not None:
                    summary["quality_llrel_rows"] = int(summary["quality_llrel_rows"]) + 1
                    llrel_values.append(llrel)
                if psf_xy is not None:
                    summary["quality_psf_xy_nm_rows"] = int(summary["quality_psf_xy_nm_rows"]) + 1
                    psf_xy_values.append(psf_xy)
                if locprec is not None:
                    summary["quality_locprec_xy_nm_rows"] = int(summary["quality_locprec_xy_nm_rows"]) + 1
                    locprec_values.append(float(locprec))
                if x_sig is not None:
                    x_sig_values.append(x_sig)
                if y_sig is not None:
                    y_sig_values.append(y_sig)
                rows.append(row)

            writer.append_rows(rows)
            summary["total_in"] = int(summary["total_in"]) + len(rows)
            summary["total_out"] = int(summary["total_out"]) + len(rows)

    total = max(int(summary["total_out"]), 1)
    summary["fit_status_good_fraction"] = float(int(summary["fit_status_good"]) / total)
    summary["postfit_status_good_fraction"] = float(int(summary["postfit_status_good"]) / total)
    summary["llrel_percentiles"] = _percentiles(llrel_values)
    summary["psf_xy_nm_percentiles"] = _percentiles(psf_xy_values)
    summary["locprec_xy_nm_percentiles"] = _percentiles(locprec_values)
    summary["x_sig_px_percentiles"] = _percentiles(x_sig_values)
    summary["y_sig_px_percentiles"] = _percentiles(y_sig_values)
    write_json(summary_json, summary)
    return summary


def main() -> int:
    args = _parse_args()
    print(
        json.dumps(
            enrich_h5(
                predictions=args.predictions,
                sample_tiff=args.sample_tiff,
                runtime_state=args.runtime_state,
                output=args.output,
                summary_json=args.summary_json,
                roi_radius_px=int(args.roi_radius_px),
                quality_metric_mode=str(args.quality_metric_mode),
                chunk_size=int(args.chunk_size),
                em_factor=float(args.em_factor),
                num_channels=int(args.num_channels),
                max_rows=args.max_rows,
            ),
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
