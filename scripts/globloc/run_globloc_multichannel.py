#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import tifffile

from globloc_core import build_dT_all, match_frame_candidates, roi_start_xy
from globloc_ctypes import GlobLocFit


ROOT = Path("/home/guest/Others/main/race")
DEFAULT_RAW = ROOT / "datasets/archive/samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif"
DEFAULT_LEFT_CANDIDATES = ROOT / "results/microtube/microtube_left/left/filter_recon_prob070_zneg400_pos200/filtered_predictions.h5"
DEFAULT_RIGHT_CANDIDATES = ROOT / "results/microtube/microtube_right/right/filter_recon_prob070_zneg400_pos200/filtered_predictions.h5"
DEFAULT_LEFT_PSF = ROOT / "results/microtube_left_retrain_20260807/calibration/decode_left_fixed_defocus_refit_v2/fitted_mean_psf_z_stack.tif"
DEFAULT_RIGHT_PSF = ROOT / "results/microtube_right_benchmark_20260711/calibration/beads_uniform/decode_neptune_fixed_defocus_refit_20260720/fitted_mean_psf_z_stack.tif"
DEFAULT_LIBRARY = ROOT / "results/microtube_globloc/provenance/lib/libglobloc_multichannel.so"
DEFAULT_GLOBLOC_SOURCE = ROOT / "results/microtube_globloc/provenance/GlobLoc-v1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_candidates(path: Path) -> dict[str, np.ndarray | str | int]:
    required = ("frame", "x_px", "y_px", "z_nm", "photon", "prob")
    with h5py.File(path, "r") as handle:
        locs = handle["locs"]
        arrays: dict[str, np.ndarray | str | int] = {
            "frame": np.asarray(locs["frame"], dtype=np.int32),
            "x_px": np.asarray(locs["x_px"], dtype=np.float32),
            "y_px": np.asarray(locs["y_px"], dtype=np.float32),
            "z_nm": np.asarray(locs["z_nm" if "z_nm" in locs else "z"], dtype=np.float32),
            "photon": np.asarray(locs["photon"], dtype=np.float32),
            "prob": np.asarray(locs["prob"], dtype=np.float32),
            "source_path": str(path.resolve()),
            "source_count": int(locs["frame"].shape[0]),
        }
        missing = [name for name in required if name not in locs and not (name == "z_nm" and "z" in locs)]
        if missing:
            raise KeyError(f"{path} is missing candidate fields: {missing}")
    frame = arrays["frame"]
    assert isinstance(frame, np.ndarray)
    if frame.ndim != 1 or np.any(np.diff(frame) < 0):
        raise ValueError(f"candidate frame order is not nondecreasing: {path}")
    return arrays


def _frame_ranges(frames: np.ndarray) -> dict[int, tuple[int, int]]:
    unique, starts, counts = np.unique(frames, return_index=True, return_counts=True)
    return {int(frame): (int(start), int(start + count)) for frame, start, count in zip(unique, starts, counts, strict=True)}


def _load_coefficients(psf_path: Path, globloc_source: Path) -> tuple[np.ndarray, dict[str, object]]:
    python_source = globloc_source / "GlobLoc_python"
    sys.path.insert(0, str(python_source))
    from psf2cspline import psf2cspline_np

    stack = np.asarray(tifffile.imread(psf_path), dtype=np.float32)
    if stack.shape != (13, 40, 40):
        raise ValueError(f"expected fitted PSF stack shape (13, 40, 40), got {stack.shape}: {psf_path}")
    plane_sums = stack.sum(axis=(1, 2), dtype=np.float64)
    if not np.all(np.isfinite(stack)) or np.any(plane_sums <= 0):
        raise ValueError(f"PSF stack contains invalid planes: {psf_path}")
    normalized = stack / plane_sums[:, None, None].astype(np.float32)
    coeff = np.asarray(psf2cspline_np(normalized), dtype=np.float32)
    if coeff.shape != (64, 12, 39, 39):
        raise ValueError(f"unexpected official spline coefficient shape {coeff.shape}: {psf_path}")
    return coeff, {
        "path": str(psf_path.resolve()),
        "sha256": _sha256(psf_path),
        "stack_shape_zyx": list(stack.shape),
        "plane_sum_min_before_normalization": float(plane_sums.min()),
        "plane_sum_max_before_normalization": float(plane_sums.max()),
        "coefficient_shape_64_zyx": list(coeff.shape),
        "z_positions_nm": list(np.arange(-600.0, 601.0, 100.0, dtype=np.float32)),
        "axis_contract": "official psf2cspline_np input z,y,x; kernel sizes x=39,y=39,z=12",
    }


def _extract_rois(
    frame: np.ndarray,
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    left_crop_x: int,
    right_crop_x: int,
    roi_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_starts = np.asarray([roi_start_xy(x, y, roi_size) for x, y in left_xy], dtype=np.int32)
    right_starts = np.asarray([roi_start_xy(x, y, roi_size) for x, y in right_xy], dtype=np.int32)
    offsets = np.arange(roi_size, dtype=np.int32)

    def gather(starts: np.ndarray, x_offset: int) -> np.ndarray:
        xs = starts[:, 0, None] + offsets[None, :] + int(x_offset)
        ys = starts[:, 1, None] + offsets[None, :]
        return frame[ys[:, :, None], xs[:, None, :]].astype(np.float32, copy=False)

    return gather(left_starts, left_crop_x), gather(right_starts, right_crop_x), left_starts, right_starts


class _H5Appender:
    def __init__(self, path: Path, fields: dict[str, np.dtype], attrs: dict[str, object]):
        self.path = path
        self.handle = h5py.File(path, "w")
        self.handle.create_group("locs")
        self.datasets = {
            name: self.handle["locs"].create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(4096,),
            )
            for name, dtype in fields.items()
        }
        for key, value in attrs.items():
            self.handle.attrs[key] = value
        self.count = 0

    def append(self, rows: dict[str, np.ndarray]) -> None:
        length = len(next(iter(rows.values())))
        if not length:
            return
        start = self.count
        stop = start + length
        for name, dataset in self.datasets.items():
            values = np.asarray(rows[name], dtype=dataset.dtype)
            dataset.resize((stop,))
            dataset[start:stop] = values
        self.count = stop

    def close(self) -> None:
        self.handle.attrs["count"] = int(self.count)
        self.handle.flush()
        self.handle.close()


def _fields() -> dict[str, np.dtype]:
    return {
        "frame": np.dtype("i4"),
        "left_candidate_index": np.dtype("i8"),
        "right_candidate_index": np.dtype("i8"),
        "x_px": np.dtype("f4"),
        "y_px": np.dtype("f4"),
        "x_px_left_local": np.dtype("f4"),
        "y_px_left_local": np.dtype("f4"),
        "x_px_right_local": np.dtype("f4"),
        "y_px_right_local": np.dtype("f4"),
        "x_px_right_full": np.dtype("f4"),
        "y_px_right_full": np.dtype("f4"),
        "x_nm": np.dtype("f4"),
        "y_nm": np.dtype("f4"),
        "z": np.dtype("f4"),
        "z_nm": np.dtype("f4"),
        "photon_left": np.dtype("f4"),
        "photon_right": np.dtype("f4"),
        "background_left": np.dtype("f4"),
        "background_right": np.dtype("f4"),
        "crlb_x": np.dtype("f4"),
        "crlb_y": np.dtype("f4"),
        "crlb_z": np.dtype("f4"),
        "log_likelihood": np.dtype("f4"),
        "iterations": np.dtype("f4"),
        "channel_match_distance": np.dtype("f4"),
        "fit_status": np.dtype("i1"),
    }


def _attrs(columns: Iterable[str]) -> dict[str, object]:
    return {
        "schema": "globloc_v1.0_multichannel_fit_v1",
        "columns_json": json.dumps(list(columns)),
        "coordinate_frame": "full_raw_1200x1200_px; x_px is left-channel reference coordinate",
        "units": json.dumps(
            {
                "x_px": "camera_pixel",
                "y_px": "camera_pixel",
                "x_nm": "nm",
                "y_nm": "nm",
                "z": "nm",
                "z_nm": "nm",
                "photon_left": "raw_camera_count_scale",
                "photon_right": "raw_camera_count_scale",
                "background_left": "raw_camera_count_scale",
                "background_right": "raw_camera_count_scale",
                "crlb_x": "nm",
                "crlb_y": "nm",
                "crlb_z": "nm",
                "log_likelihood": "Poisson_log_likelihood",
            }
        ),
        "candidate_source": "existing UnityPSF filtered_predictions.h5",
        "fit_method": "official GlobLoc v1.0 GPU spline EMCCD kernel",
        "shared_parameters": "x,y,z shared; photon/background channel-specific",
    }


def _rows_from_fit(
    frame_id: int,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    left_starts: np.ndarray,
    right_starts: np.ndarray,
    match_distances: np.ndarray,
    dts: np.ndarray,
    parameters: np.ndarray,
    crlbs: np.ndarray,
    log_likelihood: np.ndarray,
    left_z_nm: np.ndarray,
    right_z_nm: np.ndarray,
    pixel_nm_x: float,
    pixel_nm_y: float,
    z_min_nm: float,
    z_step_nm: float,
    right_crop_x: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    x_left_local = parameters[0]
    y_left_local = parameters[1]
    x_right_local = x_left_local + dts[:, 1, 0]
    y_right_local = y_left_local + dts[:, 1, 1]
    x_full = left_starts[:, 0].astype(np.float32) + x_left_local
    y_full = left_starts[:, 1].astype(np.float32) + y_left_local
    x_right_full = right_crop_x + right_starts[:, 0].astype(np.float32) + x_right_local
    y_right_full = right_starts[:, 1].astype(np.float32) + y_right_local
    z_nm = z_min_nm + parameters[2] * z_step_nm
    crlb_x = np.sqrt(np.maximum(crlbs[0], 0.0)) * float(pixel_nm_x)
    crlb_y = np.sqrt(np.maximum(crlbs[1], 0.0)) * float(pixel_nm_y)
    crlb_z = np.sqrt(np.maximum(crlbs[2], 0.0)) * float(z_step_nm)
    finite = np.all(np.isfinite(parameters[:7]), axis=0) & np.all(np.isfinite(crlbs[:3]), axis=0)
    valid = finite & (parameters[3] > 0.0) & (parameters[4] > 0.0) & (parameters[2] >= 0.0) & (parameters[2] <= 12.0)
    rows = {
        "frame": np.full(len(x_full), int(frame_id), dtype=np.int32),
        "left_candidate_index": np.asarray(left_indices, dtype=np.int64),
        "right_candidate_index": np.asarray(right_indices, dtype=np.int64),
        "x_px": x_full.astype(np.float32),
        "y_px": y_full.astype(np.float32),
        "x_px_left_local": x_left_local.astype(np.float32),
        "y_px_left_local": y_left_local.astype(np.float32),
        "x_px_right_local": x_right_local.astype(np.float32),
        "y_px_right_local": y_right_local.astype(np.float32),
        "x_px_right_full": x_right_full.astype(np.float32),
        "y_px_right_full": y_right_full.astype(np.float32),
        "x_nm": (x_full * float(pixel_nm_x)).astype(np.float32),
        "y_nm": (y_full * float(pixel_nm_y)).astype(np.float32),
        "z": z_nm.astype(np.float32),
        "z_nm": z_nm.astype(np.float32),
        "photon_left": parameters[3].astype(np.float32),
        "photon_right": parameters[4].astype(np.float32),
        "background_left": parameters[5].astype(np.float32),
        "background_right": parameters[6].astype(np.float32),
        "crlb_x": crlb_x.astype(np.float32),
        "crlb_y": crlb_y.astype(np.float32),
        "crlb_z": crlb_z.astype(np.float32),
        "log_likelihood": np.asarray(log_likelihood, dtype=np.float32),
        "iterations": parameters[7].astype(np.float32),
        "channel_match_distance": np.asarray(match_distances, dtype=np.float32),
        "fit_status": valid.astype(np.int8),
    }
    return rows, valid


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _gpu_snapshot() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"nvidia-smi unavailable: {exc}"
    return completed.stdout.strip()


def run(args: argparse.Namespace) -> dict[str, object]:
    raw_tiff = args.raw_tiff.resolve()
    left_candidates_path = args.left_candidates.resolve()
    right_candidates_path = args.right_candidates.resolve()
    left_psf_path = args.left_psf.resolve()
    right_psf_path = args.right_psf.resolve()
    library_path = args.library.resolve()
    globloc_source = args.globloc_source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_globloc.h5"
    filtered_path = output_dir / "filtered_predictions.h5"
    manifest_path = output_dir / "manifest.json"
    diagnostics_path = output_dir / "diagnostics" / "fit_summary.json"
    if predictions_path.exists() or filtered_path.exists():
        raise FileExistsError(f"refusing to overwrite existing GlobLoc outputs in {output_dir}")
    for path in (raw_tiff, left_candidates_path, right_candidates_path, left_psf_path, right_psf_path, library_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    left = _load_candidates(left_candidates_path)
    right = _load_candidates(right_candidates_path)
    left_frame = left["frame"]
    right_frame = right["frame"]
    left_x = left["x_px"]
    left_y = left["y_px"]
    left_z = left["z_nm"]
    right_x = right["x_px"]
    right_y = right["y_px"]
    right_z = right["z_nm"]
    assert isinstance(left_frame, np.ndarray) and isinstance(right_frame, np.ndarray)
    assert isinstance(left_x, np.ndarray) and isinstance(left_y, np.ndarray) and isinstance(left_z, np.ndarray)
    assert isinstance(right_x, np.ndarray) and isinstance(right_y, np.ndarray) and isinstance(right_z, np.ndarray)
    left_ranges = _frame_ranges(left_frame)
    right_ranges = _frame_ranges(right_frame)
    frame_ids = sorted(set(left_ranges).intersection(right_ranges))
    if args.max_frames is not None:
        frame_ids = frame_ids[: int(args.max_frames)]
    frame_index_offset = 0 if min(frame_ids, default=0) == 0 else 1

    left_coeff, left_psf_meta = _load_coefficients(left_psf_path, globloc_source)
    right_coeff, right_psf_meta = _load_coefficients(right_psf_path, globloc_source)
    coefficients = np.ascontiguousarray(np.stack([left_coeff, right_coeff]).astype(np.float32))
    fitter = GlobLocFit(library_path)
    fields = _fields()
    columns = list(fields)
    attrs = _attrs(columns)
    attrs.update(
        {
            "raw_tiff": str(raw_tiff),
            "left_candidates": str(left_candidates_path),
            "right_candidates": str(right_candidates_path),
            "left_psf": str(left_psf_path),
            "right_psf": str(right_psf_path),
            "library": str(library_path),
        }
    )
    manifest: dict[str, object] = {
        "schema_version": "globloc_microtube_formal_v1",
        "status": "running",
        "formal": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_snapshot": _gpu_snapshot(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "raw_tiff": str(raw_tiff),
        "raw_tiff_sha256": _sha256(raw_tiff),
        "raw_contract": {"shape_tyx": [8000, 1200, 1200], "frame_id_to_raw_index": "frame_id - 1 for this 1-based candidate set"},
        "left_candidates": {"path": str(left_candidates_path), "sha256": _sha256(left_candidates_path), "count": int(left["source_count"])},
        "right_candidates": {"path": str(right_candidates_path), "sha256": _sha256(right_candidates_path), "count": int(right["source_count"])},
        "candidate_source": "existing UnityPSF filtered predictions; GlobLoc detector is not independent",
        "matching": {"algorithm": "frame-wise greedy one-to-one nearest-distance matching", "max_distance_px": float(args.match_distance)},
        "roi": {"size_px": int(args.roi_size), "left_crop_x": 0, "right_crop_x": int(args.right_crop_x), "crop_width_px": 600, "crop_height_px": 1200},
        "globloc": {
            "source": str(globloc_source),
            "version": "v1.0",
            "library": str(library_path),
            "library_sha256": _sha256(library_path),
            "kernel": "globloc_fit_multichannel_emccd_spline",
            "iterations": int(args.iterations),
            "shared": [1, 1, 1, 0, 0],
            "parameter_order": ["x_shared", "y_shared", "z_shared", "photon_left", "photon_right", "background_left", "background_right", "iteration_count"],
            "coefficient_layout": list(coefficients.shape),
        },
        "psf": {"left": left_psf_meta, "right": right_psf_meta, "z_min_nm": float(args.z_min_nm), "z_step_nm": float(args.z_step_nm)},
        "camera": {"pixel_size_nm_x": float(args.pixel_nm_x), "pixel_size_nm_y": float(args.pixel_nm_y), "fit_noise_model": "EMCCD kernel used on raw uint16 camera values"},
        "outputs": {"predictions": str(predictions_path), "filtered_predictions": str(filtered_path), "diagnostics": str(diagnostics_path)},
    }
    _write_manifest(manifest_path, manifest)

    all_appender = _H5Appender(predictions_path, fields, attrs)
    filtered_appender = _H5Appender(filtered_path, fields, attrs)
    counts = {"frames_seen": 0, "frames_with_matches": 0, "matched_candidates": 0, "boundary_rejected": 0, "fits": 0, "valid_fits": 0}
    try:
        with tifffile.TiffFile(raw_tiff) as raw_handle:
            series = raw_handle.series[0]
            raw_shape = tuple(int(value) for value in series.shape)
            if raw_shape[1:] != (1200, 1200):
                raise ValueError(f"expected raw frame shape (1200, 1200), got {raw_shape}")
            for frame_number, frame_id in enumerate(frame_ids, start=1):
                counts["frames_seen"] += 1
                left_start, left_stop = left_ranges[frame_id]
                right_start, right_stop = right_ranges[frame_id]
                left_xy = np.column_stack([left_x[left_start:left_stop], left_y[left_start:left_stop]])
                right_xy = np.column_stack([right_x[right_start:right_stop], right_y[right_start:right_stop]])
                pairs, distances = match_frame_candidates(left_xy, right_xy, args.match_distance)
                if not len(pairs):
                    continue
                counts["frames_with_matches"] += 1
                left_indices = left_start + pairs[:, 0]
                right_indices = right_start + pairs[:, 1]
                matched_left_xy = left_xy[pairs[:, 0]]
                matched_right_xy = right_xy[pairs[:, 1]]
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
                if not np.any(boundary_ok):
                    continue
                left_indices = left_indices[boundary_ok]
                right_indices = right_indices[boundary_ok]
                matched_left_xy = matched_left_xy[boundary_ok]
                matched_right_xy = matched_right_xy[boundary_ok]
                distances = distances[boundary_ok]
                if args.max_fits is not None:
                    remaining = int(args.max_fits) - int(counts["fits"])
                    if remaining <= 0:
                        break
                    if len(left_indices) > remaining:
                        left_indices = left_indices[:remaining]
                        right_indices = right_indices[:remaining]
                        matched_left_xy = matched_left_xy[:remaining]
                        matched_right_xy = matched_right_xy[:remaining]
                        distances = distances[:remaining]
                image = np.asarray(series.asarray(key=int(frame_id) - frame_index_offset), dtype=np.float32)
                for batch_start in range(0, len(left_indices), int(args.batch_size)):
                    batch_stop = min(batch_start + int(args.batch_size), len(left_indices))
                    batch_left_xy = matched_left_xy[batch_start:batch_stop]
                    batch_right_xy = matched_right_xy[batch_start:batch_stop]
                    left_rois, right_rois, batch_left_starts, batch_right_starts = _extract_rois(
                        image,
                        batch_left_xy,
                        batch_right_xy,
                        0,
                        int(args.right_crop_x),
                        int(args.roi_size),
                    )
                    dts = build_dT_all(batch_left_xy, batch_right_xy)
                    shared = np.repeat(np.asarray([[1, 1, 1, 0, 0]], dtype=np.int32), len(batch_left_xy), axis=0)
                    initial_z_nm = 0.5 * (left_z[left_indices[batch_start:batch_stop]] + right_z[right_indices[batch_start:batch_stop]])
                    initial_z = np.clip((initial_z_nm - float(args.z_min_nm)) / float(args.z_step_nm), 0.0, 12.0).astype(np.float32)
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
                        left_z[left_indices[batch_start:batch_stop]],
                        right_z[right_indices[batch_start:batch_stop]],
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
                if frame_number == 1 or frame_number % 100 == 0:
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

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = {**counts, "output_count": int(all_appender.count), "filtered_output_count": int(filtered_appender.count)}
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    manifest["status"] = "completed"
    manifest["counts"] = diagnostics
    manifest["frame_ids_processed"] = [int(frame) for frame in frame_ids]
    _write_manifest(manifest_path, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official GlobLoc v1.0 two-channel fitting on the microtube raw TIFF.")
    parser.add_argument("--raw-tiff", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--left-candidates", type=Path, default=DEFAULT_LEFT_CANDIDATES)
    parser.add_argument("--right-candidates", type=Path, default=DEFAULT_RIGHT_CANDIDATES)
    parser.add_argument("--left-psf", type=Path, default=DEFAULT_LEFT_PSF)
    parser.add_argument("--right-psf", type=Path, default=DEFAULT_RIGHT_PSF)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--globloc-source", type=Path, default=DEFAULT_GLOBLOC_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/microtube_globloc/globloc_dual")
    parser.add_argument("--match-distance", type=float, default=2.0)
    parser.add_argument("--roi-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
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
