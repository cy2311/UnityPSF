#!/usr/bin/env python3
"""Measure UnityPSF v2 direct CUDA Zernike-map simulation throughput."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch


ROOT = Path(__file__).resolve().parents[2]
V03_ROOT = ROOT / "neptune_v0.3"
for path in (V03_ROOT, V03_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neptune_v03.localization.simulator import (  # noqa: E402
    LocalizationSimulatorConfig,
    _build_vector_renderer,
    _fourier_shift_patches,
)


DEFAULT_MAP = ROOT / (
    "results/microtube_right_benchmark_20260711/calibration/right_fd_zmap/"
    "export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_map(
    path: Path, *, roi: int, origin_x: int | None, origin_y: int | None
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[int, int], tuple[int, int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        maps = np.asarray(payload["zernike_maps_nm"], dtype=np.float32)
        mode_order = tuple(tuple(int(value) for value in row) for row in payload["mode_order"])
    if maps.ndim != 3 or maps.shape[0] != len(mode_order):
        raise ValueError("zernike_maps_nm must have shape (modes, height, width)")
    original_shape = tuple(int(value) for value in maps.shape)
    height, width = maps.shape[1:]
    if origin_x is None:
        origin_x = max(0, (width - roi) // 2)
    if origin_y is None:
        origin_y = max(0, (height - roi) // 2)
    if origin_x < 0 or origin_y < 0 or origin_x + roi > width or origin_y + roi > height:
        raise ValueError(
            f"map crop ({origin_x}, {origin_y}, {roi}, {roi}) exceeds map shape {(height, width)}"
        )
    return (
        maps[:, origin_y : origin_y + roi, origin_x : origin_x + roi].copy(),
        mode_order,
        (origin_x, origin_y),
        original_shape,
    )


def _make_events(
    *,
    frames: int,
    roi: int,
    density: float,
    pixel_size_nm: float,
    z_range_um: tuple[float, float],
    photon_mean: float,
    photon_sigma: float,
    photon_max: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    area_um2 = (roi * pixel_size_nm / 1000.0) ** 2
    counts = rng.poisson(float(density) * area_um2, size=frames).astype(np.int32)
    event_count = int(counts.sum())
    frame_ids = np.repeat(np.arange(frames, dtype=np.int64), counts)
    xs = rng.uniform(0.0, max(1.0, roi - 1.0), event_count).astype(np.float32)
    ys = rng.uniform(0.0, max(1.0, roi - 1.0), event_count).astype(np.float32)
    zs = rng.uniform(z_range_um[0], z_range_um[1], event_count).astype(np.float32)
    photons = np.clip(
        rng.normal(photon_mean, photon_sigma, event_count), 0.0, photon_max
    ).astype(np.float32)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return counts, offsets, frame_ids, xs, ys, zs, photons


def _timed_block(
    renderer: Any,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    photons: np.ndarray,
    frame_ids: np.ndarray,
    frame_count: int,
    roi: int,
) -> tuple[np.ndarray, dict[str, float]]:
    device = renderer.device
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    x_t = torch.from_numpy(xs).to(device=device, dtype=torch.float32).reshape(-1)
    y_t = torch.from_numpy(ys).to(device=device, dtype=torch.float32).reshape(-1)
    z_t = torch.from_numpy(zs).to(device=device, dtype=torch.float32).reshape(-1)
    photons_t = torch.from_numpy(photons).to(device=device, dtype=torch.float32).reshape(-1)
    frame_t = torch.from_numpy(frame_ids)

    origin = renderer.config.field_origin_xy
    cols = torch.round(x_t + int(origin[0])).to(dtype=torch.long).clamp_(0, renderer.maps_nm.shape[2] - 1)
    rows = torch.round(y_t + int(origin[1])).to(dtype=torch.long).clamp_(0, renderer.maps_nm.shape[1] - 1)
    coeffs_nm = renderer.maps_nm[:, rows, cols].transpose(0, 1).contiguous()
    coeffs_rad = coeffs_nm * (2.0 * math.pi / max(float(renderer.config.wavelength_nm), 1e-6))
    coeffs_rad = coeffs_rad * renderer.ctx.normfac[None, :]

    render_start = torch.cuda.Event(enable_timing=True)
    render_end = torch.cuda.Event(enable_timing=True)
    shift_start = torch.cuda.Event(enable_timing=True)
    shift_end = torch.cuda.Event(enable_timing=True)
    post_start = torch.cuda.Event(enable_timing=True)
    post_end = torch.cuda.Event(enable_timing=True)
    reduce_start = torch.cuda.Event(enable_timing=True)
    reduce_end = torch.cuda.Event(enable_timing=True)
    render_start.record()
    with torch.no_grad():
        patches = renderer.cuda_renderer.render(
            coeffs_rad, z_t * 1.0e-6, renderer.pupil_carrier_complex
        )
    render_end.record()
    center_x = torch.floor(x_t).to(dtype=torch.long)
    center_y = torch.floor(y_t).to(dtype=torch.long)
    local_x = x_t - (center_x.to(dtype=torch.float32) + 0.5)
    local_y = y_t - (center_y.to(dtype=torch.float32) + 0.5)
    shift_start.record()
    with torch.no_grad():
        patches = _fourier_shift_patches(patches, shift_x_px=local_x, shift_y_px=local_y)
    shift_end.record()
    post_start.record()
    with torch.no_grad():
        patches = patches.clamp_min(0.0)
        patches = patches / patches.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        patches = patches * photons_t[:, None, None]
    post_end.record()
    reduce_start.record()
    reduced = renderer.cuda_renderer.reduce_frames(
        patches.detach(),
        frame_indices=frame_t,
        center_x=center_x.detach(),
        center_y=center_y.detach(),
        frame_count=int(frame_count),
        height=int(roi),
        width=int(roi),
        background=110.0,
    )
    reduce_end.record()
    torch.cuda.synchronize(device)
    gpu_render_ms = float(render_start.elapsed_time(render_end))
    gpu_shift_ms = float(shift_start.elapsed_time(shift_end))
    gpu_postprocess_ms = float(post_start.elapsed_time(post_end))
    gpu_psf_ms = float(render_start.elapsed_time(post_end))
    gpu_reduce_ms = float(reduce_start.elapsed_time(reduce_end))
    copy_start = time.perf_counter()
    result = reduced.detach().cpu().numpy()
    copy_seconds = time.perf_counter() - copy_start
    wall_seconds = time.perf_counter() - wall_start
    return result, {
        "wall_ms": wall_seconds * 1000.0,
        "gpu_psf_ms": gpu_psf_ms,
        "gpu_render_ms": gpu_render_ms,
        "gpu_shift_ms": gpu_shift_ms,
        "gpu_postprocess_ms": gpu_postprocess_ms,
        "gpu_reduce_ms": gpu_reduce_ms,
        "host_copy_ms": copy_seconds * 1000.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal benchmark requires CUDA; CPU fallback is disabled")
    if args.frames <= 0 or args.roi <= 0 or args.npupil <= 0:
        raise ValueError("frames, roi, and npupil must be positive")
    if args.frame_block <= 0 or args.batch_size <= 0:
        raise ValueError("frame_block and batch_size must be positive")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = args.map_npz.resolve()
    maps, mode_order, map_origin, map_original_shape = _load_map(
        map_path, roi=args.roi, origin_x=args.map_origin_x, origin_y=args.map_origin_y
    )
    event_generation_start = time.perf_counter()
    counts, offsets, frame_ids, xs, ys, zs, photons = _make_events(
        frames=args.frames,
        roi=args.roi,
        density=args.density,
        pixel_size_nm=args.pixel_size_nm,
        z_range_um=(args.z_min_um, args.z_max_um),
        photon_mean=args.photon_mean,
        photon_sigma=args.photon_sigma,
        photon_max=args.photon_max,
        seed=args.seed,
    )
    event_generation_seconds = time.perf_counter() - event_generation_start
    area_um2 = (args.roi * args.pixel_size_nm / 1000.0) ** 2

    config = LocalizationSimulatorConfig(
        batch_size=args.batch_size,
        frames_per_sample=1,
        height=args.roi,
        width=args.roi,
        seed=args.seed,
        pixel_size_nm_x=args.pixel_size_nm,
        pixel_size_nm_y=args.pixel_size_nm,
        wavelength_nm=args.wavelength_nm,
        na=args.na,
        npupil=args.npupil,
        vector_psf_size=args.psf_size,
        vector_batch_size=args.batch_size,
        refmed=args.refmed,
        refcov=args.refmed,
        refimm=args.refmed,
        photon_mean=args.photon_mean,
        photon_sigma=args.photon_sigma,
        photon_range=(0.0, args.photon_max),
        background_range=(110.0, 110.0),
        z_range=(args.z_min_um, args.z_max_um),
        emitter_density_um2=args.density,
        coeff_maps_nm=torch.from_numpy(maps),
        coeff_mode_order=mode_order,
        output_device="renderer",
        simulation_backend="cuda_gemm",
    )

    setup_start = time.perf_counter()
    renderer = _build_vector_renderer(config)
    torch.cuda.synchronize(renderer.device)
    setup_seconds = time.perf_counter() - setup_start

    warmup_count = min(args.batch_size, int(frame_ids.size))
    warmup_start = time.perf_counter()
    if warmup_count:
        with torch.no_grad():
            warmup_patches, warmup_x, warmup_y = renderer._render_patches(
                xs=torch.from_numpy(xs[:warmup_count]),
                ys=torch.from_numpy(ys[:warmup_count]),
                z_um=torch.from_numpy(zs[:warmup_count]),
                photons=torch.from_numpy(photons[:warmup_count]),
            )
            renderer.cuda_renderer.reduce_frames(
                warmup_patches.detach(),
                frame_indices=torch.zeros(warmup_count, dtype=torch.long),
                center_x=warmup_x.detach(),
                center_y=warmup_y.detach(),
                frame_count=1,
                height=args.roi,
                width=args.roi,
                background=110.0,
            )
        torch.cuda.synchronize(renderer.device)
    warmup_seconds = time.perf_counter() - warmup_start

    frames_out = np.empty((args.frames, args.roi, args.roi), dtype=np.float32)
    block_rows: list[dict[str, Any]] = []
    measured_start = time.perf_counter()
    for frame_start in range(0, args.frames, args.frame_block):
        frame_stop = min(args.frames, frame_start + args.frame_block)
        event_start = int(offsets[frame_start])
        event_stop = int(offsets[frame_stop])
        if event_stop <= event_start:
            frames_out[frame_start:frame_stop] = 110.0
            continue
        local_ids = frame_ids[event_start:event_stop] - frame_start
        block_output, timing = _timed_block(
            renderer,
            xs=xs[event_start:event_stop],
            ys=ys[event_start:event_stop],
            zs=zs[event_start:event_stop],
            photons=photons[event_start:event_stop],
            frame_ids=local_ids,
            frame_count=frame_stop - frame_start,
            roi=args.roi,
        )
        frames_out[frame_start:frame_stop] = block_output
        block_rows.append(
            {
                "frame_start": frame_start,
                "frame_stop": frame_stop,
                "frame_count": frame_stop - frame_start,
                "emitter_count": event_stop - event_start,
                **timing,
            }
        )
    torch.cuda.synchronize(renderer.device)
    simulation_seconds = time.perf_counter() - measured_start

    if not np.isfinite(frames_out).all():
        raise RuntimeError("simulation output contains non-finite values")
    tiff_path = output_dir / "frames.tif"
    write_start = time.perf_counter()
    tifffile.imwrite(tiff_path, frames_out, photometric="minisblack", metadata={"axes": "TYX"})
    write_seconds = time.perf_counter() - write_start

    blocks_path = output_dir / "timing_blocks.csv"
    if block_rows:
        with blocks_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(block_rows[0]))
            writer.writeheader()
            writer.writerows(block_rows)

    wall_sum = sum(row["wall_ms"] for row in block_rows) / 1000.0
    gpu_psf_sum = sum(row["gpu_psf_ms"] for row in block_rows) / 1000.0
    gpu_render_sum = sum(row["gpu_render_ms"] for row in block_rows) / 1000.0
    gpu_shift_sum = sum(row["gpu_shift_ms"] for row in block_rows) / 1000.0
    gpu_postprocess_sum = sum(row["gpu_postprocess_ms"] for row in block_rows) / 1000.0
    gpu_reduce_sum = sum(row["gpu_reduce_ms"] for row in block_rows) / 1000.0
    host_copy_sum = sum(row["host_copy_ms"] for row in block_rows) / 1000.0
    total_events = int(frame_ids.size)
    summary = {
        "schema_version": "unitypsf_v2_cuda_zmap_benchmark.v1",
        "formal_gpu_run": True,
        "simulation_backend": "cuda_gemm",
        "simulation_backend_effective": "cuda_extension_per_emitter_zernike_map_psf",
        "cuda_kernel_variant": "phase_factored_v2",
        "frames": args.frames,
        "roi_shape_yx": [args.roi, args.roi],
        "npupil_shape_yx": [args.npupil, args.npupil],
        "psf_size": args.psf_size,
        "density_um2": args.density,
        "pixel_size_nm": args.pixel_size_nm,
        "area_um2": area_um2,
        "expected_emitters_per_frame": args.density * area_um2,
        "actual_emitters_total": total_events,
        "actual_emitters_mean_per_frame": total_events / args.frames,
        "actual_emitters_min_max": [int(counts.min()), int(counts.max())],
        "z_range_um": [args.z_min_um, args.z_max_um],
        "photon_mean_sigma": [args.photon_mean, args.photon_sigma],
        "map_npz": str(map_path),
        "map_sha256": _sha256(map_path),
        "map_original_shape": list(map_original_shape),
        "map_crop_origin_xy": list(map_origin),
        "map_crop_shape": list(maps.shape),
        "mode_order": [list(mode) for mode in mode_order],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "frame_block": args.frame_block,
        "gpu": torch.cuda.get_device_name(renderer.device),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(renderer.device)),
        "timing_seconds": {
            "event_generation_cpu": event_generation_seconds,
            "renderer_setup": setup_seconds,
            "cuda_extension_warmup_and_build": warmup_seconds,
            "measured_simulation_wall": simulation_seconds,
            "measured_block_wall_sum": wall_sum,
            "measured_gpu_psf_path_sum": gpu_psf_sum,
            "measured_gpu_psf_sum": gpu_psf_sum,
            "measured_gpu_render_kernel_sum": gpu_render_sum,
            "measured_gpu_subpixel_shift_sum": gpu_shift_sum,
            "measured_gpu_postprocess_sum": gpu_postprocess_sum,
            "measured_gpu_frame_reduction_sum": gpu_reduce_sum,
            "measured_host_copy_sum": host_copy_sum,
            "tiff_write": write_seconds,
            "end_to_end_including_setup_warmup_write": setup_seconds + warmup_seconds + simulation_seconds + write_seconds,
        },
        "throughput": {
            "frames_per_second_measured": args.frames / simulation_seconds,
            "emitters_per_second_measured": total_events / simulation_seconds,
            "ms_per_frame_measured": simulation_seconds * 1000.0 / args.frames,
            "ms_per_emitter_measured": simulation_seconds * 1000.0 / max(total_events, 1),
        },
        "output": {"frames_tiff": str(tiff_path), "timing_blocks_csv": str(blocks_path)},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-npz", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("--roi", type=int, default=64)
    parser.add_argument("--npupil", type=int, default=64)
    parser.add_argument("--psf-size", type=int, default=21)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--pixel-size-nm", type=float, default=125.211897)
    parser.add_argument("--wavelength-nm", type=float, default=642.0)
    parser.add_argument("--na", type=float, default=1.46)
    parser.add_argument("--refmed", type=float, default=1.518)
    parser.add_argument("--z-min-um", type=float, default=-0.6)
    parser.add_argument("--z-max-um", type=float, default=0.6)
    parser.add_argument("--photon-mean", type=float, default=20_000.0)
    parser.add_argument("--photon-sigma", type=float, default=1_000.0)
    parser.add_argument("--photon-max", type=float, default=31_000.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--frame-block", type=int, default=64)
    parser.add_argument("--map-origin-x", type=int)
    parser.add_argument("--map-origin-y", type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
