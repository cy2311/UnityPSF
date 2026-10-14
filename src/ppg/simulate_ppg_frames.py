#!/usr/bin/env python3
"""Render spatial PPG training frames through UnityPSF's shared simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import yaml

from unity_psf.localization import build_localization_runtime_config
from unity_psf.localization.simulator import (
    LocalizationSimulatorConfig,
    simulate_localization_batch,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/modalities/double_helix/ppg_physical.yaml"
DEFAULT_OUTPUT = ROOT / "output/ppg_simulation_64x64_density0p2_z600"


def _axis_origins(field_size: int, roi_size: int, stride_px: int) -> list[int]:
    if field_size < roi_size:
        raise ValueError("field dimensions must be at least as large as the ROI")
    if stride_px <= 0:
        raise ValueError("stride_px must be positive")
    last = field_size - roi_size
    values = list(range(0, last + 1, stride_px))
    if values[-1] != last:
        values.append(last)
    return values


def sliding_field_origins(
    *,
    field_shape_yx: tuple[int, int],
    roi_shape_yx: tuple[int, int],
    stride_px: int,
) -> np.ndarray:
    """Return deterministic x/y origins covering the full field, including edges."""
    height, width = (int(value) for value in field_shape_yx)
    roi_height, roi_width = (int(value) for value in roi_shape_yx)
    xs = _axis_origins(width, roi_width, int(stride_px))
    ys = _axis_origins(height, roi_height, int(stride_px))
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.int32)


def edge_emitter_mask(
    xy_px: np.ndarray,
    *,
    roi_shape_yx: tuple[int, int],
    psf_support_px: int,
) -> np.ndarray:
    """Mark emitters whose PSF support intersects an ROI boundary."""
    xy = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
    height, width = (int(value) for value in roi_shape_yx)
    radius = int(psf_support_px) // 2
    return (
        (xy[:, 0] < radius)
        | (xy[:, 0] >= width - radius)
        | (xy[:, 1] < radius)
        | (xy[:, 1] >= height - radius)
    )


def _simulator_config(
    provider: dict[str, Any],
    *,
    seed: int,
    origin_xy: tuple[int, int],
    roi_size_px: int,
    density_um2: float,
    z_range_um: tuple[float, float],
) -> LocalizationSimulatorConfig:
    return LocalizationSimulatorConfig(
        batch_size=1,
        frames_per_sample=1,
        height=int(roi_size_px),
        width=int(roi_size_px),
        seed=int(seed),
        background=float(provider["background_range"][0]),
        psf_type="ppg_direct",
        pixel_size_nm_x=float(provider["pixel_size_nm_x"]),
        pixel_size_nm_y=float(provider["pixel_size_nm_y"]),
        vector_psf_size=int(provider["vector_psf_size"]),
        photon_range=tuple(float(value) for value in provider["photon_range"]),
        photon_mean=float(provider["photon_mean"]),
        photon_sigma=float(provider["photon_sigma"]),
        background_range=tuple(float(value) for value in provider["background_range"]),
        z_range=tuple(float(value) for value in z_range_um),
        emitter_density_um2=float(density_um2),
        field_origin_xy=tuple(int(value) for value in origin_xy),
        ppg_map_path=str(provider["ppg_map_path"]),
        output_device="renderer",
    )


def _select_preview_indices(edge_counts: np.ndarray) -> list[int]:
    fixed = [0, 9, 40, 44, 45, 49, 50, 54, 55, 59, 90, 99]
    richest_edge = np.argsort(-np.asarray(edge_counts), kind="stable")
    selected = []
    for index in [*fixed, *richest_edge.tolist()]:
        if index not in selected:
            selected.append(int(index))
        if len(selected) == 12:
            break
    return selected


def _write_preview_figures(
    output_dir: Path,
    *,
    frames: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    origins: np.ndarray,
    edge_counts: np.ndarray,
    background: float,
    psf_support_px: int,
) -> list[int]:
    figures_dir = output_dir / "figures"
    frame_dir = figures_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    signal = np.maximum(frames - float(background), 0.0)
    positive = signal[signal > 0]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    preview_indices = _select_preview_indices(edge_counts)

    fig, axes = plt.subplots(3, 4, figsize=(10.4, 8.0), constrained_layout=True)
    for axis, index in zip(axes.flat, preview_indices, strict=True):
        axis.imshow(
            signal[index], cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest"
        )
        origin_x, origin_y = origins[index]
        count = int(masks[index].sum())
        axis.set_title(
            f"frame {index:03d} | FOV ({origin_x}, {origin_y})\n"
            f"emitters {count} | edge {int(edge_counts[index])}",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "PPG simulation: 64 x 64, density 0.2 emitters/um2, z = +/-600 nm",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(figures_dir / "selected_raw_frames.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(10.4, 8.0), constrained_layout=True)
    for axis, index in zip(axes.flat, preview_indices, strict=True):
        axis.imshow(
            signal[index], cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest"
        )
        active_xy = targets[index, masks[index], :2]
        is_edge = edge_emitter_mask(
            active_xy,
            roi_shape_yx=frames.shape[-2:],
            psf_support_px=psf_support_px,
        )
        if np.any(~is_edge):
            axis.scatter(
                active_xy[~is_edge, 0],
                active_xy[~is_edge, 1],
                s=18,
                facecolors="none",
                edgecolors="#56B4E9",
                linewidths=0.8,
            )
        if np.any(is_edge):
            axis.scatter(
                active_xy[is_edge, 0],
                active_xy[is_edge, 1],
                s=24,
                facecolors="none",
                edgecolors="#F0E442",
                linewidths=1.0,
            )
        axis.set_title(
            f"frame {index:03d} | edge-support emitters in yellow", fontsize=8
        )
        axis.set_xlim(-0.5, frames.shape[-1] - 0.5)
        axis.set_ylim(frames.shape[-2] - 0.5, -0.5)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Ground-truth emitter centers and naturally clipped edge PSFs",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(
        figures_dir / "selected_frames_with_emitters.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    individual_indices = [0, 9, 44, 55, 90, 99]
    for index in individual_indices:
        plt.imsave(
            frame_dir / f"frame_{index:03d}.png",
            signal[index],
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
    return preview_indices


def simulate_ppg_frames(
    *,
    config_path: Path,
    output_dir: Path,
    frame_count: int,
    roi_size_px: int,
    density_um2: float,
    z_range_um: tuple[float, float],
    seed: int,
    require_cuda: bool,
) -> dict[str, Any]:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("formal PPG simulation requires an available CUDA device")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime = build_localization_runtime_config(
        config,
        config_base_dir=config_path.parent,
        seed=int(seed),
    )
    provider = dict(runtime["batch_provider"]["params"])
    map_path = Path(provider["ppg_map_path"]).resolve()
    with np.load(map_path, allow_pickle=False) as payload:
        field_shape_yx = tuple(int(value) for value in payload["field_shape_yx"])
        psf_support_px = int(payload["roi_shape_yx"][0])
        schema = str(payload["schema"].item())
        boundary_anchor_strength = float(payload["boundary_anchor_strength"])
    if schema != "ppg.pca_map.v1":
        raise ValueError(f"unsupported PPG map schema: {schema}")

    origins = sliding_field_origins(
        field_shape_yx=field_shape_yx,
        roi_shape_yx=(roi_size_px, roi_size_px),
        stride_px=roi_size_px,
    )
    if len(origins) != int(frame_count):
        raise ValueError(
            f"requested {frame_count} frames, but full-field sliding coverage yields "
            f"{len(origins)}; adjust frame_count or roi_size_px"
        )

    frames = []
    targets = []
    masks = []
    counts = []
    for frame_index, origin in enumerate(origins):
        sim_config = _simulator_config(
            provider,
            seed=int(seed) + frame_index,
            origin_xy=tuple(int(value) for value in origin),
            roi_size_px=roi_size_px,
            density_um2=density_um2,
            z_range_um=z_range_um,
        )
        batch = simulate_localization_batch(
            sim_config,
            epoch=0,
            step=frame_index,
            source="ppg_spatial_preview",
        )
        frames.append(batch.model_input[0, 0].detach().cpu().numpy())
        targets.append(batch.pxyz_tar[0].detach().cpu().numpy())
        masks.append(batch.mask_tar[0].detach().cpu().numpy())
        counts.append(int(batch.mask_tar[0].sum().item()))

    frames_np = np.stack(frames).astype(np.float32, copy=False)
    targets_np = np.stack(targets).astype(np.float32, copy=False)
    masks_np = np.stack(masks).astype(bool, copy=False)
    counts_np = np.asarray(counts, dtype=np.int32)
    edge_counts = np.asarray(
        [
            int(
                edge_emitter_mask(
                    targets_np[index, masks_np[index], :2],
                    roi_shape_yx=(roi_size_px, roi_size_px),
                    psf_support_px=psf_support_px,
                ).sum()
            )
            for index in range(frame_count)
        ],
        dtype=np.int32,
    )
    background = float(provider["background_range"][0])
    emitted_photons = float(targets_np[..., 3][masks_np].sum())
    retained_signal = float(np.maximum(frames_np - background, 0.0).sum())
    retained_fraction = retained_signal / max(emitted_photons, 1e-12)

    output_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = output_dir / "ppg_simulated_100frames_64x64.tif"
    tifffile.imwrite(
        tiff_path,
        frames_np,
        photometric="minisblack",
        metadata={"axes": "TYX"},
    )
    arrays_path = output_dir / "simulation_targets.npz"
    np.savez_compressed(
        arrays_path,
        frames=frames_np,
        emitter_xy_z_um_photons=targets_np,
        emitter_mask=masks_np,
        emitter_counts=counts_np,
        edge_emitter_counts=edge_counts,
        field_origins_xy=origins,
    )
    preview_indices = _write_preview_figures(
        output_dir,
        frames=frames_np,
        targets=targets_np,
        masks=masks_np,
        origins=origins,
        edge_counts=edge_counts,
        background=background,
        psf_support_px=psf_support_px,
    )
    expected_emitters = (
        float(density_um2)
        * (roi_size_px * float(provider["pixel_size_nm_x"]) / 1000.0)
        * (roi_size_px * float(provider["pixel_size_nm_y"]) / 1000.0)
    )
    summary = {
        "schema": "ppg.simulation_preview.v1",
        "ppg_map": str(map_path),
        "ppg_schema": schema,
        "boundary_anchor_strength": boundary_anchor_strength,
        "simulation_api": "unity_psf.localization.simulator.simulate_localization_batch",
        "simulation_backend": "ppg_direct",
        "frame_shape_yx": [roi_size_px, roi_size_px],
        "frame_count": int(frame_count),
        "field_shape_yx": list(field_shape_yx),
        "field_origins_xy": origins.tolist(),
        "field_coverage": "10x10 non-overlapping windows including all field edges",
        "psf_support_px": psf_support_px,
        "edge_policy": "emitters sampled over complete ROI; PSF patches clipped at image boundary",
        "density_emitters_um2": float(density_um2),
        "expected_emitters_per_frame": expected_emitters,
        "actual_emitters_total": int(counts_np.sum()),
        "actual_emitters_mean_per_frame": float(counts_np.mean()),
        "actual_emitters_min_max": [int(counts_np.min()), int(counts_np.max())],
        "edge_emitters_total": int(edge_counts.sum()),
        "edge_emitters_mean_per_frame": float(edge_counts.mean()),
        "z_range_um": [float(z_range_um[0]), float(z_range_um[1])],
        "photon_mean_sigma": [
            float(provider["photon_mean"]),
            float(provider["photon_sigma"]),
        ],
        "photon_range": [float(value) for value in provider["photon_range"]],
        "background": background,
        "retained_signal_fraction_after_edge_clipping": retained_fraction,
        "seed": int(seed),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "preview_frame_indices": preview_indices,
        "artifacts": {
            "frames_tiff": str(tiff_path),
            "targets_npz": str(arrays_path),
            "raw_preview_png": str(output_dir / "figures/selected_raw_frames.png"),
            "overlay_preview_png": str(
                output_dir / "figures/selected_frames_with_emitters.png"
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-count", type=int, default=100)
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--z-range-um", type=float, nargs=2, default=(-0.6, 0.6))
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    summary = simulate_ppg_frames(
        config_path=args.config.resolve(),
        output_dir=args.output_dir.resolve(),
        frame_count=int(args.frame_count),
        roi_size_px=int(args.roi_size),
        density_um2=float(args.density),
        z_range_um=tuple(float(value) for value in args.z_range_um),
        seed=int(args.seed),
        require_cuda=not args.allow_cpu,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
