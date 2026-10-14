#!/usr/bin/env python3
"""Measure real-TIFF detection sensitivity to signal-domain scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppg.infer_recon_sample import _load_model
from ppg.condition_input import load_ppg_conditioning, ppg_model_input
from ppg.lunar_background import estimate_lunar_tiff_background
from unity_psf.data.normalization import LunarInputNormalization
from unity_psf.localization.legacy_decode import decode_liteloc_formal_infer_emitters


def _detect(model, raw: np.ndarray, *, threshold: float, provider=None, origin=(0, 0), source_shape=None) -> np.ndarray:
    with torch.no_grad():
        inputs = torch.as_tensor(raw[None], dtype=torch.float32, device=next(model.parameters()).device)
        inputs = ppg_model_input(inputs, provider=provider, origin_xy_px=origin, source_field_shape_yx=source_shape)
        output = model(inputs)
        decoded = decode_liteloc_formal_infer_emitters(
            output,
            accept_threshold=float(threshold),
            photon_scale=31000.0,
            z_scale=600.0,
        )
    height, width = raw.shape[-2:]
    edge = 13
    points = []
    for xyz, probability in zip(decoded.xyz_px_nm, decoded.probability):
        x, y, _ = (float(value) for value in xyz)
        if edge <= x < width - edge and edge <= y < height - edge:
            points.append((x, y, float(probability)))
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-tiff", type=Path, required=True)
    parser.add_argument("--roi-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prob-threshold", type=float, default=0.3)
    parser.add_argument("--display-threshold", type=float, default=0.7)
    parser.add_argument("--scales", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.25, 0.50, 1.00])
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, epoch = _load_model(args.checkpoint.resolve(), device)
    provider = load_ppg_conditioning(model.model_config["ppg_map_path"])
    estimate = estimate_lunar_tiff_background(args.sample_tiff.resolve(), percentile=10.0, max_size_gb=2.0, chunk_frames=64)
    bg = float(estimate.background_mean)
    photon_range = model.input_normalization.photon_range

    with h5py.File(args.roi_bank.resolve(), "r") as handle:
        rois = handle["rois"]
        raw = np.asarray(rois["raw_frames_photon"], dtype=np.float32)
        roi_ids = np.asarray(rois["roi_id"], dtype=np.int64)
        origins = np.asarray(rois["roi_origin_xy_px"], dtype=np.float32)
        frame_windows = np.asarray(rois["frame_window"], dtype=np.int64)

    selected = np.unique(np.linspace(0, len(raw) - 1, min(6, len(raw)), dtype=np.int64))
    count03 = np.zeros((len(args.scales), len(raw)), dtype=np.int32)
    count07 = np.zeros_like(count03)
    points_by_scale: dict[float, list[np.ndarray]] = {}
    for scale_index, scale in enumerate(args.scales):
        model.input_normalization = LunarInputNormalization(
            background_range=(bg, bg), photon_range=photon_range
        )
        points_for_scale = []
        for roi_index, roi_raw in enumerate(raw):
            scaled = bg + float(scale) * (roi_raw - bg)
            points03 = _detect(model, scaled, threshold=float(args.prob_threshold), provider=provider, origin=tuple(origins[roi_index].astype(int)), source_shape=(512, 512))
            points07 = points03[points03[:, 2] >= float(args.display_threshold)] if len(points03) else points03
            count03[scale_index, roi_index] = len(points03)
            count07[scale_index, roi_index] = len(points07)
            points_for_scale.append(points03)
        points_by_scale[float(scale)] = points_for_scale

    np.savez_compressed(
        output_dir / "signal_scale_counts.npz",
        scales=np.asarray(args.scales, dtype=np.float32),
        roi_ids=roi_ids,
        origins=origins,
        frame_windows=frame_windows,
        count_threshold_03=count03,
        count_threshold_07=count07,
    )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(epoch),
        "sample_tiff": str(args.sample_tiff.resolve()),
        "roi_bank": str(args.roi_bank.resolve()),
        "roi_count": int(len(raw)),
        "normalization": {
            "method": "lunar_global_temporal_mean_darkest_10_percent",
            "background_mean": bg,
            "background_range": list(estimate.background_range),
            "frame_range": list(estimate.frame_range),
            "signal_transform": "background_mean + scale * (raw - background_mean)",
        },
        "prob_threshold": float(args.prob_threshold),
        "display_threshold": float(args.display_threshold),
        "scales": {},
    }
    for index, scale in enumerate(args.scales):
        summary["scales"][str(scale)] = {
            "total_threshold_03": int(count03[index].sum()),
            "total_threshold_07": int(count07[index].sum()),
            "mean_threshold_03": float(count03[index].mean()),
            "mean_threshold_07": float(count07[index].mean()),
            "nonempty_roi_fraction_threshold_03": float(np.mean(count03[index] > 0)),
        }
    (output_dir / "signal_scale_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    columns = len(selected)
    fig, axes = plt.subplots(len(args.scales), columns, figsize=(2.7 * columns, 2.7 * len(args.scales)), squeeze=False)
    for row, scale in enumerate(args.scales):
        for col, roi_index in enumerate(selected):
            ax = axes[row, col]
            center = raw[roi_index, 1]
            vmax = float(np.percentile(center, 99.7))
            ax.imshow(center, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax)
            points = points_by_scale[float(scale)][int(roi_index)]
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], s=17, facecolors="none", edgecolors="cyan", linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"s={scale:g}\nN03={len(points)} N07={int(np.sum(points[:, 2] >= args.display_threshold)) if len(points) else 0}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"ROI {int(roi_ids[roi_index])}\nxy={origins[roi_index].astype(int).tolist()}", fontsize=8)
    fig.suptitle("Signal-scale diagnostic: fixed real-TIFF ROI bank, raw center frame + detections", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "signal_scale_raw_vs_points.png", dpi=180)
    plt.close(fig)

    for index, scale in enumerate(args.scales):
        print(f"scale={scale:g} total03={int(count03[index].sum())} total07={int(count07[index].sum())} mean03={count03[index].mean():.2f}")


if __name__ == "__main__":
    main()
