#!/usr/bin/env python3
"""Decode a fixed random subset of ROIs fully contained in the left FOV half."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

from ppg.infer_recon_sample import _load_model
from ppg.condition_input import load_ppg_conditioning, ppg_model_input
from ppg.lunar_background import estimate_lunar_background_range
from unity_psf.data.normalization import LunarInputNormalization
from unity_psf.localization.legacy_decode import decode_liteloc_formal_infer_emitters


def detect(model, raw: np.ndarray, threshold: float, *, provider=None, origin=(0, 0), source_shape=None) -> np.ndarray:
    with torch.no_grad():
        inputs = torch.as_tensor(raw[None], dtype=torch.float32, device=next(model.parameters()).device)
        inputs = ppg_model_input(inputs, provider=provider, origin_xy_px=origin, source_field_shape_yx=source_shape)
        output = model(inputs)
        decoded = decode_liteloc_formal_infer_emitters(
            output, accept_threshold=float(threshold), photon_scale=31000.0, z_scale=600.0
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
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--prob-threshold", type=float, default=0.3)
    parser.add_argument("--display-threshold", type=float, default=0.7)
    parser.add_argument("--left-width-px", type=int, default=256)
    parser.add_argument("--boundary-margin-px", type=int, default=24)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate = estimate_lunar_background_range(
        np.asarray(tifffile.imread(args.sample_tiff, key=range(4096)), dtype=np.float32)[:, :, : args.left_width_px],
        percentile=10.0,
    )
    bg_range = tuple(float(v) for v in estimate)
    bg_mean = float(sum(bg_range) / 2.0)
    device = torch.device(args.device)
    model, epoch = _load_model(args.checkpoint.resolve(), device)
    provider = load_ppg_conditioning(model.model_config["ppg_map_path"])
    model.input_normalization = LunarInputNormalization(
        background_range=bg_range,
        photon_range=model.input_normalization.photon_range,
    )

    with h5py.File(args.roi_bank.resolve(), "r") as handle:
        rois = handle["rois"]
        raw_all = np.asarray(rois["raw_frames_photon"], dtype=np.float32)
        ids_all = np.asarray(rois["roi_id"], dtype=np.int64)
        origins_all = np.asarray(rois["roi_origin_xy_px"], dtype=np.float32)
        windows_all = np.asarray(rois["frame_window"], dtype=np.int64)
    roi_h, roi_w = raw_all.shape[-2:]
    eligible = np.flatnonzero(
        (origins_all[:, 0] >= 0)
        & (origins_all[:, 0] + roi_w <= int(args.left_width_px) - int(args.boundary_margin_px))
    )
    if len(eligible) < int(args.sample_count):
        raise RuntimeError(f"only {len(eligible)} left-half ROIs available, need {args.sample_count}")
    rng = np.random.default_rng(int(args.seed))
    selected = np.sort(rng.choice(eligible, size=int(args.sample_count), replace=False))

    records = []
    fig, axes = plt.subplots(2, len(selected), figsize=(2.8 * len(selected), 6.0), squeeze=False)
    for col, roi_index in enumerate(selected):
        raw = raw_all[roi_index]
        points03 = detect(model, raw, float(args.prob_threshold), provider=provider, origin=tuple(origins_all[roi_index].astype(int)), source_shape=(512, 512))
        points07 = points03[points03[:, 2] >= float(args.display_threshold)] if len(points03) else points03
        center = raw[1]
        vmax = float(np.percentile(center, 99.7))
        axes[0, col].imshow(center, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax)
        axes[0, col].set_title(f"ROI {ids_all[roi_index]}\nraw frame {windows_all[roi_index, 0] + 1}", fontsize=8)
        axes[1, col].imshow(center, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax)
        if len(points03):
            axes[1, col].scatter(points03[:, 0], points03[:, 1], s=18, facecolors="none", edgecolors="cyan", linewidths=0.75)
        axes[1, col].set_title(f"N03={len(points03)} N07={len(points07)}", fontsize=8)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        records.append({
            "roi_id": int(ids_all[roi_index]),
            "origin_xy_px": origins_all[roi_index].tolist(),
            "frame_window": windows_all[roi_index].tolist(),
            "count_threshold_03": int(len(points03)),
            "count_threshold_07": int(len(points07)),
            "points_xy_prob": points03.tolist(),
        })
    axes[0, 0].set_ylabel("raw", fontsize=9)
    axes[1, 0].set_ylabel("detections", fontsize=9)
    fig.suptitle("Left-half independent detection: fixed random ROIs", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "left_half_random_rois_raw_vs_points.png", dpi=180)
    plt.close(fig)

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(epoch),
        "sample_tiff": str(args.sample_tiff.resolve()),
        "roi_bank": str(args.roi_bank.resolve()),
        "left_fov": {"x_min_px": 0, "x_max_px": int(args.left_width_px), "boundary_margin_px": int(args.boundary_margin_px)},
        "eligible_roi_count": int(len(eligible)),
        "selected_roi_count": int(len(selected)),
        "seed": int(args.seed),
        "normalization": {"method": "lunar_darkest_10_percent_within_left_half", "background_range": list(bg_range), "background_mean": bg_mean},
        "prob_threshold": float(args.prob_threshold),
        "display_threshold": float(args.display_threshold),
        "rois": records,
        "figure": str(output_dir / "left_half_random_rois_raw_vs_points.png"),
    }
    (output_dir / "left_half_random_rois_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"eligible_roi_count": len(eligible), "selected_roi_count": len(selected), "background_mean": bg_mean, "counts03": [r["count_threshold_03"] for r in records], "counts07": [r["count_threshold_07"] for r in records]}, sort_keys=True))


if __name__ == "__main__":
    main()
