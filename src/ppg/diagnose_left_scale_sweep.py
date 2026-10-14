#!/usr/bin/env python3
"""Sweep signal scale on the same fixed left-half ROI subset."""

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
    h, w = raw.shape[-2:]
    edge = 13
    points = []
    for xyz, probability in zip(decoded.xyz_px_nm, decoded.probability):
        x, y, _ = (float(value) for value in xyz)
        if edge <= x < w - edge and edge <= y < h - edge:
            points.append((x, y, float(probability)))
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--roi-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prob-threshold", type=float, default=0.3)
    parser.add_argument("--display-threshold", type=float, default=0.7)
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_records = json.loads(args.summary.read_text())["rois"]
    bg_cfg = json.loads(args.summary.read_text())["normalization"]
    bg = float(bg_cfg["background_mean"])
    bg_range = tuple(float(v) for v in bg_cfg["background_range"])
    selected_ids = [int(item["roi_id"]) for item in selected_records]

    with h5py.File(args.roi_bank.resolve(), "r") as handle:
        rois = handle["rois"]
        ids = np.asarray(rois["roi_id"], dtype=np.int64)
        raw_all = np.asarray(rois["raw_frames_photon"], dtype=np.float32)
    id_to_index = {int(roi_id): index for index, roi_id in enumerate(ids)}
    selected_indices = [id_to_index[roi_id] for roi_id in selected_ids]
    raw = raw_all[selected_indices]

    device = torch.device(args.device)
    model, epoch = _load_model(args.checkpoint.resolve(), device)
    provider = load_ppg_conditioning(model.model_config["ppg_map_path"])
    model.input_normalization = LunarInputNormalization(
        background_range=bg_range,
        photon_range=model.input_normalization.photon_range,
    )

    counts03 = np.zeros((len(args.scales), len(raw)), dtype=np.int32)
    counts07 = np.zeros_like(counts03)
    points_by_scale: dict[float, list[np.ndarray]] = {}
    for scale_index, scale in enumerate(args.scales):
        points_for_scale = []
        for roi_index, roi_raw in enumerate(raw):
            scaled = bg + float(scale) * (roi_raw - bg)
            points03 = detect(model, scaled, float(args.prob_threshold), provider=provider, source_shape=(512, 512))
            points07 = points03[points03[:, 2] >= float(args.display_threshold)] if len(points03) else points03
            counts03[scale_index, roi_index] = len(points03)
            counts07[scale_index, roi_index] = len(points07)
            points_for_scale.append(points03)
        points_by_scale[float(scale)] = points_for_scale

    np.savez_compressed(
        output_dir / "left_scale_sweep_counts.npz",
        scales=np.asarray(args.scales, dtype=np.float32),
        roi_ids=np.asarray(selected_ids, dtype=np.int64),
        count_threshold_03=counts03,
        count_threshold_07=counts07,
    )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(epoch),
        "source_summary": str(args.summary.resolve()),
        "roi_bank": str(args.roi_bank.resolve()),
        "background_mean": bg,
        "background_range": list(bg_range),
        "transform": "background_mean + scale * (raw - background_mean)",
        "prob_threshold": float(args.prob_threshold),
        "display_threshold": float(args.display_threshold),
        "scales": {},
    }
    for index, scale in enumerate(args.scales):
        summary["scales"][str(scale)] = {
            "counts_threshold_03": counts03[index].tolist(),
            "counts_threshold_07": counts07[index].tolist(),
            "total_threshold_03": int(counts03[index].sum()),
            "total_threshold_07": int(counts07[index].sum()),
        }
    (output_dir / "left_scale_sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    columns = len(selected_ids)
    fig, axes = plt.subplots(len(args.scales), columns, figsize=(2.8 * columns, 2.7 * len(args.scales)), squeeze=False)
    for row, scale in enumerate(args.scales):
        for col, roi_raw in enumerate(raw):
            center = roi_raw[1]
            vmax = float(np.percentile(center, 99.7))
            ax = axes[row, col]
            ax.imshow(center, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax)
            points = points_by_scale[float(scale)][col]
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], s=17, facecolors="none", edgecolors="cyan", linewidths=0.7)
            n07 = int(np.sum(points[:, 2] >= float(args.display_threshold))) if len(points) else 0
            ax.set_title(f"ROI {selected_ids[col]}\ns={scale:g} N03={len(points)} N07={n07}", fontsize=7.5)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel("scale", fontsize=9)
    fig.suptitle("Left-half fixed ROIs: background-subtracted signal-scale sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "left_scale_sweep_raw_vs_points.png", dpi=180)
    plt.close(fig)
    for index, scale in enumerate(args.scales):
        print(f"scale={scale:g} total03={int(counts03[index].sum())} total07={int(counts07[index].sum())}")


if __name__ == "__main__":
    main()
