#!/usr/bin/env python3
"""Sweep the Lunar background mean used by the detector on a fixed ROI bank."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from unity_psf.data.normalization import LunarInputNormalization
from unity_psf.localization.legacy_decode import decode_liteloc_formal_infer_emitters
from unity_psf.models.psf_moe.experts.double_helix import DoubleHelixExpert
from ppg.condition_input import load_ppg_conditioning, ppg_model_input


def load_model(checkpoint: Path, device: torch.device) -> DoubleHelixExpert:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expert = payload["experts"]["double_helix"]
    model = DoubleHelixExpert(**dict(expert["model_config"]))
    model.load_state_dict(expert["model_state_dict"])
    return model.eval().to(device)


def detect(model: DoubleHelixExpert, raw: np.ndarray, *, threshold: float, provider=None, origin=(0, 0), source_shape=None) -> list[tuple[float, float, float]]:
    with torch.no_grad():
        inputs = torch.as_tensor(raw[None], dtype=torch.float32, device=next(model.parameters()).device)
        inputs = ppg_model_input(inputs, provider=provider, origin_xy_px=origin, source_field_shape_yx=source_shape)
        output = model(inputs)
        decoded = decode_liteloc_formal_infer_emitters(
            output, accept_threshold=float(threshold), photon_scale=31000.0, z_scale=600.0
        )
    h, w = raw.shape[-2:]
    edge = 12
    points = []
    for xyz, probability in zip(decoded.xyz_px_nm, decoded.probability):
        x, y, _ = (float(value) for value in xyz)
        if edge <= x < w - edge and edge <= y < h - edge:
            points.append((x, y, float(probability)))
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--roi-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    means = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 107.2, 115.0, 125.0, 135.0, 150.0]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.checkpoint.resolve(), device)
    provider = load_ppg_conditioning(model.model_config["ppg_map_path"])
    photon_range = tuple(float(v) for v in model.input_normalization.photon_range)

    with h5py.File(args.roi_bank, "r") as handle:
        rois = handle["rois"]
        raw = np.asarray(rois["raw_frames_photon"], dtype=np.float32)
        roi_ids = np.asarray(rois["roi_id"], dtype=np.int64)
        origins = np.asarray(rois["roi_origin_xy_px"], dtype=np.float32)
        counts = np.asarray(rois["emitter_count"], dtype=np.int64)

    all_points: dict[float, list[list[tuple[float, float, float]]]] = {}
    count_matrix = np.zeros((len(means), len(raw)), dtype=np.int32)
    for mean_index, mean in enumerate(means):
        model.input_normalization = LunarInputNormalization(
            background_range=(mean, mean), photon_range=photon_range
        )
        points_for_mean = []
        for roi_index, roi_raw in enumerate(raw):
            points = detect(model, roi_raw, threshold=args.threshold, provider=provider, origin=tuple(origins[roi_index].astype(int)), source_shape=(512, 512))
            points_for_mean.append(points)
            count_matrix[mean_index, roi_index] = len(points)
        all_points[mean] = points_for_mean

    np.savez_compressed(
        output_dir / "bg_mean_sweep_counts.npz",
        bg_means=np.asarray(means),
        roi_ids=roi_ids,
        counts=count_matrix,
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(count_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(range(len(means)), [f"{v:g}" for v in means])
    ax.set_xlabel("ROI index in physical-update bank")
    ax.set_ylabel("Lunar background mean")
    ax.set_title(f"Detection count vs background mean (threshold={args.threshold:g})")
    fig.colorbar(image, ax=ax, label="detections / ROI")
    fig.tight_layout()
    fig.savefig(output_dir / "bg_mean_sweep_counts.png", dpi=180)
    plt.close(fig)

    selected = [int(i) for i in np.flatnonzero(counts > 0)[:6]]
    cols = len(means)
    fig, axes = plt.subplots(len(selected), cols, figsize=(2.1 * cols, 2.5 * len(selected)), squeeze=False)
    for row, roi_index in enumerate(selected):
        center = raw[roi_index, 1]
        vmax = float(np.percentile(center, 99.7))
        for col, mean in enumerate(means):
            ax = axes[row, col]
            ax.imshow(center, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax)
            points = all_points[mean][roi_index]
            if points:
                xy = np.asarray([(p[0], p[1]) for p in points])
                ax.scatter(xy[:, 0], xy[:, 1], s=22, facecolors="none", edgecolors="cyan", linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"bg={mean:g}\nN={len(points)}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"ROI {int(roi_ids[roi_index])}\n{origins[roi_index].astype(int).tolist()}", fontsize=8)
    fig.suptitle("Fixed physical-update ROIs: raw center frame + detections", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "bg_mean_sweep_examples.png", dpi=180)
    plt.close(fig)

    for mean_index, mean in enumerate(means):
        row = count_matrix[mean_index]
        print(f"bg={mean:g} total={int(row.sum())} median_nonempty={float(np.median(row[row > 0])) if np.any(row > 0) else 0:.1f}")


if __name__ == "__main__":
    main()
