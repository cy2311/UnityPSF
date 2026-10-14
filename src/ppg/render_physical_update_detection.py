#!/usr/bin/env python3
"""Render raw ROI center frames with current-model physical-update detections."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    with h5py.File(args.roi_bank, "r") as handle:
        rois = handle["rois"]
        emitters = handle["emitters"]
        raw = np.asarray(rois["raw_frames_photon"])
        starts = np.asarray(rois["emitter_start"], dtype=np.int64)
        counts = np.asarray(rois["emitter_count"], dtype=np.int64)
        frame_windows = np.asarray(rois["frame_window"], dtype=np.int64)
        roi_ids = np.asarray(rois["roi_id"], dtype=np.int64)
        origins = np.asarray(rois["roi_origin_xy_px"], dtype=np.float32)
        selected = np.flatnonzero(counts > 0)[: int(args.count)]
        if not len(selected):
            raise RuntimeError("ROI bank contains no detected emitters")

        cols = 4
        rows = int(np.ceil(len(selected) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.2 * rows), squeeze=False)
        for plot_index, roi_index in enumerate(selected):
            ax = axes[plot_index // cols, plot_index % cols]
            center_frame = int((frame_windows[roi_index, 0] + frame_windows[roi_index, 1]) // 2)
            image = raw[roi_index, 1]
            ax.imshow(image, cmap="magma", origin="upper", interpolation="nearest", extent=(-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5))
            start = int(starts[roi_index])
            stop = start + int(counts[roi_index])
            point_frames = np.asarray(emitters["frame_index"][start:stop])
            point_xy = np.asarray(emitters["local_xy_px"][start:stop])
            point_prob = np.asarray(emitters["probability"][start:stop])
            keep = point_frames == center_frame
            for (x, y), probability in zip(point_xy[keep], point_prob[keep]):
                ax.scatter([x], [y], s=70, facecolors="none", edgecolors="cyan", linewidths=1.4)
                ax.text(float(x) + 1.5, float(y) - 1.5, f"{float(probability):.2f}", color="white", fontsize=7)
            ax.set_title(f"ROI {int(roi_ids[roi_index])} | frame {center_frame}\n{int(keep.sum())} detections | origin {origins[roi_index].astype(int).tolist()}")
            ax.set_xlim(-0.5, image.shape[1] - 0.5)
            ax.set_ylim(image.shape[0] - 0.5, -0.5)
            ax.set_xlabel("x (ROI px)")
            ax.set_ylabel("y (ROI px)")
        for ax in axes.flat[len(selected):]:
            ax.axis("off")
        fig.suptitle("Physical update ROI bank: raw center frame + current-model detections", fontsize=16)
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print({"output": str(args.output), "roi_count": int(len(selected))})


if __name__ == "__main__":
    main()
