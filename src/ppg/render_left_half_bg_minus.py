#!/usr/bin/env python3
"""Render the same left-half ROIs before and after LUNAR background subtraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--roi-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    bg = float(summary["normalization"]["background_mean"])
    selected = summary["rois"]
    id_to_index = {}
    with h5py.File(args.roi_bank, "r") as handle:
        rois = handle["rois"]
        ids = np.asarray(rois["roi_id"], dtype=np.int64)
        raw = np.asarray(rois["raw_frames_photon"], dtype=np.float32)
        for index, roi_id in enumerate(ids):
            id_to_index[int(roi_id)] = index

    fig, axes = plt.subplots(2, len(selected), figsize=(2.8 * len(selected), 6.0), squeeze=False)
    for col, record in enumerate(selected):
        index = id_to_index[int(record["roi_id"])]
        center = raw[index, 1]
        residual = center - bg
        vmax_raw = float(np.percentile(center, 99.7))
        vmax_residual = float(np.percentile(np.clip(residual, 0.0, None), 99.7))
        axes[0, col].imshow(center, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax_raw)
        axes[1, col].imshow(np.clip(residual, 0.0, None), cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax_residual)
        axes[0, col].set_title(f"ROI {record['roi_id']}\nraw", fontsize=8)
        axes[1, col].set_title(f"raw - bg\nbg={bg:.2f}", fontsize=8)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("raw center frame", fontsize=9)
    axes[1, 0].set_ylabel("background-subtracted", fontsize=9)
    fig.suptitle("Left-half fixed ROIs: raw vs LUNAR background subtraction", fontsize=13)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    payload = {
        "summary": str(args.summary.resolve()),
        "roi_bank": str(args.roi_bank.resolve()),
        "background_mean": bg,
        "selected_roi_ids": [int(item["roi_id"]) for item in selected],
        "output": str(args.output.resolve()),
    }
    args.output.with_name(args.output.stem + "_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
