#!/usr/bin/env python3
"""Create a detailed timing breakdown figure from benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222",
    "sky": "#56B4E9",
}


def _load_blocks(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _bar_labels(ax, bars, values: list[float], *, fmt: str = "{:.3f}") -> None:
    top = max(values) if values else 1.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + top * 0.018,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=OKABE_ITO["black"],
        )


def _stacked_bar(ax, labels: list[str], values: list[float], colors: list[str], title: str, total: float) -> None:
    left = 0.0
    for label, value, color in zip(labels, values, colors):
        ax.barh([0], [value], left=left, color=color, edgecolor="white", linewidth=0.7, label=label)
        if value >= total * 0.035:
            ax.text(left + value / 2.0, 0, f"{value:.3f}", ha="center", va="center", fontsize=8)
        left += value
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Seconds")
    ax.set_xlim(0, max(total * 1.08, 1.0))
    ax.grid(axis="x", alpha=0.25)


def make_figure(summary_path: Path, blocks_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    timing = summary["timing_seconds"]
    blocks = _load_blocks(blocks_path)

    measured = [
        timing["measured_gpu_render_kernel_sum"],
        timing["measured_gpu_subpixel_shift_sum"],
        timing["measured_gpu_postprocess_sum"],
        timing["measured_gpu_frame_reduction_sum"],
        timing["measured_host_copy_sum"],
    ]
    measured_labels = ["Raw CUDA render", "Fourier subpixel shift", "Patch postprocess", "Frame reduction", "Host copy"]
    measured_colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
        OKABE_ITO["sky"],
    ]
    block_wall = timing["measured_block_wall_sum"]
    measured_residual = max(0.0, block_wall - sum(measured))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    fig.suptitle(
        "UnityPSF v2 direct CUDA Zernike-map simulation timing\n"
        f"{summary['frames']:,} frames | {summary['gpu']} | density={summary['density_um2']} / um² | npupil={summary['npupil_shape_yx'][0]}×{summary['npupil_shape_yx'][1]}",
        fontsize=14,
        fontweight="bold",
    )

    measured_names = measured_labels + ["Block scheduling / uninstrumented"]
    measured_values = measured + [measured_residual]
    measured_color_list = measured_colors + [OKABE_ITO["yellow"]]
    _stacked_bar(axes[0], measured_names, measured_values, measured_color_list, "B. Steady-state simulation path", timing["measured_block_wall_sum"])
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)

    x = np.arange(len(measured_labels))
    bars = axes[1].bar(x, measured, color=measured_colors, edgecolor="white")
    _bar_labels(axes[1], bars, measured)
    axes[1].set_title("C. Component totals over 10,000 frames", loc="left", fontweight="bold")
    axes[1].set_ylabel("Seconds")
    axes[1].set_xticks(x, ["Raw\nCUDA", "Subpixel\nshift", "Patch\npost", "Frame\nreduce", "Host\ncopy"])
    axes[1].grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--blocks", type=Path, default=Path("results/timing_blocks.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/timing_breakdown_detailed.png"))
    args = parser.parse_args()
    make_figure(args.summary, args.blocks, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
