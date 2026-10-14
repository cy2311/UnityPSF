#!/usr/bin/env python3
"""Render manuscript Figure 1: UnityPSF concept, DH PSF field model, holdout validation, renderer throughput."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import tifffile

from ppg.beads_alignment import apply_affine_z_shift, stack_ncc
from ppg.beads_initialization import (
    Z_NORMALIZED,
    extract_bead_signal,
    legendre_design,
    normalize_psf,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "configs/calibration/ppg/ppg_map_initial.npz"
DEFAULT_CATALOG = ROOT / "configs/calibration/ppg/bead_catalog_zeiss_63x1.4_642.npz"
DEFAULT_SOURCE = ROOT / "datasets/training_sets/Zeiss/63x1.4_642.tif"
DEFAULT_OUTPUT = ROOT / "docs"
HOLDOUT_ORDER = (44, 54, 11, 43, 53)
SHOWN_HOLDOUTS = (44, 11)
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#6E6E6E"
MM = 1 / 25.4

# results/microtube/unitypsf_v2_cuda_zmap_density1_roi64_npupil64_10k{,_phasefactored}/summary.json
CUDA = {
    "baseline kernel": {"frames_per_s": 1445.92, "end_to_end_s": 80.4723},
    "phase-factored v2": {"frames_per_s": 3296.56, "end_to_end_s": 5.8525},
}

plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 6,
        "axes.linewidth": 0.5,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "legend.fontsize": 5.5,
        "pdf.fonttype": 42,
    }
)


def panel_label(fig: plt.Figure, x: float, y: float, letter: str) -> None:
    fig.text(x, y, letter, fontsize=8, fontweight="bold", ha="left", va="top")


def strip_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_box(axis: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
        )
    )
    axis.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=6, fontweight="bold")
    axis.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=5.2, color="#333333")


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=6,
            linewidth=0.7,
            color=kwargs.pop("color", "black"),
            **kwargs,
        )
    )


def render_concept(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    strip_axis(axis)
    w, h, y = 0.155, 0.46, 0.40
    chain = [
        (0.00, "Bead z-stacks", "51 beads, 7 z-planes\n43 train / 5 holdout\n3 rejected", GREY),
        (0.195, "PCA morphology", "mean + 4 modes\n21 × 21 support", GREY),
        (0.39, "Legendre field", "degree-2, 9 terms\nγ: 9 × 4 = 36 params", GREY),
        (0.585, "Shared PSF state", "C(x, y) → PSF(x, y, z)\none state,\nthree consumers", BLUE),
    ]
    for x, title, body, color in chain:
        draw_box(axis, x, y, w, h, title, body, color)
    for (x0, *_), (x1, *_) in zip(chain[:-1], chain[1:]):
        arrow(axis, (x0 + w, y + h / 2), (x1, y + h / 2))

    consumers = [
        (0.80, 0.71, "Simulator", "training frames"),
        (0.80, 0.405, "FiLM network", "condition: x, y, $c_1$–$c_4$"),
        (0.80, 0.10, "Physical fit", "ROI bank, 96 × 96"),
    ]
    cw, ch = 0.185, 0.23
    for x, yc, title, body in consumers:
        draw_box(axis, x + 0.01, yc, cw, ch, title, body, GREY)
        arrow(axis, (0.585 + w, y + h / 2), (x + 0.01, yc + ch / 2))

    arrow(
        axis,
        (0.81, 0.19),
        (0.66, y),
        connectionstyle="arc3,rad=-0.25",
        color=ORANGE,
        linestyle=(0, (3, 1.5)),
    )
    axis.text(
        0.62,
        0.13,
        "held-out gate\nNLL, ROI fraction, NCC\naccept / reject γ update",
        ha="right",
        va="center",
        fontsize=5.2,
        color=ORANGE,
    )


def mode_montage(payload: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str]]:
    n_z = len(payload["train_z_um"])
    mean = payload["pca_mean"].reshape(n_z, 21, 21)
    modes = payload["pca_components"].reshape(-1, n_z, 21, 21)
    ratios = payload["pca_explained_variance_ratio"]
    rows = [mean / np.abs(mean).max()] + [mode / np.abs(mode).max() for mode in modes]
    labels = ["mean"] + [f"PC{k + 1}\n{100 * r:.1f}%" for k, r in enumerate(ratios)]
    return np.stack(rows), labels


def render_modes(fig: plt.Figure, grid, payload: np.lib.npyio.NpzFile) -> None:
    rows, labels = mode_montage(payload)
    z_um = payload["train_z_um"]
    axes = grid.subgridspec(rows.shape[0], rows.shape[1], wspace=0.04, hspace=0.06)
    for r, (row, label) in enumerate(zip(rows, labels, strict=True)):
        for c, image in enumerate(row):
            axis = fig.add_subplot(axes[r, c])
            if r == 0:
                axis.imshow(image, cmap="magma", vmin=0, vmax=1)
                axis.set_title(f"{z_um[c]:+.1f}", fontsize=5.5, pad=1.5)
            else:
                axis.imshow(image, cmap="RdBu_r", vmin=-1, vmax=1)
            strip_axis(axis)
            if c == 0:
                axis.set_ylabel(label, fontsize=5.5, rotation=0, ha="right", va="center", labelpad=2)
    fig.text(
        axis.get_position().x1,
        axes[0, 0].get_position(fig).y1 + 0.018,
        "z (µm)",
        ha="right",
        fontsize=5.5,
    )


def render_fields(fig: plt.Figure, grid, payload: np.lib.npyio.NpzFile, catalog: np.lib.npyio.NpzFile) -> None:
    terms = payload["basis_terms"].astype(int)
    gamma = payload["gamma"]
    plot_grid = np.linspace(-1.0, 1.0, 121)
    yy, xx = np.meshgrid(plot_grid, plot_grid, indexing="ij")
    x_basis = np.polynomial.legendre.legvander(xx.ravel(), 2).reshape(121, 121, 3)
    y_basis = np.polynomial.legendre.legvander(yy.ravel(), 2).reshape(121, 121, 3)
    basis = np.stack([x_basis[..., px] * y_basis[..., py] for px, py in terms])
    fields = np.einsum("byx,bm->myx", basis, gamma)

    ids = catalog["bead_ids"].astype(int)
    positions = catalog["positions_xy_normalized"] * 2.0 - 1.0
    anchors = payload["boundary_anchor_positions_xy_normalized"] * 2.0 - 1.0
    accepted = np.isin(ids, payload["accepted_ids"])
    heldout = np.isin(ids, payload["heldout_ids"])
    rejected = ~(accepted | heldout)

    axes = grid.subgridspec(2, 2, wspace=0.12, hspace=0.22)
    for mode, field in enumerate(fields):
        axis = fig.add_subplot(axes[mode // 2, mode % 2])
        limit = float(np.max(np.abs(field)))
        image = axis.imshow(
            field,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        axis.scatter(anchors[:, 0], anchors[:, 1], s=6, marker="D", color=GREY, linewidths=0, label="boundary anchor")
        axis.scatter(
            positions[accepted, 0],
            positions[accepted, 1],
            s=5,
            facecolors="none",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.7,
            label="training bead",
        )
        axis.scatter(
            positions[heldout, 0],
            positions[heldout, 1],
            s=14,
            marker="s",
            facecolors="none",
            edgecolors=BLUE,
            linewidths=0.9,
            label="holdout",
        )
        axis.scatter(
            positions[rejected, 0],
            positions[rejected, 1],
            s=12,
            marker="x",
            color=ORANGE,
            linewidths=0.8,
            label="rejected",
        )
        axis.set_xlim(-1.08, 1.08)
        axis.set_ylim(-1.08, 1.08)
        axis.set_xticks((-1, 0, 1))
        axis.set_yticks((-1, 0, 1))
        axis.tick_params(length=1.5, pad=1)
        if mode // 2 == 0:
            axis.set_xticklabels([])
        else:
            axis.set_xlabel("x (norm.)", labelpad=1)
        if mode % 2 == 1:
            axis.set_yticklabels([])
        else:
            axis.set_ylabel("y (norm.)", labelpad=1)
        axis.set_title(f"$C_{mode + 1}(x,y)$", pad=2)
        bar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        bar.ax.tick_params(labelsize=4.5, length=1.5, pad=1, width=0.4)
        bar.outline.set_linewidth(0.4)
        if mode == 0:
            handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(axes[1, 0].get_position(fig).x0 + 0.17, axes[1, 0].get_position(fig).y0 - 0.028),
        ncol=4,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
    )


def holdout_predictions(
    payload: np.lib.npyio.NpzFile, catalog: np.lib.npyio.NpzFile, stack: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ids = catalog["bead_ids"].astype(int)
    positions = catalog["positions_xy_normalized"]
    raw = normalize_psf(extract_bead_signal(stack, catalog["centers_yx"]))
    scores = legendre_design(positions) @ payload["gamma"]
    predictions = normalize_psf(
        (payload["pca_mean"] + scores @ payload["pca_components"]).reshape(
            len(ids), len(payload["train_z_um"]), 21, 21
        )
    )
    parameters = payload["alignment_parameters_dx0_dy0_dxz_dyz"]
    result = {}
    for bead_id in HOLDOUT_ORDER:
        index = int(np.flatnonzero(ids == bead_id)[0])
        registered = normalize_psf(
            apply_affine_z_shift(predictions[index], parameters[index], Z_NORMALIZED)
        )
        result[bead_id] = (raw[index], registered, stack_ncc(raw[index], registered))
    return result


def render_holdout_images(fig: plt.Figure, grid, holdouts, z_um: np.ndarray) -> None:
    rows = grid.subgridspec(len(SHOWN_HOLDOUTS), 1, hspace=0.18)
    for b, bead_id in enumerate(SHOWN_HOLDOUTS):
        raw, pred, ncc = holdouts[bead_id]
        residual = raw - pred
        vmax = float(np.percentile(np.r_[raw.ravel(), pred.ravel()], 99.5))
        rlim = float(np.max(np.abs(residual)))
        cells = rows[b].subgridspec(3, len(z_um), wspace=0.04, hspace=0.05)
        for z in range(len(z_um)):
            for r, (image, kwargs) in enumerate(
                (
                    (raw[z], {"cmap": "magma", "vmin": 0, "vmax": vmax}),
                    (pred[z], {"cmap": "magma", "vmin": 0, "vmax": vmax}),
                    (residual[z], {"cmap": "RdBu_r", "vmin": -rlim, "vmax": rlim}),
                )
            ):
                axis = fig.add_subplot(cells[r, z])
                axis.imshow(image, **kwargs)
                strip_axis(axis)
                if r == 0 and b == 0:
                    axis.set_title(f"{z_um[z]:+.1f}", fontsize=5.5, pad=1.5)
                if r == 1:
                    axis.text(10, 19.5, f"{ncc[z]:.3f}", color="white", fontsize=4.3, ha="center", va="bottom")
                if z == 0:
                    axis.set_ylabel(("measured", "model", "residual")[r], fontsize=5.5, rotation=0, ha="right", va="center", labelpad=2)
                if r == 0 and z == len(z_um) - 1:
                    axis.plot((13.5, 18.5), (18.5, 18.5), color="white", linewidth=1.0, solid_capstyle="butt")
                    axis.text(16, 17.3, "5 px", color="white", fontsize=4.3, ha="center", va="bottom")
        fig.text(
            cells[0, 0].get_position(fig).x0 - 0.052,
            cells[1, 0].get_position(fig).y0 + 0.012,
            f"bead {bead_id}",
            fontsize=5.5,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
            color=BLUE,
        )
    top = rows[0].get_position(fig)
    fig.text(top.x1, top.y1 + 0.018, "z (µm)", ha="right", fontsize=5.5)


def render_holdout_ncc(axis: plt.Axes, holdouts) -> None:
    for i, bead_id in enumerate(HOLDOUT_ORDER):
        ncc = holdouts[bead_id][2]
        jitter = np.linspace(-0.12, 0.12, len(ncc))
        axis.scatter(np.full(len(ncc), i) + jitter, ncc, s=5, color=BLUE, alpha=0.45, linewidths=0)
        axis.plot((i - 0.25, i + 0.25), (np.median(ncc),) * 2, color="black", linewidth=1.0)
    medians = [np.median(holdouts[b][2]) for b in HOLDOUT_ORDER]
    axis.axhline(np.median(medians), color=GREY, linewidth=0.5, linestyle=(0, (2, 1.5)))
    axis.text(4.55, np.median(medians) - 0.0015, f"median {np.median(medians):.3f}", fontsize=5, ha="right", va="top", color="#333333")
    axis.set_xticks(range(len(HOLDOUT_ORDER)), [str(b) for b in HOLDOUT_ORDER])
    axis.set_xlim(-0.6, len(HOLDOUT_ORDER) - 0.4)
    axis.set_xlabel("holdout bead ID", labelpad=1)
    axis.set_ylabel("NCC per z-plane", labelpad=1)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def render_throughput(fig: plt.Figure, grid) -> None:
    names = list(CUDA)
    colors = [GREY, BLUE]
    axes = grid.subgridspec(3, 1, height_ratios=(1, 1, 0.45), hspace=0.9)
    for r, (key, xlabel, fmt) in enumerate(
        (
            ("frames_per_s", "frames s$^{-1}$ (GPU PSF path)", "{:,.0f}"),
            ("end_to_end_s", "end-to-end time (s)", "{:.1f}"),
        )
    ):
        axis = fig.add_subplot(axes[r])
        values = [CUDA[name][key] for name in names]
        axis.barh(range(2), values, height=0.55, color=colors)
        for i, value in enumerate(values):
            axis.text(value + max(values) * 0.02, i, fmt.format(value), va="center", fontsize=5.2)
        axis.set_yticks(range(2), names)
        axis.set_xlim(0, max(values) * 1.3)
        axis.set_xlabel(xlabel, labelpad=1)
        axis.invert_yaxis()
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axis.text(
        -0.55,
        -0.75,
        "10,000 frames, 64 × 64 px, 64.2 emitters frame$^{-1}$\nRTX 3090, CUDA 11.8; renderer speed only",
        transform=axis.transAxes,
        fontsize=4.8,
        color="#333333",
        va="top",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppg-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--bead-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--source-tiff", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stack = tifffile.imread(args.source_tiff).astype(np.float64)
    fig = plt.figure(figsize=(180 * MM, 200 * MM))
    outer = fig.add_gridspec(
        3, 1, height_ratios=(0.62, 1.25, 1.3), hspace=0.26, left=0.075, right=0.965, top=0.975, bottom=0.05
    )
    with np.load(args.ppg_map, allow_pickle=False) as payload, np.load(
        args.bead_catalog, allow_pickle=False
    ) as catalog:
        render_concept(fig.add_subplot(outer[0]))
        middle = outer[1].subgridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.12)
        render_modes(fig, middle[0], payload)
        render_fields(fig, middle[1], payload, catalog)

        holdouts = holdout_predictions(payload, catalog, stack)
        bottom = outer[2].subgridspec(1, 3, width_ratios=(1.35, 0.6, 0.72), wspace=0.55)
        render_holdout_images(fig, bottom[0], holdouts, payload["train_z_um"])
        render_holdout_ncc(fig.add_subplot(bottom[1]), holdouts)
        render_throughput(fig, bottom[2])

    for letter, spec in zip("abcdef", (outer[0], middle[0], middle[1], bottom[0], bottom[1], bottom[2])):
        box = spec.get_position(fig)
        panel_label(fig, box.x0 - 0.06 if letter in "bd" else box.x0 - 0.045, box.y1 + 0.02, letter)

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"unitypsf_figure1.{suffix}", dpi=450, facecolor="white")
    plt.close(fig)
    print(args.output_dir / "unitypsf_figure1.png")


if __name__ == "__main__":
    main()
