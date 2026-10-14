#!/usr/bin/env python3
"""Render validation and decomposition figures for the default PPG initialization."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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
DEFAULT_OUTPUT = ROOT / "output/ppg_initialization"
HOLDOUT_ORDER = (44, 54, 11, 43, 53)


def render_holdouts(
    path: Path,
    payload: np.lib.npyio.NpzFile,
    catalog: np.lib.npyio.NpzFile,
    stack: np.ndarray,
) -> None:
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

    fig, axes = plt.subplots(
        2 * len(HOLDOUT_ORDER),
        len(payload["train_z_um"]),
        figsize=(15, 18),
        constrained_layout=True,
    )
    for bead_row, bead_id in enumerate(HOLDOUT_ORDER):
        index = int(np.flatnonzero(ids == bead_id)[0])
        registered = normalize_psf(
            apply_affine_z_shift(predictions[index], parameters[index], Z_NORMALIZED)
        )
        ncc = stack_ncc(raw[index], registered)
        vmax = float(np.percentile(np.r_[raw[index].ravel(), registered.ravel()], 99.5))
        for z_index, z_um in enumerate(payload["train_z_um"]):
            raw_axis = axes[2 * bead_row, z_index]
            recon_axis = axes[2 * bead_row + 1, z_index]
            raw_axis.imshow(raw[index, z_index], cmap="magma", vmin=0.0, vmax=vmax)
            recon_axis.imshow(registered[z_index], cmap="magma", vmin=0.0, vmax=vmax)
            for axis in (raw_axis, recon_axis):
                axis.set_xticks([])
                axis.set_yticks([])
            raw_axis.set_title(f"z={z_um:+.1f} um", fontsize=9)
            recon_axis.text(
                0.5,
                0.03,
                f"NCC {ncc[z_index]:.3f}",
                transform=recon_axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
            )
        x, y = positions[index]
        axes[2 * bead_row, 0].set_ylabel(
            f"bead {bead_id}\nFOV=({x:.2f},{y:.2f})\nraw", fontsize=9
        )
        axes[2 * bead_row + 1, 0].set_ylabel(
            f"recon\nmedian NCC={np.median(ncc):.3f}", fontsize=9
        )
    fig.suptitle(
        "Five fixed spatial holdouts: raw Beads TIFF vs default gamma36 prediction",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def render_decomposition(
    path: Path,
    payload: np.lib.npyio.NpzFile,
    catalog: np.lib.npyio.NpzFile,
) -> None:
    terms = payload["basis_terms"].astype(int)
    gamma = payload["gamma"]
    plot_grid = np.linspace(-1.0, 1.0, 121)
    yy, xx = np.meshgrid(plot_grid, plot_grid, indexing="ij")
    x_basis = np.polynomial.legendre.legvander(xx.ravel(), 2).reshape(121, 121, 3)
    y_basis = np.polynomial.legendre.legvander(yy.ravel(), 2).reshape(121, 121, 3)
    basis = np.stack([x_basis[..., px] * y_basis[..., py] for px, py in terms])
    fields = np.einsum("byx,bm->myx", basis, gamma)
    labels = [rf"$P_{{{px}}}(x)P_{{{py}}}(y)$" for px, py in terms]

    ids = catalog["bead_ids"].astype(int)
    positions = catalog["positions_xy_normalized"] * 2.0 - 1.0
    accepted = np.isin(ids, payload["accepted_ids"])
    heldout = np.isin(ids, payload["heldout_ids"])
    rejected = ~(accepted | heldout)

    fig = plt.figure(figsize=(18, 12), layout="constrained")
    rows = fig.subfigures(2, 1, height_ratios=(1.3, 1.0), hspace=0.04)
    basis_figure, coefficient_figure = rows[0].subfigures(
        1, 2, width_ratios=(1.0, 1.18), wspace=0.04
    )
    basis_axes = basis_figure.subplots(
        3, 3, squeeze=False, gridspec_kw={"wspace": 0.04, "hspace": 0.08}
    )
    for index, (image, label) in enumerate(zip(basis, labels, strict=True)):
        axis = basis_axes[index // 3, index % 3]
        shown = axis.imshow(
            image,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.set_title(label, fontsize=11, pad=2)
        axis.set_xticks((-1, 0, 1))
        axis.set_yticks((-1, 0, 1))
        axis.tick_params(labelsize=7)
    basis_figure.colorbar(shown, ax=basis_axes.ravel().tolist(), shrink=0.88, pad=0.01)
    basis_figure.suptitle(
        "A  Nine degree-2 2D Legendre basis functions",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )

    coefficient_axis = coefficient_figure.subplots()
    gamma_matrix = gamma.T
    limit = float(np.max(np.abs(gamma_matrix)))
    coefficient_image = coefficient_axis.imshow(
        gamma_matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    coefficient_axis.set_xticks(np.arange(len(labels)), labels, rotation=38, ha="right")
    coefficient_axis.set_yticks(np.arange(4), [f"PC {index}" for index in range(1, 5)])
    coefficient_axis.set_xlabel("Legendre basis")
    coefficient_axis.set_ylabel("PPG PCA mode")
    for row in range(gamma_matrix.shape[0]):
        for column in range(gamma_matrix.shape[1]):
            value = gamma_matrix[row, column]
            coefficient_axis.text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.55 * limit else "black",
            )
    coefficient_figure.colorbar(coefficient_image, ax=coefficient_axis, shrink=0.82)
    coefficient_figure.suptitle(
        "B  36 fitted gamma coefficients (4 modes x 9 basis)",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )

    field_axes = rows[1].subplots(1, 4)
    for mode, field in enumerate(fields):
        axis = field_axes[mode]
        field_limit = float(np.max(np.abs(field)))
        image = axis.imshow(
            field,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit),
        )
        axis.scatter(
            positions[accepted, 0],
            positions[accepted, 1],
            s=10,
            facecolors="none",
            edgecolors="black",
            linewidths=0.45,
            alpha=0.65,
            label="accepted bead" if mode == 0 else None,
        )
        axis.scatter(
            positions[heldout, 0],
            positions[heldout, 1],
            s=34,
            marker="s",
            facecolors="none",
            edgecolors="#0072B2",
            linewidths=1.4,
            label="fixed holdout" if mode == 0 else None,
        )
        axis.scatter(
            positions[rejected, 0],
            positions[rejected, 1],
            s=28,
            marker="x",
            color="#E69F00",
            linewidths=1.3,
            label="rejected training bead" if mode == 0 else None,
        )
        axis.set_title(f"PC {mode + 1} coefficient map", fontsize=11)
        axis.set_xlabel("normalized x")
        if mode == 0:
            axis.set_ylabel("normalized y")
            axis.legend(loc="lower left", fontsize=7)
        else:
            axis.set_yticklabels([])
        rows[1].colorbar(image, ax=axis, shrink=0.78)
    rows[1].suptitle(
        r"C  Resulting fields: $C_k(x,y)=\sum_b \gamma_{b,k}B_b(x,y)$",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.suptitle(
        "PPG default initialization: 36 gamma parameters, 21 x 21 PSF support",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppg-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--bead-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--source-tiff", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stack = tifffile.imread(args.source_tiff).astype(np.float64)
    with np.load(args.ppg_map, allow_pickle=False) as payload, np.load(
        args.bead_catalog, allow_pickle=False
    ) as catalog:
        render_holdouts(
            args.output_dir / "holdout5_raw_vs_recon.png", payload, catalog, stack
        )
        render_decomposition(
            args.output_dir / "gamma36_basis_coefficients_and_pca_maps.png",
            payload,
            catalog,
        )
    print(args.output_dir)


if __name__ == "__main__":
    main()
