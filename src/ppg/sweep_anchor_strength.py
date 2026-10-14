#!/usr/bin/env python3
"""Compare boundary-anchor strengths without changing the approved PPG map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

from ppg.beads_alignment import apply_affine_z_shift, fit_affine_z_shift, stack_ncc
from ppg.beads_initialization import (
    Z_NORMALIZED,
    extract_bead_signal,
    normalize_psf,
)
from ppg.physical_update import sample_coefficient_field
from ppg.spatial_gamma import (
    coefficient_field_from_gamma,
    degree2_legendre_design,
    fit_center_shrunk_gamma,
)
from unity_psf.localization.simulator import (
    LocalizationSimulatorConfig,
    _place_patches,
    _PPGDirectEmitterRenderer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "configs/calibration/ppg/ppg_map_initial.npz"
DEFAULT_TIFF = ROOT / "datasets/training_sets/Zeiss/63x1.4_642.tif"
DEFAULT_CATALOG = ROOT / "configs/calibration/ppg/bead_catalog_zeiss_63x1.4_642.npz"
DEFAULT_SIMULATION = (
    ROOT / "output/ppg_simulation_64x64_density0p2_z600/simulation_targets.npz"
)
DEFAULT_OUTPUT = ROOT / "output/ppg_anchor_strength_sweep_0p2_0p5"


def _reconstruct_stack(
    arrays: dict[str, np.ndarray], coefficients: np.ndarray
) -> np.ndarray:
    shape = tuple(int(value) for value in arrays["roi_shape_yx"])
    stack = (
        arrays["pca_mean"] + np.asarray(coefficients) @ arrays["pca_components"]
    ).reshape(len(arrays["train_z_um"]), *shape)
    return normalize_psf(stack)


def _reconstruct_at_z(
    arrays: dict[str, np.ndarray], coefficients: np.ndarray, z_um: float
) -> np.ndarray:
    stack = _reconstruct_stack(arrays, coefficients)
    z_train = arrays["train_z_um"]
    upper = int(np.clip(np.searchsorted(z_train, z_um), 1, len(z_train) - 1))
    lower = upper - 1
    weight = float(
        np.clip(
            (z_um - z_train[lower]) / (z_train[upper] - z_train[lower]),
            0.0,
            1.0,
        )
    )
    return normalize_psf(stack[lower] * (1.0 - weight) + stack[upper] * weight)


def _fit_candidate(
    base: dict[str, np.ndarray], *, strength: float
) -> dict[str, np.ndarray]:
    arrays = {name: np.array(value, copy=True) for name, value in base.items()}
    gamma = fit_center_shrunk_gamma(
        degree2_legendre_design(arrays["coefficient_positions_xy_normalized"]),
        arrays["coefficient_values"],
        degree2_legendre_design(arrays["boundary_anchor_positions_xy_normalized"]),
        arrays["boundary_anchor_density_weights"],
        anchor_target=arrays["boundary_anchor_center_score_target"],
        anchor_strength=float(strength),
        ridge=float(arrays["gamma_ridge"]),
    )
    arrays["gamma"] = gamma
    arrays["coefficient_field"] = coefficient_field_from_gamma(
        gamma, arrays["xy_grid_normalized"]
    )
    arrays["boundary_anchor_strength"] = np.asarray(strength)
    return arrays


def _holdout_metrics(
    arrays: dict[str, np.ndarray],
    *,
    raw_calibration: np.ndarray,
    bead_ids: np.ndarray,
    positions: np.ndarray,
) -> tuple[list[float], float, float]:
    holdout = np.isin(bead_ids, arrays["heldout_ids"])
    scores = degree2_legendre_design(positions[holdout]) @ arrays["gamma"]
    predictions = normalize_psf(
        (arrays["pca_mean"][None] + scores @ arrays["pca_components"]).reshape(
            int(holdout.sum()), len(arrays["train_z_um"]), 21, 21
        )
    )
    values = []
    for observed, prediction in zip(raw_calibration[holdout], predictions, strict=True):
        parameters = fit_affine_z_shift(
            observed,
            prediction,
            Z_NORMALIZED,
            constant_bound_px=3.0,
            slope_bound_px=1.0,
        )
        registered = normalize_psf(
            apply_affine_z_shift(prediction, parameters, Z_NORMALIZED)
        )
        values.append(float(np.median(stack_ncc(observed, registered))))
    return values, float(np.median(values)), float(np.min(values))


def _registered_holdout_predictions(
    arrays: dict[str, np.ndarray],
    *,
    raw_calibration: np.ndarray,
    bead_ids: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    holdout = np.isin(bead_ids, arrays["heldout_ids"])
    scores = degree2_legendre_design(positions[holdout]) @ arrays["gamma"]
    predictions = normalize_psf(
        (arrays["pca_mean"][None] + scores @ arrays["pca_components"]).reshape(
            int(holdout.sum()), len(arrays["train_z_um"]), 21, 21
        )
    )
    registered = []
    ncc = []
    for observed, prediction in zip(raw_calibration[holdout], predictions, strict=True):
        parameters = fit_affine_z_shift(
            observed,
            prediction,
            Z_NORMALIZED,
            constant_bound_px=3.0,
            slope_bound_px=1.0,
        )
        item = normalize_psf(apply_affine_z_shift(prediction, parameters, Z_NORMALIZED))
        registered.append(item)
        ncc.append(stack_ncc(observed, item))
    return raw_calibration[holdout], np.stack(registered), np.stack(ncc)


def _coefficient_metrics(
    arrays: dict[str, np.ndarray],
    *,
    frame009_origin_xy: np.ndarray,
) -> dict[str, object]:
    values = arrays["coefficient_values"]
    lower, upper = np.percentile(values, [1.0, 99.0], axis=0)
    field = arrays["coefficient_field"]
    grid = arrays["xy_grid_normalized"]
    x0, y0 = frame009_origin_xy / (arrays["field_shape_yx"] - 1)[::-1]
    x1 = (frame009_origin_xy[0] + 63.0) / (arrays["field_shape_yx"][1] - 1)
    y1 = (frame009_origin_xy[1] + 63.0) / (arrays["field_shape_yx"][0] - 1)
    x_mask = (grid >= x0) & (grid <= x1)
    y_mask = (grid >= y0) & (grid <= y1)
    region = field[:, y_mask][:, :, x_mask]
    outside = (region < lower[:, None, None]) | (region > upper[:, None, None])
    center_target = arrays["boundary_anchor_center_score_target"]
    top_right = field[:, 0, -1]
    scale = np.maximum(np.std(values, axis=0), 1e-12)
    return {
        "top_right_coefficients": top_right.tolist(),
        "top_right_center_distance_standardized": float(
            np.linalg.norm((top_right - center_target) / scale)
        ),
        "frame009_region_outside_training_1_99_fraction": float(outside.mean()),
        "frame009_region_coeff_min": region.min(axis=(1, 2)).tolist(),
        "frame009_region_coeff_max": region.max(axis=(1, 2)).tolist(),
    }


def _plot_corner_z_stacks(
    output: Path,
    *,
    candidates: dict[float, dict[str, np.ndarray]],
    location_xy_normalized: tuple[float, float],
) -> None:
    strengths = list(candidates)
    z_values = candidates[strengths[0]]["train_z_um"]
    fig, axes = plt.subplots(
        len(z_values),
        len(strengths),
        figsize=(2.15 * len(strengths), 1.85 * len(z_values)),
        constrained_layout=True,
    )
    for column, strength in enumerate(strengths):
        arrays = candidates[strength]
        coefficients = sample_coefficient_field(
            arrays["coefficient_field"], location_xy_normalized
        )
        stack = _reconstruct_stack(arrays, coefficients)
        for row, (z_um, image) in enumerate(zip(z_values, stack, strict=True)):
            axis = axes[row, column]
            axis.imshow(
                image,
                cmap="magma",
                vmin=0.0,
                vmax=float(np.percentile(image, 99.5)),
                interpolation="nearest",
            )
            if row == 0:
                axis.set_title(f"anchor {strength:.1f}", fontsize=10)
            if column == 0:
                axis.set_ylabel(f"z={z_um:+.1f} um", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Right-top sparse corner PSF: frame 009 center (x=608, y=32)",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output / "right_top_corner_zstack.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_frame009_emitters(
    output: Path,
    *,
    candidates: dict[float, dict[str, np.ndarray]],
    targets: np.ndarray,
    origin_xy: np.ndarray,
) -> None:
    strengths = list(candidates)
    emitter_indices = [0, 1, 5, 7]
    fig, axes = plt.subplots(
        len(emitter_indices),
        len(strengths),
        figsize=(2.15 * len(strengths), 2.05 * len(emitter_indices)),
        constrained_layout=True,
    )
    for column, strength in enumerate(strengths):
        arrays = candidates[strength]
        field_denominator = (arrays["field_shape_yx"] - 1)[::-1]
        for row, emitter_index in enumerate(emitter_indices):
            emitter = targets[emitter_index]
            global_xy = emitter[:2] + origin_xy
            normalized_xy = tuple(global_xy / field_denominator)
            coefficients = sample_coefficient_field(
                arrays["coefficient_field"], normalized_xy
            )
            image = _reconstruct_at_z(arrays, coefficients, float(emitter[2]))
            axis = axes[row, column]
            axis.imshow(
                image,
                cmap="magma",
                vmin=0.0,
                vmax=float(np.percentile(image, 99.5)),
                interpolation="nearest",
            )
            if row == 0:
                axis.set_title(f"anchor {strength:.1f}", fontsize=10)
            if column == 0:
                axis.set_ylabel(
                    f"emitter #{emitter_index}\nz={emitter[2] * 1000:+.0f} nm",
                    fontsize=8,
                )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Frame 009 individual PSFs before overlap and edge clipping",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output / "frame009_emitter_sweep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_frame009_frames(
    output: Path,
    *,
    candidates: dict[float, dict[str, np.ndarray]],
    targets: np.ndarray,
    origin_xy: np.ndarray,
) -> None:
    strengths = list(candidates)
    fig, axes = plt.subplots(
        1,
        len(strengths),
        figsize=(2.35 * len(strengths), 2.65),
        constrained_layout=True,
    )
    frames = []
    for strength, axis in zip(strengths, axes, strict=True):
        token = str(strength).replace(".", "p")
        map_path = output / "maps" / f"ppg_map_anchor_{token}.npz"
        renderer = _PPGDirectEmitterRenderer(
            LocalizationSimulatorConfig(
                batch_size=1,
                frames_per_sample=1,
                height=64,
                width=64,
                psf_type="ppg_direct",
                vector_psf_size=21,
                field_origin_xy=tuple(int(value) for value in origin_xy),
                ppg_map_path=str(map_path),
            )
        )
        patches, center_x, center_y = renderer._render_patches(
            xs=torch.as_tensor(targets[:, 0]).contiguous(),
            ys=torch.as_tensor(targets[:, 1]).contiguous(),
            z_um=torch.as_tensor(targets[:, 2]).contiguous(),
            photons=torch.as_tensor(targets[:, 3]).contiguous(),
            field_origin_xy=tuple(int(value) for value in origin_xy),
        )
        frame = (
            _place_patches(
                patches,
                height=64,
                width=64,
                center_x=center_x,
                center_y=center_y,
            )
            .detach()
            .cpu()
            .numpy()
        )
        frames.append(frame)
    vmax = float(
        np.percentile(np.concatenate([frame.ravel() for frame in frames]), 99.5)
    )
    for strength, frame, axis in zip(strengths, frames, axes, strict=True):
        axis.imshow(
            frame,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(f"anchor {strength:.1f}", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Frame 009: identical emitters, photons, subpixel shifts, and edge clipping",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output / "frame009_full_frame_sweep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics(output: Path, rows: list[dict[str, object]]) -> None:
    strengths = np.asarray([row["anchor_strength"] for row in rows])
    holdout = np.asarray([row["holdout_ncc_median"] for row in rows])
    corner = np.asarray([row["top_right_center_distance_standardized"] for row in rows])
    outside = np.asarray(
        [row["frame009_region_outside_training_1_99_fraction"] for row in rows]
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True)
    axes[0].plot(strengths, holdout, marker="o", color="#0072B2")
    axes[0].set_ylabel("Median holdout NCC")
    axes[1].plot(strengths, corner, marker="o", color="#D55E00")
    axes[1].set_ylabel("Corner-to-center distance (std.)")
    axes[2].plot(strengths, outside, marker="o", color="#009E73")
    axes[2].set_ylabel("Frame 009 coeff. outside 1-99%")
    for axis in axes:
        axis.set_xlabel("Boundary anchor strength")
        axis.grid(alpha=0.2)
    fig.suptitle("PPG boundary-anchor sweep", fontsize=12, fontweight="bold")
    fig.savefig(output / "anchor_strength_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_nearest_raw_beads(
    output: Path,
    *,
    base: dict[str, np.ndarray],
    raw_calibration: np.ndarray,
    bead_ids: np.ndarray,
    positions: np.ndarray,
    frame009_origin_xy: np.ndarray,
) -> list[int]:
    accepted = np.isin(bead_ids, base["accepted_ids"])
    field_denominator = (base["field_shape_yx"] - 1)[::-1]
    corner_center = (frame009_origin_xy + 31.5) / field_denominator
    distances = np.linalg.norm(positions[accepted] - corner_center, axis=1)
    accepted_indices = np.flatnonzero(accepted)
    nearest_indices = accepted_indices[np.argsort(distances)[:2]]
    nearest_ids = [int(bead_ids[index]) for index in nearest_indices]
    z_values = base["train_z_um"]
    fig, axes = plt.subplots(
        len(nearest_indices),
        len(z_values),
        figsize=(12.2, 3.8),
        constrained_layout=True,
    )
    for row, index in enumerate(nearest_indices):
        for column, (z_um, image) in enumerate(
            zip(z_values, raw_calibration[index], strict=True)
        ):
            axis = axes[row, column]
            axis.imshow(
                image,
                cmap="magma",
                vmin=0.0,
                vmax=float(np.percentile(image, 99.5)),
                interpolation="nearest",
            )
            if row == 0:
                axis.set_title(f"z={z_um:+.1f} um", fontsize=8)
            if column == 0:
                x, y = positions[index]
                axis.set_ylabel(
                    f"raw bead {int(bead_ids[index])}\nFOV=({x:.2f}, {y:.2f})",
                    fontsize=8,
                )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Nearest accepted Beads TIFF sources to the frame 009 sparse corner",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output / "frame009_nearest_raw_beads.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return nearest_ids


def _plot_holdout_comparisons(
    output: Path,
    *,
    candidates: dict[float, dict[str, np.ndarray]],
    raw_calibration: np.ndarray,
    bead_ids: np.ndarray,
    positions: np.ndarray,
) -> None:
    strengths = list(candidates)
    holdout_ids = candidates[strengths[0]]["heldout_ids"]
    holdout_mask = np.isin(bead_ids, holdout_ids)
    holdout_positions = positions[holdout_mask]
    raw = raw_calibration[holdout_mask]
    predictions = {}
    ncc = {}
    for strength, arrays in candidates.items():
        _, registered, values = _registered_holdout_predictions(
            arrays,
            raw_calibration=raw_calibration,
            bead_ids=bead_ids,
            positions=positions,
        )
        predictions[strength] = registered
        ncc[strength] = values
    z_values = candidates[strengths[0]]["train_z_um"]
    holdout_dir = output / "holdouts"
    holdout_dir.mkdir(exist_ok=True)
    for bead_index, (bead_id, fov_xy) in enumerate(
        zip(holdout_ids, holdout_positions, strict=True)
    ):
        labels = ["raw", *[f"anchor {strength:.1f}" for strength in strengths]]
        stacks = [raw[bead_index], *[predictions[s][bead_index] for s in strengths]]
        fig, axes = plt.subplots(
            len(stacks),
            len(z_values),
            figsize=(12.2, 1.85 * len(stacks)),
            constrained_layout=True,
        )
        for row, (label, stack) in enumerate(zip(labels, stacks, strict=True)):
            for column, (z_um, image) in enumerate(zip(z_values, stack, strict=True)):
                axis = axes[row, column]
                axis.imshow(
                    image,
                    cmap="magma",
                    vmin=0.0,
                    vmax=float(np.percentile(image, 99.5)),
                    interpolation="nearest",
                )
                if row == 0:
                    axis.set_title(f"z={z_um:+.1f} um", fontsize=8)
                if column == 0:
                    suffix = ""
                    if row > 0:
                        strength = strengths[row - 1]
                        suffix = (
                            f"\nmedian NCC={np.median(ncc[strength][bead_index]):.3f}"
                        )
                    axis.set_ylabel(f"{label}{suffix}", fontsize=8)
                axis.set_xticks([])
                axis.set_yticks([])
        fig.suptitle(
            f"Holdout bead {int(bead_id)} at FOV=({fov_xy[0]:.2f}, {fov_xy[1]:.2f})",
            fontsize=12,
            fontweight="bold",
        )
        fig.savefig(
            holdout_dir / f"holdout_bead_{int(bead_id)}_raw_vs_anchor.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def _plot_full_field_atlas(
    output: Path,
    *,
    arrays: dict[str, np.ndarray],
    strength: float,
    samples_per_axis: int = 5,
) -> None:
    coordinates = np.linspace(0.0, 1.0, samples_per_axis)
    atlas_dir = output / f"field_atlas_anchor_{str(strength).replace('.', 'p')}"
    atlas_dir.mkdir(exist_ok=True)
    for z_um in arrays["train_z_um"]:
        fig, axes = plt.subplots(
            samples_per_axis,
            samples_per_axis,
            figsize=(8.2, 8.2),
            constrained_layout=True,
        )
        for row, y in enumerate(coordinates):
            for column, x in enumerate(coordinates):
                coefficients = sample_coefficient_field(
                    arrays["coefficient_field"], (float(x), float(y))
                )
                image = _reconstruct_at_z(arrays, coefficients, float(z_um))
                axis = axes[row, column]
                axis.imshow(
                    image,
                    cmap="magma",
                    vmin=0.0,
                    vmax=float(np.percentile(image, 99.5)),
                    interpolation="nearest",
                )
                axis.set_title(f"FOV=({x:.2f}, {y:.2f})", fontsize=7)
                axis.set_xticks([])
                axis.set_yticks([])
        fig.suptitle(
            f"Full-field PPG atlas: anchor={strength:.1f}, z={z_um:+.1f} um",
            fontsize=12,
            fontweight="bold",
        )
        z_token = f"{z_um:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
        fig.savefig(
            atlas_dir / f"field_psf_atlas_z_{z_token}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def _plot_nine_region_strength_comparison(
    output: Path,
    *,
    candidates: dict[float, dict[str, np.ndarray]],
) -> None:
    coordinates = (0.05, 0.50, 0.95)
    positions = [(x, y) for y in coordinates for x in coordinates]
    strengths = list(candidates)
    for z_um in (-0.6, 0.0, 0.6):
        fig, axes = plt.subplots(
            len(positions),
            len(strengths),
            figsize=(2.0 * len(strengths), 1.65 * len(positions)),
            constrained_layout=True,
        )
        for column, strength in enumerate(strengths):
            arrays = candidates[strength]
            for row, (x, y) in enumerate(positions):
                coefficients = sample_coefficient_field(
                    arrays["coefficient_field"], (x, y)
                )
                image = _reconstruct_at_z(arrays, coefficients, z_um)
                axis = axes[row, column]
                axis.imshow(
                    image,
                    cmap="magma",
                    vmin=0.0,
                    vmax=float(np.percentile(image, 99.5)),
                    interpolation="nearest",
                )
                if row == 0:
                    axis.set_title(f"anchor {strength:.1f}", fontsize=9)
                if column == 0:
                    axis.set_ylabel(f"FOV=({x:.2f}, {y:.2f})", fontsize=7)
                axis.set_xticks([])
                axis.set_yticks([])
        fig.suptitle(
            f"Nine field regions across anchor strengths: z={z_um:+.1f} um",
            fontsize=12,
            fontweight="bold",
        )
        z_token = f"{z_um:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
        fig.savefig(
            output / f"nine_region_anchor_comparison_z_{z_token}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def run_sweep(
    *,
    map_path: Path,
    source_tiff: Path,
    catalog_path: Path,
    simulation_path: Path,
    output: Path,
    strengths: list[float],
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    maps_dir = output / "maps"
    maps_dir.mkdir(exist_ok=True)
    with np.load(map_path, allow_pickle=False) as payload:
        base = {name: payload[name] for name in payload.files}
    with np.load(catalog_path, allow_pickle=False) as payload:
        bead_ids = payload["bead_ids"]
        centers_yx = payload["centers_yx"]
        positions = payload["positions_xy_normalized"]
    raw_calibration = normalize_psf(
        extract_bead_signal(tifffile.imread(source_tiff), centers_yx)
    )
    with np.load(simulation_path, allow_pickle=False) as payload:
        origin009 = payload["field_origins_xy"][9]
        targets009 = payload["emitter_xy_z_um_photons"][9][payload["emitter_mask"][9]]

    candidates = {}
    rows = []
    for strength in strengths:
        arrays = _fit_candidate(base, strength=strength)
        candidates[strength] = arrays
        token = str(strength).replace(".", "p")
        map_output = maps_dir / f"ppg_map_anchor_{token}.npz"
        np.savez_compressed(map_output, **arrays)
        holdout_values, holdout_median, holdout_min = _holdout_metrics(
            arrays,
            raw_calibration=raw_calibration,
            bead_ids=bead_ids,
            positions=positions,
        )
        row = {
            "anchor_strength": strength,
            "map": str(map_output),
            "holdout_ncc_per_bead": {
                str(bead_id): value
                for bead_id, value in zip(
                    arrays["heldout_ids"], holdout_values, strict=True
                )
            },
            "holdout_ncc_median": holdout_median,
            "holdout_ncc_min": holdout_min,
            **_coefficient_metrics(arrays, frame009_origin_xy=origin009),
        }
        rows.append(row)

    field_shape = base["field_shape_yx"]
    frame009_center = origin009 + 31.5
    normalized_center = tuple(frame009_center / (field_shape - 1)[::-1])
    _plot_corner_z_stacks(
        output,
        candidates=candidates,
        location_xy_normalized=normalized_center,
    )
    _plot_frame009_emitters(
        output,
        candidates=candidates,
        targets=targets009,
        origin_xy=origin009,
    )
    _plot_frame009_frames(
        output,
        candidates=candidates,
        targets=targets009,
        origin_xy=origin009,
    )
    _plot_metrics(output, rows)
    nearest_ids = _plot_nearest_raw_beads(
        output,
        base=base,
        raw_calibration=raw_calibration,
        bead_ids=bead_ids,
        positions=positions,
        frame009_origin_xy=origin009,
    )
    _plot_holdout_comparisons(
        output,
        candidates=candidates,
        raw_calibration=raw_calibration,
        bead_ids=bead_ids,
        positions=positions,
    )
    strongest = max(candidates)
    _plot_full_field_atlas(
        output,
        arrays=candidates[strongest],
        strength=strongest,
    )
    _plot_nine_region_strength_comparison(output, candidates=candidates)
    summary = {
        "schema": "ppg.anchor_strength_sweep.v1",
        "base_map": str(map_path),
        "strengths": strengths,
        "frame009_origin_xy": origin009.tolist(),
        "frame009_nearest_accepted_bead_ids": nearest_ids,
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--source-tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--simulation", type=Path, default=DEFAULT_SIMULATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    args = parser.parse_args()
    summary = run_sweep(
        map_path=args.map.resolve(),
        source_tiff=args.source_tiff.resolve(),
        catalog_path=args.catalog.resolve(),
        simulation_path=args.simulation.resolve(),
        output=args.output.resolve(),
        strengths=[float(value) for value in args.strengths],
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
