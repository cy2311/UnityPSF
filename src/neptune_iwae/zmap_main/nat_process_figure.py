from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import textwrap
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path("/tmp") / "matplotlib-codex").resolve())

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from .nat_field import NATFieldConfig, NATGamma


@dataclass(frozen=True)
class NATProcessRun:
    run_dir: Path
    side: str | None
    diagnostics_config: dict[str, Any]
    pixel_size_x_nm: float
    pixel_size_y_nm: float
    gamma: torch.Tensor
    roixy_px: torch.Tensor
    raw_patches: torch.Tensor
    recon_patches: torch.Tensor
    patch_ncc: torch.Tensor
    coeff_stack: np.ndarray
    mode_order: list[tuple[int, int]]
    scalar_map: np.ndarray
    selection_count: int
    image_shape_hw: tuple[int, int]


@dataclass(frozen=True)
class LegendrePanel:
    gamma_index: int
    values: np.ndarray
    basis_label: str
    target_label: str


@dataclass(frozen=True)
class PatchExample:
    patch_index: int
    ncc: float
    brightness: float
    raw_display: np.ndarray
    recon_display: np.ndarray


@dataclass(frozen=True)
class GammaDisplaySummary:
    raw_values: np.ndarray
    normalized_values: np.ndarray
    normalization: str
    compression_scale: float
    max_abs_value: float


@dataclass(frozen=True)
class PeakOverlayFrame:
    frame_index: int
    image_adu: np.ndarray
    roixy_px: np.ndarray
    patch_size_px: int
    roi_shape_hw: tuple[int, int]
    roi_origin_xy_px: tuple[float, float]


@dataclass(frozen=True)
class NATProcessFigureResult:
    figure_path: Path
    summary_path: Path
    panel_paths: dict[str, Path]
    selected_patch_indices: torch.Tensor
    gamma_display: GammaDisplaySummary


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ncc(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_centered = pred - pred.mean(dim=(-2, -1), keepdim=True)
    denom = torch.sqrt(
        target_centered.square().sum(dim=(-2, -1)).clamp_min(1e-8)
        * pred_centered.square().sum(dim=(-2, -1)).clamp_min(1e-8)
    )
    return (target_centered * pred_centered).sum(dim=(-2, -1)) / denom


def _image_shape_from_payload(summary: dict[str, Any], scalar_payload: np.lib.npyio.NpzFile) -> tuple[int, int]:
    image_shape = ((summary.get("export_summary") or {}).get("image_shape_hw")) or None
    if image_shape is not None:
        return int(image_shape[0]), int(image_shape[1])
    metadata_json = scalar_payload.get("metadata_json")
    if metadata_json is not None:
        metadata = json.loads(str(metadata_json.item()))
        if "image_shape_hw" in metadata:
            return tuple(int(v) for v in metadata["image_shape_hw"])
    values_nm = np.asarray(scalar_payload["values_nm"], dtype=np.float32)
    return int(values_nm.shape[0]), int(values_nm.shape[1])


def _pixel_size_from_summary(summary: dict[str, Any]) -> tuple[float, float]:
    candidate_configs = (
        ((summary.get("harvest_summary") or {}).get("config")) or {},
        ((summary.get("diagnostics_summary") or {}).get("config")) or {},
        ((summary.get("export_summary") or {})),
    )
    for config in candidate_configs:
        pixel_size_x_nm = config.get("pixel_size_x_nm")
        pixel_size_y_nm = config.get("pixel_size_y_nm")
        if pixel_size_x_nm is not None and pixel_size_y_nm is not None:
            return float(pixel_size_x_nm), float(pixel_size_y_nm)
    return 1.0, 1.0


def _legendre_basis(order: int, values: np.ndarray) -> np.ndarray:
    coeffs = np.zeros(int(order) + 1, dtype=np.float64)
    coeffs[int(order)] = 1.0
    return np.polynomial.legendre.legval(values, coeffs)


def _patch_brightness(patches: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(patches, dtype=torch.float32)
    return values.sum(dim=(-2, -1))


def summarize_gamma_for_display(
    gamma: torch.Tensor | np.ndarray,
    *,
    compression_percentile: float = 75.0,
) -> GammaDisplaySummary:
    raw_values = np.asarray(torch.as_tensor(gamma, dtype=torch.float32).reshape(-1).cpu().numpy(), dtype=np.float32)
    abs_values = np.abs(raw_values)
    max_abs_value = float(abs_values.max()) if abs_values.size else 0.0
    if max_abs_value <= 0.0 or not np.isfinite(max_abs_value):
        return GammaDisplaySummary(
            raw_values=raw_values,
            normalized_values=np.zeros_like(raw_values, dtype=np.float32),
            normalization="signed_log_maxabs",
            compression_scale=1.0,
            max_abs_value=0.0,
        )

    compression_scale = float(np.percentile(abs_values, compression_percentile))
    if not np.isfinite(compression_scale) or compression_scale <= 0.0:
        compression_scale = max_abs_value
    compression_scale = min(compression_scale, max_abs_value)
    denominator = np.log1p(max_abs_value / compression_scale)
    if denominator <= 0.0 or not np.isfinite(denominator):
        normalized_values = raw_values / max_abs_value
    else:
        normalized_values = np.sign(raw_values) * np.log1p(abs_values / compression_scale) / denominator
    return GammaDisplaySummary(
        raw_values=raw_values,
        normalized_values=np.asarray(normalized_values, dtype=np.float32),
        normalization="signed_log_maxabs",
        compression_scale=float(compression_scale),
        max_abs_value=max_abs_value,
    )


def select_patch_indices_by_ncc(
    ncc: torch.Tensor,
    *,
    brightness: torch.Tensor | None = None,
    count: int = 8,
) -> torch.Tensor:
    values = torch.as_tensor(ncc, dtype=torch.float32).reshape(-1)
    if values.numel() == 0 or int(count) <= 0:
        return torch.zeros((0,), dtype=torch.int64)
    ordered = torch.argsort(values, stable=True)
    target = min(int(count), int(ordered.numel()))
    if brightness is not None:
        brightness_values = torch.as_tensor(brightness, dtype=torch.float32).reshape(-1)
        if brightness_values.shape != values.shape:
            raise ValueError("brightness must match ncc shape.")
        chosen = []
        seen: set[int] = set()
        for chunk in np.array_split(ordered.cpu().numpy(), target):
            candidates = [int(index) for index in chunk.tolist() if int(index) not in seen]
            if not candidates:
                continue
            best = max(candidates, key=lambda index: (float(brightness_values[index].item()), float(values[index].item())))
            seen.add(best)
            chosen.append(best)
        if len(chosen) < target:
            remaining = [int(index) for index in ordered.tolist() if int(index) not in seen]
            remaining.sort(
                key=lambda index: (float(brightness_values[index].item()), float(values[index].item())),
                reverse=True,
            )
            chosen.extend(remaining[: target - len(chosen)])
        return torch.as_tensor(chosen, dtype=torch.int64)
    anchors = np.linspace(0, ordered.numel() - 1, num=target)
    chosen = []
    seen: set[int] = set()
    for anchor in anchors:
        index = int(ordered[int(round(float(anchor)))].item())
        if index not in seen:
            seen.add(index)
            chosen.append(index)
    if len(chosen) < target:
        for index in ordered.tolist():
            if index not in seen:
                seen.add(index)
                chosen.append(int(index))
            if len(chosen) == target:
                break
    return torch.as_tensor(chosen, dtype=torch.int64)


def describe_gamma_targets(gamma: NATGamma) -> str:
    seen: list[tuple[int, int]] = []
    for component in gamma.components:
        mode = (int(component.n), int(component.m))
        if mode not in seen:
            seen.append(mode)
    return ", ".join(f"Z({n},{m})" for n, m in seen)


def _describe_gamma_basis(gamma: NATGamma) -> str:
    seen: list[tuple[int, int]] = []
    for component in gamma.components:
        order = (int(component.px), int(component.py))
        if order not in seen:
            seen.append(order)
    return " + ".join(f"P({px},{py})" for px, py in seen)


def build_legendre_basis_planes(cfg: NATFieldConfig, *, width: int, height: int) -> list[LegendrePanel]:
    xs = np.linspace(-1.0, 1.0, int(width), dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, int(height), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    panels: list[LegendrePanel] = []
    for gamma_index, gamma in enumerate(cfg.gammas):
        plane = np.zeros((int(height), int(width)), dtype=np.float64)
        for component in gamma.components:
            norm = np.sqrt((1.0 + 2.0 * float(component.px)) * (1.0 + 2.0 * float(component.py)))
            plane += (
                float(component.weight)
                * norm
                * _legendre_basis(int(component.px), xx)
                * _legendre_basis(int(component.py), yy)
            )
        panels.append(
            LegendrePanel(
                gamma_index=int(gamma_index),
                values=plane.astype(np.float32, copy=False),
                basis_label=_describe_gamma_basis(gamma),
                target_label=describe_gamma_targets(gamma),
            )
        )
    return panels


def _symmetric_limit(values: np.ndarray, percentile: float = 99.5) -> float:
    vmax = float(np.percentile(np.abs(values), percentile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return 1.0
    return vmax


def _nonnegative_limit(values: np.ndarray, percentile: float = 99.5) -> float:
    vmax = float(np.percentile(values, percentile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return 1.0
    return vmax


def _style_axis_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def field_extent_um(run: NATProcessRun) -> tuple[float, float, float, float]:
    width_um = float(run.image_shape_hw[1]) * float(run.pixel_size_x_nm) * 1e-3
    height_um = float(run.image_shape_hw[0]) * float(run.pixel_size_y_nm) * 1e-3
    return 0.0, width_um, 0.0, height_um


def _field_width_height_um(run: NATProcessRun) -> tuple[float, float]:
    _, width_um, _, height_um = field_extent_um(run)
    return width_um, height_um


def _standalone_field_figsize(
    run: NATProcessRun,
    *,
    width_in: float,
    min_height_in: float = 5.5,
    max_height_in: float = 10.5,
) -> tuple[float, float]:
    width_um, height_um = _field_width_height_um(run)
    if width_um <= 0.0 or height_um <= 0.0:
        return width_in, min_height_in
    height_in = float(np.clip(width_in * (height_um / width_um), min_height_in, max_height_in))
    return width_in, height_in


def select_overlay_frame_index(
    frame_index: torch.Tensor,
    probability: torch.Tensor | None = None,
) -> int:
    frames = torch.as_tensor(frame_index, dtype=torch.int64).reshape(-1)
    if frames.numel() == 0:
        raise ValueError("frame_index must not be empty.")
    probs = (
        torch.ones_like(frames, dtype=torch.float32)
        if probability is None
        else torch.as_tensor(probability, dtype=torch.float32).reshape(-1)
    )
    if probs.shape != frames.shape:
        raise ValueError("probability must match frame_index shape.")
    counts: dict[int, int] = {}
    scores: dict[int, float] = {}
    for frame, score in zip(frames.tolist(), probs.tolist(), strict=False):
        frame_id = int(frame)
        counts[frame_id] = counts.get(frame_id, 0) + 1
        scores[frame_id] = scores.get(frame_id, 0.0) + float(score)
    return max(counts, key=lambda frame_id: (counts[frame_id], scores[frame_id], -frame_id))


def _normalize_display_image(
    image: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.8,
    gamma: float = 0.85,
) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    vmin = float(np.percentile(values, lower_percentile))
    vmax = float(np.percentile(values, upper_percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(values.min())
        vmax = float(values.max())
    scaled = np.clip((values - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    if gamma != 1.0:
        scaled = np.power(scaled, gamma)
    return scaled.astype(np.float32, copy=False)


def _build_overlay_fit_config(run: NATProcessRun):
    config = run.diagnostics_config or {}
    if not config.get("pipeline_config_path") or not config.get("tiff_path"):
        return None
    from .nat_fit import build_nat_spot_fitter_config

    return build_nat_spot_fitter_config(
        pipeline_config_path=config["pipeline_config_path"],
        tiff_path=config["tiff_path"],
        nat_config_kind=str(config.get("nat_config_kind", "order1")),
        device="cpu",
        patch_size_px=int(config.get("patch_size_px", run.raw_patches.shape[-1])),
        max_emitters=int(config.get("max_emitters", run.selection_count)),
        min_neighbor_distance_px=config.get("min_neighbor_distance_px"),
        global_projected_min_distance_px=float(config.get("global_projected_min_distance_px", 0.0)),
        selection_mode=str(config.get("selection_mode", "probability_topk")),
        spatial_balance_grid_px=int(config.get("spatial_balance_grid_px", 128)),
        spatial_balance_min_per_cell=int(config.get("spatial_balance_min_per_cell", 0)),
        spatial_balance_max_per_cell=int(config.get("spatial_balance_max_per_cell", 0)),
        max_patch_peak_distance_px=float(config.get("max_patch_peak_distance_px", 2.5)),
        secondary_peak_exclusion_radius_px=float(config.get("secondary_peak_exclusion_radius_px", 3.5)),
        max_secondary_peak_fraction=float(config.get("max_secondary_peak_fraction", 0.45)),
        rank_spots_by_quality=bool(config.get("rank_spots_by_quality", False)),
        min_center_peak_norm=float(config.get("min_center_peak_norm", 0.0)),
        min_signal_sum_norm=float(config.get("min_signal_sum_norm", 0.0)),
        alternating_min_photon_fraction=float(config.get("alternating_min_photon_fraction", 0.0)),
        alternating_min_background_fraction=float(config.get("alternating_min_background_fraction", 0.0)),
        roi_x_min_px=float(config.get("roi_x_min_px", 0.0)),
        roi_x_max_px=None if config.get("roi_x_max_px") is None else float(config["roi_x_max_px"]),
        roi_y_min_px=float(config.get("roi_y_min_px", 0.0)),
        roi_y_max_px=None if config.get("roi_y_max_px") is None else float(config["roi_y_max_px"]),
        roi_edge_margin_px=float(config.get("roi_edge_margin_px", 0.0)),
        input_offset=config.get("input_offset"),
        input_scale=config.get("input_scale"),
        patch_adu_mode=str(config.get("patch_adu_mode", "baseline_clamp")),
        baseline_adu=config.get("baseline_adu"),
    )


def build_peak_overlay_frame(run: NATProcessRun) -> PeakOverlayFrame | None:
    fit_cfg = _build_overlay_fit_config(run)
    if fit_cfg is None:
        return None
    harvest_path = run.run_dir / "peak_harvest" / "harvest.pt"
    if not harvest_path.exists():
        return None
    from .harvest import HarvestedEmitters, load_tiff_frame_stack
    from .nat_fit import build_patch_dataset_from_emitters

    harvest_payload = torch.load(harvest_path, map_location="cpu", weights_only=False)
    payload = harvest_payload.get("payload")
    if not isinstance(payload, dict):
        return None
    dataset = build_patch_dataset_from_emitters(HarvestedEmitters(**payload), fit_cfg)
    if dataset.roixy_px.shape != run.roixy_px.shape or not torch.equal(dataset.roixy_px, run.roixy_px):
        return None
    frame_idx = select_overlay_frame_index(dataset.frame_index, dataset.probability)
    raw_frame = load_tiff_frame_stack(fit_cfg.tiff_path, frame_idx, frame_idx + 1)[0].cpu().numpy().astype(np.float32, copy=False)
    x0 = int(round(float(fit_cfg.roi_x_min_px)))
    x1 = raw_frame.shape[1] if fit_cfg.roi_x_max_px is None else int(round(float(fit_cfg.roi_x_max_px)))
    y0 = int(round(float(fit_cfg.roi_y_min_px)))
    y1 = raw_frame.shape[0] if fit_cfg.roi_y_max_px is None else int(round(float(fit_cfg.roi_y_max_px)))
    mask = dataset.frame_index == int(frame_idx)
    return PeakOverlayFrame(
        frame_index=int(frame_idx),
        image_adu=np.asarray(raw_frame[y0:y1, x0:x1], dtype=np.float32),
        roixy_px=np.asarray(dataset.roixy_px[mask].cpu().numpy(), dtype=np.float32),
        patch_size_px=int(fit_cfg.patch_size_px),
        roi_shape_hw=(int(y1 - y0), int(x1 - x0)),
        roi_origin_xy_px=(float(x0), float(y0)),
    )


def draw_selected_peak_overlay(
    ax,
    overlay: PeakOverlayFrame,
    *,
    pixel_size_x_nm: float,
    pixel_size_y_nm: float,
    box_color: str = "#56cfe1",
) -> None:
    display = _normalize_display_image(overlay.image_adu)
    width_um = float(overlay.roi_shape_hw[1]) * float(pixel_size_x_nm) * 1e-3
    height_um = float(overlay.roi_shape_hw[0]) * float(pixel_size_y_nm) * 1e-3
    ax.imshow(display, cmap="gray", extent=(0.0, width_um, height_um, 0.0), aspect="equal")
    half_x_um = 0.5 * float(overlay.patch_size_px) * float(pixel_size_x_nm) * 1e-3
    half_y_um = 0.5 * float(overlay.patch_size_px) * float(pixel_size_y_nm) * 1e-3
    for center_x_px, center_y_px in overlay.roixy_px.tolist():
        x0_um = float(center_x_px) * float(pixel_size_x_nm) * 1e-3 - half_x_um
        y0_um = float(center_y_px) * float(pixel_size_y_nm) * 1e-3 - half_y_um
        ax.add_patch(
            Rectangle(
                (x0_um, y0_um),
                2.0 * half_x_um,
                2.0 * half_y_um,
                fill=False,
                edgecolor=box_color,
                linewidth=1.1,
                alpha=0.9,
            )
        )
    ax.set_title(
        f"Representative raw TIFF frame with selected peak boxes\nframe {overlay.frame_index}, n={overlay.roixy_px.shape[0]}",
        fontsize=10,
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_xlim(0.0, width_um)
    ax.set_ylim(height_um, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.10, linewidth=0.4, color="white")


def _write_nat_process_summary(
    run: NATProcessRun,
    *,
    figure_path: Path,
    summary_path: Path,
    panel_paths: dict[str, Path],
    selected_patch_indices: torch.Tensor,
    selected_patch_brightness: torch.Tensor,
    gamma_display: GammaDisplaySummary,
    overlay_frame: PeakOverlayFrame | None,
) -> None:
    payload = {
        "run_dir": str(run.run_dir),
        "side": run.side,
        "figure_path": str(figure_path),
        "panel_paths": {label: str(path) for label, path in panel_paths.items()},
        "image_shape_hw": [int(run.image_shape_hw[0]), int(run.image_shape_hw[1])],
        "pixel_size_nm": [float(run.pixel_size_x_nm), float(run.pixel_size_y_nm)],
        "field_extent_um": [float(v) for v in field_extent_um(run)],
        "selection_count": int(run.selection_count),
        "gamma_count": int(run.gamma.shape[0]),
        "mode_order": [[int(n), int(m)] for n, m in run.mode_order],
        "selected_patch_indices": [int(v) for v in selected_patch_indices.tolist()],
        "selected_patch_ncc": [float(v) for v in run.patch_ncc[selected_patch_indices].tolist()],
        "selected_patch_brightness": [float(v) for v in selected_patch_brightness.tolist()],
        "gamma_display": {
            "normalization": gamma_display.normalization,
            "compression_scale": float(gamma_display.compression_scale),
            "max_abs_value": float(gamma_display.max_abs_value),
            "raw_values": [float(v) for v in gamma_display.raw_values.tolist()],
            "normalized_values": [float(v) for v in gamma_display.normalized_values.tolist()],
        },
        "panel_d_definition": (
            "Per-gamma Legendre basis planes used as the NAT spatial basis; "
            "these are basis planes, not the final fitted Zernike coefficient maps."
        ),
        "panel_e_layout": {
            "includes_panel_f": True,
            "panel_f_position": "right",
            "coeff_grid_shape": [2, 4],
            "coeff_column_spacing": "tight",
            "coeff_colorbar_mode": "dedicated_axis_between_e_and_f",
            "coeff_colorbar_pad": None,
            "outer_width_ratios": [3.9, 0.18, 1.5],
            "panel_f_height": "matches_two_coeff_rows",
        },
        "scalar_summary_label": "Derived summary: non-astig RMS (nm)",
        "source_files": {
            "peak_summary": str(run.run_dir / "peak_nat_zmap_summary.json"),
            "diagnostics_payload": str(run.run_dir / "real_nat_diagnostics" / "real_nat_diagnostics_payload.pt"),
            "coeff_stack_npz": str(run.run_dir / "export_nat_zmap" / "alternating_full_roi_zernike_maps_nm.npz"),
            "scalar_summary_npz": str(run.run_dir / "export_nat_zmap" / "provisional_non_astig_rms_nm.npz"),
        },
    }
    if overlay_frame is not None:
        payload["panel_a_overlay"] = {
            "frame_index": int(overlay_frame.frame_index),
            "selected_peak_count": int(overlay_frame.roixy_px.shape[0]),
            "patch_size_px": int(overlay_frame.patch_size_px),
            "roi_shape_hw": [int(overlay_frame.roi_shape_hw[0]), int(overlay_frame.roi_shape_hw[1])],
            "roi_origin_xy_px": [float(overlay_frame.roi_origin_xy_px[0]), float(overlay_frame.roi_origin_xy_px[1])],
        }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _normalize_patch_pair_for_display(
    raw_patch: np.ndarray,
    recon_patch: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    gamma: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([np.asarray(raw_patch, dtype=np.float32), np.asarray(recon_patch, dtype=np.float32)], axis=0)
    vmin = float(np.percentile(stacked, lower_percentile))
    vmax = float(np.percentile(stacked, upper_percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(stacked.min())
        vmax = float(stacked.max())
    scale = max(vmax - vmin, 1e-6)
    raw_display = np.clip((stacked[0] - vmin) / scale, 0.0, 1.0)
    recon_display = np.clip((stacked[1] - vmin) / scale, 0.0, 1.0)
    if gamma != 1.0:
        raw_display = np.power(raw_display, gamma)
        recon_display = np.power(recon_display, gamma)
    return raw_display.astype(np.float32, copy=False), recon_display.astype(np.float32, copy=False)


def _order_patch_indices_for_display(indices: torch.Tensor, brightness: torch.Tensor) -> torch.Tensor:
    if indices.numel() <= 1:
        return indices
    order = torch.argsort(brightness[indices], descending=True, stable=True)
    return indices[order]


def _build_patch_examples(run: NATProcessRun, selected_patch_indices: torch.Tensor) -> list[PatchExample]:
    brightness_values = _patch_brightness(run.raw_patches)
    examples: list[PatchExample] = []
    for patch_index in selected_patch_indices.tolist():
        raw_display, recon_display = _normalize_patch_pair_for_display(
            run.raw_patches[patch_index].numpy(),
            run.recon_patches[patch_index].numpy(),
        )
        examples.append(
            PatchExample(
                patch_index=int(patch_index),
                ncc=float(run.patch_ncc[patch_index].item()),
                brightness=float(brightness_values[patch_index].item()),
                raw_display=raw_display,
                recon_display=recon_display,
            )
        )
    return examples


def _draw_panel_a(
    ax,
    run: NATProcessRun,
    *,
    label: str | None = None,
    physical_units: bool = False,
) -> None:
    if label:
        _style_axis_label(ax, label)
    ax.set_title(f"Selected peak coverage (n={run.selection_count})")
    if physical_units:
        x_values = run.roixy_px[:, 0].numpy() * float(run.pixel_size_x_nm) * 1e-3
        y_values = run.roixy_px[:, 1].numpy() * float(run.pixel_size_y_nm) * 1e-3
        _, width_um, _, height_um = field_extent_um(run)
        ax.scatter(x_values, y_values, s=12, c="#d1495b", alpha=0.75, edgecolors="none")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_xlim(0.0, width_um)
        ax.set_ylim(height_um, 0.0)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.scatter(run.roixy_px[:, 0].numpy(), run.roixy_px[:, 1].numpy(), s=12, c="#d1495b", alpha=0.75, edgecolors="none")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_xlim(0, run.image_shape_hw[1])
        ax.set_ylim(run.image_shape_hw[0], 0)
    ax.grid(alpha=0.15, linewidth=0.5)


def _draw_panel_c(ax, gamma_display: GammaDisplaySummary, *, label: str | None = None) -> None:
    if label:
        _style_axis_label(ax, label)
    gamma_x = np.arange(1, gamma_display.raw_values.shape[0] + 1, dtype=np.int64)
    ax.fill_between(gamma_x, 0.0, gamma_display.normalized_values, color="#a8c7e8", alpha=0.4)
    ax.plot(
        gamma_x,
        gamma_display.normalized_values,
        color="#1f77b4",
        linewidth=2.0,
        marker="o",
        markersize=3.8,
    )
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_title("Shared gamma parameters (normalized for display)")
    ax.set_xlabel("gamma index")
    ax.set_ylabel("signed-log normalized value")
    ax.set_xlim(1, gamma_display.raw_values.shape[0])
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.text(
        0.99,
        0.03,
        f"raw max |gamma|={gamma_display.max_abs_value:.2f}\nscale={gamma_display.compression_scale:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.0},
    )


def _draw_panel_b(
    fig: plt.Figure,
    patch_grid,
    patch_examples: list[PatchExample],
    *,
    layout: str = "horizontal",
    label: str | None = None,
) -> tuple[list[Any], Any | None]:
    if not patch_examples:
        ax = fig.add_subplot(patch_grid[0, 0])
        if label:
            _style_axis_label(ax, label)
        ax.axis("off")
        ax.text(0.5, 0.5, "No patch examples available", transform=ax.transAxes, ha="center", va="center")
        return [ax], None

    patch_axes = []
    last_im = None
    if layout == "vertical":
        for row, example in enumerate(patch_examples):
            ax_raw = fig.add_subplot(patch_grid[row, 0])
            ax_recon = fig.add_subplot(patch_grid[row, 1])
            if row == 0 and label:
                _style_axis_label(ax_raw, label)
                ax_raw.set_title("raw", fontsize=9, pad=6)
                ax_recon.set_title("recon", fontsize=9, pad=6)
            last_im = ax_raw.imshow(example.raw_display, cmap="magma", vmin=0.0, vmax=1.0, origin="lower")
            ax_recon.imshow(example.recon_display, cmap="magma", vmin=0.0, vmax=1.0, origin="lower")
            ax_raw.set_xticks([])
            ax_raw.set_yticks([])
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])
            ax_raw.text(
                -0.10,
                0.5,
                f"#{example.patch_index}\nNCC={example.ncc:.3f}",
                transform=ax_raw.transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )
            patch_axes.extend([ax_raw, ax_recon])
        return patch_axes, last_im

    for col, example in enumerate(patch_examples):
        ax_raw = fig.add_subplot(patch_grid[0, col])
        ax_recon = fig.add_subplot(patch_grid[1, col])
        if col == 0 and label:
            _style_axis_label(ax_raw, label)
        last_im = ax_raw.imshow(example.raw_display, cmap="magma", vmin=0.0, vmax=1.0, origin="lower")
        ax_recon.imshow(example.recon_display, cmap="magma", vmin=0.0, vmax=1.0, origin="lower")
        ax_raw.set_title(f"#{example.patch_index} NCC={example.ncc:.3f}", fontsize=8)
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        ax_recon.set_xticks([])
        ax_recon.set_yticks([])
        if col == 0:
            ax_raw.set_ylabel("raw")
            ax_recon.set_ylabel("recon")
        patch_axes.extend([ax_raw, ax_recon])
    return patch_axes, last_im


def _draw_panel_d(
    fig: plt.Figure,
    basis_grid,
    basis_panels: list[LegendrePanel],
    *,
    label: str | None = None,
    title_fontsize: float = 6.0,
    subtitle_width: int = 30,
) -> tuple[list[Any], Any | None]:
    basis_axes = []
    last_im = None
    total_axes = basis_grid.nrows * basis_grid.ncols
    for idx, panel in enumerate(basis_panels):
        ax = fig.add_subplot(basis_grid[idx // basis_grid.ncols, idx % basis_grid.ncols])
        if idx == 0 and label:
            _style_axis_label(ax, label)
        vmax = _symmetric_limit(panel.values)
        last_im = ax.imshow(panel.values, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
        target_label = textwrap.shorten(panel.target_label, width=subtitle_width, placeholder="...")
        ax.set_title(
            f"g{panel.gamma_index + 1}: {panel.basis_label}\n-> {target_label}",
            fontsize=title_fontsize,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        basis_axes.append(ax)
    for idx in range(len(basis_panels), total_axes):
        ax = fig.add_subplot(basis_grid[idx // basis_grid.ncols, idx % basis_grid.ncols])
        ax.axis("off")
    return basis_axes, last_im


def _draw_panel_e(
    fig: plt.Figure,
    coeff_grid,
    coeff_stack: np.ndarray,
    mode_order: list[tuple[int, int]],
    coeff_norm: colors.Normalize,
    *,
    label: str | None = None,
) -> tuple[list[Any], Any | None]:
    coeff_axes = []
    last_im = None
    for idx, mode in enumerate(mode_order):
        ax = fig.add_subplot(coeff_grid[idx // coeff_grid.ncols, idx % coeff_grid.ncols])
        if idx == 0 and label:
            _style_axis_label(ax, label)
        last_im = ax.imshow(coeff_stack[idx], cmap="coolwarm", norm=coeff_norm, origin="lower")
        ax.set_title(f"Z{mode}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        coeff_axes.append(ax)
    return coeff_axes, last_im


def _draw_panel_f(
    ax,
    run: NATProcessRun,
    scalar_map: np.ndarray,
    *,
    label: str | None = None,
    physical_units: bool = False,
) -> Any:
    if label:
        _style_axis_label(ax, label)
    scalar_vmax = _nonnegative_limit(scalar_map)
    if physical_units:
        scalar_im = ax.imshow(
            scalar_map,
            cmap="viridis",
            vmin=0.0,
            vmax=scalar_vmax,
            origin="lower",
            extent=field_extent_um(run),
            aspect="equal",
        )
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
    else:
        scalar_im = ax.imshow(scalar_map, cmap="viridis", vmin=0.0, vmax=scalar_vmax, origin="lower", aspect="auto")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    ax.set_title("Derived summary: non-astig RMS (nm)")
    return scalar_im


def _panel_output_path(panel_dir: Path, label: str, stem: str, side: str | None) -> Path:
    suffix = f"_{side}" if side else ""
    return panel_dir / f"panel_{label.lower()}_{stem}{suffix}.png"


def _write_panel_pngs(
    run: NATProcessRun,
    *,
    output_dir: Path,
    dpi: int,
    patch_examples: list[PatchExample],
    gamma_display: GammaDisplaySummary,
    basis_panels: list[LegendrePanel],
    coeff_norm: colors.Normalize,
    overlay_frame: PeakOverlayFrame | None,
) -> dict[str, Path]:
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_paths: dict[str, Path] = {}

    panel_paths["A"] = _panel_output_path(panel_dir, "A", "selected_peak_coverage", run.side)
    if overlay_frame is None:
        fig = plt.figure(figsize=_standalone_field_figsize(run, width_in=6.2), dpi=int(dpi), constrained_layout=True)
        ax = fig.add_subplot(111)
        _draw_panel_a(ax, run, physical_units=True)
    else:
        panel_width, panel_height = _standalone_field_figsize(run, width_in=6.1)
        fig = plt.figure(figsize=(panel_width * 2.05, panel_height), dpi=int(dpi), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.08)
        ax_left = fig.add_subplot(grid[0, 0])
        _draw_panel_a(ax_left, run, physical_units=True)
        ax_right = fig.add_subplot(grid[0, 1])
        draw_selected_peak_overlay(
            ax_right,
            overlay_frame,
            pixel_size_x_nm=run.pixel_size_x_nm,
            pixel_size_y_nm=run.pixel_size_y_nm,
        )
    fig.savefig(panel_paths["A"], facecolor="white")
    plt.close(fig)

    panel_paths["B"] = _panel_output_path(panel_dir, "B", "raw_recon_examples", run.side)
    fig = plt.figure(figsize=(max(14.0, 2.2 * max(len(patch_examples), 1)), 5.6), dpi=int(dpi), constrained_layout=True)
    patch_grid = fig.add_gridspec(2, max(len(patch_examples), 1), hspace=0.06, wspace=0.08)
    patch_axes, patch_im = _draw_panel_b(fig, patch_grid, patch_examples)
    if patch_im is not None:
        fig.colorbar(patch_im, ax=patch_axes, fraction=0.02, pad=0.01, label="normalized intensity")
    fig.suptitle("Brightness-prioritized raw/recon patches (paired contrast normalized)", fontsize=14)
    fig.savefig(panel_paths["B"], facecolor="white")
    plt.close(fig)

    panel_paths["B_vertical"] = _panel_output_path(panel_dir, "B", "raw_recon_examples_vertical", run.side)
    fig = plt.figure(
        figsize=(5.8, max(12.0, 1.55 * max(len(patch_examples), 1))),
        dpi=int(dpi),
        constrained_layout=True,
    )
    patch_grid = fig.add_gridspec(max(len(patch_examples), 1), 2, hspace=0.08, wspace=0.08)
    patch_axes, patch_im = _draw_panel_b(fig, patch_grid, patch_examples, layout="vertical")
    if patch_im is not None:
        fig.colorbar(patch_im, ax=patch_axes, fraction=0.03, pad=0.01, label="normalized intensity")
    fig.suptitle("Brightness-prioritized raw/recon patches (portrait layout)", fontsize=14)
    fig.savefig(panel_paths["B_vertical"], facecolor="white")
    plt.close(fig)

    panel_paths["C"] = _panel_output_path(panel_dir, "C", "shared_gamma", run.side)
    fig = plt.figure(figsize=(10.0, 4.8), dpi=int(dpi), constrained_layout=True)
    ax = fig.add_subplot(111)
    _draw_panel_c(ax, gamma_display)
    fig.savefig(panel_paths["C"], facecolor="white")
    plt.close(fig)

    panel_paths["D"] = _panel_output_path(panel_dir, "D", "legendre_basis", run.side)
    d_ncols = 7 if len(basis_panels) > 15 else 5
    d_nrows = max(int(np.ceil(len(basis_panels) / d_ncols)), 1)
    fig = plt.figure(figsize=(d_ncols * 2.9, d_nrows * 3.0), dpi=int(dpi), constrained_layout=True)
    basis_grid = fig.add_gridspec(d_nrows, d_ncols, hspace=0.32, wspace=0.16)
    basis_axes, basis_im = _draw_panel_d(fig, basis_grid, basis_panels, title_fontsize=7.2, subtitle_width=44)
    if basis_im is not None:
        fig.colorbar(basis_im, ax=basis_axes, fraction=0.02, pad=0.01, label="basis value")
    fig.suptitle("Per-gamma Legendre basis planes (basis only, not final fitted mode maps)", fontsize=14)
    fig.savefig(panel_paths["D"], facecolor="white")
    plt.close(fig)

    panel_paths["E"] = _panel_output_path(panel_dir, "E", "zernike_maps", run.side)
    ef_width = 15.2
    ef_height = 7.2
    fig = plt.figure(figsize=(ef_width, ef_height), dpi=int(dpi), constrained_layout=True)
    outer = fig.add_gridspec(1, 3, width_ratios=(3.9, 0.18, 1.5), wspace=0.10)
    coeff_grid = outer[0, 0].subgridspec(2, 4, hspace=0.18, wspace=0.06)
    coeff_axes, coeff_im = _draw_panel_e(fig, coeff_grid, run.coeff_stack, run.mode_order, coeff_norm)
    if coeff_im is not None:
        coeff_cax = fig.add_subplot(outer[0, 1])
        fig.colorbar(coeff_im, cax=coeff_cax, label="nm")
    ax_scalar = fig.add_subplot(outer[0, 2])
    scalar_im = _draw_panel_f(ax_scalar, run, run.scalar_map, physical_units=True)
    fig.colorbar(scalar_im, ax=ax_scalar, fraction=0.05, pad=0.02, label="nm")
    fig.suptitle("Exported Zernike coefficient maps with derived scalar summary", fontsize=14)
    fig.savefig(panel_paths["E"], facecolor="white")
    plt.close(fig)

    panel_paths["F"] = _panel_output_path(panel_dir, "F", "non_astig_rms", run.side)
    fig = plt.figure(figsize=_standalone_field_figsize(run, width_in=6.6), dpi=int(dpi), constrained_layout=True)
    ax = fig.add_subplot(111)
    scalar_im = _draw_panel_f(ax, run, run.scalar_map, physical_units=True)
    fig.colorbar(scalar_im, ax=ax, fraction=0.04, pad=0.02, label="nm")
    fig.savefig(panel_paths["F"], facecolor="white")
    plt.close(fig)

    return panel_paths


def render_nat_process_figure(
    run: NATProcessRun,
    *,
    output_dir: str | Path,
    patch_count: int = 8,
    dpi: int = 180,
    figure_name: str = "nat_zmap_process_figure.png",
    summary_name: str = "nat_zmap_process_summary.json",
) -> NATProcessFigureResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_brightness = _patch_brightness(run.raw_patches)
    selected_patch_indices = select_patch_indices_by_ncc(
        run.patch_ncc,
        brightness=patch_brightness,
        count=int(patch_count),
    )
    selected_patch_indices = _order_patch_indices_for_display(selected_patch_indices, patch_brightness)
    patch_examples = _build_patch_examples(run, selected_patch_indices)
    gamma_display = summarize_gamma_for_display(run.gamma)
    overlay_frame = build_peak_overlay_frame(run)
    figure_path = output_dir / str(figure_name)
    summary_path = output_dir / str(summary_name)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )
    cfg_width = max(int(run.image_shape_hw[1]), 4)
    cfg_height = max(int(run.image_shape_hw[0]), 4)
    from .nat_field import default_order1_config

    basis_panels = build_legendre_basis_planes(
        default_order1_config(
            img_size_x=cfg_width,
            img_size_y=cfg_height,
            pixel_size_x_nm=run.pixel_size_x_nm,
            pixel_size_y_nm=run.pixel_size_y_nm,
        ),
        width=72,
        height=72,
    )
    coeff_vmax = _symmetric_limit(run.coeff_stack)
    coeff_norm = colors.SymLogNorm(linthresh=max(coeff_vmax * 1e-3, 1e-3), vmin=-coeff_vmax, vmax=coeff_vmax)

    panel_paths = _write_panel_pngs(
        run,
        output_dir=output_dir,
        dpi=int(dpi),
        patch_examples=patch_examples,
        gamma_display=gamma_display,
        basis_panels=basis_panels,
        coeff_norm=coeff_norm,
        overlay_frame=overlay_frame,
    )

    fig = plt.figure(figsize=(26, 20), dpi=int(dpi), constrained_layout=True)
    outer = fig.add_gridspec(10, 12)

    ax_a = fig.add_subplot(outer[0:2, 0:6])
    _draw_panel_a(ax_a, run, label="A")

    ax_c = fig.add_subplot(outer[0:2, 6:12])
    _draw_panel_c(ax_c, gamma_display, label="C")

    patch_grid = outer[2:4, 0:12].subgridspec(2, max(len(patch_examples), 1), hspace=0.06, wspace=0.05)
    patch_axes, patch_im = _draw_panel_b(fig, patch_grid, patch_examples, label="B")
    if patch_im is not None:
        fig.colorbar(
            patch_im,
            ax=patch_axes,
            fraction=0.012,
            pad=0.01,
            label="normalized intensity",
        )

    basis_grid = outer[4:9, 0:6].subgridspec(4, 5, hspace=0.32, wspace=0.18)
    basis_axes, basis_im = _draw_panel_d(fig, basis_grid, basis_panels, label="D", title_fontsize=6.4, subtitle_width=32)
    if basis_im is not None:
        fig.colorbar(basis_im, ax=basis_axes, fraction=0.015, pad=0.01, label="basis value")

    coeff_grid = outer[4:9, 6:12].subgridspec(2, 4, hspace=0.22, wspace=0.14)
    coeff_axes, coeff_im = _draw_panel_e(fig, coeff_grid, run.coeff_stack, run.mode_order, coeff_norm, label="E")
    if coeff_im is not None:
        fig.colorbar(coeff_im, ax=coeff_axes, fraction=0.015, pad=0.01, label="nm")

    ax_f = fig.add_subplot(outer[9:10, 0:12])
    scalar_im = _draw_panel_f(ax_f, run, run.scalar_map, label="F")
    fig.colorbar(scalar_im, ax=ax_f, fraction=0.02, pad=0.015, label="nm")

    fig.suptitle(
        f"{(run.side or 'left').capitalize()} NAT zmap process: peaks -> shared gamma -> basis planes -> Zernike maps -> summary",
        fontsize=16,
        y=1.01,
    )
    fig.savefig(figure_path, facecolor="white")
    plt.close(fig)
    _write_nat_process_summary(
        run,
        figure_path=figure_path,
        summary_path=summary_path,
        panel_paths=panel_paths,
        selected_patch_indices=selected_patch_indices,
        selected_patch_brightness=patch_brightness[selected_patch_indices],
        gamma_display=gamma_display,
        overlay_frame=overlay_frame,
    )
    return NATProcessFigureResult(
        figure_path=figure_path,
        summary_path=summary_path,
        panel_paths=panel_paths,
        selected_patch_indices=selected_patch_indices,
        gamma_display=gamma_display,
    )


def load_nat_process_run(run_dir: str | Path) -> NATProcessRun:
    run_dir = Path(run_dir)
    peak_summary = _load_json(run_dir / "peak_nat_zmap_summary.json")
    diagnostics_summary_path = run_dir / "real_nat_diagnostics" / "real_nat_diagnostics_summary.json"
    diagnostics_summary = _load_json(diagnostics_summary_path) if diagnostics_summary_path.exists() else {}
    diagnostics_payload = torch.load(
        run_dir / "real_nat_diagnostics" / "real_nat_diagnostics_payload.pt",
        map_location="cpu",
        weights_only=False,
    )
    alternating = diagnostics_payload["alternating"]
    raw_patches = diagnostics_payload["raw_patches"].detach().cpu().to(dtype=torch.float32)
    recon_patches = diagnostics_payload["alternating_recon"].detach().cpu().to(dtype=torch.float32)
    coeff_payload = np.load(run_dir / "export_nat_zmap" / "alternating_full_roi_zernike_maps_nm.npz", allow_pickle=False)
    scalar_payload = np.load(run_dir / "export_nat_zmap" / "provisional_non_astig_rms_nm.npz", allow_pickle=False)
    image_shape_hw = _image_shape_from_payload(peak_summary, scalar_payload)
    pixel_size_x_nm, pixel_size_y_nm = _pixel_size_from_summary(peak_summary)
    comparison_metrics = diagnostics_payload.get("comparison_metrics") or {}
    selection_count = int(
        comparison_metrics.get("selected_emitters")
        or ((peak_summary.get("ncc_summary") or {}).get("count"))
        or raw_patches.shape[0]
    )
    return NATProcessRun(
        run_dir=run_dir,
        side=((peak_summary.get("config") or {}).get("side")) or None,
        diagnostics_config=dict((diagnostics_summary.get("config") or {})),
        pixel_size_x_nm=float(pixel_size_x_nm),
        pixel_size_y_nm=float(pixel_size_y_nm),
        gamma=alternating.gamma.detach().cpu().to(dtype=torch.float32),
        roixy_px=alternating.roixy_px.detach().cpu().to(dtype=torch.float32),
        raw_patches=raw_patches,
        recon_patches=recon_patches,
        patch_ncc=_ncc(raw_patches, recon_patches).detach().cpu().to(dtype=torch.float32),
        coeff_stack=np.asarray(coeff_payload["zernike_maps_nm"], dtype=np.float32),
        mode_order=[tuple(int(v) for v in row) for row in np.asarray(coeff_payload["mode_order"], dtype=np.int64)],
        scalar_map=np.asarray(scalar_payload["values_nm"], dtype=np.float32),
        selection_count=selection_count,
        image_shape_hw=image_shape_hw,
    )

__all__ = [
    "GammaDisplaySummary",
    "LegendrePanel",
    "NATProcessFigureResult",
    "NATProcessRun",
    "PatchExample",
    "PeakOverlayFrame",
    "build_legendre_basis_planes",
    "build_peak_overlay_frame",
    "describe_gamma_targets",
    "draw_selected_peak_overlay",
    "field_extent_um",
    "load_nat_process_run",
    "render_nat_process_figure",
    "select_overlay_frame_index",
    "summarize_gamma_for_display",
    "select_patch_indices_by_ncc",
]
