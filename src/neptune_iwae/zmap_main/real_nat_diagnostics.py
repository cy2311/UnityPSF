from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .harvest import HarvestedEmitters
from .nat_fit import (
    NATSpotFitResult,
    build_nat_spot_fitter_config,
    build_patch_dataset_from_emitters,
    fit_nat_alternating_from_harvested_emitters,
    fit_nat_from_harvested_emitters,
    initial_astig_standard_coefficients_nm,
    materialize_selected_harvested_emitters,
)
from .nat_torch import evaluate_zernike_from_roi_positions_torch


@dataclass(frozen=True)
class RealNATDiagnosticsConfig:
    output_dir: Path
    harvest_pt: Path
    pipeline_config_path: Path
    tiff_path: Path
    nat_config_kind: str = "order1"
    device: str = "cpu"
    patch_size_px: int = 15
    max_emitters: int = 32
    approximate_fit_steps: int = 60
    alternating_rounds: int = 10
    alternating_local_steps: int = 50
    alternating_global_steps: int = 50
    alternating_local_warmup_rounds: int = 0
    alternating_local_warmup_steps: int = 0
    approximate_fit_lr: float = 5e-2
    alternating_local_lr: float = 1e-2
    alternating_local_z_lr_multiplier: float = 1.0
    alternating_gamma_lr: float = 2e-3
    alternating_optimizer_kind: str = "adam"
    coeff_l2: float = 1e-4
    ridge: float = 1e-4
    gamma_l2: float = 0.0
    coeff_map_grid: int = 64
    patch_examples: int = 8
    min_neighbor_distance_px: float = 15.0
    global_projected_min_distance_px: float = 0.0
    selection_pool_multiplier: int = 50
    selection_mode: str = "probability_topk"
    spatial_balance_grid_px: int = 128
    spatial_balance_min_per_cell: int = 0
    spatial_balance_max_per_cell: int = 0
    max_patch_peak_distance_px: float = 2.5
    secondary_peak_exclusion_radius_px: float = 3.5
    max_secondary_peak_fraction: float = 0.45
    rank_spots_by_quality: bool = False
    min_center_peak_norm: float = 0.0
    min_signal_sum_norm: float = 0.0
    input_offset: float | None = None
    input_scale: float | None = None
    patch_adu_mode: str = "baseline_clamp"
    baseline_adu: float | None = None
    alternating_min_photon_fraction: float = 0.0
    alternating_min_background_fraction: float = 0.0
    roi_x_min_px: float = 0.0
    roi_x_max_px: float | None = 600.0
    roi_y_min_px: float = 0.0
    roi_y_max_px: float | None = 1200.0
    roi_edge_margin_px: float = 3.0
    freeze_initial_astig_standard: bool = True
    freeze_defocus_zero_gauge: bool = False
    vectorfit_astig_gauge: bool = False
    vectorfit_astig_anchor_nm: float | None = None
    vectorfit_astig_anchor_mode: str = "init_only"
    vectorfit_phasor_z_init: bool = False
    include_approx_outputs: bool = False


@dataclass(frozen=True)
class RealNATDiagnosticsResult:
    config: RealNATDiagnosticsConfig
    approximate_metrics: dict[str, Any]
    alternating_metrics: dict[str, Any]
    comparison_metrics: dict[str, Any]
    figures: dict[str, str]
    output_dir: Path

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "config": {
                **asdict(self.config),
                "output_dir": str(self.config.output_dir),
                "harvest_pt": str(self.config.harvest_pt),
                "pipeline_config_path": str(self.config.pipeline_config_path),
                "tiff_path": str(self.config.tiff_path),
            },
            "approximate_metrics": self.approximate_metrics,
            "alternating_metrics": self.alternating_metrics,
            "comparison_metrics": self.comparison_metrics,
            "figures": self.figures,
            "output_dir": str(self.output_dir),
        }


def load_harvested_emitters_from_pt(path: str | Path) -> HarvestedEmitters:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    emitters = payload.get("payload", payload)
    if isinstance(emitters, HarvestedEmitters):
        return emitters
    if not isinstance(emitters, dict):
        raise TypeError(f"Unsupported harvest payload type: {type(emitters)!r}")
    return HarvestedEmitters(**{key: value for key, value in emitters.items()})


def _record_progress(output_dir: Path, stage: str, **details: Any) -> None:
    event = {
        "time_unix": time.time(),
        "stage": str(stage),
        **details,
    }
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    suffix = f" {detail_text}" if detail_text else ""
    print(f"[real_nat_diagnostics] stage={stage}{suffix}", flush=True)


def _render_patches(cfg, result: NATSpotFitResult) -> torch.Tensor:
    from .nat_fit import _build_fourier_psf

    psf = _build_fourier_psf(cfg)
    device = torch.device(str(cfg.device))
    signal = psf.simulate(
        result.local_x_nm.to(device=device, dtype=torch.float32),
        result.local_y_nm.to(device=device, dtype=torch.float32),
        result.z_nm.to(device=device, dtype=torch.float32),
        result.photons.to(device=device, dtype=torch.float32),
        zernike_coefs=result.coefficients_nm.to(device=device, dtype=torch.float32),
    )
    return (float(cfg.emit_scale_adu) * signal + result.background_adu.to(device=device, dtype=torch.float32)[:, None, None]).detach().cpu()


def _ncc(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_centered = pred - pred.mean(dim=(-2, -1), keepdim=True)
    denom = torch.sqrt(
        target_centered.square().sum(dim=(-2, -1)).clamp_min(1e-8)
        * pred_centered.square().sum(dim=(-2, -1)).clamp_min(1e-8)
    )
    return (target_centered * pred_centered).sum(dim=(-2, -1)) / denom


def _patch_metrics(raw: torch.Tensor, recon: torch.Tensor) -> dict[str, float]:
    if raw.shape[0] == 0 or recon.shape[0] == 0:
        return {
            "patch_mse_mean": 0.0,
            "patch_mse_median": 0.0,
            "patch_ncc_mean": 0.0,
            "patch_ncc_median": 0.0,
            "patch_mse_mean_nonzero_raw": 0.0,
            "patch_mse_median_nonzero_raw": 0.0,
            "patch_ncc_mean_nonzero_raw": 0.0,
            "patch_ncc_median_nonzero_raw": 0.0,
            "nonzero_raw_patch_count": 0,
            "zero_raw_patch_count": 0,
        }
    mse = (recon - raw).square().mean(dim=(-2, -1))
    ncc = _ncc(raw, recon)
    nonzero_raw = raw.abs().sum(dim=(-2, -1)) > 1e-6
    zero_raw_count = int((~nonzero_raw).sum().item())
    if bool(nonzero_raw.any()):
        mse_nonzero = mse[nonzero_raw]
        ncc_nonzero = ncc[nonzero_raw]
        patch_mse_mean_nonzero_raw = float(mse_nonzero.mean().item())
        patch_mse_median_nonzero_raw = float(mse_nonzero.median().item())
        patch_ncc_mean_nonzero_raw = float(ncc_nonzero.mean().item())
        patch_ncc_median_nonzero_raw = float(ncc_nonzero.median().item())
    else:
        patch_mse_mean_nonzero_raw = 0.0
        patch_mse_median_nonzero_raw = 0.0
        patch_ncc_mean_nonzero_raw = 0.0
        patch_ncc_median_nonzero_raw = 0.0
    return {
        "patch_mse_mean": float(mse.mean().item()),
        "patch_mse_median": float(mse.median().item()),
        "patch_ncc_mean": float(ncc.mean().item()),
        "patch_ncc_median": float(ncc.median().item()),
        "patch_mse_mean_nonzero_raw": patch_mse_mean_nonzero_raw,
        "patch_mse_median_nonzero_raw": patch_mse_median_nonzero_raw,
        "patch_ncc_mean_nonzero_raw": patch_ncc_mean_nonzero_raw,
        "patch_ncc_median_nonzero_raw": patch_ncc_median_nonzero_raw,
        "nonzero_raw_patch_count": int(nonzero_raw.sum().item()),
        "zero_raw_patch_count": zero_raw_count,
    }


def _safe_abs_mean(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.abs().mean().item())


def _safe_abs_max(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.abs().max().item())


def _preferred_patch_ncc(metrics: dict[str, Any]) -> float:
    value = metrics.get("patch_ncc_mean_nonzero_raw")
    if value is not None:
        return float(value)
    return float(metrics.get("patch_ncc_mean", 0.0))


def _preferred_recon_stage(approximate_metrics: dict[str, Any], alternating_metrics: dict[str, Any]) -> str:
    return (
        "approximate"
        if _preferred_patch_ncc(approximate_metrics) > _preferred_patch_ncc(alternating_metrics)
        else "alternating"
    )


def _save_coeff_maps(
    path: Path,
    *,
    cfg,
    approximate: NATSpotFitResult,
    alternating: NATSpotFitResult,
    grid_size: int,
    include_approx: bool,
) -> None:
    xs = torch.linspace(0.0, float(cfg.nat_config.img_size_x - 1), int(grid_size))
    ys = torch.linspace(0.0, float(cfg.nat_config.img_size_y - 1), int(grid_size))
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    roixy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    approx_coeffs, _, _ = evaluate_zernike_from_roi_positions_torch(roixy, approximate.gamma, cfg.nat_config)
    alt_coeffs, _, _ = evaluate_zernike_from_roi_positions_torch(roixy, alternating.gamma, cfg.nat_config)
    approx_maps = approx_coeffs.reshape(grid_size, grid_size, -1).permute(2, 0, 1).detach().cpu().numpy()
    alt_maps = alt_coeffs.reshape(grid_size, grid_size, -1).permute(2, 0, 1).detach().cpu().numpy()
    modes = list(cfg.nat_config.aberrations)
    columns = 3 if include_approx else 1
    fig, axes = plt.subplots(len(modes), columns, figsize=(3.6 * columns, 2.1 * len(modes)), dpi=160, constrained_layout=True)
    if len(modes) == 1:
        axes = np.asarray(axes)[None, ...]
    if columns == 1:
        axes = np.asarray(axes)[:, None]
    for row, mode in enumerate(modes):
        if include_approx:
            diff_map = alt_maps[row] - approx_maps[row]
            panels = [approx_maps[row], alt_maps[row], diff_map]
            titles = [f"approx Z{mode}", f"alternating Z{mode}", "alt - approx"]
        else:
            panels = [alt_maps[row]]
            titles = [f"alternating Z{mode}"]
        vmax = max(float(np.max(np.abs(approx_maps[row]))), float(np.max(np.abs(alt_maps[row]))), 1e-6)
        for col, panel in enumerate(panels):
            limit = max(float(np.max(np.abs(panel))), 1e-6) if not include_approx else (max(float(np.max(np.abs(panel))), 1e-6) if col == 2 else vmax)
            im = axes[row, col].imshow(panel, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[row, col].set_title(titles[col], fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_patch_grid(
    path: Path,
    *,
    raw: torch.Tensor,
    recon: torch.Tensor,
    max_examples: int,
    recon_label: str,
) -> None:
    count = min(int(max_examples), int(raw.shape[0]))
    if count <= 0:
        return
    rows = 3
    fig, axes = plt.subplots(rows, count, figsize=(1.8 * count, 1.6 * rows), dpi=160, constrained_layout=True)
    labels = ["raw", str(recon_label), f"|raw-{recon_label}|"]
    for col in range(count):
        panels = [
            raw[col].numpy(),
            recon[col].numpy(),
            (raw[col] - recon[col]).abs().numpy(),
        ]
        for row, panel in enumerate(panels):
            ax = axes[row, col] if count > 1 else axes[row]
            is_error = row == 2
            if not is_error:
                vmin = float(np.percentile(raw[col].numpy(), 1))
                vmax = float(np.percentile(raw[col].numpy(), 99))
                cmap = "magma"
            else:
                vmin = 0.0
                vmax = max(float(np.percentile(panel, 99)), 1e-6)
                cmap = "viridis"
            ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=8)
            if row == 0:
                ax.set_title(f"spot {col}", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_loss_curve(path: Path, *, approximate: NATSpotFitResult, alternating: NATSpotFitResult, include_approx: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160, constrained_layout=True)
    if include_approx and approximate.loss_history:
        ax.plot(approximate.loss_history, label="approx coeff fit")
    if alternating.loss_history:
        ax.plot(alternating.loss_history, label="alternating local/global")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("loss")
    ax.set_title("NAT fit loss history")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_selected_emitter_distribution_map(
    path: Path,
    *,
    emitters: HarvestedEmitters,
    width: int,
    height: int,
    roi_x_min_px: float,
    roi_x_max_px: float | None,
    roi_y_min_px: float,
    roi_y_max_px: float | None,
    selection_mode: str,
    spatial_balance_grid_px: int,
) -> None:
    x_min = float(roi_x_min_px)
    x_max = float(width) if roi_x_max_px is None else float(roi_x_max_px)
    y_min = float(roi_y_min_px)
    y_max = float(height) if roi_y_max_px is None else float(roi_y_max_px)
    if not np.isfinite(x_max) or x_max <= x_min:
        x_min, x_max = 0.0, float(width)
    if not np.isfinite(y_max) or y_max <= y_min:
        y_min, y_max = 0.0, float(height)
    selection_mode = str(selection_mode).lower()

    xs = emitters.x_px.detach().cpu().numpy().astype(np.float32, copy=False)
    ys = emitters.y_px.detach().cpu().numpy().astype(np.float32, copy=False)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=180, constrained_layout=True)
    ax.set_facecolor("#081420")
    if xs.size > 0:
        if selection_mode == "spatial_balanced" and int(spatial_balance_grid_px) > 0:
            grid_px = float(spatial_balance_grid_px)
            bins_x = max(1, int(np.ceil((x_max - x_min) / grid_px)))
            bins_y = max(1, int(np.ceil((y_max - y_min) / grid_px)))
        else:
            bins_x = max(8, min(64, int(np.ceil((x_max - x_min) / 32.0))))
            bins_y = max(8, min(64, int(np.ceil((y_max - y_min) / 32.0))))
        hist = ax.hist2d(
            xs,
            ys,
            bins=(bins_x, bins_y),
            range=((x_min, x_max), (y_min, y_max)),
            cmap="viridis",
            cmin=1.0,
        )
        fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.04, label="selected emitters")
        ax.scatter(xs, ys, s=8.0, c="white", alpha=0.25, linewidths=0.0)
    else:
        ax.text(0.5, 0.5, "No selected emitters", ha="center", va="center", transform=ax.transAxes, color="white")

    if selection_mode == "spatial_balanced" and int(spatial_balance_grid_px) > 0:
        grid_px = float(spatial_balance_grid_px)
        for x_pos in np.arange(x_min + grid_px, x_max, grid_px):
            ax.axvline(float(x_pos), color="white", alpha=0.14, linewidth=0.8)
        for y_pos in np.arange(y_min + grid_px, y_max, grid_px):
            ax.axhline(float(y_pos), color="white", alpha=0.14, linewidth=0.8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Selected emitter spatial distribution")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_report(path: Path, result: RealNATDiagnosticsResult) -> None:
    mse_reduction = 1.0 - (
        float(result.alternating_metrics["patch_mse_mean"])
        / max(float(result.approximate_metrics["patch_mse_mean"]), 1e-12)
    )
    lines = [
        "# Real TIFF alternating NAT diagnostics",
        "",
        "## Summary",
        "",
        f"- output: `{result.output_dir}`",
        f"- selected_emitters: `{result.comparison_metrics['selected_emitters']}`",
        f"- nat_config_kind: `{result.config.nat_config_kind}`",
        f"- freeze_initial_astig_standard: `{result.config.freeze_initial_astig_standard}`",
        f"- min_neighbor_distance_px: `{result.config.min_neighbor_distance_px}`",
        f"- selection_pool_multiplier: `{result.config.selection_pool_multiplier}`",
        f"- gamma_delta_norm: `{result.comparison_metrics['gamma_delta_norm']:.6f}`",
        f"- coeff_delta_mean_abs: `{result.comparison_metrics['coeff_delta_mean_abs']:.6f}`",
        f"- alternating patch MSE: `{result.alternating_metrics['patch_mse_mean']:.6f}`",
        f"- patch MSE reduction: `{100.0 * mse_reduction:.2f}%`",
        f"- alternating patch NCC: `{result.alternating_metrics['patch_ncc_mean']:.6f}`",
        f"- alternating patch NCC (nonzero raw): `{result.alternating_metrics['patch_ncc_mean_nonzero_raw']:.6f}`",
        f"- nonzero raw patches: `{result.alternating_metrics['nonzero_raw_patch_count']}`",
        f"- preferred recon stage: `{result.comparison_metrics.get('preferred_recon_stage', 'alternating')}`",
        f"- include_approx_outputs: `{result.config.include_approx_outputs}`",
        "- The default raw-vs-recon figure shows the preferred stage selected by patch NCC.",
        "",
        "## Figures",
        "",
    ]
    for name, figure_path in result.figures.items():
        lines.append(f"- {name}: `{figure_path}`")
    lines.extend(
        [
            "",
            "## Visual Preview",
            "",
            "### Coefficient maps",
            "",
            "![coefficient maps](figures/real_nat_coeff_maps_approx_vs_alternating.png)",
            "",
            "### Selected emitter distribution",
            "",
            "![selected emitter distribution](figures/selected_emitter_distribution_map.png)",
            "",
            "### Raw vs reconstruction",
            "",
            "![raw vs reconstruction](figures/real_nat_raw_vs_recon_approx_vs_alternating.png)",
            "",
            "### Loss curves",
            "",
            "![loss curves](figures/real_nat_loss_curves.png)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_real_nat_diagnostics(config: RealNATDiagnosticsConfig) -> RealNATDiagnosticsResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _record_progress(output_dir, "load_harvest", harvest_pt=str(config.harvest_pt))
    emitters = load_harvested_emitters_from_pt(config.harvest_pt)
    _record_progress(
        output_dir,
        "build_fitter_config",
        raw_emitters=int(len(emitters)),
        device=str(config.device),
        max_emitters=int(config.max_emitters),
    )
    fitter_cfg = build_nat_spot_fitter_config(
        pipeline_config_path=config.pipeline_config_path,
        tiff_path=config.tiff_path,
        nat_config_kind=config.nat_config_kind,
        device=config.device,
        patch_size_px=config.patch_size_px,
        max_emitters=config.max_emitters,
        fit_steps=config.approximate_fit_steps,
        fit_lr=config.approximate_fit_lr,
        coeff_l2=config.coeff_l2,
        ridge=config.ridge,
        min_neighbor_distance_px=config.min_neighbor_distance_px,
        global_projected_min_distance_px=config.global_projected_min_distance_px,
        selection_mode=config.selection_mode,
        spatial_balance_grid_px=config.spatial_balance_grid_px,
        spatial_balance_min_per_cell=config.spatial_balance_min_per_cell,
        spatial_balance_max_per_cell=config.spatial_balance_max_per_cell,
        max_patch_peak_distance_px=config.max_patch_peak_distance_px,
        secondary_peak_exclusion_radius_px=config.secondary_peak_exclusion_radius_px,
        max_secondary_peak_fraction=config.max_secondary_peak_fraction,
        rank_spots_by_quality=config.rank_spots_by_quality,
        min_center_peak_norm=config.min_center_peak_norm,
        min_signal_sum_norm=config.min_signal_sum_norm,
        input_offset=config.input_offset,
        input_scale=config.input_scale,
        patch_adu_mode=config.patch_adu_mode,
        baseline_adu=config.baseline_adu,
        alternating_min_photon_fraction=config.alternating_min_photon_fraction,
        alternating_min_background_fraction=config.alternating_min_background_fraction,
        roi_x_min_px=config.roi_x_min_px,
        roi_x_max_px=config.roi_x_max_px,
        roi_y_min_px=config.roi_y_min_px,
        roi_y_max_px=config.roi_y_max_px,
        roi_edge_margin_px=config.roi_edge_margin_px,
        freeze_initial_astig_standard=bool(config.freeze_initial_astig_standard),
        freeze_defocus_zero_gauge=bool(config.freeze_defocus_zero_gauge),
        vectorfit_astig_gauge=bool(config.vectorfit_astig_gauge),
        vectorfit_astig_anchor_nm=config.vectorfit_astig_anchor_nm,
        vectorfit_astig_anchor_mode=str(config.vectorfit_astig_anchor_mode),
        vectorfit_phasor_z_init=bool(config.vectorfit_phasor_z_init),
        alternating_optimizer_kind=str(config.alternating_optimizer_kind),
    )
    fitter_cfg = replace(
        fitter_cfg,
        selection_pool_multiplier=int(config.selection_pool_multiplier),
    )
    selected_harvest_dir = output_dir / "selected_harvest"
    selected_harvest_dir.mkdir(parents=True, exist_ok=True)
    _record_progress(output_dir, "select_emitters", raw_emitters=int(len(emitters)))
    selected_emitters, selected_harvest_summary = materialize_selected_harvested_emitters(emitters, fitter_cfg)
    selected_harvest_path = selected_harvest_dir / "selected_harvest.pt"
    torch.save({"payload": selected_emitters}, selected_harvest_path)
    (selected_harvest_dir / "selected_harvest_summary.json").write_text(
        json.dumps(selected_harvest_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    preselected_cfg = replace(
        fitter_cfg,
        emitters_preselected=True,
    )
    _record_progress(
        output_dir,
        "approximate_fit",
        selected_emitters=int(len(selected_emitters)),
        fit_steps=int(config.approximate_fit_steps),
    )
    approximate = fit_nat_from_harvested_emitters(selected_emitters, preselected_cfg)
    _record_progress(
        output_dir,
        "alternating_fit",
        selected_emitters=int(len(selected_emitters)),
        rounds=int(config.alternating_rounds),
        local_steps=int(config.alternating_local_steps),
        global_steps=int(config.alternating_global_steps),
        optimizer=str(config.alternating_optimizer_kind),
    )

    def alternating_progress(event: dict[str, int | float | str]) -> None:
        _record_progress(output_dir, "alternating_round", **event)

    alternating = fit_nat_alternating_from_harvested_emitters(
        selected_emitters,
        preselected_cfg,
        gamma_init=approximate.gamma.detach().clone(),
        approximate_init=approximate,
        rounds=config.alternating_rounds,
        local_steps=config.alternating_local_steps,
        global_steps=config.alternating_global_steps,
        local_warmup_rounds=config.alternating_local_warmup_rounds,
        local_warmup_steps=config.alternating_local_warmup_steps,
        local_lr=config.alternating_local_lr,
        local_z_lr_multiplier=config.alternating_local_z_lr_multiplier,
        gamma_lr=config.alternating_gamma_lr,
        gamma_l2=config.gamma_l2,
        progress_callback=alternating_progress,
    )

    _record_progress(output_dir, "render_patches", selected_emitters=int(len(selected_emitters)))
    dataset = build_patch_dataset_from_emitters(selected_emitters, preselected_cfg)
    raw = dataset.patches_adu.detach().cpu()
    approximate_recon = _render_patches(preselected_cfg, approximate)
    alternating_recon = _render_patches(preselected_cfg, alternating)
    approximate_metrics = {**approximate.metrics, **_patch_metrics(raw, approximate_recon)}
    alternating_metrics = {**alternating.metrics, **_patch_metrics(raw, alternating_recon)}
    preferred_recon_stage = _preferred_recon_stage(approximate_metrics, alternating_metrics)
    preferred_recon = approximate_recon if preferred_recon_stage == "approximate" else alternating_recon
    gamma_delta = alternating.gamma - approximate.gamma
    coeff_delta = alternating.coefficients_nm - approximate.coefficients_nm
    comparison_metrics = {
        "gamma_delta_norm": float(torch.linalg.norm(gamma_delta).item()),
        "gamma_delta_max_abs": _safe_abs_max(gamma_delta),
        "coeff_delta_mean_abs": _safe_abs_mean(coeff_delta),
        "coeff_delta_max_abs": _safe_abs_max(coeff_delta),
        "selected_emitters": int(dataset.patches_norm.shape[0]),
        "raw_harvest_emitters": int(len(emitters)),
        "selected_harvest_emitters": int(len(selected_emitters)),
        "selected_harvest_path": str(selected_harvest_path),
        "selection_mode": str(config.selection_mode),
        "global_projected_min_distance_px": float(config.global_projected_min_distance_px),
        "spatial_balance_grid_px": int(config.spatial_balance_grid_px),
        "spatial_balance_min_per_cell": int(config.spatial_balance_min_per_cell),
        "spatial_balance_max_per_cell": int(config.spatial_balance_max_per_cell),
    }
    if bool(config.freeze_initial_astig_standard):
        comparison_metrics["fixed_initial_astig_standard_nm"] = [
            float(v) for v in initial_astig_standard_coefficients_nm(fitter_cfg.nat_config).tolist()
        ]
    comparison_metrics["freeze_defocus_zero_gauge"] = bool(config.freeze_defocus_zero_gauge)
    comparison_metrics["vectorfit_astig_gauge"] = bool(config.vectorfit_astig_gauge)
    comparison_metrics["vectorfit_astig_anchor_nm"] = config.vectorfit_astig_anchor_nm
    comparison_metrics["vectorfit_astig_anchor_mode"] = str(config.vectorfit_astig_anchor_mode)
    comparison_metrics["vectorfit_phasor_z_init"] = bool(config.vectorfit_phasor_z_init)
    comparison_metrics["alternating_optimizer_kind"] = str(config.alternating_optimizer_kind)
    comparison_metrics["alternating_local_warmup_rounds"] = int(config.alternating_local_warmup_rounds)
    comparison_metrics["alternating_local_warmup_steps"] = int(config.alternating_local_warmup_steps)
    comparison_metrics["alternating_local_z_lr_multiplier"] = float(config.alternating_local_z_lr_multiplier)
    comparison_metrics["preferred_recon_stage"] = str(preferred_recon_stage)

    coeff_maps = figures_dir / "real_nat_coeff_maps_approx_vs_alternating.png"
    distribution_map = figures_dir / "selected_emitter_distribution_map.png"
    approximate_patch_grid = figures_dir / "real_nat_raw_vs_recon_approximate.png"
    alternating_patch_grid = figures_dir / "real_nat_raw_vs_recon_alternating.png"
    patch_grid = figures_dir / "real_nat_raw_vs_recon_preferred.png"
    loss_curve = figures_dir / "real_nat_loss_curves.png"
    _record_progress(output_dir, "write_figures", figures_dir=str(figures_dir))
    _save_coeff_maps(
        coeff_maps,
        cfg=preselected_cfg,
        approximate=approximate,
        alternating=alternating,
        grid_size=config.coeff_map_grid,
        include_approx=bool(config.include_approx_outputs),
    )
    _save_patch_grid(
        approximate_patch_grid,
        raw=raw,
        recon=approximate_recon,
        max_examples=config.patch_examples,
        recon_label="approximate",
    )
    _save_patch_grid(
        alternating_patch_grid,
        raw=raw,
        recon=alternating_recon,
        max_examples=config.patch_examples,
        recon_label="alternating",
    )
    _save_patch_grid(
        patch_grid,
        raw=raw,
        recon=preferred_recon,
        max_examples=config.patch_examples,
        recon_label=f"preferred ({preferred_recon_stage})",
    )
    _save_loss_curve(
        loss_curve,
        approximate=approximate,
        alternating=alternating,
        include_approx=bool(config.include_approx_outputs),
    )
    _save_selected_emitter_distribution_map(
        distribution_map,
        emitters=selected_emitters,
        width=int(fitter_cfg.nat_config.img_size_x),
        height=int(fitter_cfg.nat_config.img_size_y),
        roi_x_min_px=float(config.roi_x_min_px),
        roi_x_max_px=None if config.roi_x_max_px is None else float(config.roi_x_max_px),
        roi_y_min_px=float(config.roi_y_min_px),
        roi_y_max_px=None if config.roi_y_max_px is None else float(config.roi_y_max_px),
        selection_mode=str(config.selection_mode),
        spatial_balance_grid_px=int(config.spatial_balance_grid_px),
    )

    torch.save(
        {
            "approximate": approximate,
            "alternating": alternating,
            "raw_patches": raw,
            "approximate_recon": approximate_recon,
            "alternating_recon": alternating_recon,
            "preferred_recon_stage": preferred_recon_stage,
            "selected_harvest_summary": selected_harvest_summary,
            "comparison_metrics": comparison_metrics,
        },
        output_dir / "real_nat_diagnostics_payload.pt",
    )
    result = RealNATDiagnosticsResult(
        config=config,
        approximate_metrics=approximate_metrics,
        alternating_metrics=alternating_metrics,
        comparison_metrics=comparison_metrics,
        figures={
            "coeff_maps": str(coeff_maps),
            "distribution_map": str(distribution_map),
            "raw_vs_recon_approximate": str(approximate_patch_grid),
            "raw_vs_recon_alternating": str(alternating_patch_grid),
            "raw_vs_recon": str(patch_grid),
            "loss_curves": str(loss_curve),
        },
        output_dir=output_dir,
    )
    _record_progress(output_dir, "write_summary", summary_path=str(output_dir / "real_nat_diagnostics_summary.json"))
    (output_dir / "real_nat_diagnostics_summary.json").write_text(
        json.dumps(result.to_serializable_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", result)
    return result


__all__ = [
    "RealNATDiagnosticsConfig",
    "RealNATDiagnosticsResult",
    "load_harvested_emitters_from_pt",
    "run_real_nat_diagnostics",
]
