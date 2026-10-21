from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .nat_fit import build_nat_spot_fitter_config, initial_astig_standard_coefficients_nm
from .zmap_contract import build_full_roi_zmap
from .nat_torch import evaluate_zernike_from_roi_positions_torch


@dataclass(frozen=True)
class ExportNATZMapConfig:
    diagnostics_dir: Path
    output_dir: Path
    export_alternating: bool = True
    export_approximate: bool = False
    preferred_stage: str = "alternating"
    include_fixed_astig_baseline: bool = True
    export_provisional_scalar_zmap: bool = True
    scalar_mode: str = "non_astig_rms_nm"
    figure_percentile: float = 99.5
    surface_max_grid: int = 160
    sync_diagnostics_figures: bool = True


@dataclass(frozen=True)
class ExportNATZMapResult:
    config: ExportNATZMapConfig
    mode_order: list[tuple[int, int]]
    image_shape_hw: tuple[int, int]
    alternating_stack_npz: str | None
    approximate_stack_npz: str | None
    preferred_stack_npz: str | None
    provisional_scalar_zmap_npz: str | None
    figures: dict[str, str]
    output_dir: Path

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "config": {
                **asdict(self.config),
                "diagnostics_dir": str(self.config.diagnostics_dir),
                "output_dir": str(self.config.output_dir),
            },
            "mode_order": [[int(n), int(m)] for n, m in self.mode_order],
            "image_shape_hw": [int(self.image_shape_hw[0]), int(self.image_shape_hw[1])],
            "alternating_stack_npz": self.alternating_stack_npz,
            "approximate_stack_npz": self.approximate_stack_npz,
            "preferred_stack_npz": self.preferred_stack_npz,
            "provisional_scalar_zmap_npz": self.provisional_scalar_zmap_npz,
            "figures": self.figures,
            "output_dir": str(self.output_dir),
        }


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_figure_source_path(diagnostics_dir: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.is_file():
        return path
    return diagnostics_dir / path


def _sync_selected_diagnostic_figures(
    *,
    diagnostics_dir: Path,
    diagnostics_summary: dict[str, Any],
    figures_dir: Path,
) -> dict[str, str]:
    figures = diagnostics_summary.get("figures")
    if not isinstance(figures, dict):
        return {}
    synced: dict[str, str] = {}
    figure_aliases = {
        "raw_vs_recon": ("raw_vs_recon",),
        "raw_vs_recon_approximate": ("raw_vs_recon_approximate",),
        "raw_vs_recon_alternating": ("raw_vs_recon_alternating",),
        "distribution_map": ("distribution_map", "selected_emitter_distribution_map"),
    }
    for output_key, aliases in figure_aliases.items():
        source_path = None
        for alias in aliases:
            candidate = figures.get(alias)
            if candidate:
                source_path = _resolve_figure_source_path(diagnostics_dir, candidate)
                break
        if source_path is None or not source_path.is_file():
            continue
        target_path = figures_dir / source_path.name
        if source_path != target_path:
            shutil.copy2(source_path, target_path)
        synced[output_key] = str(target_path)
    return synced


def _fitter_config_from_summary(summary: dict[str, Any]):
    cfg = summary["config"]
    return build_nat_spot_fitter_config(
        pipeline_config_path=cfg["pipeline_config_path"],
        tiff_path=cfg["tiff_path"],
        nat_config_kind=str(cfg.get("nat_config_kind", "order1")),
        device="cpu",
        patch_size_px=int(cfg["patch_size_px"]),
        max_emitters=int(cfg["max_emitters"]),
        fit_steps=int(cfg["approximate_fit_steps"]),
        fit_lr=float(cfg["approximate_fit_lr"]),
        coeff_l2=float(cfg["coeff_l2"]),
        ridge=float(cfg["ridge"]),
        min_neighbor_distance_px=float(cfg["min_neighbor_distance_px"]),
        max_patch_peak_distance_px=float(cfg["max_patch_peak_distance_px"]),
        secondary_peak_exclusion_radius_px=float(cfg["secondary_peak_exclusion_radius_px"]),
        max_secondary_peak_fraction=float(cfg["max_secondary_peak_fraction"]),
        roi_x_min_px=float(cfg.get("roi_x_min_px", 0.0)),
        roi_x_max_px=None if cfg.get("roi_x_max_px") is None else float(cfg["roi_x_max_px"]),
        roi_y_min_px=float(cfg.get("roi_y_min_px", 0.0)),
        roi_y_max_px=None if cfg.get("roi_y_max_px") is None else float(cfg["roi_y_max_px"]),
        roi_edge_margin_px=float(cfg.get("roi_edge_margin_px", 0.0)),
        freeze_initial_astig_standard=bool(cfg.get("freeze_initial_astig_standard", False)),
        freeze_defocus_zero_gauge=bool(cfg.get("freeze_defocus_zero_gauge", False)),
    )


def _full_roi_positions(width: int, height: int) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def _evaluate_full_roi_coeff_stack(
    *,
    gamma: torch.Tensor,
    nat_config,
    width: int,
    height: int,
    include_fixed_astig_baseline: bool,
) -> torch.Tensor:
    roixy = _full_roi_positions(width, height)
    coeffs = evaluate_zernike_from_roi_positions_torch(
        roixy,
        gamma.to(dtype=torch.float32, device="cpu"),
        nat_config,
        dtype=torch.float32,
        device="cpu",
    )[0]
    coeffs = coeffs.reshape(height, width, -1).permute(2, 0, 1).contiguous()
    if include_fixed_astig_baseline:
        coeffs = coeffs + initial_astig_standard_coefficients_nm(nat_config).reshape(-1, 1, 1)
    return coeffs


def _save_stack_npz(path: Path, *, stack_nm: torch.Tensor, mode_order: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        zernike_maps_nm=stack_nm.detach().cpu().numpy().astype(np.float32, copy=False),
        mode_order=np.asarray(mode_order, dtype=np.int64),
    )


def _save_mode_gallery(path: Path, *, stack_nm: torch.Tensor, mode_order: list[tuple[int, int]], percentile: float) -> None:
    maps = stack_nm.detach().cpu().numpy()
    count = int(maps.shape[0])
    cols = 2
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.6 * rows), dpi=180, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx >= count:
            ax.axis("off")
            continue
        panel = maps[idx]
        vmax = float(np.percentile(np.abs(panel), percentile))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        im = ax.imshow(panel, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Z{mode_order[idx]}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _compute_scalar_map(stack_nm: torch.Tensor, *, mode_order: list[tuple[int, int]], scalar_mode: str) -> np.ndarray:
    values = stack_nm.detach().cpu().numpy().astype(np.float32, copy=False)
    if scalar_mode == "all_mode_rms_nm":
        return np.sqrt(np.sum(values ** 2, axis=0, dtype=np.float32), dtype=np.float32)
    if scalar_mode == "non_astig_rms_nm":
        keep = [idx for idx, mode in enumerate(mode_order) if mode not in {(2, 2), (2, -2)}]
        reduced = values[np.asarray(keep, dtype=np.int64)]
        return np.sqrt(np.sum(reduced ** 2, axis=0, dtype=np.float32), dtype=np.float32)
    raise ValueError(f"Unsupported scalar_mode: {scalar_mode!r}")


def _save_scalar_figure(path: Path, *, values_nm: np.ndarray, title: str, percentile: float) -> None:
    vmax = float(np.percentile(values_nm, percentile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=180, constrained_layout=True)
    im = ax.imshow(values_nm, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _downsample_surface(values: np.ndarray, *, max_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = values.shape
    step_y = max(1, int(np.ceil(height / max_grid)))
    step_x = max(1, int(np.ceil(width / max_grid)))
    reduced = values[::step_y, ::step_x]
    ys = np.arange(0, height, step_y, dtype=np.float32)[: reduced.shape[0]]
    xs = np.arange(0, width, step_x, dtype=np.float32)[: reduced.shape[1]]
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return xx, yy, reduced


def _save_surface_figure(
    path: Path,
    *,
    values_nm: np.ndarray,
    title: str,
    percentile: float,
    max_grid: int,
    cmap: str,
) -> None:
    xx, yy, zz = _downsample_surface(values_nm, max_grid=max_grid)
    vmax = float(np.percentile(np.abs(zz), percentile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    fig = plt.figure(figsize=(7.0, 5.4), dpi=180, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap=cmap,
        linewidth=0.0,
        antialiased=True,
        vmin=-vmax if np.nanmin(zz) < 0.0 else 0.0,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_zlabel("nm")
    ax.view_init(elev=28, azim=-130)
    fig.colorbar(surf, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_mode_surface_gallery(
    path: Path,
    *,
    stack_nm: torch.Tensor,
    mode_order: list[tuple[int, int]],
    percentile: float,
    max_grid: int,
) -> None:
    maps = stack_nm.detach().cpu().numpy()
    count = int(maps.shape[0])
    cols = 2
    rows = int(np.ceil(count / cols))
    fig = plt.figure(figsize=(7.2 * cols, 5.0 * rows), dpi=180, constrained_layout=True)
    for idx in range(count):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        panel = maps[idx]
        xx, yy, zz = _downsample_surface(panel, max_grid=max_grid)
        vmax = float(np.percentile(np.abs(zz), percentile))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        surf = ax.plot_surface(
            xx,
            yy,
            zz,
            cmap="coolwarm",
            linewidth=0.0,
            antialiased=True,
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"Z{mode_order[idx]}", fontsize=9)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("nm", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=28, azim=-130)
        fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_report(path: Path, result: ExportNATZMapResult) -> None:
    lines = [
        "# Export NAT full-ROI maps",
        "",
        "## Summary",
        "",
        f"- diagnostics_dir: `{result.config.diagnostics_dir}`",
        f"- image_shape_hw: `{result.image_shape_hw}`",
        f"- mode_count: `{len(result.mode_order)}`",
        f"- export_alternating: `{result.config.export_alternating}`",
        f"- export_approximate: `{result.config.export_approximate}`",
        f"- include_fixed_astig_baseline: `{result.config.include_fixed_astig_baseline}`",
        f"- export_provisional_scalar_zmap: `{result.config.export_provisional_scalar_zmap}`",
        f"- scalar_mode: `{result.config.scalar_mode}`",
        "",
        "## Outputs",
        "",
    ]
    if result.alternating_stack_npz:
        lines.append(f"- alternating coefficient stack: `{result.alternating_stack_npz}`")
    if result.approximate_stack_npz:
        lines.append(f"- approximate coefficient stack: `{result.approximate_stack_npz}`")
    if result.provisional_scalar_zmap_npz:
        lines.append(f"- provisional scalar zmap: `{result.provisional_scalar_zmap_npz}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This export writes the NAT full-ROI coefficient maps directly; these are the closest thing to a true NAT full-field output.",
            "- The provisional scalar zmap is a single-channel descriptor derived from the coefficient stack, not a finalized physical phase-map definition.",
            "- For now, the safest interpretation is: NAT already gives a full-ROI spatial optical prior, while the final single-channel training `Zmap` semantic still needs an explicit design choice.",
        ]
    )
    for name, figure_path in result.figures.items():
        lines.append(f"- {name}: `{figure_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_export_nat_zmap(config: ExportNATZMapConfig) -> ExportNATZMapResult:
    diagnostics_dir = Path(config.diagnostics_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(diagnostics_dir / "real_nat_diagnostics_summary.json")
    payload = torch.load(diagnostics_dir / "real_nat_diagnostics_payload.pt", map_location="cpu", weights_only=False)
    alternating = payload["alternating"]
    approximate = payload["approximate"]
    fitter_cfg = _fitter_config_from_summary(summary)
    width = int(fitter_cfg.nat_config.img_size_x)
    height = int(fitter_cfg.nat_config.img_size_y)
    mode_order = [tuple(int(v) for v in mode) for mode in fitter_cfg.nat_config.aberrations]

    alternating_stack_path: str | None = None
    approximate_stack_path: str | None = None
    preferred_stack_path: str | None = None
    provisional_scalar_path: str | None = None
    figures: dict[str, str] = {}

    alternating_stack = None
    if bool(config.export_alternating):
        alternating_stack = _evaluate_full_roi_coeff_stack(
            gamma=alternating.gamma,
            nat_config=fitter_cfg.nat_config,
            width=width,
            height=height,
            include_fixed_astig_baseline=bool(config.include_fixed_astig_baseline),
        )
        stack_path = output_dir / "alternating_full_roi_zernike_maps_nm.npz"
        _save_stack_npz(stack_path, stack_nm=alternating_stack, mode_order=mode_order)
        alternating_stack_path = str(stack_path)
        gallery_path = figures_dir / "alternating_full_roi_zernike_maps_nm.png"
        _save_mode_gallery(gallery_path, stack_nm=alternating_stack, mode_order=mode_order, percentile=float(config.figure_percentile))
        figures["alternating_coeff_maps"] = str(gallery_path)
        gallery_3d_path = figures_dir / "alternating_full_roi_zernike_maps_3d.png"
        _save_mode_surface_gallery(
            gallery_3d_path,
            stack_nm=alternating_stack,
            mode_order=mode_order,
            percentile=float(config.figure_percentile),
            max_grid=int(config.surface_max_grid),
        )
        figures["alternating_coeff_maps_3d"] = str(gallery_3d_path)

    if bool(config.export_approximate):
        approximate_stack = _evaluate_full_roi_coeff_stack(
            gamma=approximate.gamma,
            nat_config=fitter_cfg.nat_config,
            width=width,
            height=height,
            include_fixed_astig_baseline=bool(config.include_fixed_astig_baseline),
        )
        stack_path = output_dir / "approximate_full_roi_zernike_maps_nm.npz"
        _save_stack_npz(stack_path, stack_nm=approximate_stack, mode_order=mode_order)
        approximate_stack_path = str(stack_path)
        gallery_path = figures_dir / "approximate_full_roi_zernike_maps_nm.png"
        _save_mode_gallery(gallery_path, stack_nm=approximate_stack, mode_order=mode_order, percentile=float(config.figure_percentile))
        figures["approximate_coeff_maps"] = str(gallery_path)

    preferred_stage = str(config.preferred_stage).strip().lower()
    if preferred_stage not in {"alternating", "approximate"}:
        preferred_stage = "alternating"
    preferred_source = approximate_stack_path if preferred_stage == "approximate" else alternating_stack_path
    if preferred_source is not None:
        preferred_path = output_dir / "preferred_full_roi_zernike_maps_nm.npz"
        shutil.copy2(Path(preferred_source), preferred_path)
        preferred_stack_path = str(preferred_path)

    if bool(config.export_provisional_scalar_zmap):
        if alternating_stack is None:
            alternating_stack = _evaluate_full_roi_coeff_stack(
                gamma=alternating.gamma,
                nat_config=fitter_cfg.nat_config,
                width=width,
                height=height,
                include_fixed_astig_baseline=bool(config.include_fixed_astig_baseline),
            )
        scalar_map_nm = _compute_scalar_map(alternating_stack, mode_order=mode_order, scalar_mode=str(config.scalar_mode))
        finite = scalar_map_nm[np.isfinite(scalar_map_nm)]
        z_max = float(np.quantile(finite, 0.999)) if finite.size else 1.0
        z_max = max(z_max, 1e-6)
        phase_map = build_full_roi_zmap(
            scalar_map_nm,
            image_shape_hw=(height, width),
            pixel_size_nm_xy=(float(fitter_cfg.pixel_size_x_nm), float(fitter_cfg.pixel_size_y_nm)),
            value_range_nm=(0.0, z_max),
            source=f"nat_{config.scalar_mode}_provisional",
            continuous=True,
            pixel_aligned=True,
        )
        zmap_path = output_dir / f"provisional_{config.scalar_mode}.npz"
        phase_map.save(zmap_path)
        provisional_scalar_path = str(zmap_path)
        figure_path = figures_dir / f"provisional_{config.scalar_mode}.png"
        _save_scalar_figure(
            figure_path,
            values_nm=scalar_map_nm,
            title=f"Provisional {config.scalar_mode}",
            percentile=float(config.figure_percentile),
        )
        figures["provisional_scalar_zmap"] = str(figure_path)
        figure_3d_path = figures_dir / f"provisional_{config.scalar_mode}_3d.png"
        _save_surface_figure(
            figure_3d_path,
            values_nm=scalar_map_nm,
            title=f"Provisional {config.scalar_mode} (3D)",
            percentile=float(config.figure_percentile),
            max_grid=int(config.surface_max_grid),
            cmap="viridis",
        )
        figures["provisional_scalar_zmap_3d"] = str(figure_3d_path)

    if bool(config.sync_diagnostics_figures):
        figures.update(
            _sync_selected_diagnostic_figures(
                diagnostics_dir=diagnostics_dir,
                diagnostics_summary=summary,
                figures_dir=figures_dir,
            )
        )

    result = ExportNATZMapResult(
        config=config,
        mode_order=mode_order,
        image_shape_hw=(height, width),
        alternating_stack_npz=alternating_stack_path,
        approximate_stack_npz=approximate_stack_path,
        preferred_stack_npz=preferred_stack_path,
        provisional_scalar_zmap_npz=provisional_scalar_path,
        figures=figures,
        output_dir=output_dir,
    )
    (output_dir / "export_nat_zmap_summary.json").write_text(
        json.dumps(result.to_serializable_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", result)
    return result


__all__ = [
    "ExportNATZMapConfig",
    "ExportNATZMapResult",
    "run_export_nat_zmap",
]
