from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json

from .clean_reharvest import CleanReharvestConfig, run_clean_reharvest
from .export_nat_zmap import ExportNATZMapConfig, run_export_nat_zmap
from .pipeline_presets import build_microtubule_preset, build_zmap_preset
from .real_nat_diagnostics import RealNATDiagnosticsConfig, run_real_nat_diagnostics
from .standardize import promote_standard_zmap


def _default_run_root(sample: str, side: str) -> Path:
    sample_slug = str(sample).strip().lower().replace("_", "-")
    return Path(f"neptune/output/2026-05-24-{sample_slug}-{side}-zmap-main-localvector")


def _diagnostics_selected_emitters(metrics: dict[str, object]) -> int:
    keys = (
        "selection_final_selected_count",
        "selected_emitters",
    )
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return int(value)
    return 0


def _diagnostics_underfilled_cells(metrics: dict[str, object]) -> int | None:
    value = metrics.get("selection_spatial_balance_underfilled_cell_count")
    if value is None:
        return None
    return int(value)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical raw TIFF -> NAT zmap pipeline.")
    parser.add_argument(
        "--sample",
        choices=("microtube", "microtubule", "paint", "ncp", "tripple_color", "triple_color"),
        default="microtube",
    )
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--promote-standard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standard-root", type=Path, default=Path("param/zmap/microtube"))
    parser.add_argument("--standard-side-name", type=str, default=None)
    parser.add_argument("--min-neighbor-distance-px", type=float, default=None)
    parser.add_argument("--global-projected-min-distance-px", type=float, default=None)
    parser.add_argument("--max-emitters", type=int, default=None)
    parser.add_argument("--spatial-balance-grid-px", type=int, default=None)
    parser.add_argument("--spatial-balance-min-per-cell", type=int, default=None)
    parser.add_argument("--spatial-balance-max-per-cell", type=int, default=None)
    parser.add_argument("--max-patch-peak-distance-px", type=float, default=None)
    parser.add_argument("--max-secondary-peak-fraction", type=float, default=None)
    parser.add_argument("--rank-spots-by-quality", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def run_microtubule_pipeline(
    *,
    side: str,
    run_root: str | Path | None = None,
    device: str = "cuda:0",
    promote_standard: bool = True,
    standard_root: str | Path = "param/zmap/microtube",
    standard_side_name: str | None = None,
    min_neighbor_distance_px: float | None = None,
    global_projected_min_distance_px: float | None = None,
    max_emitters: int | None = None,
    spatial_balance_grid_px: int | None = None,
    spatial_balance_min_per_cell: int | None = None,
    spatial_balance_max_per_cell: int | None = None,
    max_patch_peak_distance_px: float | None = None,
    max_secondary_peak_fraction: float | None = None,
    rank_spots_by_quality: bool | None = None,
) -> dict[str, object]:
    return run_zmap_pipeline(
        sample="microtube",
        side=side,
        run_root=run_root,
        device=device,
        promote_standard=promote_standard,
        standard_root=standard_root,
        standard_side_name=standard_side_name,
        min_neighbor_distance_px=min_neighbor_distance_px,
        global_projected_min_distance_px=global_projected_min_distance_px,
        max_emitters=max_emitters,
        spatial_balance_grid_px=spatial_balance_grid_px,
        spatial_balance_min_per_cell=spatial_balance_min_per_cell,
        spatial_balance_max_per_cell=spatial_balance_max_per_cell,
        max_patch_peak_distance_px=max_patch_peak_distance_px,
        max_secondary_peak_fraction=max_secondary_peak_fraction,
        rank_spots_by_quality=rank_spots_by_quality,
    )


def run_zmap_pipeline(
    *,
    sample: str,
    side: str,
    run_root: str | Path | None = None,
    device: str = "cuda:0",
    promote_standard: bool = True,
    standard_root: str | Path = "param/zmap/microtube",
    standard_side_name: str | None = None,
    min_neighbor_distance_px: float | None = None,
    global_projected_min_distance_px: float | None = None,
    max_emitters: int | None = None,
    spatial_balance_grid_px: int | None = None,
    spatial_balance_min_per_cell: int | None = None,
    spatial_balance_max_per_cell: int | None = None,
    max_patch_peak_distance_px: float | None = None,
    max_secondary_peak_fraction: float | None = None,
    rank_spots_by_quality: bool | None = None,
) -> dict[str, object]:
    preset = build_zmap_preset(sample, side)
    preset_updates = {}
    if min_neighbor_distance_px is not None:
        preset_updates["min_neighbor_distance_px"] = float(min_neighbor_distance_px)
    if global_projected_min_distance_px is not None:
        preset_updates["global_projected_min_distance_px"] = float(global_projected_min_distance_px)
    if max_emitters is not None:
        preset_updates["max_emitters"] = int(max_emitters)
    if spatial_balance_grid_px is not None:
        preset_updates["spatial_balance_grid_px"] = int(spatial_balance_grid_px)
    if spatial_balance_min_per_cell is not None:
        preset_updates["spatial_balance_min_per_cell"] = int(spatial_balance_min_per_cell)
    if spatial_balance_max_per_cell is not None:
        preset_updates["spatial_balance_max_per_cell"] = int(spatial_balance_max_per_cell)
    if max_patch_peak_distance_px is not None:
        preset_updates["max_patch_peak_distance_px"] = float(max_patch_peak_distance_px)
    if max_secondary_peak_fraction is not None:
        preset_updates["max_secondary_peak_fraction"] = float(max_secondary_peak_fraction)
    if rank_spots_by_quality is not None:
        preset_updates["rank_spots_by_quality"] = bool(rank_spots_by_quality)
    if preset_updates:
        preset = replace(preset, **preset_updates)
    run_root = _default_run_root(sample, side) if run_root is None else Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    harvest_dir = run_root / "clean_reharvest"
    diagnostics_dir = run_root / "real_nat_diagnostics"
    export_dir = run_root / "export_nat_zmap"
    harvest = run_clean_reharvest(
        CleanReharvestConfig(
            output_dir=harvest_dir,
            tiff_path=preset.tiff_path,
            pipeline_config_path=preset.pipeline_config_path,
            checkpoint_path=preset.checkpoint_path,
            device=device,
            frame_start=preset.frame_start,
            frame_stop=preset.frame_stop,
            temporal_batch_size=preset.temporal_batch_size,
            spatial_tile_batch_size=preset.spatial_tile_batch_size,
            clean_library_target=preset.clean_library_target,
            gallery_count=preset.gallery_count,
            crop_x0=preset.crop_x0,
            crop_x1=preset.crop_x1,
            crop_y0=preset.crop_y0,
            crop_y1=preset.crop_y1,
            input_offset=preset.input_offset,
            input_scale=preset.input_scale,
            accept_prob_threshold=preset.accept_prob_threshold,
            phot_min=preset.phot_min,
            sigma_max_px=preset.sigma_max_px,
            baseline_adu=preset.baseline_adu,
        )
    )
    diagnostics = run_real_nat_diagnostics(
        RealNATDiagnosticsConfig(
            output_dir=diagnostics_dir,
            harvest_pt=harvest_dir / "harvest.pt",
            pipeline_config_path=preset.pipeline_config_path,
            tiff_path=preset.tiff_path,
            nat_config_kind=preset.nat_config_kind,
            device=device,
            max_emitters=preset.max_emitters,
            min_neighbor_distance_px=preset.min_neighbor_distance_px,
            global_projected_min_distance_px=preset.global_projected_min_distance_px,
            selection_pool_multiplier=preset.selection_pool_multiplier,
            selection_mode=preset.selection_mode,
            spatial_balance_grid_px=preset.spatial_balance_grid_px,
            spatial_balance_min_per_cell=preset.spatial_balance_min_per_cell,
            spatial_balance_max_per_cell=preset.spatial_balance_max_per_cell,
            max_patch_peak_distance_px=preset.max_patch_peak_distance_px,
            secondary_peak_exclusion_radius_px=preset.secondary_peak_exclusion_radius_px,
            max_secondary_peak_fraction=preset.max_secondary_peak_fraction,
            roi_x_min_px=preset.roi_x_min_px,
            roi_x_max_px=preset.roi_x_max_px,
            roi_y_min_px=preset.roi_y_min_px,
            roi_y_max_px=preset.roi_y_max_px,
            roi_edge_margin_px=preset.roi_edge_margin_px,
            freeze_initial_astig_standard=True,
            include_approx_outputs=False,
            alternating_rounds=preset.alternating_rounds,
            alternating_local_steps=preset.alternating_local_steps,
            alternating_global_steps=preset.alternating_global_steps,
            input_offset=preset.input_offset,
            input_scale=preset.input_scale,
            baseline_adu=preset.baseline_adu,
        )
    )
    final_stop = preset.frame_stop
    export = run_export_nat_zmap(
        ExportNATZMapConfig(
            diagnostics_dir=diagnostics_dir,
            output_dir=export_dir,
            export_alternating=True,
            export_approximate=False,
            include_fixed_astig_baseline=True,
            export_provisional_scalar_zmap=True,
        )
    )

    promoted = None
    if promote_standard:
        target_leaf = str(side if standard_side_name is None else standard_side_name)
        sample_slug = str(sample).strip().lower().replace("-", "_")
        promoted = promote_standard_zmap(
            side=side,
            harvest_dir=harvest_dir,
            diagnostics_dir=diagnostics_dir,
            export_dir=export_dir,
            target_root=Path(standard_root) / target_leaf,
            notes=(
                f"Canonical {sample_slug} zmap generated by neptune.zmap_main pipeline "
                f"with spatially balanced {preset.spatial_balance_grid_px}px NAT emitter sampling; "
                f"min_neighbor_distance_px={preset.min_neighbor_distance_px}, "
                f"global_projected_min_distance_px={preset.global_projected_min_distance_px}, "
                f"min_per_cell={preset.spatial_balance_min_per_cell}, "
                f"max_per_cell={preset.spatial_balance_max_per_cell}."
            ),
        )

    result = {
        "sample": str(sample),
        "side": side,
        "preset": _jsonable(asdict(preset)),
        "run_root": str(run_root),
        "harvest_dir": str(harvest.output_dir),
        "diagnostics_dir": str(diagnostics.output_dir),
        "export_dir": str(export.output_dir),
        "promoted_root": None if promoted is None else str(promoted.root_dir),
        "effective_frame_stop": None if final_stop is None else int(final_stop),
    }
    (run_root / "pipeline_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    result = run_zmap_pipeline(
        sample=args.sample,
        side=args.side,
        run_root=args.run_root,
        device=args.device,
        promote_standard=args.promote_standard,
        standard_root=args.standard_root,
        standard_side_name=args.standard_side_name,
        min_neighbor_distance_px=args.min_neighbor_distance_px,
        global_projected_min_distance_px=args.global_projected_min_distance_px,
        max_emitters=args.max_emitters,
        spatial_balance_grid_px=args.spatial_balance_grid_px,
        spatial_balance_min_per_cell=args.spatial_balance_min_per_cell,
        spatial_balance_max_per_cell=args.spatial_balance_max_per_cell,
        max_patch_peak_distance_px=args.max_patch_peak_distance_px,
        max_secondary_peak_fraction=args.max_secondary_peak_fraction,
        rank_spots_by_quality=args.rank_spots_by_quality,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
