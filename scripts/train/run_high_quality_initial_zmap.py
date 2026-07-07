from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from neptune_iwae.zmap_main.clean_reharvest import CleanReharvestConfig, run_clean_reharvest
from neptune_iwae.zmap_main.export_nat_zmap import ExportNATZMapConfig, run_export_nat_zmap
from neptune_iwae.zmap_main.harvest import _canonicalize_stack_array
from neptune_iwae.zmap_main.pipeline_presets import build_zmap_preset
from neptune_iwae.zmap_main.real_nat_diagnostics import RealNATDiagnosticsConfig, run_real_nat_diagnostics


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _resolve_existing(path: str | Path, roots: tuple[Path, ...]) -> Path:
    candidate = Path(path)
    candidates = [candidate] if candidate.is_absolute() else [*(root / candidate for root in roots), candidate]
    for item in candidates:
        expanded = item.expanduser()
        if expanded.exists():
            return expanded.resolve()
    raise FileNotFoundError(f"Could not resolve path {path!s} under roots {[str(root) for root in roots]}")


def _read_frame_window(tiff_path: Path, frame_start: int, frame_stop: int) -> np.ndarray:
    try:
        stack = _canonicalize_stack_array(np.asarray(tifffile.memmap(tiff_path)))
        return np.asarray(stack[int(frame_start) : int(frame_stop)], dtype=np.float32)
    except ValueError:
        with tifffile.TiffFile(tiff_path) as tif:
            if not tif.series:
                raise ValueError(f"TIFF has no image series: {tiff_path}")
            frames = tif.series[0].asarray(key=range(int(frame_start), int(frame_stop)))
        return np.asarray(_canonicalize_stack_array(frames), dtype=np.float32)


def _estimate_channel_baseline(
    *,
    tiff_path: Path,
    side: str,
    frame_start: int,
    frame_stop: int,
    percentile: float,
) -> float:
    frames = _read_frame_window(tiff_path, frame_start, frame_stop)
    width = int(frames.shape[2])
    if side == "left":
        crop = frames[:, :, : width // 2]
    elif side == "right":
        crop = frames[:, :, width // 2 :]
    else:
        raise ValueError(f"Unsupported side: {side!r}")
    return float(np.percentile(crop, float(percentile)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neptune v0.3 high-quality initial zmap bootstrap.")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--repo-root", type=Path, default=Path("/home/guest/Others/main/race"))
    parser.add_argument("--raw-tiff", type=Path, default=None)
    parser.add_argument("--max-emitters", type=int, default=500)
    parser.add_argument("--alternating-rounds", type=int, default=20)
    parser.add_argument("--alternating-local-steps", type=int, default=100)
    parser.add_argument("--alternating-global-steps", type=int, default=100)
    parser.add_argument("--selection-pool-multiplier", type=int, default=None)
    parser.add_argument("--spatial-balance-grid-px", type=int, default=None)
    parser.add_argument("--spatial-balance-min-per-cell", type=int, default=None)
    parser.add_argument("--spatial-balance-max-per-cell", type=int, default=None)
    parser.add_argument("--baseline-percentile", type=float, default=1.0)
    parser.add_argument("--baseline-frame-start", type=int, default=0)
    parser.add_argument("--baseline-frame-stop", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    neptune_iwae_root = repo_root / "neptune_iwae"
    raw_tiff = (
        _resolve_existing(args.raw_tiff, (repo_root, neptune_iwae_root))
        if args.raw_tiff is not None
        else _resolve_existing(
            "test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif",
            (neptune_iwae_root, repo_root),
        )
    )

    preset = build_zmap_preset("microtube", args.side)
    preset = replace(
        preset,
        tiff_path=raw_tiff,
        pipeline_config_path=_resolve_existing(preset.pipeline_config_path, (neptune_iwae_root, repo_root)),
        checkpoint_path=_resolve_existing(preset.checkpoint_path, (neptune_iwae_root, repo_root)),
        max_emitters=int(args.max_emitters),
        alternating_rounds=int(args.alternating_rounds),
        alternating_local_steps=int(args.alternating_local_steps),
        alternating_global_steps=int(args.alternating_global_steps),
        selection_pool_multiplier=(
            preset.selection_pool_multiplier
            if args.selection_pool_multiplier is None
            else int(args.selection_pool_multiplier)
        ),
        spatial_balance_grid_px=(
            preset.spatial_balance_grid_px if args.spatial_balance_grid_px is None else int(args.spatial_balance_grid_px)
        ),
        spatial_balance_min_per_cell=(
            preset.spatial_balance_min_per_cell
            if args.spatial_balance_min_per_cell is None
            else int(args.spatial_balance_min_per_cell)
        ),
        spatial_balance_max_per_cell=(
            preset.spatial_balance_max_per_cell
            if args.spatial_balance_max_per_cell is None
            else int(args.spatial_balance_max_per_cell)
        ),
    )

    baseline_adu = _estimate_channel_baseline(
        tiff_path=raw_tiff,
        side=args.side,
        frame_start=int(args.baseline_frame_start),
        frame_stop=int(args.baseline_frame_stop),
        percentile=float(args.baseline_percentile),
    )
    preset = replace(preset, baseline_adu=baseline_adu)

    run_root = args.run_root.resolve()
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
            device=args.device,
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
            device=args.device,
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

    alternating_stack = export_dir / "alternating_full_roi_zernike_maps_nm.npz"
    result = {
        "sample": "microtube",
        "side": args.side,
        "preset": _jsonable(asdict(preset)),
        "run_root": str(run_root),
        "harvest_dir": str(harvest.output_dir),
        "diagnostics_dir": str(diagnostics.output_dir),
        "export_dir": str(export.output_dir),
        "alternating_stack_npz": str(alternating_stack),
        "clean_patch_count": int(harvest.clean_patch_count),
        "baseline_mode": "percentile",
        "baseline_adu": float(baseline_adu),
        "baseline_percentile": float(args.baseline_percentile),
        "baseline_frame_range": [int(args.baseline_frame_start), int(args.baseline_frame_stop)],
        "diagnostics_selected_emitters": int(diagnostics.comparison_metrics.get("selected_emitters", 0)),
    }
    (run_root / "pipeline_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not alternating_stack.exists():
        raise FileNotFoundError(f"Expected zmap export was not created: {alternating_stack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
