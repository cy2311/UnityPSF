from __future__ import annotations

import argparse
from pathlib import Path

from .real_nat_diagnostics import RealNATDiagnosticsConfig, run_real_nat_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real harvested-emitter NAT diagnostics with approximate and alternating fitters.")
    parser.add_argument("--output-dir", type=Path, default=Path("neptune/output/real_nat_alternating_diagnostics"))
    parser.add_argument("--harvest-pt", type=Path, default=Path("output/neptune/2026-04-21-first-tiff-bootstrap-v1/rounds/round_000/harvest.pt"))
    parser.add_argument("--pipeline-config", type=Path, default=Path("neptune/neptune_core/starters/density_1/pipeline_density_1_real_sample_default.json"))
    parser.add_argument("--tiff-path", type=Path, default=Path("datasets/raw_samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif"))
    parser.add_argument("--nat-config-kind", type=str, default="order1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-emitters", type=int, default=32)
    parser.add_argument("--min-neighbor-distance-px", type=float, default=15.0)
    parser.add_argument("--selection-pool-multiplier", type=int, default=50)
    parser.add_argument("--selection-mode", choices=("probability_topk", "spatial_balanced"), default="probability_topk")
    parser.add_argument("--spatial-balance-grid-px", type=int, default=128)
    parser.add_argument("--spatial-balance-min-per-cell", type=int, default=0)
    parser.add_argument("--spatial-balance-max-per-cell", type=int, default=0)
    parser.add_argument("--rank-spots-by-quality", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-center-peak-norm", type=float, default=0.0)
    parser.add_argument("--min-signal-sum-norm", type=float, default=0.0)
    parser.add_argument("--input-offset", type=float, default=None)
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument("--patch-adu-mode", choices=("baseline_clamp", "raw", "border_subtract"), default="baseline_clamp")
    parser.add_argument("--max-patch-peak-distance-px", type=float, default=2.5)
    parser.add_argument("--secondary-peak-exclusion-radius-px", type=float, default=3.5)
    parser.add_argument("--max-secondary-peak-fraction", type=float, default=0.45)
    parser.add_argument("--roi-x-min-px", type=float, default=0.0)
    parser.add_argument("--roi-x-max-px", type=float, default=600.0)
    parser.add_argument("--roi-y-min-px", type=float, default=0.0)
    parser.add_argument("--roi-y-max-px", type=float, default=1200.0)
    parser.add_argument("--roi-edge-margin-px", type=float, default=3.0)
    parser.add_argument("--freeze-initial-astig-standard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-defocus-zero-gauge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vectorfit-astig-gauge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vectorfit-astig-anchor-nm", type=float, default=None)
    parser.add_argument("--vectorfit-astig-anchor-mode", choices=("init_only", "hard_fixed"), default="init_only")
    parser.add_argument("--vectorfit-phasor-z-init", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-approx-outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alternating-rounds", type=int, default=10)
    parser.add_argument("--alternating-local-steps", type=int, default=50)
    parser.add_argument("--alternating-global-steps", type=int, default=50)
    parser.add_argument("--alternating-local-warmup-rounds", type=int, default=0)
    parser.add_argument("--alternating-local-warmup-steps", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_real_nat_diagnostics(
        RealNATDiagnosticsConfig(
            output_dir=args.output_dir,
            harvest_pt=args.harvest_pt,
            pipeline_config_path=args.pipeline_config,
            tiff_path=args.tiff_path,
            nat_config_kind=args.nat_config_kind,
            device=args.device,
            max_emitters=args.max_emitters,
            min_neighbor_distance_px=args.min_neighbor_distance_px,
            selection_pool_multiplier=args.selection_pool_multiplier,
            selection_mode=args.selection_mode,
            spatial_balance_grid_px=args.spatial_balance_grid_px,
            spatial_balance_min_per_cell=args.spatial_balance_min_per_cell,
            spatial_balance_max_per_cell=args.spatial_balance_max_per_cell,
            rank_spots_by_quality=args.rank_spots_by_quality,
            min_center_peak_norm=args.min_center_peak_norm,
            min_signal_sum_norm=args.min_signal_sum_norm,
            input_offset=args.input_offset,
            input_scale=args.input_scale,
            patch_adu_mode=args.patch_adu_mode,
            max_patch_peak_distance_px=args.max_patch_peak_distance_px,
            secondary_peak_exclusion_radius_px=args.secondary_peak_exclusion_radius_px,
            max_secondary_peak_fraction=args.max_secondary_peak_fraction,
            roi_x_min_px=args.roi_x_min_px,
            roi_x_max_px=args.roi_x_max_px,
            roi_y_min_px=args.roi_y_min_px,
            roi_y_max_px=args.roi_y_max_px,
            roi_edge_margin_px=args.roi_edge_margin_px,
            freeze_initial_astig_standard=args.freeze_initial_astig_standard,
            freeze_defocus_zero_gauge=args.freeze_defocus_zero_gauge,
            vectorfit_astig_gauge=args.vectorfit_astig_gauge,
            vectorfit_astig_anchor_nm=args.vectorfit_astig_anchor_nm,
            vectorfit_astig_anchor_mode=args.vectorfit_astig_anchor_mode,
            vectorfit_phasor_z_init=args.vectorfit_phasor_z_init,
            include_approx_outputs=args.include_approx_outputs,
            alternating_rounds=args.alternating_rounds,
            alternating_local_steps=args.alternating_local_steps,
            alternating_global_steps=args.alternating_global_steps,
            alternating_local_warmup_rounds=args.alternating_local_warmup_rounds,
            alternating_local_warmup_steps=args.alternating_local_warmup_steps,
        )
    )
    print(f"output_dir={result.output_dir}")
    print(f"gamma_delta_norm={result.comparison_metrics['gamma_delta_norm']:.6f}")
    print(f"approximate_patch_mse={result.approximate_metrics['patch_mse_mean']:.6f}")
    print(f"alternating_patch_mse={result.alternating_metrics['patch_mse_mean']:.6f}")
    for name, path in result.figures.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
