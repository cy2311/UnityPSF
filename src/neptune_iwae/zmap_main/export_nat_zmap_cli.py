from __future__ import annotations

import argparse
from pathlib import Path

from .export_nat_zmap import ExportNATZMapConfig, run_export_nat_zmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NAT full-ROI coefficient maps and provisional scalar zmap.")
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("neptune/nat_maps"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("neptune/nat_maps"),
    )
    parser.add_argument("--export-alternating", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-approximate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-fixed-astig-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-provisional-scalar-zmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scalar-mode", type=str, default="non_astig_rms_nm")
    parser.add_argument("--figure-percentile", type=float, default=99.5)
    parser.add_argument("--surface-max-grid", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_export_nat_zmap(
        ExportNATZMapConfig(
            diagnostics_dir=args.diagnostics_dir,
            output_dir=args.output_dir,
            export_alternating=args.export_alternating,
            export_approximate=args.export_approximate,
            include_fixed_astig_baseline=args.include_fixed_astig_baseline,
            export_provisional_scalar_zmap=args.export_provisional_scalar_zmap,
            scalar_mode=args.scalar_mode,
            figure_percentile=args.figure_percentile,
            surface_max_grid=args.surface_max_grid,
        )
    )
    print(f"output_dir={result.output_dir}")
    print(f"image_shape_hw={result.image_shape_hw}")
    print(f"mode_count={len(result.mode_order)}")
    if result.alternating_stack_npz:
        print(f"alternating_stack_npz={result.alternating_stack_npz}")
    if result.approximate_stack_npz:
        print(f"approximate_stack_npz={result.approximate_stack_npz}")
    if result.provisional_scalar_zmap_npz:
        print(f"provisional_scalar_zmap_npz={result.provisional_scalar_zmap_npz}")
    for name, path in result.figures.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
