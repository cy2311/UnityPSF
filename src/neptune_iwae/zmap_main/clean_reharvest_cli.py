from __future__ import annotations

import argparse
from pathlib import Path

from .clean_reharvest import CleanReharvestConfig, run_clean_reharvest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a larger real harvest and summarize the clean single-emitter patch library.")
    parser.add_argument("--output-dir", type=Path, default=Path("neptune/output/clean_reharvest"))
    parser.add_argument("--tiff-path", type=Path, default=Path("datasets/raw_samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif"))
    parser.add_argument("--pipeline-config", type=Path, default=Path("neptune/neptune_core/starters/density_1/pipeline_density_1_real_sample_default.json"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=512)
    parser.add_argument("--temporal-batch-size", type=int, default=64)
    parser.add_argument("--spatial-tile-batch-size", type=int, default=64)
    parser.add_argument("--clean-library-target", type=int, default=256)
    parser.add_argument("--gallery-count", type=int, default=16)
    parser.add_argument("--crop-x0", type=int, default=0)
    parser.add_argument("--crop-x1", type=int, default=None)
    parser.add_argument("--crop-y0", type=int, default=0)
    parser.add_argument("--crop-y1", type=int, default=None)
    parser.add_argument("--input-offset", type=float, default=None)
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument("--raw-th", type=float, default=None)
    parser.add_argument("--split-th", type=float, default=None)
    parser.add_argument("--accept-prob-threshold", type=float, default=0.7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_clean_reharvest(
        CleanReharvestConfig(
            output_dir=args.output_dir,
            tiff_path=args.tiff_path,
            pipeline_config_path=args.pipeline_config,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            frame_start=args.frame_start,
            frame_stop=args.frame_stop,
            temporal_batch_size=args.temporal_batch_size,
            spatial_tile_batch_size=args.spatial_tile_batch_size,
            clean_library_target=args.clean_library_target,
            gallery_count=args.gallery_count,
            crop_x0=args.crop_x0,
            crop_x1=args.crop_x1,
            crop_y0=args.crop_y0,
            crop_y1=args.crop_y1,
            input_offset=args.input_offset,
            input_scale=args.input_scale,
            raw_th=args.raw_th,
            split_th=args.split_th,
            accept_prob_threshold=args.accept_prob_threshold,
        )
    )
    print(f"output_dir={result.output_dir}")
    print(f"candidate_count={result.harvest_metrics['candidate_count']}")
    print(f"accepted_count={result.harvest_metrics['accepted_count']}")
    print(f"clean_patch_count={result.clean_patch_count}")
    print(f"gallery={result.figures['clean_patch_gallery']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
