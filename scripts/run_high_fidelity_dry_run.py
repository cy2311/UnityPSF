from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neptune_v03.config import materialize_config
from neptune_v03.training.run_high_fidelity import main as run_high_fidelity_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny high-fidelity raw-TIFF ROI gamma dry run.")
    parser.add_argument("--raw-tiff", type=Path, default=None, help="Optional real raw TIFF. Defaults to a synthetic tiny TIFF.")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / ".local" / "tmp" / "high_fidelity_dry_run")
    parser.add_argument("--run-name", default="synthetic_tiff_smoke")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=None)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=None)
    parser.add_argument("--crop-height", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="Torch device for the dry run, e.g. cpu or cuda:0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = args.run_root / "_inputs"
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_tiff.resolve() if args.raw_tiff is not None else _write_synthetic_tiff(work_dir / "synthetic_raw.tif")
    override_path = work_dir / "dry_run_override.yaml"
    resolved_path = work_dir / "resolved_dry_run.yaml"
    maps_dir = work_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    left_map, right_map = _write_tiny_coeff_maps(maps_dir)

    frames_shape = _tiff_shape(raw_path)
    crop_height = min(int(frames_shape[-2]) - int(args.crop_top), 12) if args.crop_height is None else int(args.crop_height)
    crop_width = min(int(frames_shape[-1]) - int(args.crop_left), 12) if args.crop_width is None else int(args.crop_width)
    frame_stop = min(int(frames_shape[0]), 3) if args.frame_stop is None else int(args.frame_stop)
    if frame_stop - int(args.frame_start) < 3:
        raise ValueError(
            f"dry run requires at least 3 TIFF frames, got range [{int(args.frame_start)}, {frame_stop}): {raw_path}"
        )

    override_path.write_text(
        yaml.safe_dump(
            _override_payload(
                raw_path=raw_path,
                crop_left=int(args.crop_left),
                crop_top=int(args.crop_top),
                crop_height=crop_height,
                crop_width=crop_width,
                frame_start=int(args.frame_start),
                frame_stop=frame_stop,
                epochs=args.epochs,
                device=str(args.device),
                left_map=left_map,
                right_map=right_map,
            )
        ),
        encoding="utf-8",
    )
    resolved = materialize_config(PROJECT_ROOT / "configs" / "microtube_base.yaml", override_path)
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    old_argv = sys.argv[:]
    sys.argv = [
        "neptune-v03-train-high-fidelity",
        "--config",
        str(resolved_path),
        "--run-root",
        str(args.run_root),
        "--run-name",
        args.run_name,
        "--seed",
        str(args.seed),
    ]
    try:
        exit_code = run_high_fidelity_main()
    finally:
        sys.argv = old_argv

    run_dir = args.run_root / args.run_name
    status_path = run_dir / "metadata" / "stage_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        print(json.dumps(status.get("high_fidelity_localization", {}), indent=2, sort_keys=True))
    print(f"run_dir={run_dir}")
    print(f"resolved_config={resolved_path}")
    return int(exit_code)


def _write_synthetic_tiff(path: Path) -> Path:
    frames = np.full((3, 12, 12), 0.2, dtype=np.float32)
    frames[:, 4, 4] = 9.0
    frames[:, 7, 7] = 8.0
    tifffile.imwrite(path, frames, photometric="minisblack")
    return path.resolve()


def _write_tiny_coeff_maps(path: Path) -> tuple[Path, Path]:
    mode_order = np.asarray([(2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0), (3, -3), (3, 3)], dtype=np.int64)
    left = path / "left.npz"
    right = path / "right.npz"
    np.savez_compressed(left, zernike_maps_nm=np.ones((8, 8, 8), dtype=np.float32), mode_order=mode_order)
    np.savez_compressed(right, zernike_maps_nm=np.ones((8, 8, 8), dtype=np.float32) * 2.0, mode_order=mode_order)
    return left.resolve(), right.resolve()


def _tiff_shape(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"TIFF has no image series: {path}")
        shape = tuple(int(v) for v in tif.series[0].shape)
    if len(shape) == 2:
        return 1, int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise ValueError(f"expected raw TIFF stack shape (T,H,W), got {shape}: {path}")


def _override_payload(
    *,
    raw_path: Path,
    crop_left: int,
    crop_top: int,
    crop_height: int,
    crop_width: int,
    frame_start: int,
    frame_stop: int,
    epochs: int,
    device: str,
    left_map: Path,
    right_map: Path,
) -> dict:
    return {
        "localization_overrides": {
            "depth_shared": 1,
            "depth_union": 1,
            "nfeatures_init": 4,
            "nfeatures_inter": 4,
            "film_hidden_dim": 8,
            "z_mu_activation": "tanh",
        },
        "train": {
            "device": str(device),
            "epochs": int(epochs),
            "batch_size": 1,
            "learning_rate": 0.001,
            "online_generation": {
                "enabled": True,
                "steps_per_epoch": 1,
                "height": 8,
                "width": 8,
                "channels": 3,
                "emitters_per_sample": 1,
                "signal": 10.0,
                "background": 0.1,
                "conditioning_mode": "film",
                "expert_mode": "soft_moe",
                "condition_feature_dim": 8,
                "condition_dim": 10,
                "domain_count": 2,
                "append_domain_onehot": True,
                "dual_domain_coeff_maps": [
                    {"name": "left", "coeff_maps_npz": str(left_map)},
                    {"name": "right", "coeff_maps_npz": str(right_map)},
                ],
            },
            "real_tiff_wake": {
                "enabled": True,
                "tiff_path": str(raw_path),
                "domains": [
                    {
                        "name": "left",
                        "crop_left": int(crop_left),
                        "crop_top": int(crop_top),
                        "crop_width": int(crop_width),
                        "crop_height": int(crop_height),
                    }
                ],
            },
            "roi_bank_gamma": {
                "enabled": True,
                "start_epoch": 1,
                "stop_epoch": int(epochs),
                "update_interval_epochs": 1,
                "gamma_steps": 2,
                "gamma_lr": 0.05,
                "num_posterior_samples": 2,
                "roi_size_px": min(8, int(crop_height), int(crop_width)),
                "roi_bank_frame_range": [int(frame_start), int(frame_stop)],
                "roi_bank_grid_shape": [2, 2],
                "roi_library_max_rois": 2,
                "target_projected_emitters": 4,
                "roi_bank_over_cut_px": 1,
                "probability_threshold": 0.0,
                "roi_bank_probability_threshold": 0.0,
                "auto_heldout_min_rois": 0,
                "auto_heldout_max_rois": 0,
            },
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
