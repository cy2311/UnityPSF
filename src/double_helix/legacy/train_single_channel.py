from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def build_single_channel_training_config(
    *,
    reference_config_path: str | Path,
    zmap_path: str | Path,
    carrier_path: str | Path,
    roi_tiff_path: str | Path,
    density_um2: float,
    seed: int,
) -> dict[str, Any]:
    reference = Path(reference_config_path).resolve()
    zmap = Path(zmap_path).resolve()
    carrier = Path(carrier_path).resolve()
    roi_tiff = Path(roi_tiff_path).resolve()
    if float(density_um2) <= 0.0:
        raise ValueError("density_um2 must be positive")
    for path in (reference, zmap, carrier, roi_tiff):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(zmap, allow_pickle=False) as payload:
        mode_order = np.asarray(payload["mode_order"], dtype=np.int64)
        maps_shape = tuple(int(value) for value in payload["zernike_maps_nm"].shape)
    if maps_shape[0] != 22 or tuple(mode_order[0]) != (2, 0):
        raise ValueError("DH training requires zero defocus plus exactly 21 residual Zernike maps")

    config = _load_yaml(reference)
    config["optical"].update({"NA": 1.27, "n_medium": 1.33})
    simulation = config["simulation"]
    simulation["background_uniform"] = [110.0, 110.0]
    simulation["emitter"].update(
        {
            "emitter_extent": [[-0.5, 95.5], [-0.5, 95.5], [-1.9647, 1.9647]],
            "z_range": [-1.9647, 1.9647],
            "intensity_mu_sig": [20000.0, 1000.0],
            "intensity_clip": [0.0, 31000.0],
        }
    )
    simulation["psf"]["vector"].update(
        {
            "npupil": 128,
            "psf_size": 63,
            "refmed": 1.33,
            "refcov": 1.33,
            "refimm": 1.33,
            "zemit0": 1998.0,
        }
    )
    simulation.setdefault("simulation", {})["random_seed"] = int(seed)

    train = config["train"]
    train["epochs"] = 300
    train["batch_size"] = 24
    train["scaling"].update({"z_max": 1.9647, "photon_max": 31000.0})
    online = train["online_generation"]
    online.update(
        {
            "steps_per_epoch": 417,
            "sequence_count": 16,
            "cached_window_max_gpu_sequences": 1,
            "domain_count": 1,
            "append_domain_onehot": True,
            "domain_balance_mode": "fixed",
            "condition_feature_dim": 13,
            "condition_dim": 14,
            "emitter_density_um2": float(density_um2),
            "dual_domain_coeff_maps": [{"name": "double_helix", "coeff_maps_npz": str(zmap)}],
            "pupil_carrier_complex_npz": str(carrier),
            "NA": 1.27,
            "refmed": 1.33,
            "refcov": 1.33,
            "refimm": 1.33,
            "npupil": 128,
            "vector_psf_size": 63,
            "zemit0": 1998.0,
            "z_range": [-1.9647, 1.9647],
            "nat_grid_z_steps": 131,
            "field_origin_sampling_mode": "sliding_window",
            "field_origin_stride_px": 88,
        }
    )
    online["lut_simulation"].update(
        {
            "psf_size": 63,
            "z_steps": 131,
            "field_mode": "global_field",
            "storage_dtype": "fp16",
        }
    )

    gamma = train["roi_bank_gamma"]
    gamma.update(
        {
            "enabled": True,
            "start_epoch": 30,
            "update_interval_epochs": 10,
            "stop_epoch": 300,
            "nat_config_kind": "order1_21",
            "pupil_carrier_complex_npz": str(carrier),
            "preserve_full_base_modes": True,
            "NA": 1.27,
            "refmed": 1.33,
            "refcov": 1.33,
            "refimm": 1.33,
            "npupil": 128,
            "patch_size_px": 63,
            "zemit0": 1998.0,
            "image_size_x": maps_shape[2],
            "image_size_y": maps_shape[1],
            "pixel_size_x_nm": 101.11,
            "pixel_size_y_nm": 98.83,
            "roi_bank_source": {
                "mode": "loc_infer_raw_tiff",
                "raw_path": str(roi_tiff),
                "frame_range": [0, 100],
                "candidate_mode": "dense_tile_temporal",
                "domains": [
                    {
                        "name": "double_helix",
                        "crop_left": 0,
                        "crop_top": 0,
                        "crop_width": 96,
                        "crop_height": 96,
                    }
                ],
            },
        }
    )
    train["peak_zmap_bootstrap"]["enabled"] = False
    train["joint_training"]["enabled"] = False
    train["real_tiff_wake"].update(
        {
            "enabled": False,
            "tiff_path": str(roi_tiff),
            "domains": gamma["roi_bank_source"]["domains"],
        }
    )
    train["eval"]["enabled"] = True
    config["phase_retrieval"]["input"]["tiff_path"] = str(roi_tiff)
    config.setdefault("metadata", {}).update(
        {
            "training_mode": "double_helix_neptune_v03_single_channel_lut63",
            "psf_model": "independent shared DH carrier + 21 FD residual Zernike maps",
            "dual_channel_enabled": False,
            "domain_count": 1,
            "emitter_density_um2": float(density_um2),
            "epochs": 300,
            "steps_per_epoch": 417,
            "gamma_update_start_epoch": 30,
            "gamma_update_interval_epochs": 10,
            "photon_mean": 20000.0,
            "photon_std": 1000.0,
            "background_photons": 110.0,
            "seed": int(seed),
        }
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-channel Neptune v0.4 DH training config.")
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--zmap", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--roi-tiff", type=Path, required=True)
    parser.add_argument("--density-um2", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output-config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_single_channel_training_config(
        reference_config_path=args.reference_config,
        zmap_path=args.zmap,
        carrier_path=args.carrier,
        roi_tiff_path=args.roi_tiff,
        density_um2=args.density_um2,
        seed=args.seed,
    )
    output = args.output_config.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"resolved_config": str(output), "density_um2": args.density_um2}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
