#!/usr/bin/env python3
"""Run full-frame DH inference and standard infer_recon rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import tifffile
import torch

from unity_psf.infer_recon.predictions_io import H5PredictionWriter
from ppg.lunar_background import estimate_lunar_tiff_background
from ppg.condition_input import load_ppg_conditioning, ppg_model_input
from unity_psf.localization.legacy_decode import decode_liteloc_formal_infer_emitters
from unity_psf.data.normalization import LunarInputNormalization
from unity_psf.models.psf_moe.experts.double_helix import DoubleHelixExpert


FIELDS = ("frame", "x_px", "y_px", "z", "photon", "prob", "sample_tiff")


def _load_model(
    checkpoint: Path, device: torch.device
) -> tuple[DoubleHelixExpert, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    progress = payload["training_state"]["double_helix"]
    expert = payload["experts"]["double_helix"]
    model_config = expert.get("model_config", {})
    model = DoubleHelixExpert(**dict(model_config))
    if model.input_normalization is None:
        raise RuntimeError("checkpoint does not contain LUNAR input normalization")
    model.load_state_dict(expert["model_state_dict"])
    return model.eval().to(device), int(progress["epoch"])


def _decode_batch(
    output: torch.Tensor, *, frame_ids: list[int], threshold: float, sample_tiff: Path
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    emitters = decode_liteloc_formal_infer_emitters(
        output,
        accept_threshold=float(threshold),
        photon_scale=31000.0,
        z_scale=0.6,
    )
    edge = 13
    height, width = output.shape[-2:]
    for index in range(len(emitters.probability)):
        batch_index = int(emitters.batch_index[index])
        x, y, z_nm = (float(value) for value in emitters.xyz_px_nm[index])
        if x < edge or x >= width - edge or y < edge or y >= height - edge:
            continue
        rows.append(
            {
                "frame": frame_ids[batch_index],
                "x_px": x,
                "y_px": y,
                "z": z_nm / 1000.0,
                "photon": float(emitters.photons[index]),
                "prob": float(emitters.probability[index]),
                "sample_tiff": str(sample_tiff),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-tiff", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--prob-threshold", type=float, default=0.3)
    parser.add_argument("--render-prob-threshold", type=float, default=0.7)
    parser.add_argument(
        "--background-percentile",
        type=float,
        default=10.0,
        help="LUNAR darkest temporal-mean percentile used for this inference run",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_tiff = args.sample_tiff.resolve()
    device = torch.device(args.device)
    model, checkpoint_epoch = _load_model(args.checkpoint.resolve(), device)
    ppg_map_path = model.model_config.get("ppg_map_path")
    condition_provider = load_ppg_conditioning(ppg_map_path) if ppg_map_path else None
    estimated_background = estimate_lunar_tiff_background(
        sample_tiff,
        percentile=float(args.background_percentile),
        max_size_gb=2.0,
        chunk_frames=64,
    )
    checkpoint_normalization = model.input_normalization
    if checkpoint_normalization is None:
        raise RuntimeError("checkpoint does not contain LUNAR input normalization")
    estimated_range = np.asarray(estimated_background.background_range, dtype=np.float64)
    checkpoint_range = np.asarray(checkpoint_normalization.background_range, dtype=np.float64)
    checkpoint_matches_estimate = bool(
        np.allclose(estimated_range, checkpoint_range, rtol=0.0, atol=1e-4)
    )
    # Keep the LUNAR photon contract learned by the checkpoint, while allowing
    # explicit darkest-3/5/7 diagnostic runs to use their own measured bg mean.
    model.input_normalization = LunarInputNormalization(
        background_range=tuple(float(value) for value in estimated_range),
        photon_range=checkpoint_normalization.photon_range,
    )
    predictions = output_dir / "predictions_merged.h5"
    total_rows = 0
    with tifffile.TiffFile(sample_tiff) as handle, H5PredictionWriter(
        predictions, fieldnames=FIELDS
    ) as writer:
        frame_count, height, width = (int(v) for v in handle.series[0].shape)
        for center_start in range(1, frame_count - 1, int(args.chunk_frames)):
            center_stop = min(frame_count - 1, center_start + int(args.chunk_frames))
            raw = np.asarray(
                handle.asarray(key=range(center_start - 1, center_stop + 1)),
                dtype=np.float32,
            )
            windows = np.stack(
                [
                    raw[index : index + 3]
                    for index in range(center_stop - center_start)
                ]
            )
            for batch_start in range(0, len(windows), int(args.batch_size)):
                batch_stop = min(len(windows), batch_start + int(args.batch_size))
                frame_ids = list(
                    range(center_start + batch_start, center_start + batch_stop)
                )
                inputs = torch.as_tensor(
                    windows[batch_start:batch_stop], dtype=torch.float32, device=device
                )
                inputs = ppg_model_input(
                    inputs,
                    provider=condition_provider,
                    origin_xy_px=(0, 0),
                    source_field_shape_yx=(height, width),
                )
                with torch.no_grad():
                    output = model(inputs)
                rows = _decode_batch(
                    output,
                    frame_ids=frame_ids,
                    threshold=float(args.prob_threshold),
                    sample_tiff=sample_tiff,
                )
                writer.append_rows(rows)
                total_rows += len(rows)

    runtime_state = {
        "crop_width": width,
        "crop_height": height,
        "pixel_size_nm_x": 125.211897,
        "pixel_size_nm_y": 125.211897,
        "camera_pixel_nm_x": 125.211897,
        "camera_pixel_nm_y": 125.211897,
    }
    (output_dir / "derived_runtime_state.json").write_text(
        json.dumps(runtime_state, indent=2) + "\n"
    )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "sample_tiff": str(sample_tiff),
        "frame_count": frame_count,
        "processed_center_frames": frame_count - 2,
        "prediction_count": total_rows,
        "prob_threshold": float(args.prob_threshold),
        "input_intensity_contract": "raw_identity_camera_then_lunar_normalization",
        "input_normalization": (
            model.input_normalization.to_config()
            if model.input_normalization is not None
            else None
        ),
        "lunar_background_estimate": {
            "background_range": list(estimated_background.background_range),
            "background_mean": estimated_background.background_mean,
            "frame_range": list(estimated_background.frame_range),
            "percentile": estimated_background.percentile,
            "estimator": "lunar_get_bg_stats_gauss_global_temporal_mean_dark_pixels",
            "matches_checkpoint": checkpoint_matches_estimate,
            "checkpoint_background_range": list(checkpoint_normalization.background_range),
        },
        "predictions": str(predictions),
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            "infer_recon/recon/render_standard.py",
            "--infer-dir",
            str(output_dir),
            "--output-dir",
            str(output_dir / "reconstruction"),
            "--runtime-state",
            str(output_dir / "derived_runtime_state.json"),
            "--prob-threshold",
            str(args.render_prob_threshold),
            "--z-min",
            "-0.6",
            "--z-max",
            "0.6",
            "--suffix",
            "ppg_dh_epoch300_full15000",
        ],
        check=True,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
