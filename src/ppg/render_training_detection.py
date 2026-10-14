#!/usr/bin/env python3
"""Render one fixed online evaluation batch with decoded DH detections."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from ppg.infer_recon_sample import _load_model
from unity_psf.localization import build_localization_runtime_config, pixel_index_image_extent
from unity_psf.localization.data.online_types import OnlineBatchProviderConfig
from unity_psf.localization.data.online_provider import build_online_batch_provider
from unity_psf.localization.legacy_decode import (
    decode_legacy_targets,
    decode_liteloc_formal_infer_emitters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=104311)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    runtime = build_localization_runtime_config(
        config, config_base_dir=args.config.parent, seed=args.seed
    )
    provider_cfg = runtime["batch_provider"]
    params = dict(provider_cfg["params"])
    params.update({"batch_size": 1, "steps_per_epoch": 1, "seed": args.seed})
    batch = next(iter(build_online_batch_provider(OnlineBatchProviderConfig(**params))(1)))
    inputs = batch.inputs.model_input if hasattr(batch.inputs, "model_input") else batch.inputs
    if isinstance(inputs, (tuple, list)):
        inputs = inputs[0]
    device = torch.device(args.device)
    model, epoch = _load_model(args.checkpoint, device)
    with torch.no_grad():
        output = model(inputs.to(device))
    decoded = decode_liteloc_formal_infer_emitters(
        output, accept_threshold=0.3, photon_scale=31000.0, z_scale=0.6
    )
    targets = decode_legacy_targets(
        batch.inputs.pxyz_tar,
        batch.inputs.mask_tar,
        target_order="legacy_iwae",
        photon_scale=31000.0,
        z_scale=0.6,
    )
    image = inputs[0, 1].detach().cpu()
    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    height, width = image.shape[-2:]
    ax.imshow(
        image,
        cmap="magma",
        origin="upper",
        extent=pixel_index_image_extent(height, width),
        interpolation="nearest",
    )
    mask = decoded.batch_index == 0
    if mask.any():
        xy = decoded.xyz_px_nm[mask].detach().cpu()
        prob = decoded.probability[mask].detach().cpu()
        ax.scatter(xy[:, 0], xy[:, 1], s=42, facecolors="none", edgecolors="cyan", linewidths=1.2)
        for x, y, p in zip(xy[:, 0], xy[:, 1], prob):
            ax.text(float(x) + 1.5, float(y) - 1.5, f"{float(p):.2f}", color="white", fontsize=5)
    target_mask = targets.batch_index == 0
    if target_mask.any():
        gt = targets.xyz_px_nm[target_mask].detach().cpu()
        ax.scatter(gt[:, 0], gt[:, 1], s=34, marker="x", color="yellow", linewidths=1.0)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="cyan", markerfacecolor="none", linestyle="None", label="prediction"),
            plt.Line2D([], [], marker="x", color="yellow", linestyle="None", label="ground truth"),
        ],
        loc="upper right",
        fontsize=7,
    )
    ax.set_title(f"Training evaluation: raw + GMM detections and GT (epoch {epoch})")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(
        {
            "checkpoint_epoch": epoch,
            "detections": int(mask.sum()),
            "ground_truth": int(target_mask.sum()),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
