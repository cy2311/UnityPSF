#!/usr/bin/env python3
"""Verify one GPU DH vector/LUT batch before formal expert-parallel training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from unity_psf.localization.dh_raw_tiff import (
    DHDirectXYZLossAdapter,
    DHRawTiffBatchProviderConfig,
    DoubleHelixRuntimeModel,
    build_dh_raw_tiff_batch_provider,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DH preflight requires CUDA; CPU fallback is invalid")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["train"]
    dh_config = dict(config["dh_raw_tiff"])
    dh_config.pop("enabled", None)
    batch = next(iter(build_dh_raw_tiff_batch_provider(DHRawTiffBatchProviderConfig(**dh_config))(1)))
    model = DoubleHelixRuntimeModel(**dict(config["model"]["params"])).cuda().eval()
    with torch.no_grad():
        output = model(batch.inputs.cuda(non_blocking=True))
        loss = DHDirectXYZLossAdapter(dict(config["loss"].get("params", {}))).from_output(output, batch)
    if not torch.isfinite(loss):
        raise RuntimeError("DH vector/LUT preflight produced a non-finite direct-XYZ loss")
    print(json.dumps({"device": torch.cuda.get_device_name(0), "batch_shape": list(batch.inputs.shape), "loss": float(loss.item()), "source": batch.metadata["source"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
