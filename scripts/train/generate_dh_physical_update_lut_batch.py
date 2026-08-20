#!/usr/bin/env python3
"""Materialize a reproducible DH vector/LUT batch from a gamma-feedback map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from unity_psf.localization.data.online import OnlineBatchProviderConfig, build_online_batch_provider
from unity_psf.localization.runtime.config import build_localization_runtime_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _updated_map(state_path: Path) -> Path:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("source") != "gamma_feedback":
        raise ValueError("DH LUT generation requires gamma_feedback physical state")
    entries = state.get("coeff_maps", [])
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("DH physical state must contain exactly one coefficient map")
    value = entries[0].get("coeff_maps_npz") if isinstance(entries[0], dict) else None
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DH physical-state coefficient map is missing: {path}")
    return path


def _online_params(source_config: Path, coeff_map: Path, *, batch_size: int, steps: int, seed: int) -> dict[str, object]:
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    train = config["train"]
    online = dict(train["online_generation"])
    online.update(
        {
            "simulation_backend": "lut",
            "simulation_output_device": "renderer",
            "domain_count": 1,
            "dual_domain_coeff_maps": [{"name": "double_helix", "coeff_maps_npz": str(coeff_map)}],
            "batch_strategy": "triplet",
            "steps_per_epoch": int(steps),
        }
    )
    train["batch_size"] = int(batch_size)
    train["online_generation"] = online
    runtime = build_localization_runtime_config(config, config_base_dir=source_config.parent, seed=int(seed))
    provider = runtime["batch_provider"]
    if provider["name"] != "online_train_batch":
        raise ValueError("DH source config did not resolve to the online simulator")
    params = dict(provider["params"])
    if params["simulation_backend"] != "lut" or params["psf_type"] != "vector":
        raise ValueError("DH batch must resolve to vector/LUT simulation")
    return params


def _write_batch(output: Path, params: dict[str, object], *, steps: int) -> dict[str, object]:
    provider = build_online_batch_provider(OnlineBatchProviderConfig(**params))
    frames: list[np.ndarray] = []
    detection: list[np.ndarray] = []
    background: list[np.ndarray] = []
    emitters: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    batch_metadata: list[dict[str, object]] = []
    for item in provider(1):
        batch = item.inputs
        image = batch.model_input[0] if isinstance(batch.model_input, tuple) else batch.model_input
        frames.append(image.detach().cpu().numpy().astype(np.float32, copy=False))
        detection.append(batch.detect_tar.detach().cpu().numpy().astype(np.float32, copy=False))
        background.append(batch.bkg_tar.detach().cpu().numpy().astype(np.float32, copy=False))
        emitters.append(batch.pxyz_tar.detach().cpu().numpy().astype(np.float32, copy=False))
        masks.append(batch.mask_tar.detach().cpu().numpy().astype(bool, copy=False))
        batch_metadata.append(dict(batch.metadata))
    if len(frames) != int(steps):
        raise RuntimeError("DH simulator returned an unexpected number of batches")
    frames_adu = np.concatenate(frames, axis=0)
    max_emitters = max(int(value.shape[1]) for value in emitters)
    emitter_xy_z_um_photons = np.concatenate(
        [
            np.pad(value, ((0, 0), (0, max_emitters - value.shape[1]), (0, 0)))
            for value in emitters
        ],
        axis=0,
    )
    emitter_mask = np.concatenate(
        [np.pad(value, ((0, 0), (0, max_emitters - value.shape[1]))) for value in masks],
        axis=0,
    )
    sample_count, max_emitters, _ = emitter_xy_z_um_photons.shape
    height, width = frames_adu.shape[-2:]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames_adu=frames_adu,
        detect_target=np.concatenate(detection, axis=0),
        background_target=np.concatenate(background, axis=0),
        emitter_xy_z_um_photons=emitter_xy_z_um_photons,
        emitter_mask=emitter_mask,
        dh_lobe_targets=np.zeros((sample_count, max_emitters * 2, 3), dtype=np.float32),
        dh_lobe_mask=np.zeros((sample_count, max_emitters * 2), dtype=bool),
    )
    return {
        "sample_count": int(sample_count),
        "frame_shape": [int(value) for value in frames_adu.shape[1:]],
        "max_emitters": int(max_emitters),
        "target_order": ["x_px", "y_px", "z_um", "photons"],
        "batch_metadata": batch_metadata,
        "height": int(height),
        "width": int(width),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--physical-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=4311)
    args = parser.parse_args()

    state_path = args.physical_state.resolve()
    coeff_map = _updated_map(state_path)
    params = _online_params(args.source_config.resolve(), coeff_map, batch_size=args.batch_size, steps=args.steps, seed=args.seed)
    result = _write_batch(args.output.resolve(), params, steps=args.steps)
    metadata = {
        "source": "dh_vector_lut_simulation_from_physical_update",
        "physical_state_path": str(state_path),
        "physical_state_source": "gamma_feedback",
        "coefficient_map": str(coeff_map),
        "coefficient_map_sha256": _sha256(coeff_map),
        "simulation_backend": "lut",
        "psf_type": "vector",
        "channels": int(params["channels"]),
        "pupil_carrier": "configured_from_source_dh_contract",
        "online_params": {key: value for key, value in params.items() if key != "pupil_carrier_complex"},
        **result,
    }
    metadata_path = args.output.with_name("simulated_training_batch.metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"batch": str(args.output.resolve()), "metadata": str(metadata_path), **metadata}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
