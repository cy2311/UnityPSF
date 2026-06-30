#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from neptune_v03.config import load_config
from neptune_v03.localization import build_localization_model_registry, build_localization_runtime_config
from neptune_v03.roi_library import ROIBank
from neptune_v03.runtime import ensure_run_layout
from neptune_v03.training import build_trainer_runtime
from neptune_v03.training.loop import load_training_checkpoint
from neptune_v03.training.run_high_fidelity import (
    _auto_build_roi_bank,
    _build_vector_roi_gamma_objective,
    _record_projection_tensors,
    _resolve_roi_bank_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a strict same-ROI raw vs initial vs latest physical-model projection triplet."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. neptune_v0.3/output/..._3198")
    parser.add_argument("--step", type=int, required=True, help="Gamma update step to match, e.g. 12500")
    parser.add_argument(
        "--roi-index",
        type=int,
        default=0,
        help="Index within the selected ROI/frame samples after filtering to selected_roi_ids.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output PNG path. Defaults under artifacts/roi_bank_gamma/step_xxxxxxxx/...",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary_path = (
        run_dir
        / "artifacts"
        / "roi_bank_gamma"
        / f"step_{int(args.step):08d}"
        / "source_auto_built"
        / "domain_multi"
        / "gamma_alternation_summary.json"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing gamma summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_roi_ids = {int(v) for v in summary.get("selected_roi_ids", [])}
    if not selected_roi_ids:
        raise ValueError(f"No selected_roi_ids in {summary_path}")

    manifest_path = run_dir / "metadata" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = _resolve_manifest_config_path(run_dir=run_dir, manifest_config_path=str(manifest["config_path"]))
    config = _config_with_current_physical_maps(load_config(config_path), run_dir=run_dir)
    train_cfg = _mapping(config.get("train"), "train")
    gamma_cfg = _mapping(train_cfg.get("roi_bank_gamma"), "train.roi_bank_gamma")
    config_base_dir = config_path.parent

    layout = ensure_run_layout(run_dir.parent, run_dir.name)
    runtime_config = build_localization_runtime_config(config, config_base_dir=config_base_dir, seed=int(manifest.get("seed", 0)))
    runtime = build_trainer_runtime(
        runtime_config,
        layout=layout,
        model_registry=build_localization_model_registry(),
    )
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_latest.pt"
    load_training_checkpoint(checkpoint_path, model=runtime.model, map_location=next(runtime.model.parameters()).device)

    roi_source = _resolve_roi_bank_source(
        gamma_cfg,
        train_cfg=train_cfg,
        config=config,
        config_base_dir=config_base_dir,
    )
    if roi_source is None:
        raise ValueError("Could not resolve ROI bank source from run config")

    bank = _auto_build_roi_bank(
        gamma_cfg,
        roi_source=roi_source,
        model=runtime.model,
        train_cfg=train_cfg,
    )
    selected_records = tuple(record for record in bank.records if int(record.roi_id) in selected_roi_ids)
    if not selected_records:
        raise ValueError("Rebuilt ROI bank did not contain any selected ROI ids from the target gamma step")
    selected_bank = ROIBank(
        records=selected_records,
        config=bank.config,
        metadata=bank.metadata,
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    )

    raw_frames, background, samples, roi_origin_xy_px, domain_names = _record_projection_tensors(selected_bank)
    roi_index = int(args.roi_index)
    if not (0 <= roi_index < int(raw_frames.shape[0])):
        raise IndexError(f"roi-index {roi_index} out of range [0, {int(raw_frames.shape[0]) - 1}]")

    objective = _build_vector_roi_gamma_objective(gamma_cfg, train_cfg=train_cfg, config=config, model=runtime.model)
    initial_gamma = objective.initial_gamma().detach()

    latest_feedback = summary.get("feedback_coeff_maps")
    if not isinstance(latest_feedback, dict) or not latest_feedback:
        raise ValueError(f"Missing feedback_coeff_maps in {summary_path}")
    latest_base_maps = tuple(
        (str(name), str((run_dir / Path(path_str)).resolve()) if not Path(path_str).is_absolute() else str(Path(path_str)))
        for name, path_str in sorted(latest_feedback.items())
    )
    latest_objective = _build_vector_roi_gamma_objective(
        {**gamma_cfg, "base_coeff_maps": latest_base_maps},
        train_cfg=train_cfg,
        config=config,
        model=runtime.model,
    )
    latest_gamma = latest_objective.initial_gamma().detach()

    raw_frame = raw_frames[roi_index].detach().cpu()
    initial_recon = objective.render_reconstruction(
        gamma=initial_gamma,
        samples=samples,
        background=background,
        batch_index=roi_index,
        roi_origin_xy_px=roi_origin_xy_px,
        domain_names=domain_names,
    )
    latest_recon = latest_objective.render_reconstruction(
        gamma=latest_gamma,
        samples=samples,
        background=background,
        batch_index=roi_index,
        roi_origin_xy_px=roi_origin_xy_px,
        domain_names=domain_names,
    )

    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else run_dir
        / "artifacts"
        / "roi_bank_gamma"
        / f"step_{int(args.step):08d}"
        / "source_auto_built"
        / "domain_multi"
        / f"strict_roi_triplet_idx_{roi_index:03d}.png"
    )
    _write_triplet_png(output_path, raw_frame=raw_frame, initial_recon=initial_recon, latest_recon=latest_recon)

    meta_path = output_path.with_suffix(".json")
    payload = {
        "schema_version": "strict_roi_triplet.v1",
        "run_dir": str(run_dir),
        "step": int(args.step),
        "roi_index": roi_index,
        "selected_roi_id_count": len(selected_roi_ids),
        "sample_count": int(raw_frames.shape[0]),
        "domain_name": str(domain_names[roi_index]),
        "roi_origin_xy_px": [float(v) for v in roi_origin_xy_px[roi_index].tolist()],
        "output_png": str(output_path),
        "initial_base_maps": dict(objective.config.base_coeff_maps),
        "latest_feedback_maps": {k: str(v) for k, v in latest_feedback.items()},
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _write_triplet_png(path: Path, *, raw_frame: torch.Tensor, initial_recon: torch.Tensor, latest_recon: torch.Tensor) -> None:
    raw_u8 = _to_uint8(raw_frame)
    initial_u8 = _to_uint8(initial_recon)
    latest_u8 = _to_uint8(latest_recon)
    canvas = np.concatenate(
        [
            _gray_to_rgb(raw_u8),
            _gray_to_rgb(initial_u8),
            _gray_to_rgb(latest_u8),
        ],
        axis=1,
    )
    _write_rgb_png(path, canvas)


def _to_uint8(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    array = np.asarray(frame, dtype=np.float32)
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(gray, dtype=np.uint8)[..., None], 3, axis=2)


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width = int(image.shape[0]), int(image.shape[1])
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw_rows)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _resolve_manifest_config_path(*, run_dir: Path, manifest_config_path: str) -> Path:
    path = Path(manifest_config_path)
    if path.is_file():
        return path.resolve()
    alt = run_dir.parents[1] / ".local" / "tmp" / "standard" / path.name
    if alt.is_file():
        return alt.resolve()
    alt2 = run_dir.parents[2] / ".local" / "tmp" / "standard" / path.name
    if alt2.is_file():
        return alt2.resolve()
    raise FileNotFoundError(
        f"Could not resolve config path from manifest: {manifest_config_path} "
        f"(tried {path}, {alt}, {alt2})"
    )


def _config_with_current_physical_maps(config: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    coeff_maps = _current_physical_coeff_maps(run_dir)
    if not coeff_maps:
        return config
    updated = dict(config)
    train_cfg = dict(_mapping(updated.get("train"), "train"))
    online_cfg = dict(_mapping(train_cfg.get("online_generation"), "train.online_generation"))
    online_cfg["dual_domain_coeff_maps"] = coeff_maps
    train_cfg["online_generation"] = online_cfg
    updated["train"] = train_cfg
    return updated


def _current_physical_coeff_maps(run_dir: Path):
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_latest.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        entries = checkpoint.get("physical_coeff_maps")
        if entries:
            return entries
    state_path = run_dir / "metadata" / "current_physical_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        entries = state.get("coeff_maps")
        if entries:
            return entries
    manifest_path = run_dir / "metadata" / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("current_physical_coeff_maps")
        if entries:
            return entries
    return None


if __name__ == "__main__":
    raise SystemExit(main())
