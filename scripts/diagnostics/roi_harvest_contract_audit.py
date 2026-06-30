from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[3]
V03_ROOT = REPO_ROOT / "neptune_v0.3"
IWAE_ROOT = REPO_ROOT / "neptune_iwae"


def _install_import_paths() -> None:
    for path in reversed((str(V03_ROOT / "src"), str(IWAE_ROOT))):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "p01": float(np.nanpercentile(arr, 1)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p99": float(np.nanpercentile(arr, 99)),
    }


def _resolve_raw_tiff(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_file():
        return raw
    if raw.is_dir():
        candidates = sorted(
            list(raw.glob("*.tif"))
            + list(raw.glob("*.tiff"))
            + list(raw.glob("*.ome.tif"))
            + list(raw.glob("*.ome.tiff"))
        )
        if not candidates:
            raise FileNotFoundError(f"No TIFF found under {raw}")
        return candidates[0]
    raise FileNotFoundError(str(raw))


def audit_normalization(*, config_path: Path, raw_path: Path, frame_start: int, crop: tuple[int, int, int, int]) -> dict[str, Any]:
    from Normalization import build_inference_frame_normalizer
    from neptune_v03.roi_library.loc_harvest import _normalize_input

    cfg = _load_config(config_path)
    x0, y0, width, height = crop
    raw_tiff = _resolve_raw_tiff(raw_path)
    with tifffile.TiffFile(str(raw_tiff)) as tif:
        frames = np.asarray(tif.series[0].asarray(key=range(frame_start, frame_start + 3)), dtype=np.float32)
    frames = np.squeeze(frames)
    if frames.ndim == 2:
        frames = frames[None]
    tile = torch.as_tensor(frames[:, y0 : y0 + height, x0 : x0 + width], dtype=torch.float32)
    train_cfg = cfg.get("train") or {}
    scaling = train_cfg.get("scaling") or {}
    simple = _normalize_input(
        tile,
        offset=float(scaling.get("input_offset", 0.0)),
        scale=float(scaling.get("input_scale", 1.0)),
    )
    frame_proc = build_inference_frame_normalizer(cfg)
    fd = frame_proc.forward(tile).to(dtype=torch.float32)
    return {
        "raw_tiff": str(raw_tiff),
        "config_path": str(config_path),
        "normalization_config": train_cfg.get("normalization") or {},
        "old_frame_proc_type": type(frame_proc).__name__,
        "old_frame_proc_mode": frame_proc.to_dict() if hasattr(frame_proc, "to_dict") else type(frame_proc).__name__,
        "v03_current_input": "simple_offset_scale",
        "crop_xywh": [int(x0), int(y0), int(width), int(height)],
        "frame_window": [int(frame_start), int(frame_start + 3)],
        "raw_adu": _stats(tile),
        "v03_simple_input": _stats(simple),
        "old_fd_recenter_input": _stats(fd),
        "old_minus_v03": _stats(fd - simple),
        "mean_abs_delta": float((fd - simple).abs().mean().item()),
    }


def audit_spatial_thresholds() -> dict[str, Any]:
    from neptune_v03.roi_library.loc_harvest import _spatial_integration

    p = torch.zeros((1, 9, 9), dtype=torch.float32)
    p[0, 2, 2] = 0.34
    p[0, 2, 3] = 0.31
    p[0, 4, 4] = 0.46
    p[0, 4, 5] = 0.42
    p[0, 6, 6] = 0.55
    p[0, 6, 7] = 0.48
    p[0, 1, 6] = 0.29
    old_style = _spatial_integration(p, raw_th=0.3, split_th=0.6)
    mixed_v03 = _spatial_integration(p, raw_th=0.5, split_th=0.6)
    return {
        "synthetic_p_values": [float(v) for v in p[p > 0].tolist()],
        "old_style": {
            "spatial_raw_th": 0.3,
            "accept_threshold": 0.5,
            "integrated_ge_raw": int((old_style >= 0.3).sum().item()),
            "integrated_ge_accept": int((old_style >= 0.5).sum().item()),
            "max": float(old_style.max().item()),
        },
        "v03_current_mixed": {
            "spatial_raw_th": 0.5,
            "accept_threshold": 0.5,
            "integrated_ge_raw": int((mixed_v03 >= 0.5).sum().item()),
            "integrated_ge_accept": int((mixed_v03 >= 0.5).sum().item()),
            "max": float(mixed_v03.max().item()),
        },
        "candidate_delta_ge_accept": int((mixed_v03 >= 0.5).sum().item() - (old_style >= 0.5).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit real TIFF ROI harvest input and threshold contracts.")
    parser.add_argument(
        "--config",
        default=str(V03_ROOT / ".local/tmp/parity/resolved_v03_current_route_epoch30_3174.yaml"),
    )
    parser.add_argument(
        "--raw",
        default=os.environ.get("NEPTUNE_V03_RAW_TIFF_PATH", ""),
    )
    parser.add_argument(
        "--output",
        default=str(V03_ROOT / ".local/tmp/diagnostics/roi_harvest_contract_audit.json"),
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--crop", nargs=4, type=int, default=(0, 0, 128, 128), metavar=("X", "Y", "W", "H"))
    args = parser.parse_args()

    _install_import_paths()
    report = {
        "normalization": audit_normalization(
            config_path=Path(args.config),
            raw_path=Path(args.raw),
            frame_start=int(args.frame_start),
            crop=tuple(int(v) for v in args.crop),
        ),
        "spatial_thresholds": audit_spatial_thresholds(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
