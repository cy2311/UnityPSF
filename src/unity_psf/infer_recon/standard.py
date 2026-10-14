from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STANDARD_ROI_SIZE = 128
STANDARD_VALID_ROI_SIZE = 108
STANDARD_TILING_MODE = "edgecover"
STANDARD_TILE_SHIFTS = "0,0"
STANDARD_BATCH_SIZE = 256
STANDARD_FRAME_BLOCK = 128
STANDARD_MAX_FRAMES = 8000
STANDARD_EARLY_PROB_THRESHOLD = 0.70
STANDARD_INFER_AMP = True
STANDARD_STREAM_OUTPUT = True
STANDARD_FD_PADDING_PX = 20
STANDARD_CONDITIONING_MODE = "zmap_spatial_film"
STANDARD_ALLOWED_CONDITIONING_MODES = {"zmap_spatial_film", "film"}
RECENTER_INFER_MODES = {
    "fd_deeploc_exact_recenter",
    "fd_deeploc_style",
    "fd-style",
    "fd_style",
    "fd_deeploc",
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_standard_geometry(*, roi_size: int, valid_roi_size: int | None, tiling_mode: str) -> None:
    if int(roi_size) != STANDARD_ROI_SIZE:
        raise ValueError(f"standard infer requires roi_size={STANDARD_ROI_SIZE}, got {roi_size}")
    if valid_roi_size is None:
        raise ValueError(f"standard infer requires valid_roi_size={STANDARD_VALID_ROI_SIZE}; do not omit it")
    if int(valid_roi_size) != STANDARD_VALID_ROI_SIZE:
        raise ValueError(f"standard infer requires valid_roi_size={STANDARD_VALID_ROI_SIZE}, got {valid_roi_size}")
    if str(tiling_mode) != STANDARD_TILING_MODE:
        raise ValueError(f"standard infer requires tiling_mode={STANDARD_TILING_MODE!r}, got {tiling_mode!r}")


def validate_recenter_normalization(config: dict[str, Any]) -> None:
    train = config.get("train") or {}
    norm = train.get("normalization") or {}
    mode = str(norm.get("infer_mode", norm.get("mode", ""))).lower()
    if mode not in RECENTER_INFER_MODES:
        allowed = ", ".join(sorted(RECENTER_INFER_MODES))
        raise ValueError(
            "standard infer requires FD-DeepLoc-style recenter normalization via "
            f"train.normalization.infer_mode or mode; got {mode!r}. Allowed: {allowed}"
        )
    required = ["baseline_adu", "e_per_adu", "train_background_adu", "qe", "em_gain", "spurious_charge", "photon_scale"]
    missing = [key for key in required if norm.get(key) is None]
    if missing:
        raise ValueError(f"standard recenter normalization is missing required keys: {missing}")


def validate_continuous_conditioning(runtime: dict[str, Any]) -> None:
    mode = str(runtime.get("conditioning_mode", runtime.get("condition_mode", ""))).lower()
    if mode not in STANDARD_ALLOWED_CONDITIONING_MODES:
        allowed = ", ".join(sorted(STANDARD_ALLOWED_CONDITIONING_MODES))
        raise ValueError(f"standard infer requires conditioning_mode in {{{allowed}}}, got {mode!r}")
    coeff_maps = runtime.get("coeff_maps_npz") or runtime.get("nat_coeff_maps_path")
    if not coeff_maps:
        raise ValueError("standard infer requires coeff_maps_npz/nat_coeff_maps_path for continuous condition")
    if int(runtime.get("conditioning_vector_dim", runtime.get("condition_channels", 0))) <= 0:
        raise ValueError("standard infer requires positive conditioning_vector_dim/condition_channels")
    if mode == "film" and not bool(runtime.get("append_domain_onehot", False)):
        raise ValueError("dual film standard infer requires append_domain_onehot=true")


def camera_pixels_from_runtime(runtime: dict[str, Any]) -> tuple[float, float]:
    return float(runtime["camera_pixel_nm_x"]), float(runtime["camera_pixel_nm_y"])
