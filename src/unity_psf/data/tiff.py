from __future__ import annotations

from pathlib import Path

import numpy as np


def read_tiff_stack(path: str | Path) -> np.ndarray:
    import tifffile

    frames = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    frames = np.squeeze(frames)
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim != 3:
        raise ValueError(f"TIFF stack must have shape (T,H,W), got {frames.shape}")
    return np.ascontiguousarray(frames, dtype=np.float32)
