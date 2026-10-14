from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True)
class EmpiricalFocalPSF:
    kernel: np.ndarray
    source_path: str
    source_sha256: str
    kernel_sha256: str
    channel_id: str
    focus_index: int
    source_channel_slice: tuple[int, int]
    background: float
    source_centroid_yx: tuple[float, float]

    def metadata(self) -> dict[str, object]:
        return {
            "psf_type": "empirical_focal",
            "empirical_psf_path": self.source_path,
            "empirical_psf_source_sha256": self.source_sha256,
            "empirical_psf_kernel_sha256": self.kernel_sha256,
            "empirical_psf_channel": self.channel_id,
            "empirical_psf_focus_index": self.focus_index,
            "empirical_psf_source_channel_slice": list(self.source_channel_slice),
            "empirical_psf_background": self.background,
            "empirical_psf_source_centroid_yx": list(self.source_centroid_yx),
        }


def load_empirical_focal_psf(
    path: str | Path,
    *,
    channel_id: str,
    focus_index: int,
) -> EmpiricalFocalPSF:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"empirical PSF TIFF does not exist: {source}")
    stack = np.asarray(tifffile.imread(source))
    if stack.ndim != 3:
        raise ValueError("empirical PSF TIFF must have shape (z, y, 2*x)")
    depth, height, joined_width = (int(value) for value in stack.shape)
    half_width = joined_width // 2
    if joined_width % 2 or height != half_width or height % 2 == 0:
        raise ValueError("empirical PSF TIFF must contain two equal square channel patches with odd support")
    focus = int(focus_index)
    if focus < 0 or focus >= depth:
        raise ValueError(f"empirical PSF focus index {focus} is outside [0, {depth})")
    channel = str(channel_id).strip().lower()
    if channel == "left":
        x0, x1 = 0, half_width
    elif channel == "right":
        x0, x1 = half_width, joined_width
    else:
        raise ValueError("empirical PSF channel_id must be 'left' or 'right'")

    patch = stack[focus, :, x0:x1].astype(np.float64, copy=False)
    border = np.concatenate((patch[0], patch[-1], patch[:, 0], patch[:, -1]))
    background = float(np.median(border))
    signal = np.clip(patch - background, 0.0, None)
    total = float(signal.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("empirical focal PSF contains no positive signal after background subtraction")
    yy, xx = np.indices(signal.shape, dtype=np.float64)
    centroid_y = float((signal * yy).sum() / total)
    centroid_x = float((signal * xx).sum() / total)
    center = (height - 1) / 2.0
    kernel = _fourier_shift(signal, shift_y=center - centroid_y, shift_x=center - centroid_x)
    kernel = np.clip(kernel, 0.0, None).astype(np.float32)
    kernel /= np.maximum(kernel.sum(dtype=np.float64), 1e-12)
    kernel = np.ascontiguousarray(kernel)
    return EmpiricalFocalPSF(
        kernel=kernel,
        source_path=str(source),
        source_sha256=_sha256_file(source),
        kernel_sha256=hashlib.sha256(kernel.tobytes(order="C")).hexdigest(),
        channel_id=channel,
        focus_index=focus,
        source_channel_slice=(x0, x1),
        background=background,
        source_centroid_yx=(centroid_y, centroid_x),
    )


def _fourier_shift(image: np.ndarray, *, shift_y: float, shift_x: float) -> np.ndarray:
    freq_y = np.fft.fftfreq(image.shape[0])[:, None]
    freq_x = np.fft.fftfreq(image.shape[1])[None, :]
    phase = np.exp(-2j * np.pi * (float(shift_y) * freq_y + float(shift_x) * freq_x))
    return np.fft.ifft2(np.fft.fft2(image) * phase).real


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["EmpiricalFocalPSF", "load_empirical_focal_psf"]
