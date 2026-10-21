from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationLUT:
    planes: np.ndarray
    z_step_nm: float
    z_index_origin: float
    z_sign: int = 1

    @classmethod
    def from_array(
        cls,
        planes: np.ndarray,
        *,
        z_step_nm: float,
        z_index_origin: float,
        z_sign: int = 1,
    ) -> "CalibrationLUT":
        values = np.asarray(planes, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(f"Calibration stack must have shape (Z,H,W), got {values.shape}")
        if values.shape[1] != values.shape[2] or values.shape[1] % 2 != 1:
            raise ValueError("Calibration planes must be odd, square patches.")
        if float(z_step_nm) <= 0.0:
            raise ValueError("z_step_nm must be positive.")
        if int(z_sign) not in (-1, 1):
            raise ValueError("z_sign must be -1 or +1.")

        normalized = np.empty_like(values)
        for index, plane in enumerate(values):
            rim = np.concatenate((plane[0], plane[-1], plane[1:-1, 0], plane[1:-1, -1]))
            signal = np.clip(plane - np.median(rim), 0.0, None)
            flux = float(signal.sum())
            if flux <= 0.0:
                raise ValueError(f"Calibration plane {index} has no positive signal after background removal.")
            normalized[index] = signal / flux
        return cls(
            planes=np.ascontiguousarray(normalized),
            z_step_nm=float(z_step_nm),
            z_index_origin=float(z_index_origin),
            z_sign=int(z_sign),
        )

    @property
    def z_nm(self) -> np.ndarray:
        indices = np.arange(self.planes.shape[0], dtype=np.float64)
        return self.z_sign * (indices + self.z_index_origin) * self.z_step_nm

    def render(
        self,
        z_nm: float,
        *,
        shift_x_px: float = 0.0,
        shift_y_px: float = 0.0,
    ) -> np.ndarray:
        continuous_index = float(z_nm) / (self.z_sign * self.z_step_nm) - self.z_index_origin
        continuous_index = float(np.clip(continuous_index, 0.0, self.planes.shape[0] - 1.0))
        low = int(np.floor(continuous_index))
        high = min(low + 1, self.planes.shape[0] - 1)
        fraction = continuous_index - low
        plane = (1.0 - fraction) * self.planes[low] + fraction * self.planes[high]
        if shift_x_px or shift_y_px:
            plane = _fourier_shift(plane, shift_x_px=shift_x_px, shift_y_px=shift_y_px)
        plane = np.clip(plane, 0.0, None)
        return np.asarray(plane / plane.sum(), dtype=np.float32)


def _fourier_shift(image: np.ndarray, *, shift_x_px: float, shift_y_px: float) -> np.ndarray:
    height, width = image.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    phase = np.exp(-2j * np.pi * (fy * float(shift_y_px) + fx * float(shift_x_px)))
    shifted = np.fft.ifft2(np.fft.fft2(image) * phase).real
    return shifted.astype(np.float32, copy=False)


__all__ = ["CalibrationLUT"]
