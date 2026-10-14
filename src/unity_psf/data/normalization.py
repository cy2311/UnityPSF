from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CameraCalibration:
    baseline_adu: float
    e_per_adu: float
    qe: float
    em_gain: float = 1.0
    spurious_charge: float = 0.0


@dataclass(frozen=True)
class TrainNormalization:
    input_offset: float = 0.0
    input_scale: float = 1.0
    photon_scale: float = 1.0


@dataclass(frozen=True)
class LunarInputNormalization:
    background_range: tuple[float, float]
    photon_range: tuple[float, float]

    def __post_init__(self) -> None:
        if float(self.photon_range[1]) <= 0.0:
            raise ValueError("LUNAR photon_range maximum must be positive")

    @property
    def background_mean(self) -> float:
        return (float(self.background_range[0]) + float(self.background_range[1])) / 2.0

    @property
    def scale_ph_factor(self) -> float:
        return float(self.photon_range[1]) / 50.0

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.background_mean) / self.scale_ph_factor

    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.scale_ph_factor + self.background_mean

    def to_config(self) -> dict[str, object]:
        return {
            "name": "lunar",
            "background_range": [float(value) for value in self.background_range],
            "photon_range": [float(value) for value in self.photon_range],
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "LunarInputNormalization":
        if str(config.get("name", "")).lower() != "lunar":
            raise ValueError("input normalization config must use name='lunar'")
        background_range = config.get("background_range")
        photon_range = config.get("photon_range")
        if not isinstance(background_range, (list, tuple)) or len(background_range) != 2:
            raise ValueError("LUNAR normalization requires a two-value background_range")
        if not isinstance(photon_range, (list, tuple)) or len(photon_range) != 2:
            raise ValueError("LUNAR normalization requires a two-value photon_range")
        return cls(
            background_range=tuple(float(value) for value in background_range),
            photon_range=tuple(float(value) for value in photon_range),
        )


def adu_to_photons(frames_adu: ArrayLike, calibration: CameraCalibration) -> np.ndarray:
    frames = np.asarray(frames_adu, dtype=np.float32)
    electrons = (frames - float(calibration.baseline_adu)) * float(calibration.e_per_adu)
    electrons = electrons - float(calibration.spurious_charge)
    photons = electrons / max(float(calibration.em_gain) * float(calibration.qe), 1e-12)
    return np.clip(photons, 0.0, None).astype(np.float32, copy=False)


def normalize_train_input(frames_photon: ArrayLike, normalization: TrainNormalization) -> np.ndarray:
    frames = np.asarray(frames_photon, dtype=np.float32)
    denominator = max(float(normalization.photon_scale) * float(normalization.input_scale), 1e-12)
    return ((frames - float(normalization.input_offset)) / denominator).astype(np.float32, copy=False)
