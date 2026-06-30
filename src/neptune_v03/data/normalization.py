from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
