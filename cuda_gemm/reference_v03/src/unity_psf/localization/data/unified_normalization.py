"""Shared raw-ADU normalization contract for simulation and TIFF inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class UnifiedNormalizationContract:
    simulation_background_photons: float
    camera_baseline_adu: float
    camera_qe: float
    camera_e_per_adu: float
    camera_em_gain: float
    camera_spurious_charge: float
    train_background_adu: float
    background_percentile: float = 50.0

    @classmethod
    def from_runtime(
        cls,
        *,
        background_range: tuple[float, float] | list[float],
        camera_baseline_adu: float,
        camera_qe: float,
        camera_e_per_adu: float,
        camera_em_gain: float,
        camera_spurious_charge: float,
        background_percentile: float = 50.0,
    ) -> "UnifiedNormalizationContract":
        if len(background_range) != 2 or float(background_range[0]) != float(background_range[1]):
            raise ValueError("unified normalization requires a single simulation background value")
        background = float(background_range[0])
        train_background_adu = float(camera_baseline_adu) + (
            background * float(camera_qe) + float(camera_spurious_charge)
        ) / max(float(camera_e_per_adu), 1e-12)
        return cls(
            simulation_background_photons=background,
            camera_baseline_adu=float(camera_baseline_adu),
            camera_qe=float(camera_qe),
            camera_e_per_adu=float(camera_e_per_adu),
            camera_em_gain=float(camera_em_gain),
            camera_spurious_charge=float(camera_spurious_charge),
            train_background_adu=train_background_adu,
            background_percentile=float(background_percentile),
        )

    def canonicalize_raw_adu(
        self,
        frames: torch.Tensor,
        *,
        background_adu: float | None = None,
    ) -> torch.Tensor:
        reference = self.train_background_adu if background_adu is None else float(background_adu)
        if background_adu is None:
            background = torch.quantile(
                frames.detach().to(dtype=torch.float32).reshape(-1),
                float(self.background_percentile) / 100.0,
            )
        else:
            background = frames.new_tensor(reference)
        return self.train_background_adu + frames.to(dtype=torch.float32) - background

    def as_dict(self) -> dict[str, float | str]:
        return {
            "mode": "unified_domain_aware_raw_adu",
            "model_input": "canonical_recentered_raw_adu",
            "simulation_background_photons": self.simulation_background_photons,
            "camera_baseline_adu": self.camera_baseline_adu,
            "camera_qe": self.camera_qe,
            "camera_e_per_adu": self.camera_e_per_adu,
            "camera_em_gain": self.camera_em_gain,
            "camera_spurious_charge": self.camera_spurious_charge,
            "train_background_adu": self.train_background_adu,
            "background_percentile": self.background_percentile,
        }


def estimate_domain_baseline_adu(frames_adu: np.ndarray, *, percentile: float) -> float:
    frames = np.asarray(frames_adu, dtype=np.float32)
    if frames.size == 0:
        raise ValueError("cannot estimate baseline from empty frames")
    per_frame = np.percentile(frames.reshape(frames.shape[0], -1), float(percentile), axis=1)
    return float(np.median(per_frame))


def contract_from_mapping(normalization: Mapping[str, object], camera: Mapping[str, object], online: Mapping[str, object]) -> UnifiedNormalizationContract:
    background = online.get("background_range", (online.get("background", 0.0), online.get("background", 0.0)))
    if not isinstance(background, (list, tuple)):
        background = (float(background), float(background))
    return UnifiedNormalizationContract.from_runtime(
        background_range=background,
        camera_baseline_adu=float(camera["baseline"]),
        camera_qe=float(camera["qe"]),
        camera_e_per_adu=float(camera["e_per_adu"]),
        camera_em_gain=float(camera.get("em_gain", 1.0)),
        camera_spurious_charge=float(camera.get("spurious_charge", 0.0)),
        background_percentile=float(normalization.get("background_percentile", 50.0)),
    )
