from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


FORMAT_VERSION = 1

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class EmitterPosterior:
    probability: float
    cell_xy_px: tuple[float, float]
    mu_xy_px: tuple[float, float]
    sigma_xy_px: tuple[float, float]
    mu_z_nm: float
    sigma_z_nm: float
    mu_photons: float
    sigma_photons: float
    local_xy_px: tuple[float, float]
    full_xy_px: tuple[float, float]
    frame_index: int = 0


@dataclass(frozen=True)
class ROIRecord:
    roi_id: int
    domain_name: str
    frame_window: tuple[int, int]
    roi_origin_xy_px: tuple[float, float]
    raw_frames_photon: FloatArray
    background_mu: FloatArray
    background_smoothed: FloatArray
    grid_cell_id: int
    emitters: tuple[EmitterPosterior, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ROIBank:
    records: tuple[ROIRecord, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    empty_grid_cell_ids: tuple[int, ...] = ()
    format_version: int = FORMAT_VERSION


@dataclass(frozen=True)
class ROICandidate:
    candidate_id: int
    origin_xy_px: tuple[int, int]
    center_xy_px: tuple[float, float]
    grid_cell_id: int
    emitter_indices: tuple[int, ...]
    emitter_count: int
    quality_score: float
    origin_bank_index: int | None = None
    valid_core_origin_xy_px: tuple[int, int] | None = None
    valid_core_offset_xy_px: tuple[int, int] | None = None
    valid_core_size_px: int | None = None


@dataclass(frozen=True)
class FOVSelection:
    candidates: tuple[ROICandidate, ...]
    summary: dict[str, Any]
