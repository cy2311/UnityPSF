from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ZMapContract:
    image_shape_hw: tuple[int, int]
    pixel_size_nm_xy: tuple[float, float]
    phase_unit: str = "nm"
    value_range_nm: tuple[float, float] = (-500.0, 500.0)
    x_axis_points_right: bool = True
    y_axis_points_down: bool = True
    pixel_aligned: bool = True
    continuous: bool = True
    source: str = "nat_full_roi"
    version: int = 1

    def __post_init__(self) -> None:
        height, width = self.image_shape_hw
        if int(height) <= 0 or int(width) <= 0:
            raise ValueError("image_shape_hw must contain positive integers.")

        px_x, px_y = self.pixel_size_nm_xy
        if float(px_x) <= 0.0 or float(px_y) <= 0.0:
            raise ValueError("pixel_size_nm_xy must contain positive values.")

        z_min, z_max = self.value_range_nm
        if not np.isfinite([z_min, z_max]).all():
            raise ValueError("value_range_nm must be finite.")
        if float(z_max) <= float(z_min):
            raise ValueError("value_range_nm must satisfy z_max > z_min.")

    @property
    def height(self) -> int:
        return int(self.image_shape_hw[0])

    @property
    def width(self) -> int:
        return int(self.image_shape_hw[1])

    @property
    def z_min(self) -> float:
        return float(self.value_range_nm[0])

    @property
    def z_max(self) -> float:
        return float(self.value_range_nm[1])

    def to_metadata_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_shape_hw"] = [self.height, self.width]
        payload["pixel_size_nm_xy"] = [float(self.pixel_size_nm_xy[0]), float(self.pixel_size_nm_xy[1])]
        payload["value_range_nm"] = [self.z_min, self.z_max]
        return payload

    @classmethod
    def from_metadata_dict(cls, payload: dict[str, Any]) -> "ZMapContract":
        return cls(
            image_shape_hw=tuple(int(v) for v in payload["image_shape_hw"]),
            pixel_size_nm_xy=tuple(float(v) for v in payload["pixel_size_nm_xy"]),
            phase_unit=str(payload.get("phase_unit", "nm")),
            value_range_nm=tuple(float(v) for v in payload.get("value_range_nm", (-500.0, 500.0))),
            x_axis_points_right=bool(payload.get("x_axis_points_right", True)),
            y_axis_points_down=bool(payload.get("y_axis_points_down", True)),
            pixel_aligned=bool(payload.get("pixel_aligned", True)),
            continuous=bool(payload.get("continuous", True)),
            source=str(payload.get("source", "nat_full_roi")),
            version=int(payload.get("version", 1)),
        )


@dataclass(frozen=True)
class ZMapAlignmentReport:
    matches_shape: bool
    matches_pixel_size: bool
    pixel_aligned: bool
    continuous: bool
    max_abs_jump_nm: float
    p99_abs_jump_nm: float

    @property
    def is_usable(self) -> bool:
        return bool(self.matches_shape and self.matches_pixel_size and self.pixel_aligned and self.continuous)


@dataclass(frozen=True)
class FullROIPhaseMap:
    values_nm: np.ndarray
    contract: ZMapContract

    def __post_init__(self) -> None:
        values = np.asarray(self.values_nm, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("values_nm must be a 2D array.")
        if tuple(values.shape) != tuple(self.contract.image_shape_hw):
            raise ValueError("values_nm shape must match contract.image_shape_hw.")
        if not np.isfinite(values).all():
            raise ValueError("values_nm must be finite.")
        object.__setattr__(self, "values_nm", values)

    def normalize(self) -> np.ndarray:
        scale = self.contract.z_max - self.contract.z_min
        normalized = 2.0 * (self.values_nm - self.contract.z_min) / scale - 1.0
        return normalized.astype(np.float32, copy=False)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float32)
        if values.shape != self.values_nm.shape:
            raise ValueError("normalized must match the stored zmap shape.")
        restored = (values + 1.0) * 0.5 * (self.contract.z_max - self.contract.z_min) + self.contract.z_min
        return restored.astype(np.float32, copy=False)

    def crop_patch(self, *, top: int, left: int, height: int, width: int, normalized: bool = True) -> np.ndarray:
        top_i = int(top)
        left_i = int(left)
        height_i = int(height)
        width_i = int(width)
        if top_i < 0 or left_i < 0 or height_i <= 0 or width_i <= 0:
            raise ValueError("Patch bounds must be positive and in range.")
        if top_i + height_i > self.contract.height or left_i + width_i > self.contract.width:
            raise ValueError("Patch exceeds full-ROI zmap bounds.")
        source = self.normalize() if normalized else self.values_nm
        return source[top_i : top_i + height_i, left_i : left_i + width_i].copy()

    def alignment_report(
        self,
        *,
        image_shape_hw: tuple[int, int],
        pixel_size_nm_xy: tuple[float, float] | None = None,
    ) -> ZMapAlignmentReport:
        jump_metrics = _compute_jump_metrics_nm(self.values_nm)
        return ZMapAlignmentReport(
            matches_shape=tuple(int(v) for v in image_shape_hw) == self.contract.image_shape_hw,
            matches_pixel_size=(
                True
                if pixel_size_nm_xy is None
                else np.allclose(np.asarray(pixel_size_nm_xy, dtype=np.float32), np.asarray(self.contract.pixel_size_nm_xy, dtype=np.float32))
            ),
            pixel_aligned=bool(self.contract.pixel_aligned),
            continuous=bool(self.contract.continuous),
            max_abs_jump_nm=float(jump_metrics["max_abs_jump_nm"]),
            p99_abs_jump_nm=float(jump_metrics["p99_abs_jump_nm"]),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            values_nm=self.values_nm.astype(np.float32, copy=False),
            metadata_json=np.asarray(json.dumps(self.contract.to_metadata_dict())),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "FullROIPhaseMap":
        payload = np.load(Path(path), allow_pickle=False)
        try:
            values_nm = np.asarray(payload["values_nm"], dtype=np.float32)
            metadata = json.loads(str(payload["metadata_json"].item()))
        finally:
            payload.close()
        return cls(values_nm=values_nm, contract=ZMapContract.from_metadata_dict(metadata))


def build_full_roi_zmap(
    values_nm: np.ndarray,
    *,
    image_shape_hw: tuple[int, int],
    pixel_size_nm_xy: tuple[float, float],
    value_range_nm: tuple[float, float],
    phase_unit: str = "nm",
    source: str = "nat_full_roi",
    continuous: bool = True,
    pixel_aligned: bool = True,
) -> FullROIPhaseMap:
    contract = ZMapContract(
        image_shape_hw=tuple(int(v) for v in image_shape_hw),
        pixel_size_nm_xy=tuple(float(v) for v in pixel_size_nm_xy),
        phase_unit=str(phase_unit),
        value_range_nm=tuple(float(v) for v in value_range_nm),
        source=str(source),
        continuous=bool(continuous),
        pixel_aligned=bool(pixel_aligned),
    )
    return FullROIPhaseMap(values_nm=np.asarray(values_nm, dtype=np.float32), contract=contract)


def _compute_jump_metrics_nm(values_nm: np.ndarray) -> dict[str, float]:
    values = np.asarray(values_nm, dtype=np.float32)
    deltas: list[np.ndarray] = []
    if values.shape[0] > 1:
        deltas.append(np.abs(np.diff(values, axis=0)).reshape(-1))
    if values.shape[1] > 1:
        deltas.append(np.abs(np.diff(values, axis=1)).reshape(-1))
    if not deltas:
        return {"max_abs_jump_nm": 0.0, "p99_abs_jump_nm": 0.0}
    merged = np.concatenate(deltas, axis=0)
    return {
        "max_abs_jump_nm": float(np.max(merged)),
        "p99_abs_jump_nm": float(np.quantile(merged, 0.99)),
    }


__all__ = [
    "FullROIPhaseMap",
    "ZMapAlignmentReport",
    "ZMapContract",
    "build_full_roi_zmap",
]
