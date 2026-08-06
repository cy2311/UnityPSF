from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True)
class Microscope1Config:
    pixel_size_nm: float = 200.0
    calibration_step_nm: float = 33.3
    frame_shape_hw: tuple[int, int] = (150, 150)
    calibration_patch_shape_hw: tuple[int, int] = (31, 31)
    calibration_planes: int = 119
    frame_count: int = 5000
    emitters_per_frame: int = 5
    gt_frame_base: int = 1
    image_origin_xy_px: tuple[float, float] = (15.0, 15.0)
    z_index_origin: float = 1.0
    z_sign: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.pixel_size_nm) or self.pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm must be finite and positive.")
        if not np.isfinite(self.calibration_step_nm) or self.calibration_step_nm <= 0:
            raise ValueError("calibration_step_nm must be finite and positive.")
        if len(self.frame_shape_hw) != 2 or any(int(value) <= 0 for value in self.frame_shape_hw):
            raise ValueError("frame_shape_hw must contain two positive dimensions.")
        if len(self.calibration_patch_shape_hw) != 2 or any(
            int(value) <= 0 for value in self.calibration_patch_shape_hw
        ):
            raise ValueError("calibration_patch_shape_hw must contain two positive dimensions.")
        if self.calibration_planes <= 0 or self.frame_count <= 0 or self.emitters_per_frame <= 0:
            raise ValueError("Dataset counts must be positive.")
        if self.gt_frame_base < 0:
            raise ValueError("gt_frame_base must be non-negative.")
        if len(self.image_origin_xy_px) != 2 or not np.all(np.isfinite(self.image_origin_xy_px)):
            raise ValueError("image_origin_xy_px must contain two finite coordinates.")
        if not np.isfinite(self.z_index_origin) or self.z_sign not in (-1, 1):
            raise ValueError("z_index_origin must be finite and z_sign must be -1 or 1.")


@dataclass(frozen=True)
class DatasetContract:
    frame_shape: tuple[int, int, int]
    calibration_shape: tuple[int, int, int]
    gt_rows: int
    pixel_size_nm: float
    z_step_nm: float
    calibration_origin_xy_px: tuple[float, float]
    z_index_origin: float
    z_sign: int


@dataclass(frozen=True)
class GroundTruth:
    x_nm: np.ndarray
    y_nm: np.ndarray
    z_nm: np.ndarray
    frame_index: np.ndarray

    def __len__(self) -> int:
        return int(self.x_nm.shape[0])


@dataclass(frozen=True)
class FrameSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


class Microscope1Dataset:
    def __init__(self, root: str | Path, *, config: Microscope1Config | None = None) -> None:
        self.root = Path(root)
        self.config = Microscope1Config() if config is None else config
        if not isinstance(self.config, Microscope1Config):
            raise TypeError("config must be a Microscope1Config.")
        self.calibration_path = self.root / "Calib.tif"
        self.frames_path = self.root / "Dens5_noisy5000.tif"
        self.gt_path = self.root / "Dens5_noisy5000_GT.txt"

    @property
    def pixel_size_nm(self) -> float:
        return self.config.pixel_size_nm

    @property
    def z_step_nm(self) -> float:
        return self.config.calibration_step_nm

    @property
    def calibration_origin_xy_px(self) -> tuple[float, float]:
        return self.config.image_origin_xy_px

    @property
    def z_index_origin(self) -> float:
        return self.config.z_index_origin

    @property
    def z_sign(self) -> int:
        return self.config.z_sign

    def validate(self) -> DatasetContract:
        for path in (self.calibration_path, self.frames_path, self.gt_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        calibration_shape = self._tiff_shape(self.calibration_path)
        frame_shape = self._tiff_shape(self.frames_path)
        gt = self.load_ground_truth()
        contract = DatasetContract(
            frame_shape=frame_shape,
            calibration_shape=calibration_shape,
            gt_rows=len(gt),
            pixel_size_nm=self.pixel_size_nm,
            z_step_nm=self.z_step_nm,
            calibration_origin_xy_px=self.calibration_origin_xy_px,
            z_index_origin=self.z_index_origin,
            z_sign=self.z_sign,
        )
        expected_frame_shape = (self.config.frame_count, *self.config.frame_shape_hw)
        expected_calibration_shape = (
            self.config.calibration_planes,
            *self.config.calibration_patch_shape_hw,
        )
        if contract.frame_shape != expected_frame_shape:
            raise ValueError(f"Unexpected frame stack shape: {contract.frame_shape}")
        if contract.calibration_shape != expected_calibration_shape:
            raise ValueError(f"Unexpected calibration stack shape: {contract.calibration_shape}")
        expected_gt_rows = self.config.frame_count * self.config.emitters_per_frame
        if contract.gt_rows != expected_gt_rows:
            raise ValueError(f"Unexpected GT row count: {contract.gt_rows}")
        per_frame = np.bincount(gt.frame_index, minlength=self.config.frame_count)
        expected_per_frame = np.full(self.config.frame_count, self.config.emitters_per_frame)
        if not np.array_equal(per_frame, expected_per_frame):
            raise ValueError("Ground truth does not contain the configured emitters per frame.")
        return contract

    def load_ground_truth(self) -> GroundTruth:
        values = np.loadtxt(self.gt_path, skiprows=1, dtype=np.float64, ndmin=2)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(f"Ground truth must have four columns, got {values.shape}")
        if values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Ground truth must contain finite rows.")
        frame_ids = values[:, 3].astype(np.int64)
        if not np.array_equal(values[:, 3], frame_ids.astype(np.float64)):
            raise ValueError("Ground-truth frame identifiers must be integers.")
        frame_indices = frame_ids - self.config.gt_frame_base
        if np.any(frame_indices < 0) or np.any(frame_indices >= self.config.frame_count):
            raise ValueError("Ground-truth frame identifiers are outside the configured frame range.")
        return GroundTruth(
            x_nm=values[:, 0],
            y_nm=values[:, 1],
            z_nm=values[:, 2],
            frame_index=frame_indices,
        )

    def ground_truth_xy_px(self, ground_truth: GroundTruth) -> np.ndarray:
        if not isinstance(ground_truth, GroundTruth):
            raise TypeError("ground_truth must be a GroundTruth instance.")
        return self.gt_xy_to_image_px(ground_truth.x_nm, ground_truth.y_nm)

    def gt_xy_to_image_px(
        self,
        x_nm: np.ndarray | float,
        y_nm: np.ndarray | float,
    ) -> np.ndarray:
        x_px = np.asarray(x_nm, dtype=np.float64) / self.pixel_size_nm + self.calibration_origin_xy_px[0]
        y_px = np.asarray(y_nm, dtype=np.float64) / self.pixel_size_nm + self.calibration_origin_xy_px[1]
        return np.stack([x_px, y_px], axis=-1)

    def read_calibration(self) -> np.ndarray:
        return np.asarray(tifffile.imread(self.calibration_path), dtype=np.float32)

    def open_frames(self) -> np.ndarray:
        return tifffile.memmap(self.frames_path)

    @staticmethod
    def _tiff_shape(path: Path) -> tuple[int, int, int]:
        with tifffile.TiffFile(path) as tif:
            shape = tuple(int(value) for value in tif.series[0].shape)
        if len(shape) != 3:
            raise ValueError(f"Expected a three-dimensional TIFF stack, got {shape}")
        return shape


def deterministic_frame_split(
    frame_indices: np.ndarray,
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> FrameSplit:
    values = np.asarray(frame_indices)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("frame_indices must be a non-empty one-dimensional array.")
    integer_values = values.astype(np.int64)
    if not np.array_equal(values, integer_values.astype(values.dtype)) or np.any(integer_values < 0):
        raise ValueError("frame_indices must contain non-negative integers.")

    weights = np.asarray(fractions, dtype=np.float64)
    if weights.shape != (3,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("fractions must contain three finite non-negative values.")
    if not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("fractions must sum to one.")

    shuffled = np.random.default_rng(int(seed)).permutation(np.unique(integer_values))
    train_stop = int(np.floor(shuffled.size * weights[0]))
    validation_stop = train_stop + int(np.floor(shuffled.size * weights[1]))
    return FrameSplit(
        train=shuffled[:train_stop],
        validation=shuffled[train_stop:validation_stop],
        test=shuffled[validation_stop:],
    )


__all__ = [
    "DatasetContract",
    "FrameSplit",
    "GroundTruth",
    "Microscope1Config",
    "Microscope1Dataset",
    "deterministic_frame_split",
]
