from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import Microscope1Dataset
from .lut import CalibrationLUT


@dataclass(frozen=True)
class OracleObservations:
    patches: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    z_gt_nm: np.ndarray
    frame_index: np.ndarray
    nearest_neighbor_px: np.ndarray
    gt_index: np.ndarray


@dataclass(frozen=True)
class LocalZFit:
    z_fit_nm: np.ndarray
    z_residual_nm: np.ndarray
    ncc: np.ndarray
    photons_adu: np.ndarray
    background_adu: np.ndarray
    residual_variance_adu2: np.ndarray


def harvest_oracle_patches(
    dataset: Microscope1Dataset,
    *,
    min_neighbor_distance_px: float = 31.0,
    max_emitters: int | None = None,
) -> OracleObservations:
    gt = dataset.load_ground_truth()
    xy_px = dataset.gt_xy_to_image_px(gt.x_nm, gt.y_nm)
    nearest = _same_frame_nearest_distances(xy_px, gt.frame_index)
    selected = np.flatnonzero(nearest >= float(min_neighbor_distance_px))
    if max_emitters is not None:
        selected = selected[: int(max_emitters)]

    selected_xy = xy_px[selected]
    centers = np.floor(selected_xy + 0.5).astype(np.int64)
    radius = 15
    frames = dataset.open_frames()
    patches = np.empty((selected.shape[0], 31, 31), dtype=np.float32)
    for output_index, (gt_index, center) in enumerate(zip(selected, centers, strict=True)):
        center_x, center_y = int(center[0]), int(center[1])
        patch = frames[
            int(gt.frame_index[gt_index]),
            center_y - radius : center_y + radius + 1,
            center_x - radius : center_x + radius + 1,
        ]
        if patch.shape != (31, 31):
            raise ValueError(f"GT row {gt_index} produces out-of-bounds patch {patch.shape}.")
        patches[output_index] = patch

    fractional_xy = selected_xy - centers
    patches = _fourier_shift_batch(
        patches,
        shift_x_px=-fractional_xy[:, 0],
        shift_y_px=-fractional_xy[:, 1],
    )
    return OracleObservations(
        patches=patches,
        x_px=selected_xy[:, 0],
        y_px=selected_xy[:, 1],
        z_gt_nm=gt.z_nm[selected],
        frame_index=gt.frame_index[selected],
        nearest_neighbor_px=nearest[selected],
        gt_index=selected,
    )


def estimate_local_z(
    patches: np.ndarray,
    z_gt_nm: np.ndarray,
    lut: CalibrationLUT,
    *,
    search_radius_planes: int = 3,
    batch_size: int = 512,
) -> LocalZFit:
    observed = np.asarray(patches, dtype=np.float32)
    z_gt = np.asarray(z_gt_nm, dtype=np.float64).reshape(-1)
    if observed.ndim != 3 or observed.shape[1:] != lut.planes.shape[1:]:
        raise ValueError(
            f"patches must have shape (N,{lut.planes.shape[1]},{lut.planes.shape[2]}), got {observed.shape}"
        )
    if observed.shape[0] != z_gt.shape[0]:
        raise ValueError("patches and z_gt_nm must have the same length.")

    template_unit = _unit_centered(lut.planes)
    z_fit = np.empty_like(z_gt)
    peak_ncc = np.empty(z_gt.shape, dtype=np.float32)
    radius = int(search_radius_planes)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    nominal_index = z_gt / (lut.z_sign * lut.z_step_nm) - lut.z_index_origin

    for start in range(0, observed.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), observed.shape[0])
        patch_unit = _unit_centered(observed[start:stop])
        centers = np.floor(nominal_index[start:stop] + 0.5).astype(np.int64)
        candidates = np.clip(centers[:, None] + offsets[None, :], 0, lut.planes.shape[0] - 1)
        scores = np.einsum(
            "bp,bkp->bk",
            patch_unit.reshape(stop - start, -1),
            template_unit[candidates].reshape(stop - start, offsets.size, -1),
            optimize=True,
        )
        best_column = np.argmax(scores, axis=1)
        rows = np.arange(stop - start)
        best_index = candidates[rows, best_column]
        peak_ncc[start:stop] = scores[rows, best_column]

        delta = np.zeros(best_index.shape, dtype=np.float64)
        interior = (best_index > 0) & (best_index < lut.planes.shape[0] - 1)
        if np.any(interior):
            flat_patches = patch_unit[interior].reshape(int(interior.sum()), -1)
            interior_index = best_index[interior]
            left = np.einsum("bp,bp->b", flat_patches, template_unit[interior_index - 1].reshape(flat_patches.shape))
            center = np.einsum("bp,bp->b", flat_patches, template_unit[interior_index].reshape(flat_patches.shape))
            right = np.einsum("bp,bp->b", flat_patches, template_unit[interior_index + 1].reshape(flat_patches.shape))
            denominator = left - 2.0 * center + right
            valid = denominator < -1e-8
            local_delta = np.zeros_like(center, dtype=np.float64)
            local_delta[valid] = 0.5 * (left[valid] - right[valid]) / denominator[valid]
            delta[interior] = np.clip(local_delta, -1.0, 1.0)

        fitted_index = best_index.astype(np.float64) + delta
        z_fit[start:stop] = lut.z_sign * (fitted_index + lut.z_index_origin) * lut.z_step_nm

    fitted_templates = _interpolate_planes(lut.planes, z_fit, lut)
    flat_observed = observed.reshape(observed.shape[0], -1).astype(np.float64)
    flat_templates = fitted_templates.reshape(fitted_templates.shape[0], -1).astype(np.float64)
    observed_mean = flat_observed.mean(axis=1)
    template_mean = flat_templates.mean(axis=1)
    observed_centered = flat_observed - observed_mean[:, None]
    template_centered = flat_templates - template_mean[:, None]
    denominator = np.sum(template_centered**2, axis=1).clip(1e-12)
    photons = np.sum(observed_centered * template_centered, axis=1) / denominator
    background = observed_mean - photons * template_mean
    expected = photons[:, None] * flat_templates + background[:, None]
    residual_variance = np.mean((flat_observed - expected) ** 2, axis=1)

    return LocalZFit(
        z_fit_nm=z_fit,
        z_residual_nm=z_fit - z_gt,
        ncc=peak_ncc,
        photons_adu=photons.astype(np.float32),
        background_adu=background.astype(np.float32),
        residual_variance_adu2=residual_variance.astype(np.float32),
    )


def _same_frame_nearest_distances(xy_px: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
    nearest = np.empty(frame_index.shape[0], dtype=np.float64)
    for frame in np.unique(frame_index):
        rows = np.flatnonzero(frame_index == frame)
        delta = xy_px[rows, None, :] - xy_px[None, rows, :]
        distances = np.sqrt(np.sum(delta**2, axis=2))
        np.fill_diagonal(distances, np.inf)
        nearest[rows] = distances.min(axis=1)
    return nearest


def _fourier_shift_batch(
    images: np.ndarray,
    *,
    shift_x_px: np.ndarray,
    shift_y_px: np.ndarray,
) -> np.ndarray:
    height, width = images.shape[-2:]
    fy = np.fft.fftfreq(height)[None, :, None]
    fx = np.fft.fftfreq(width)[None, None, :]
    phase = np.exp(
        -2j
        * np.pi
        * (
            fy * np.asarray(shift_y_px, dtype=np.float64)[:, None, None]
            + fx * np.asarray(shift_x_px, dtype=np.float64)[:, None, None]
        )
    )
    shifted = np.fft.ifft2(np.fft.fft2(images, axes=(-2, -1)) * phase, axes=(-2, -1)).real
    return shifted.astype(np.float32, copy=False)


def _unit_centered(images: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float32)
    centered = values - values.mean(axis=(-2, -1), keepdims=True)
    norm = np.sqrt(np.sum(centered**2, axis=(-2, -1), keepdims=True)).clip(1e-12)
    return centered / norm


def _interpolate_planes(planes: np.ndarray, z_nm: np.ndarray, lut: CalibrationLUT) -> np.ndarray:
    index = np.asarray(z_nm, dtype=np.float64) / (lut.z_sign * lut.z_step_nm) - lut.z_index_origin
    index = np.clip(index, 0.0, planes.shape[0] - 1.0)
    low = np.floor(index).astype(np.int64)
    high = np.minimum(low + 1, planes.shape[0] - 1)
    fraction = (index - low).astype(np.float32)[:, None, None]
    return (1.0 - fraction) * planes[low] + fraction * planes[high]


__all__ = [
    "LocalZFit",
    "OracleObservations",
    "estimate_local_z",
    "harvest_oracle_patches",
]
