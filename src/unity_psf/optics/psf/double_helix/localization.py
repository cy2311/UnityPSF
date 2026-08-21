from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from .calibration import profile_photometry
from .gamma_field import DirectGammaZernikeField
from .numerics import normalized_cross_correlation
from .vector_model import DoubleHelixVectorPSF


@dataclass(frozen=True)
class LobeDetectionConfig:
    threshold_sigma: float = 15.0
    minimum_separation_px: float = 3.0
    maximum_separation_px: float = 10.0
    target_separation_px: float = 6.0
    maximum_peaks: int = 20


@dataclass(frozen=True)
class LobeDetections:
    x_px: np.ndarray
    y_px: np.ndarray
    angle_rad: np.ndarray
    separation_px: np.ndarray
    score: np.ndarray


@dataclass(frozen=True)
class AngleZCalibration:
    angle_unwrapped_rad: np.ndarray
    z_nm: np.ndarray

    @classmethod
    def from_angles(cls, angle_rad: np.ndarray, z_nm: np.ndarray) -> "AngleZCalibration":
        angles = np.asarray(angle_rad, dtype=np.float64).reshape(-1)
        z_values = np.asarray(z_nm, dtype=np.float64).reshape(-1)
        if angles.shape != z_values.shape or angles.size < 2:
            raise ValueError("angle_rad and z_nm must have matching lengths of at least two.")
        unwrapped = np.unwrap(2.0 * angles) / 2.0
        return cls(angle_unwrapped_rad=unwrapped, z_nm=z_values)

    @classmethod
    def from_stack(
        cls,
        calibration_stack: np.ndarray,
        z_nm: np.ndarray,
        *,
        detection_config: LobeDetectionConfig | None = None,
    ) -> "AngleZCalibration":
        stack = np.asarray(calibration_stack, dtype=np.float32)
        angles = np.empty(stack.shape[0], dtype=np.float64)
        for plane_index, image in enumerate(stack):
            angles[plane_index] = _principal_axis_angle(image)
        return cls.from_angles(angles, z_nm)

    def estimate_z(self, angle_rad: np.ndarray | float) -> np.ndarray:
        wrapped = np.asarray(angle_rad, dtype=np.float64)
        center = float(np.mean(self.angle_unwrapped_rad))
        cycle = np.round((center - wrapped) / np.pi)
        aligned = wrapped + cycle * np.pi
        order = np.argsort(self.angle_unwrapped_rad)
        angle_sorted = self.angle_unwrapped_rad[order]
        z_sorted = self.z_nm[order]
        unique_angle, unique_index = np.unique(angle_sorted, return_index=True)
        return np.interp(aligned, unique_angle, z_sorted[unique_index])


@dataclass(frozen=True)
class LocalizationConfig:
    detection: LobeDetectionConfig = LobeDetectionConfig()
    batch_size: int = 64
    refinement_steps: int = 60
    xy_learning_rate_px: float = 0.02
    z_learning_rate_nm: float = 8.0
    minimum_ncc: float = 0.35
    z_range_nm: tuple[float, float] = (33.3, 3962.7)
    pixel_size_nm: float = 200.0
    image_origin_xy_px: tuple[float, float] = (15.0, 15.0)
    image_shape_hw: tuple[int, int] = (150, 150)
    na: float = 1.27
    wavelength_nm: float = 660.0
    refractive_index: float = 1.33
    npupil: int = 128
    psf_size: int = 31
    device: str = "cuda"


@dataclass(frozen=True)
class IndependentLocalizations:
    frame_index: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    z_nm: np.ndarray
    photons_adu: np.ndarray
    background_adu: np.ndarray
    ncc: np.ndarray
    lobe_angle_rad: np.ndarray
    lobe_separation_px: np.ndarray


@dataclass(frozen=True)
class MatchResult:
    prediction_to_gt: np.ndarray
    gt_to_prediction: np.ndarray
    true_positives: int
    false_positives: int
    false_negatives: int
    dx_nm: np.ndarray
    dy_nm: np.ndarray
    dz_nm: np.ndarray


def detect_lobe_pairs(
    frame: np.ndarray,
    *,
    config: LobeDetectionConfig | None = None,
) -> LobeDetections:
    cfg = LobeDetectionConfig() if config is None else config
    image = np.asarray(frame, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("frame must be a two-dimensional image.")
    background = float(np.median(image))
    sigma = max(1.4826 * float(np.median(np.abs(image - background))), 1.0)
    maxima = (image == maximum_filter(image, size=3, mode="nearest")) & (
        image > background + cfg.threshold_sigma * sigma
    )
    peak_y, peak_x = np.nonzero(maxima)
    intensities = image[peak_y, peak_x] - background
    if intensities.size > cfg.maximum_peaks:
        keep = np.argsort(intensities)[-int(cfg.maximum_peaks) :]
        peak_x, peak_y, intensities = peak_x[keep], peak_y[keep], intensities[keep]
    edges: list[tuple[float, int, int, float]] = []
    for first in range(peak_x.size):
        for second in range(first + 1, peak_x.size):
            dx = float(peak_x[second] - peak_x[first])
            dy = float(peak_y[second] - peak_y[first])
            distance = float(np.hypot(dx, dy))
            if cfg.minimum_separation_px <= distance <= cfg.maximum_separation_px:
                intensity_penalty = abs(np.log(max(float(intensities[first]), 1.0) / max(float(intensities[second]), 1.0)))
                pair_cost = abs(distance - cfg.target_separation_px) + 0.5 * intensity_penalty
                edges.append((pair_cost, first, second, distance))
    used: set[int] = set()
    pairs = []
    for pair_cost, first, second, distance in sorted(edges):
        if first in used or second in used:
            continue
        used.update((first, second))
        dx = float(peak_x[second] - peak_x[first])
        dy = float(peak_y[second] - peak_y[first])
        pairs.append(
            (
                0.5 * (peak_x[first] + peak_x[second]),
                0.5 * (peak_y[first] + peak_y[second]),
                np.mod(np.arctan2(dy, dx), np.pi),
                distance,
                float(intensities[first] + intensities[second]) / (1.0 + pair_cost),
            )
        )
    if not pairs:
        empty = np.empty(0, dtype=np.float32)
        return LobeDetections(empty, empty, empty, empty, empty)
    values = np.asarray(pairs, dtype=np.float64)
    order = np.argsort(values[:, 0])
    return LobeDetections(
        x_px=values[order, 0].astype(np.float32),
        y_px=values[order, 1].astype(np.float32),
        angle_rad=values[order, 2].astype(np.float32),
        separation_px=values[order, 3].astype(np.float32),
        score=values[order, 4].astype(np.float32),
    )


def _principal_axis_angle(image: np.ndarray) -> float:
    values = np.asarray(image, dtype=np.float64)
    border = np.concatenate((values[:3].ravel(), values[-3:].ravel(), values[3:-3, :3].ravel(), values[3:-3, -3:].ravel()))
    signal = np.maximum(values - np.median(border), 0.0)
    threshold = np.percentile(signal[signal > 0], 75.0)
    weights = np.where(signal >= threshold, signal, 0.0)
    yy, xx = np.indices(values.shape, dtype=np.float64)
    total = weights.sum()
    center_x = float((weights * xx).sum() / total)
    center_y = float((weights * yy).sum() / total)
    covariance_xx = float((weights * (xx - center_x) ** 2).sum() / total)
    covariance_yy = float((weights * (yy - center_y) ** 2).sum() / total)
    covariance_xy = float((weights * (xx - center_x) * (yy - center_y)).sum() / total)
    return float(np.mod(0.5 * np.arctan2(2.0 * covariance_xy, covariance_xx - covariance_yy), np.pi))


def localize_frames(
    frames: np.ndarray,
    *,
    gamma_nm: np.ndarray,
    mode_order: tuple[tuple[int, int], ...],
    angle_calibration: AngleZCalibration,
    config: LocalizationConfig,
    carrier_complex: np.ndarray | torch.Tensor | None = None,
) -> IndependentLocalizations:
    images = np.asarray(frames)
    if images.ndim != 3 or images.shape[1:] != config.image_shape_hw:
        raise ValueError("frames must have shape (N,H,W) matching image_shape_hw.")
    candidates: list[tuple[int, float, float, float, float]] = []
    radius = config.psf_size // 2
    for frame_index, image in enumerate(images):
        detected = detect_lobe_pairs(image, config=config.detection)
        z_initial = angle_calibration.estimate_z(detected.angle_rad)
        for x, y, z, angle, separation in zip(
            detected.x_px,
            detected.y_px,
            z_initial,
            detected.angle_rad,
            detected.separation_px,
            strict=True,
        ):
            center_x, center_y = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
            if (
                center_x - radius >= 0
                and center_y - radius >= 0
                and center_x + radius < images.shape[2]
                and center_y + radius < images.shape[1]
            ):
                candidates.append((frame_index, float(x), float(y), float(z), float(angle), float(separation)))
    if not candidates:
        empty = np.empty(0, dtype=np.float32)
        return IndependentLocalizations(np.empty(0, dtype=np.int64), empty, empty, empty, empty, empty, empty, empty, empty)
    candidate_values = np.asarray(candidates, dtype=np.float64)
    return _refine_candidates(
        images,
        candidate_values,
        gamma_nm=gamma_nm,
        mode_order=mode_order,
        config=config,
        carrier_complex=carrier_complex,
    )


def _refine_candidates(
    frames: np.ndarray,
    candidates: np.ndarray,
    *,
    gamma_nm: np.ndarray,
    mode_order: tuple[tuple[int, int], ...],
    config: LocalizationConfig,
    carrier_complex: np.ndarray | torch.Tensor | None = None,
) -> IndependentLocalizations:
    device = torch.device(config.device)
    model = DoubleHelixVectorPSF(
        mode_order=mode_order,
        na=config.na,
        wavelength_nm=config.wavelength_nm,
        pixel_size_nm=config.pixel_size_nm,
        refractive_index=config.refractive_index,
        npupil=config.npupil,
        psf_size=config.psf_size,
        device=device,
    )
    field = DirectGammaZernikeField(
        gamma_nm=torch.as_tensor(gamma_nm, dtype=torch.float32, device=device),
        mode_order=mode_order,
    )
    carrier = None
    if carrier_complex is not None:
        carrier = torch.as_tensor(carrier_complex, device=device)
    radius = config.psf_size // 2
    outputs = []
    for start in range(0, candidates.shape[0], config.batch_size):
        batch = candidates[start : start + config.batch_size]
        centers_x = np.floor(batch[:, 1] + 0.5).astype(np.int64)
        centers_y = np.floor(batch[:, 2] + 0.5).astype(np.int64)
        patches = np.stack(
            [
                frames[int(row[0]), cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
                for row, cx, cy in zip(batch, centers_x, centers_y, strict=True)
            ]
        ).astype(np.float32)
        observed = torch.as_tensor(patches, dtype=torch.float32, device=device)
        dx = torch.tensor(batch[:, 1] - centers_x, dtype=torch.float32, device=device, requires_grad=True)
        dy = torch.tensor(batch[:, 2] - centers_y, dtype=torch.float32, device=device, requires_grad=True)
        z = torch.tensor(batch[:, 3], dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam(
            (
                {"params": (dx, dy), "lr": config.xy_learning_rate_px},
                {"params": (z,), "lr": config.z_learning_rate_nm},
            )
        )
        centers_x_t = torch.as_tensor(centers_x, dtype=torch.float32, device=device)
        centers_y_t = torch.as_tensor(centers_y, dtype=torch.float32, device=device)
        for _ in range(config.refinement_steps):
            optimizer.zero_grad(set_to_none=True)
            x_global = centers_x_t + dx
            y_global = centers_y_t + dy
            x_normalized = -1.0 + 2.0 * x_global / float(config.image_shape_hw[1])
            y_normalized = -1.0 + 2.0 * y_global / float(config.image_shape_hw[0])
            coefficients = field.evaluate(x_normalized, y_normalized)
            unit_flux = model.render(
                coefficients_nm=coefficients,
                z_nm=z,
                carrier_complex=carrier,
                dx_px=dx,
                dy_px=dy,
            )
            reconstruction, _, _ = profile_photometry(observed, unit_flux)
            (observed - reconstruction).square().mean().backward()
            optimizer.step()
            with torch.no_grad():
                dx.clamp_(-2.0, 2.0)
                dy.clamp_(-2.0, 2.0)
                z.clamp_(*config.z_range_nm)
        with torch.no_grad():
            x_global = centers_x_t + dx
            y_global = centers_y_t + dy
            coefficients = field.evaluate(
                -1.0 + 2.0 * x_global / float(config.image_shape_hw[1]),
                -1.0 + 2.0 * y_global / float(config.image_shape_hw[0]),
            )
            unit_flux = model.render(
                coefficients_nm=coefficients,
                z_nm=z,
                carrier_complex=carrier,
                dx_px=dx,
                dy_px=dy,
            )
            _, photons, background = profile_photometry(observed, unit_flux)
            observed_unit = ((observed - background[:, None, None]) / photons[:, None, None]).clamp_min(0.0)
            observed_unit = observed_unit / observed_unit.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            ncc = normalized_cross_correlation(observed_unit, unit_flux)
        outputs.append(
            np.column_stack(
                (
                    batch[:, 0],
                    x_global.cpu().numpy(),
                    y_global.cpu().numpy(),
                    z.detach().cpu().numpy(),
                    photons.cpu().numpy(),
                    background.cpu().numpy(),
                    ncc.cpu().numpy(),
                    batch[:, 4],
                    batch[:, 5],
                )
            )
        )
    values = np.concatenate(outputs, axis=0)
    keep = values[:, 6] >= config.minimum_ncc
    values = values[keep]
    return IndependentLocalizations(
        frame_index=values[:, 0].astype(np.int64),
        x_px=values[:, 1].astype(np.float32),
        y_px=values[:, 2].astype(np.float32),
        z_nm=values[:, 3].astype(np.float32),
        photons_adu=values[:, 4].astype(np.float32),
        background_adu=values[:, 5].astype(np.float32),
        ncc=values[:, 6].astype(np.float32),
        lobe_angle_rad=values[:, 7].astype(np.float32),
        lobe_separation_px=values[:, 8].astype(np.float32),
    )


def match_localizations(
    prediction_frame: np.ndarray,
    prediction_xyz_nm: np.ndarray,
    gt_frame: np.ndarray,
    gt_xyz_nm: np.ndarray,
    *,
    lateral_tolerance_nm: float,
    axial_tolerance_nm: float,
) -> MatchResult:
    prediction_frames = np.asarray(prediction_frame, dtype=np.int64).reshape(-1)
    gt_frames = np.asarray(gt_frame, dtype=np.int64).reshape(-1)
    prediction_xyz = np.asarray(prediction_xyz_nm, dtype=np.float64)
    ground_truth_xyz = np.asarray(gt_xyz_nm, dtype=np.float64)
    prediction_to_gt = np.full(prediction_frames.size, -1, dtype=np.int64)
    gt_to_prediction = np.full(gt_frames.size, -1, dtype=np.int64)
    for frame in np.union1d(prediction_frames, gt_frames):
        prediction_rows = np.flatnonzero(prediction_frames == frame)
        gt_rows = np.flatnonzero(gt_frames == frame)
        if prediction_rows.size == 0 or gt_rows.size == 0:
            continue
        delta = prediction_xyz[prediction_rows, None] - ground_truth_xyz[None, gt_rows]
        lateral = np.hypot(delta[:, :, 0], delta[:, :, 1])
        axial = np.abs(delta[:, :, 2])
        valid = (lateral <= lateral_tolerance_nm) & (axial <= axial_tolerance_nm)
        cost = np.sqrt((lateral / lateral_tolerance_nm) ** 2 + (axial / axial_tolerance_nm) ** 2)
        cost[~valid] = 1e6
        prediction_assignment, gt_assignment = linear_sum_assignment(cost)
        for prediction_local, gt_local in zip(prediction_assignment, gt_assignment, strict=True):
            if cost[prediction_local, gt_local] >= 1e6:
                continue
            prediction_index = prediction_rows[prediction_local]
            gt_index = gt_rows[gt_local]
            prediction_to_gt[prediction_index] = gt_index
            gt_to_prediction[gt_index] = prediction_index
    matched_prediction = np.flatnonzero(prediction_to_gt >= 0)
    matched_gt = prediction_to_gt[matched_prediction]
    errors = prediction_xyz[matched_prediction] - ground_truth_xyz[matched_gt]
    return MatchResult(
        prediction_to_gt=prediction_to_gt,
        gt_to_prediction=gt_to_prediction,
        true_positives=int(matched_prediction.size),
        false_positives=int(np.count_nonzero(prediction_to_gt < 0)),
        false_negatives=int(np.count_nonzero(gt_to_prediction < 0)),
        dx_nm=errors[:, 0],
        dy_nm=errors[:, 1],
        dz_nm=errors[:, 2],
    )


__all__ = [
    "AngleZCalibration",
    "IndependentLocalizations",
    "LobeDetectionConfig",
    "LobeDetections",
    "LocalizationConfig",
    "MatchResult",
    "detect_lobe_pairs",
    "localize_frames",
    "match_localizations",
]
