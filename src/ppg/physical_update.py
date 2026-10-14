"""Gamma-constrained physical updates for a spatial PPG map from sample ROIs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ppg.spatial_gamma import apply_gamma_updates, coefficient_field_from_gamma
from unity_psf.data.normalization import LunarInputNormalization
from unity_psf.localization.legacy_decode import decode_liteloc_formal_infer_emitters

from unity_psf.roi_library import (
    InferredEmitter,
    ROIBank,
    ROIBankBuildConfig,
    ROIBankDomain,
    RawInferenceResult,
    build_roi_bank_from_inference,
    save_roi_bank,
)


def should_update_ppg(epoch: int, *, start_epoch: int = 30, interval: int = 10) -> bool:
    return (
        int(epoch) >= int(start_epoch)
        and (int(epoch) - int(start_epoch)) % int(interval) == 0
    )


def resolve_roi_update_budget(config: dict[str, Any]) -> tuple[int, int]:
    """Return the UnityPSF ROI pool cap and per-update cap."""
    pool_max = int(config.get("roi_library_max_rois", config.get("max_rois", 256)))
    update_max = int(
        config.get("roi_update_max_rois", config.get("max_rois", pool_max))
    )
    if pool_max <= 0 or update_max <= 0:
        raise ValueError("ROI pool and update budgets must be positive")
    if update_max > pool_max:
        raise ValueError("roi_update_max_rois cannot exceed roi_library_max_rois")
    return pool_max, update_max


def normalized_field_xy(
    x_px: float,
    y_px: float,
    *,
    field_shape_yx: tuple[int, int],
) -> tuple[float, float]:
    height, width = (int(value) for value in field_shape_yx)
    return (
        float(np.clip(float(x_px) / max(width - 1, 1), 0.0, 1.0)),
        float(np.clip(float(y_px) / max(height - 1, 1), 0.0, 1.0)),
    )


def _normalized_signal(image: torch.Tensor) -> torch.Tensor:
    signal = image - torch.quantile(image.flatten(), 0.2)
    signal = signal.clamp_min(0.0)
    return signal / signal.sum().clamp_min(1e-8)


def _ncc(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    return (left_centered * right_centered).sum() / (
        left_centered.square().sum().sqrt() * right_centered.square().sum().sqrt()
    ).clamp_min(1e-8)


def _poisson_nll_from_lunar_normalized(
    expected_normalized: torch.Tensor,
    observed_normalized: torch.Tensor,
    *,
    normalization: LunarInputNormalization,
) -> torch.Tensor:
    expected_counts = normalization.denormalize(expected_normalized).clamp_min(1e-6)
    observed_counts = normalization.denormalize(observed_normalized)
    return (
        expected_counts - observed_counts * torch.log(expected_counts)
    ).mean() / normalization.scale_ph_factor


def physical_update_quality_reasons(
    *,
    ncc_after: float,
    poisson_nll_before: float,
    poisson_nll_after: float,
    ncc_threshold: float,
) -> list[str]:
    values = (float(ncc_after), float(poisson_nll_before), float(poisson_nll_after))
    if not all(np.isfinite(value) for value in values):
        return ["non_finite_metric"]
    reasons = []
    if float(ncc_after) < float(ncc_threshold):
        reasons.append("ncc_below_threshold")
    if float(poisson_nll_after) >= float(poisson_nll_before):
        reasons.append("poisson_nll_not_improved")
    return reasons


def _reconstruct_at_z(
    coefficients: torch.Tensor,
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    z_train_um: torch.Tensor,
    z_um: float | torch.Tensor,
    roi_shape_yx: tuple[int, int],
) -> torch.Tensor:
    stack = (mean + coefficients @ components).reshape(len(z_train_um), *roi_shape_yx)
    z = torch.as_tensor(z_um, dtype=stack.dtype, device=stack.device).clamp(
        float(z_train_um[0]), float(z_train_um[-1])
    )
    upper = torch.searchsorted(z_train_um, z).clamp(1, len(z_train_um) - 1)
    lower = upper - 1
    weight = (z - z_train_um[lower]) / (z_train_um[upper] - z_train_um[lower])
    return _normalized_signal(stack[lower] * (1.0 - weight) + stack[upper] * weight)


def sample_coefficient_field(
    field: np.ndarray, xy_normalized: tuple[float, float]
) -> np.ndarray:
    values = torch.as_tensor(field, dtype=torch.float32)[None]
    x, y = xy_normalized
    grid = torch.tensor([[[[2.0 * x - 1.0, 2.0 * y - 1.0]]]], dtype=torch.float32)
    return F.grid_sample(
        values, grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, :, 0, 0].numpy()


@dataclass(frozen=True)
class PPGROIEmitter:
    xy_normalized: tuple[float, float]
    local_xy_px: tuple[float, float]
    z_um: float
    photons: float
    weight: float


@dataclass(frozen=True)
class PPGROIUpdateSample:
    observed: np.ndarray
    background: np.ndarray
    emitters: tuple[PPGROIEmitter, ...]
    roi_id: int
    loss_mask: np.ndarray | None = None
    frame_index: int = -1
    frame_window: tuple[int, int] = (-1, -1)


def _emitter_union_mask(
    sample: PPGROIUpdateSample, *, psf_shape_yx: tuple[int, int]
) -> torch.Tensor:
    height, width = (int(v) for v in sample.observed.shape)
    mask = torch.zeros((height, width), dtype=torch.bool)
    for emitter in sample.emitters:
        mask |= _emitter_mask(sample, emitter=emitter, psf_shape_yx=psf_shape_yx)
    return mask


def _emitter_mask(
    sample: PPGROIUpdateSample,
    *,
    emitter: PPGROIEmitter,
    psf_shape_yx: tuple[int, int],
) -> torch.Tensor:
    height, width = (int(v) for v in sample.observed.shape)
    patch_height, patch_width = (int(v) for v in psf_shape_yx)
    mask = torch.zeros((height, width), dtype=torch.bool)
    center_x = int(round(float(emitter.local_xy_px[0])))
    center_y = int(round(float(emitter.local_xy_px[1])))
    patch_left = center_x - patch_width // 2
    patch_top = center_y - patch_height // 2
    x0 = max(0, patch_left)
    y0 = max(0, patch_top)
    x1 = min(width, patch_left + patch_width)
    y1 = min(height, patch_top + patch_height)
    mask[y0:y1, x0:x1] = True
    return mask


def _masked_normalized_signal(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = image[mask]
    signal = (selected - torch.quantile(selected, 0.2)).clamp_min(0.0)
    return signal / signal.sum().clamp_min(1e-8)


def _project_patch_to_roi(
    patch: torch.Tensor,
    *,
    local_xy_px: tuple[float, float],
    roi_shape_yx: tuple[int, int],
) -> torch.Tensor:
    height, width = (int(v) for v in roi_shape_yx)
    patch_height, patch_width = (int(v) for v in patch.shape)
    center_x = int(round(float(local_xy_px[0])))
    center_y = int(round(float(local_xy_px[1])))
    shift_x = (
        torch.as_tensor(local_xy_px[0], dtype=patch.dtype, device=patch.device)
        - center_x
    )
    shift_y = (
        torch.as_tensor(local_xy_px[1], dtype=patch.dtype, device=patch.device)
        - center_y
    )
    theta = torch.eye(2, 3, dtype=patch.dtype, device=patch.device)[None]
    theta[0, 0, 2] = -2.0 * shift_x / max(patch_width - 1, 1)
    theta[0, 1, 2] = -2.0 * shift_y / max(patch_height - 1, 1)
    grid = F.affine_grid(theta, (1, 1, patch_height, patch_width), align_corners=True)
    shifted = F.grid_sample(patch[None, None], grid, align_corners=True)[0, 0]
    left = center_x - patch_width // 2
    top = center_y - patch_height // 2
    source_x0 = max(0, -left)
    source_y0 = max(0, -top)
    source_x1 = min(patch_width, width - left)
    source_y1 = min(patch_height, height - top)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return shifted.new_zeros((height, width))
    clipped = shifted[source_y0:source_y1, source_x0:source_x1]
    target_left = max(0, left)
    target_top = max(0, top)
    return F.pad(
        clipped,
        (
            target_left,
            width - target_left - int(clipped.shape[1]),
            target_top,
            height - target_top - int(clipped.shape[0]),
        ),
    )


def _reconstruct_roi(
    coefficients: torch.Tensor,
    *,
    sample: PPGROIUpdateSample,
    mean: torch.Tensor,
    components: torch.Tensor,
    z_train_um: torch.Tensor,
    psf_shape_yx: tuple[int, int],
    z_delta_raw: torch.Tensor | None = None,
    shift_raw: torch.Tensor | None = None,
    photon_scale_raw: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction = coefficients.new_zeros(sample.observed.shape)
    for index, emitter in enumerate(sample.emitters):
        z_um = torch.as_tensor(
            emitter.z_um, dtype=coefficients.dtype, device=coefficients.device
        )
        local_xy = emitter.local_xy_px
        photons = torch.as_tensor(
            max(emitter.photons, 1.0),
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        if z_delta_raw is not None:
            z_um = z_um + 0.1 * torch.tanh(z_delta_raw[index])
        if shift_raw is not None:
            shift = 0.75 * torch.tanh(shift_raw[index])
            local_xy = (
                torch.as_tensor(emitter.local_xy_px[0], device=shift.device) + shift[0],
                torch.as_tensor(emitter.local_xy_px[1], device=shift.device) + shift[1],
            )
        if photon_scale_raw is not None:
            photons = photons * torch.exp(0.5 * torch.tanh(photon_scale_raw[index]))
        patch = _reconstruct_at_z(
            coefficients[index],
            mean=mean,
            components=components,
            z_train_um=z_train_um,
            z_um=z_um,
            roi_shape_yx=psf_shape_yx,
        )
        prediction = prediction + photons * _project_patch_to_roi(
            patch,
            local_xy_px=local_xy,
            roi_shape_yx=sample.observed.shape,
        )
    return prediction


def fit_roi_emitter_coefficients(
    *,
    sample: PPGROIUpdateSample,
    mean: np.ndarray,
    components: np.ndarray,
    initial_coefficients: np.ndarray,
    z_train_um: np.ndarray,
    psf_shape_yx: tuple[int, int],
    steps: int,
    learning_rate: float,
    regularization: float,
    coefficient_delta_limit: np.ndarray,
    over_cut_px: int,
    normalization: LunarInputNormalization,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_counts = torch.as_tensor(
        sample.observed, dtype=torch.float32, device=device
    )
    target = normalization.normalize(target_counts)
    background = torch.as_tensor(sample.background, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    components_t = torch.as_tensor(components, dtype=torch.float32, device=device)
    z_train_t = torch.as_tensor(z_train_um, dtype=torch.float32, device=device)
    initial = torch.as_tensor(initial_coefficients, dtype=torch.float32, device=device)
    coefficients = initial.clone().requires_grad_(True)
    z_delta_raw = torch.zeros(
        (len(sample.emitters),), dtype=torch.float32, device=device, requires_grad=True
    )
    shift_raw = torch.zeros(
        (len(sample.emitters), 2),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    photon_scale_raw = torch.zeros(
        (len(sample.emitters),), dtype=torch.float32, device=device, requires_grad=True
    )
    delta_limit = torch.as_tensor(
        coefficient_delta_limit, dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(
        (coefficients, z_delta_raw, shift_raw, photon_scale_raw),
        lr=float(learning_rate),
    )
    if sample.loss_mask is None:
        quality_mask = _emitter_union_mask(sample, psf_shape_yx=psf_shape_yx).to(
            device=device
        )
    else:
        quality_mask = torch.as_tensor(
            sample.loss_mask, dtype=torch.bool, device=device
        )
    fit_mask = quality_mask.clone()
    over = int(over_cut_px)
    if over > 0:
        interior = torch.zeros_like(fit_mask)
        interior[over:-over, over:-over] = True
        fit_mask &= interior
    if not bool(fit_mask.any()):
        raise ValueError("physical-update loss mask is empty")

    def reconstruction() -> torch.Tensor:
        signal = _reconstruct_roi(
            coefficients,
            sample=sample,
            mean=mean_t,
            components=components_t,
            z_train_um=z_train_t,
            psf_shape_yx=psf_shape_yx,
            z_delta_raw=z_delta_raw,
            shift_raw=shift_raw,
            photon_scale_raw=photon_scale_raw,
        )
        return normalization.normalize(signal + background)

    def poisson_loss(prediction: torch.Tensor) -> torch.Tensor:
        return _poisson_nll_from_lunar_normalized(
            prediction[fit_mask],
            target[fit_mask],
            normalization=normalization,
        )

    def masked_ncc(prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _ncc(prediction[mask], target[mask])

    with torch.no_grad():
        before_prediction = reconstruction()
        nll_before = poisson_loss(before_prediction)
        ncc_before = masked_ncc(before_prediction, quality_mask)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = reconstruction()
        loss = poisson_loss(prediction) + float(regularization) * F.mse_loss(
            coefficients, initial
        )
        loss = loss + 1e-5 * (
            z_delta_raw.square().mean()
            + shift_raw.square().mean()
            + photon_scale_raw.square().mean()
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            coefficients.copy_(
                initial + (coefficients - initial).clamp(-delta_limit, delta_limit)
            )
    with torch.no_grad():
        after_prediction = reconstruction()
        nll_after = poisson_loss(after_prediction)
        ncc_after = masked_ncc(after_prediction, quality_mask)
        emitter_metrics = []
        for emitter in sample.emitters:
            emitter_mask = _emitter_mask(
                sample, emitter=emitter, psf_shape_yx=psf_shape_yx
            ).to(device=device)
            emitter_fit_mask = emitter_mask & fit_mask
            emitter_metrics.append(
                {
                    "poisson_nll_before": float(
                        _poisson_nll_from_lunar_normalized(
                            before_prediction[emitter_fit_mask],
                            target[emitter_fit_mask],
                            normalization=normalization,
                        ).item()
                    ),
                    "poisson_nll_after": float(
                        _poisson_nll_from_lunar_normalized(
                            after_prediction[emitter_fit_mask],
                            target[emitter_fit_mask],
                            normalization=normalization,
                        ).item()
                    ),
                    "ncc_before": float(
                        masked_ncc(before_prediction, emitter_mask).item()
                    ),
                    "ncc_after": float(
                        masked_ncc(after_prediction, emitter_mask).item()
                    ),
                }
            )
    return coefficients.detach().cpu().numpy(), {
        "poisson_nll_before": float(nll_before.item()),
        "poisson_nll_after": float(nll_after.item()),
        "ncc_before": float(ncc_before.item()),
        "ncc_after": float(ncc_after.item()),
        "emitter_count": len(sample.emitters),
        "mean_abs_z_delta_nm": float(
            (100.0 * torch.tanh(z_delta_raw).abs()).mean().item()
        ),
        "mean_abs_shift_px": float((0.75 * torch.tanh(shift_raw).abs()).mean().item()),
        "mean_abs_log_photon_scale": float(
            (0.5 * torch.tanh(photon_scale_raw).abs()).mean().item()
        ),
        "fitted_coefficients": coefficients.detach().cpu().numpy().tolist(),
        "fitted_z_delta_um": (0.1 * torch.tanh(z_delta_raw))
        .detach()
        .cpu()
        .numpy()
        .tolist(),
        "fitted_shift_xy_px": (0.75 * torch.tanh(shift_raw))
        .detach()
        .cpu()
        .numpy()
        .tolist(),
        "fitted_photon_scale": torch.exp(0.5 * torch.tanh(photon_scale_raw))
        .detach()
        .cpu()
        .numpy()
        .tolist(),
        "emitter_metrics": emitter_metrics,
    }


def _split_roi_samples_by_frame(
    samples: list[Any], *, heldout_fraction: float
) -> tuple[list[Any], list[Any]]:
    ordered = sorted(
        samples, key=lambda item: (int(item.frame_index), int(item.roi_id))
    )
    groups: list[list[Any]] = []
    group_stop = -1
    for sample in ordered:
        window = getattr(
            sample,
            "frame_window",
            (int(sample.frame_index), int(sample.frame_index) + 1),
        )
        start, stop = (int(v) for v in window)
        if not groups or start >= group_stop:
            groups.append([])
            group_stop = stop
        else:
            group_stop = max(group_stop, stop)
        groups[-1].append(sample)
    heldout_stride = max(2, int(round(1.0 / float(heldout_fraction))))
    heldout_group_ids = set(range(0, len(groups), heldout_stride))
    training = [
        sample
        for index, group in enumerate(groups)
        if index not in heldout_group_ids
        for sample in group
    ]
    heldout = [
        sample
        for index, group in enumerate(groups)
        if index in heldout_group_ids
        for sample in group
    ]
    return training, heldout


def _offset_roi_bank_frame_indices(bank: ROIBank, *, frame_offset: int) -> ROIBank:
    offset = int(frame_offset)
    records = tuple(
        replace(
            record,
            frame_window=(
                int(record.frame_window[0]) + offset,
                int(record.frame_window[1]) + offset,
            ),
            emitters=tuple(
                replace(emitter, frame_index=int(emitter.frame_index) + offset)
                for emitter in record.emitters
            ),
        )
        for record in bank.records
    )
    return replace(
        bank,
        records=records,
        metadata={**bank.metadata, "global_frame_offset": offset},
    )


def update_ppg_map(
    source_path: str | Path,
    output_path: str | Path,
    *,
    samples: list[PPGROIUpdateSample],
    device: str,
    steps: int = 60,
    learning_rate: float = 0.03,
    regularization: float = 1e-3,
    normalization: LunarInputNormalization,
    gamma_ridge: float = 0.01,
    gamma_boundary_anchor_strength: float = 1.0,
    gamma_max_abs_delta: float = 0.05,
    ncc_threshold: float = 0.75,
    heldout_fraction: float = 0.2,
    heldout_min_nll_improvement: float = 0.0,
    heldout_min_improved_fraction: float = 0.5,
    over_cut_px: int = 8,
) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    field = arrays["coefficient_field"]
    if "gamma" not in arrays or tuple(arrays["gamma"].shape) != (9, field.shape[0]):
        raise ValueError("physical update requires a 9 x PCA-mode gamma state")
    base_gamma = arrays["gamma"].copy()
    ordered = sorted(samples, key=lambda item: item.roi_id)
    training, heldout = _split_roi_samples_by_frame(
        ordered, heldout_fraction=heldout_fraction
    )
    if len(training) < 4 or len(heldout) < 1:
        raise ValueError(
            "PPG physical update requires at least five spatially distributed ROIs"
        )
    deltas = []
    positions = []
    weights = []
    metrics = []
    coefficient_scale = np.std(arrays["coefficient_values"], axis=0)
    coefficient_delta_limit = np.maximum(2.0 * coefficient_scale, 1e-6)
    rejected_metrics = []
    accepted_emitter_count = 0
    rejected_emitter_count = 0
    psf_shape_yx = tuple(int(v) for v in arrays["roi_shape_yx"])
    for sample in training:
        initial = np.stack(
            [
                sample_coefficient_field(field, emitter.xy_normalized)
                for emitter in sample.emitters
            ],
            axis=0,
        )
        coefficients, item_metrics = fit_roi_emitter_coefficients(
            sample=sample,
            mean=arrays["pca_mean"],
            components=arrays["pca_components"],
            initial_coefficients=initial,
            z_train_um=arrays["train_z_um"],
            psf_shape_yx=psf_shape_yx,
            steps=steps,
            learning_rate=learning_rate,
            regularization=regularization,
            coefficient_delta_limit=coefficient_delta_limit,
            over_cut_px=over_cut_px,
            normalization=normalization,
            device=device,
        )
        emitter_quality = []
        for emitter_index, emitter_metric in enumerate(item_metrics["emitter_metrics"]):
            reasons = physical_update_quality_reasons(
                ncc_after=emitter_metric["ncc_after"],
                poisson_nll_before=emitter_metric["poisson_nll_before"],
                poisson_nll_after=emitter_metric["poisson_nll_after"],
                ncc_threshold=ncc_threshold,
            )
            emitter_quality.append(
                {
                    "emitter_index": emitter_index,
                    **emitter_metric,
                    "accepted": not reasons,
                    "rejection_reasons": reasons,
                }
            )
        item_metrics["emitter_metrics"] = emitter_quality
        accepted = any(item["accepted"] for item in emitter_quality)
        item = {"roi_id": sample.roi_id, "accepted": accepted, **item_metrics}
        (metrics if accepted else rejected_metrics).append(item)
        if accepted:
            for emitter, fitted, initial_emitter, quality in zip(
                sample.emitters, coefficients, initial, emitter_quality, strict=True
            ):
                if not quality["accepted"]:
                    rejected_emitter_count += 1
                    continue
                radius_y, radius_x = psf_shape_yx[0] // 2, psf_shape_yx[1] // 2
                x_px, y_px = emitter.local_xy_px
                if not (
                    radius_x <= x_px < sample.observed.shape[1] - radius_x
                    and radius_y <= y_px < sample.observed.shape[0] - radius_y
                ):
                    continue
                deltas.append(fitted - initial_emitter)
                positions.append(emitter.xy_normalized)
                weights.append(emitter.weight)
                accepted_emitter_count += 1
        else:
            rejected_emitter_count += len(emitter_quality)
    insufficient_rois = len(metrics) < 4
    insufficient_emitters = accepted_emitter_count < 4
    if insufficient_rois or insufficient_emitters:
        arrays["coefficient_field"] = field
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
        return {
            "map_accepted": False,
            "rejection_reason": (
                "insufficient_quality_controlled_rois"
                if insufficient_rois
                else "insufficient_quality_controlled_emitters"
            ),
            "candidate_roi_count": len(samples),
            "training_roi_count": len(training),
            "heldout_roi_count": len(heldout),
            "accepted_roi_count": len(metrics),
            "rejected_roi_count": len(rejected_metrics),
            "accepted_emitter_count": accepted_emitter_count,
            "rejected_emitter_count": rejected_emitter_count,
            "coefficient_field_delta_l2": 0.0,
            "gamma_delta_l2": 0.0,
            "update_parameterization": "36_gamma_coefficients",
            "input_normalization": normalization.to_config(),
            "roi_metrics": metrics,
            "rejected_roi_metrics": rejected_metrics,
        }
    candidate_gamma = apply_gamma_updates(
        base_gamma,
        positions_xy_normalized=np.asarray(positions),
        deltas=np.asarray(deltas),
        weights=np.asarray(weights),
        ridge=gamma_ridge,
        boundary_anchor_strength=gamma_boundary_anchor_strength,
        max_abs_delta=gamma_max_abs_delta,
    )
    candidate_field = coefficient_field_from_gamma(
        candidate_gamma, arrays["xy_grid_normalized"]
    ).astype(field.dtype)
    heldout_before = _evaluate_roi_samples(
        heldout,
        field=field,
        arrays=arrays,
        device=device,
        normalization=normalization,
        over_cut_px=over_cut_px,
    )
    heldout_after = _evaluate_roi_samples(
        heldout,
        field=candidate_field,
        arrays=arrays,
        device=device,
        normalization=normalization,
        over_cut_px=over_cut_px,
    )
    heldout_improvement = (
        heldout_before["mean_poisson_nll"] - heldout_after["mean_poisson_nll"]
    )
    improved_fraction = float(
        np.mean(
            np.asarray(heldout_after["poisson_nll_values"])
            < np.asarray(heldout_before["poisson_nll_values"])
        )
    )
    heldout_ncc_pass_fraction = float(
        np.mean(np.asarray(heldout_after["ncc_values"]) >= float(ncc_threshold))
    )
    map_accepted = (
        heldout_improvement > float(heldout_min_nll_improvement)
        and improved_fraction >= float(heldout_min_improved_fraction)
        and heldout_ncc_pass_fraction >= float(heldout_min_improved_fraction)
    )
    if map_accepted:
        arrays["gamma"] = candidate_gamma
        arrays["coefficient_field"] = candidate_field
    else:
        arrays["coefficient_field"] = field
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {
        "map_accepted": bool(map_accepted),
        "candidate_roi_count": len(samples),
        "training_roi_count": len(training),
        "heldout_roi_count": len(heldout),
        "accepted_roi_count": len(metrics),
        "rejected_roi_count": len(rejected_metrics),
        "accepted_emitter_count": accepted_emitter_count,
        "rejected_emitter_count": rejected_emitter_count,
        "mean_poisson_nll_before": float(
            np.mean([item["poisson_nll_before"] for item in metrics])
        ),
        "mean_poisson_nll_after": float(
            np.mean([item["poisson_nll_after"] for item in metrics])
        ),
        "coefficient_field_delta_l2": float(
            np.linalg.norm(arrays["coefficient_field"] - field)
        ),
        "gamma_delta_l2": float(np.linalg.norm(candidate_gamma - base_gamma))
        if map_accepted
        else 0.0,
        "update_parameterization": "36_gamma_coefficients",
        "input_normalization": normalization.to_config(),
        "heldout_mean_poisson_nll_before": heldout_before["mean_poisson_nll"],
        "heldout_mean_poisson_nll_after": heldout_after["mean_poisson_nll"],
        "heldout_mean_poisson_nll_improvement": heldout_improvement,
        "heldout_improved_fraction": improved_fraction,
        "heldout_ncc_pass_fraction": heldout_ncc_pass_fraction,
        "roi_metrics": metrics,
        "rejected_roi_metrics": rejected_metrics,
    }


def _evaluate_roi_samples(
    samples: list[PPGROIUpdateSample],
    *,
    field: np.ndarray,
    arrays: dict[str, np.ndarray],
    device: str,
    normalization: LunarInputNormalization,
    over_cut_px: int = 8,
) -> dict[str, float]:
    ncc = []
    poisson_nll = []
    for sample in samples:
        coefficients = np.stack(
            [
                sample_coefficient_field(field, emitter.xy_normalized)
                for emitter in sample.emitters
            ],
            axis=0,
        )
        target = normalization.normalize(
            torch.as_tensor(sample.observed, dtype=torch.float32, device=device)
        )
        prediction = _reconstruct_roi(
            torch.as_tensor(coefficients, dtype=torch.float32, device=device),
            sample=sample,
            mean=torch.as_tensor(
                arrays["pca_mean"], dtype=torch.float32, device=device
            ),
            components=torch.as_tensor(
                arrays["pca_components"], dtype=torch.float32, device=device
            ),
            z_train_um=torch.as_tensor(
                arrays["train_z_um"], dtype=torch.float32, device=device
            ),
            psf_shape_yx=tuple(int(v) for v in arrays["roi_shape_yx"]),
        )
        prediction = normalization.normalize(
            prediction
            + torch.as_tensor(sample.background, dtype=torch.float32, device=device)
        )
        if sample.loss_mask is None:
            quality_mask = _emitter_union_mask(
                sample, psf_shape_yx=tuple(int(v) for v in arrays["roi_shape_yx"])
            ).to(device=device)
        else:
            quality_mask = torch.as_tensor(
                sample.loss_mask, dtype=torch.bool, device=device
            )
        over = int(over_cut_px)
        interior = torch.zeros_like(quality_mask)
        interior[over:-over, over:-over] = True
        fit_mask = quality_mask & interior
        ncc.append(float(_ncc(prediction[quality_mask], target[quality_mask]).item()))
        poisson_nll.append(
            float(
                _poisson_nll_from_lunar_normalized(
                    prediction[fit_mask],
                    target[fit_mask],
                    normalization=normalization,
                ).item()
            )
        )
    return {
        "mean_ncc": float(np.mean(ncc)),
        "ncc_values": ncc,
        "mean_poisson_nll": float(np.mean(poisson_nll)),
        "poisson_nll_values": poisson_nll,
    }


def build_sample_roi_bank(
    *,
    model: torch.nn.Module,
    sample_tiff: str | Path,
    output_h5: str | Path,
    frame_shape_yx: tuple[int, int],
    frame_range: tuple[int, int] = (0, 32),
    probability_threshold: float = 0.3,
    grid_shape: tuple[int, int] = (8, 8),
    max_rois: int = 256,
    roi_size_px: int = 96,
    psf_support_px: int = 21,
    z_scale_nm: float = 600.0,
    photon_scale: float = 31000.0,
    condition_provider=None,
) -> ROIBank:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    normalization = getattr(model, "input_normalization", None)
    if normalization is None:
        raise RuntimeError("sample ROI detection requires LUNAR input normalization")
    background_mean = float(normalization.background_mean)

    def infer_fn(*, domain, frame_window, raw_domain_frames_photon):
        raw = np.asarray(raw_domain_frames_photon, dtype=np.float32)
        inputs = torch.as_tensor(raw[None], device=device, dtype=torch.float32)
        if condition_provider is not None:
            from ppg.condition_input import ppg_model_input

            inputs = ppg_model_input(
                inputs,
                provider=condition_provider,
                origin_xy_px=(0, 0),
                source_field_shape_yx=frame_shape_yx,
            )
        with torch.no_grad():
            output = model(inputs)
            decoded = decode_liteloc_formal_infer_emitters(
                output,
                accept_threshold=float(probability_threshold),
                photon_scale=float(photon_scale),
                z_scale=float(z_scale_nm),
            )
            emitters = []
            edge = int(psf_support_px) // 2 + 1
            height, width = raw.shape[-2:]
            accepted = [
                index
                for index, xyz in enumerate(decoded.xyz_px_nm)
                if edge <= float(xyz[0]) < width - edge
                and edge <= float(xyz[1]) < height - edge
            ]
            if len(accepted) > 512:
                scores = decoded.probability[accepted]
                accepted = [accepted[index] for index in torch.topk(scores, 512).indices.tolist()]
            for index in accepted:
                x, y, z_nm = (float(value) for value in decoded.xyz_px_nm[index])
                emitters.append(
                    InferredEmitter(
                        probability=float(decoded.probability[index]),
                        mu_xy_px=(x, y),
                        sigma_xy_px=tuple(float(value) for value in decoded.sigma_xy_px[index]),
                        mu_z_nm=z_nm,
                        sigma_z_nm=float(decoded.sigma_z_nm[index]),
                        mu_photons=float(decoded.photons[index]),
                        sigma_photons=float(decoded.sigma_photons[index]),
                        cell_xy_px=(float(np.floor(x)), float(np.floor(y))),
                        frame_index=int(frame_window[0]) + len(raw) // 2,
                    )
                )
        background = np.full(raw.shape[-2:], background_mean, dtype=np.float32)
        return RawInferenceResult(emitters=tuple(emitters), background_mu=background)

    height, width = (int(v) for v in frame_shape_yx)
    bank = build_roi_bank_from_inference(
        raw_frames_photon=sample_tiff,
        domains=(ROIBankDomain("sample", 0, 0, width, height),),
        infer_fn=infer_fn,
        config=ROIBankBuildConfig(
            roi_size_px=int(roi_size_px),
            window_size=3,
            frame_range=tuple(int(v) for v in frame_range),
            grid_shape=tuple(int(v) for v in grid_shape),
            max_rois=int(max_rois),
            target_emitters=int(max_rois),
            candidate_probability_threshold=float(probability_threshold),
            probability_threshold=float(probability_threshold),
            max_overlap_fraction=0.95,
            background_smoothing_kernel=3,
            over_cut_px=2,
        ),
    )
    bank = _offset_roi_bank_frame_indices(bank, frame_offset=int(frame_range[0]))
    multi_emitter_bank = ROIBank(
        records=bank.records,
        config=bank.config,
        metadata={
            **bank.metadata,
            "source_tiff": str(Path(sample_tiff).resolve()),
            "source_field_shape_yx": [height, width],
            "coordinate_mapping": "xy/(size-1)",
            "multi_emitter_policy": "retain_all_quality_controlled_emitters_for_joint_projection",
            "psf_support_px": int(psf_support_px),
            "intensity_contract": "raw_identity_camera_with_global_lunar_background",
            "input_normalization": normalization.to_config(),
        },
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
    )
    save_roi_bank(multi_emitter_bank, output_h5)
    if was_training:
        model.train()
    return multi_emitter_bank


def samples_from_roi_bank(
    bank: ROIBank, *, field_shape_yx: tuple[int, int]
) -> list[PPGROIUpdateSample]:
    samples = []
    for record in bank.records:
        if not record.emitters:
            continue
        frame_indices = {int(emitter.frame_index) for emitter in record.emitters}
        if len(frame_indices) != 1:
            continue
        frame_index = frame_indices.pop()
        frame_offset = frame_index - int(record.frame_window[0])
        if not 0 <= frame_offset < len(record.raw_frames_photon):
            continue
        observed = np.asarray(record.raw_frames_photon[frame_offset], dtype=np.float32)
        background = np.asarray(record.background_smoothed, dtype=np.float32)
        sample = PPGROIUpdateSample(
            observed=np.asarray(observed, dtype=np.float32),
            background=np.asarray(background, dtype=np.float32),
            emitters=tuple(
                PPGROIEmitter(
                    xy_normalized=normalized_field_xy(
                        emitter.full_xy_px[0],
                        emitter.full_xy_px[1],
                        field_shape_yx=field_shape_yx,
                    ),
                    local_xy_px=tuple(float(v) for v in emitter.local_xy_px),
                    z_um=float(np.clip(emitter.mu_z_nm / 1000.0, -0.6, 0.6)),
                    photons=float(emitter.mu_photons),
                    weight=float(emitter.probability),
                )
                for emitter in record.emitters
            ),
            roi_id=int(record.roi_id),
        )
        mask = _emitter_union_mask(
            sample,
            psf_shape_yx=(int(bank.metadata.get("psf_support_px", 21)),) * 2,
        )
        samples.append(
            PPGROIUpdateSample(
                observed=sample.observed,
                background=sample.background,
                emitters=sample.emitters,
                roi_id=sample.roi_id,
                loss_mask=mask.numpy(),
                frame_index=frame_index,
                frame_window=tuple(int(v) for v in record.frame_window),
            )
        )
    return samples
