from __future__ import annotations

from dataclasses import dataclass
import torch

from unity_psf.localization.smlm_output import SMLMOutputChannels


LITELOC_CANDIDATE_THRESHOLD = 0.3
LITELOC_ADJACENT_THRESHOLD = 0.6
LITELOC_EVAL_THRESHOLD = 0.3
LITELOC_FORMAL_INFER_THRESHOLD = 0.7


@dataclass(frozen=True)
class LegacyEmitterSet:
    batch_index: torch.Tensor
    probability: torch.Tensor
    xyz_px_nm: torch.Tensor
    photons: torch.Tensor
    sigma_xy_px: torch.Tensor
    sigma_z_nm: torch.Tensor | None = None
    sigma_photons: torch.Tensor | None = None


@dataclass(frozen=True)
class LegacyEvalMetrics:
    precision: float
    recall: float
    jaccard: float
    rmse_lat: float | None
    rmse_ax: float | None
    predicted_emitters: float
    target_emitters: float
    true_positive: int
    false_positive: int
    false_negative: int
    lateral_sq_error_nm2_sum: float
    axial_sq_error_nm2_sum: float
    photon_relative_error: float | None
    photon_relative_error_sum: float
    matched_photons: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "jaccard": self.jaccard,
            "rmse_lat": self.rmse_lat,
            "rmse_ax": self.rmse_ax,
            "predicted_emitters": self.predicted_emitters,
            "target_emitters": self.target_emitters,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "lateral_sq_error_nm2_sum": self.lateral_sq_error_nm2_sum,
            "axial_sq_error_nm2_sum": self.axial_sq_error_nm2_sum,
            "photon_relative_error": self.photon_relative_error,
            "photon_relative_error_sum": self.photon_relative_error_sum,
            "matched_photons": self.matched_photons,
        }


@dataclass(frozen=True)
class LegacyLocalizationMatches:
    matched_prediction_indices: tuple[int, ...]
    matched_target_indices: tuple[int, ...]
    lateral_sq_errors_nm2: tuple[float, ...]
    axial_sq_errors_nm2: tuple[float, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_target_indices: tuple[int, ...]


def liteloc_spatial_integration_probability(p: torch.Tensor) -> torch.Tensor:
    if p.ndim != 3:
        raise ValueError(f"p must have shape (N,H,W), got {tuple(p.shape)}")
    filt = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=p.dtype,
        device=p.device,
    ).view(1, 1, 3, 3)
    conv = torch.nn.functional.conv2d(p.unsqueeze(1), filt, padding=1)
    p_clip = torch.where(p > LITELOC_CANDIDATE_THRESHOLD, p, torch.zeros_like(p))
    pool = torch.nn.functional.max_pool2d(p_clip.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    max_mask1 = torch.eq(p.unsqueeze(1), pool)
    p_ps1 = max_mask1.to(dtype=p.dtype) * conv

    p_copy = p.unsqueeze(1).clone()
    p_copy *= 1.0 - max_mask1.to(dtype=p.dtype)
    max_mask2 = torch.where(p_copy > LITELOC_ADJACENT_THRESHOLD, torch.ones_like(p_copy), torch.zeros_like(p_copy))
    p_ps2 = max_mask2 * conv
    return (p_ps1 + p_ps2).squeeze(1)


def decode_liteloc_eval_emitters(
    y_out: torch.Tensor,
    *,
    photon_scale: float | None = None,
    z_scale: float | None = None,
) -> LegacyEmitterSet:
    return _decode_liteloc_emitters(
        y_out,
        accept_threshold=LITELOC_EVAL_THRESHOLD,
        photon_scale=photon_scale,
        z_scale=z_scale,
    )


def decode_liteloc_formal_infer_emitters(
    y_out: torch.Tensor,
    *,
    accept_threshold: float = LITELOC_FORMAL_INFER_THRESHOLD,
    photon_scale: float | None = None,
    z_scale: float | None = None,
) -> LegacyEmitterSet:
    return _decode_liteloc_emitters(
        y_out,
        accept_threshold=float(accept_threshold),
        photon_scale=photon_scale,
        z_scale=z_scale,
    )


def _decode_liteloc_emitters(
    y_out: torch.Tensor,
    *,
    accept_threshold: float,
    photon_scale: float | None,
    z_scale: float | None,
) -> LegacyEmitterSet:
    if y_out.ndim != 4 or int(y_out.shape[1]) != SMLMOutputChannels.count:
        raise ValueError(f"expected SMLM output shape (N,10,H,W), got {tuple(y_out.shape)}")
    out = y_out.detach()
    p = liteloc_spatial_integration_probability(out[:, SMLMOutputChannels.p])
    height, width = int(p.shape[1]), int(p.shape[2])
    rows, cols = torch.meshgrid(
        torch.arange(height, dtype=out.dtype, device=out.device),
        torch.arange(width, dtype=out.dtype, device=out.device),
        indexing="ij",
    )
    active = p > float(accept_threshold)
    if not bool(active.any()):
        return _empty_emitters(device=out.device)

    batch_ix = active.nonzero(as_tuple=False)[:, 0]
    row_ix = active.nonzero(as_tuple=False)[:, 1]
    col_ix = active.nonzero(as_tuple=False)[:, 2]
    z = out[batch_ix, SMLMOutputChannels.z_mu, row_ix, col_ix]
    photons = out[batch_ix, SMLMOutputChannels.photons_mu, row_ix, col_ix]
    sigma_z = out[batch_ix, SMLMOutputChannels.z_sigma, row_ix, col_ix]
    sigma_photons = out[batch_ix, SMLMOutputChannels.photons_sigma, row_ix, col_ix]
    if z_scale is not None:
        scale = abs(float(z_scale))
        scale_nm = scale * 1000.0 if scale <= 10.0 else scale
        z = z * scale_nm
        sigma_z = sigma_z * scale_nm
    if photon_scale is not None:
        photons = photons * float(photon_scale)
        sigma_photons = sigma_photons * float(photon_scale)
    x = cols[row_ix, col_ix] + 0.5 + out[batch_ix, SMLMOutputChannels.x_mu, row_ix, col_ix]
    y = rows[row_ix, col_ix] + 0.5 + out[batch_ix, SMLMOutputChannels.y_mu, row_ix, col_ix]
    return LegacyEmitterSet(
        batch_index=batch_ix.detach().cpu(),
        probability=p[batch_ix, row_ix, col_ix].detach().cpu().to(dtype=torch.float32),
        xyz_px_nm=torch.stack((x, y, z), dim=1).detach().cpu().to(dtype=torch.float32),
        photons=photons.detach().cpu().to(dtype=torch.float32),
        sigma_xy_px=torch.stack(
            (
                out[batch_ix, SMLMOutputChannels.x_sigma, row_ix, col_ix],
                out[batch_ix, SMLMOutputChannels.y_sigma, row_ix, col_ix],
            ),
            dim=1,
        )
        .detach()
        .cpu()
        .to(dtype=torch.float32),
        sigma_z_nm=sigma_z.detach().cpu().to(dtype=torch.float32),
        sigma_photons=sigma_photons.detach().cpu().to(dtype=torch.float32),
    )


def decode_legacy_targets(
    pxyz_tar: torch.Tensor,
    mask_tar: torch.Tensor,
    *,
    target_order: str = "legacy_iwae",
    photon_scale: float | None = None,
    z_scale: float | None = None,
) -> LegacyEmitterSet:
    if pxyz_tar.ndim != 3 or int(pxyz_tar.shape[-1]) != 4:
        raise ValueError(f"pxyz_tar must have shape (N,M,4), got {tuple(pxyz_tar.shape)}")
    if tuple(mask_tar.shape) != (int(pxyz_tar.shape[0]), int(pxyz_tar.shape[1])):
        raise ValueError(f"mask_tar shape mismatch: {tuple(mask_tar.shape)}")
    mask = mask_tar.detach().to(dtype=torch.bool)
    if not bool(mask.any()):
        return _empty_emitters(device=pxyz_tar.device)
    batch_ix, target_ix = mask.nonzero(as_tuple=True)
    values = pxyz_tar.detach()[batch_ix, target_ix]
    order = str(target_order or "legacy_iwae").lower()
    if order in {"legacy_iwae", "iwae", "old", "phot_xyz", "phot,x,y,z", "photons_x_y_z"}:
        photons = values[:, 0]
        x = values[:, 1]
        y = values[:, 2]
        z = values[:, 3]
        if photon_scale is not None:
            photons = photons * float(photon_scale)
        if z_scale is not None:
            scale = abs(float(z_scale))
            scale_nm = scale * 1000.0 if scale <= 10.0 else scale
            z = z * scale_nm
    elif order in {"v03", "xyzph", "x,y,z,phot", "x_y_z_photons"}:
        x = values[:, 0]
        y = values[:, 1]
        z = values[:, 2]
        photons = values[:, 3]
    else:
        raise ValueError(f"unsupported target_order: {target_order!r}")
    return LegacyEmitterSet(
        batch_index=batch_ix.detach().cpu(),
        probability=torch.ones((int(values.shape[0]),), dtype=torch.float32),
        xyz_px_nm=torch.stack((x, y, z), dim=1).detach().cpu().to(dtype=torch.float32),
        photons=photons.detach().cpu().to(dtype=torch.float32),
        sigma_xy_px=torch.zeros((int(values.shape[0]), 2), dtype=torch.float32),
        sigma_z_nm=torch.zeros((int(values.shape[0]),), dtype=torch.float32),
        sigma_photons=torch.zeros((int(values.shape[0]),), dtype=torch.float32),
    )


def evaluate_legacy_localizations(
    pred: LegacyEmitterSet,
    target: LegacyEmitterSet,
    *,
    dist_tol_xy_px: float | None = 1.0,
    dist_tol_xy_nm: float | None = None,
    dist_tol_z_nm: float | None = None,
    pixel_size_nm_x: float = 1.0,
    pixel_size_nm_y: float = 1.0,
    match_dims: int = 2,
) -> LegacyEvalMetrics:
    matches = match_legacy_localizations(
        pred,
        target,
        dist_tol_xy_px=dist_tol_xy_px,
        dist_tol_xy_nm=dist_tol_xy_nm,
        dist_tol_z_nm=dist_tol_z_nm,
        pixel_size_nm_x=pixel_size_nm_x,
        pixel_size_nm_y=pixel_size_nm_y,
        match_dims=match_dims,
    )
    true_positive = len(matches.matched_prediction_indices)
    false_positive = len(matches.unmatched_prediction_indices)
    false_negative = len(matches.unmatched_target_indices)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    jaccard = true_positive / max(true_positive + false_positive + false_negative, 1)
    rmse_lat = (
        (sum(matches.lateral_sq_errors_nm2) / true_positive) ** 0.5
        if true_positive
        else None
    )
    rmse_ax = (
        (sum(matches.axial_sq_errors_nm2) / true_positive) ** 0.5
        if true_positive
        else None
    )
    photon_relative_errors = [
        abs(float(pred.photons[pred_index].item()) - float(target.photons[target_index].item()))
        / max(abs(float(target.photons[target_index].item())), 1e-12)
        for pred_index, target_index in zip(
            matches.matched_prediction_indices,
            matches.matched_target_indices,
        )
    ]
    photon_relative_error_sum = sum(photon_relative_errors)
    return LegacyEvalMetrics(
        precision=float(precision),
        recall=float(recall),
        jaccard=float(jaccard),
        rmse_lat=None if rmse_lat is None else float(rmse_lat),
        rmse_ax=None if rmse_ax is None else float(rmse_ax),
        predicted_emitters=float(true_positive + false_positive),
        target_emitters=float(true_positive + false_negative),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        lateral_sq_error_nm2_sum=float(sum(matches.lateral_sq_errors_nm2)),
        axial_sq_error_nm2_sum=float(sum(matches.axial_sq_errors_nm2)),
        photon_relative_error=(
            float(photon_relative_error_sum / len(photon_relative_errors))
            if photon_relative_errors
            else None
        ),
        photon_relative_error_sum=float(photon_relative_error_sum),
        matched_photons=len(photon_relative_errors),
    )


def match_legacy_localizations(
    pred: LegacyEmitterSet,
    target: LegacyEmitterSet,
    *,
    dist_tol_xy_px: float | None = 1.0,
    dist_tol_xy_nm: float | None = None,
    dist_tol_z_nm: float | None = None,
    pixel_size_nm_x: float = 1.0,
    pixel_size_nm_y: float = 1.0,
    match_dims: int = 2,
) -> LegacyLocalizationMatches:
    if int(match_dims) not in {2, 3}:
        raise ValueError(f"match_dims must be 2 or 3, got {match_dims!r}")
    matched_prediction_indices: list[int] = []
    matched_target_indices: list[int] = []
    lateral_sq_errors_nm2: list[float] = []
    axial_sq_errors_nm2: list[float] = []
    for batch in sorted(set(pred.batch_index.tolist()) | set(target.batch_index.tolist())):
        pred_ix = (pred.batch_index == int(batch)).nonzero(as_tuple=False).flatten()
        target_ix = (target.batch_index == int(batch)).nonzero(as_tuple=False).flatten()
        matched_predictions: set[int] = set()
        matched_targets: set[int] = set()
        candidates: list[tuple[float, int, int, float, float]] = []
        for pi in pred_ix.tolist():
            for ti in target_ix.tolist():
                delta = pred.xyz_px_nm[pi] - target.xyz_px_nm[ti]
                dx_nm = float(delta[0].item()) * float(pixel_size_nm_x)
                dy_nm = float(delta[1].item()) * float(pixel_size_nm_y)
                lateral_sq_nm = dx_nm * dx_nm + dy_nm * dy_nm
                lateral_nm = lateral_sq_nm**0.5
                lateral_px = float(torch.linalg.vector_norm(delta[:2]).item())
                axial = abs(float(delta[2].item()))
                if dist_tol_xy_nm is not None:
                    lateral_ok = lateral_nm <= float(dist_tol_xy_nm)
                elif dist_tol_xy_px is not None:
                    lateral_ok = lateral_px <= float(dist_tol_xy_px)
                else:
                    lateral_ok = True
                axial_ok = dist_tol_z_nm is None or axial <= float(dist_tol_z_nm)
                if lateral_ok and axial_ok:
                    sort_distance = (lateral_sq_nm + axial * axial) ** 0.5 if int(match_dims) == 3 else lateral_nm
                    candidates.append((sort_distance, int(pi), int(ti), lateral_sq_nm, axial * axial))
        for _, pi, ti, lat_sq, ax_sq in sorted(candidates, key=lambda item: item[0]):
            if int(pi) in matched_predictions or int(ti) in matched_targets:
                continue
            matched_predictions.add(int(pi))
            matched_targets.add(int(ti))
            matched_prediction_indices.append(int(pi))
            matched_target_indices.append(int(ti))
            lateral_sq_errors_nm2.append(float(lat_sq))
            axial_sq_errors_nm2.append(float(ax_sq))
    matched_prediction_set = set(matched_prediction_indices)
    matched_target_set = set(matched_target_indices)
    return LegacyLocalizationMatches(
        matched_prediction_indices=tuple(matched_prediction_indices),
        matched_target_indices=tuple(matched_target_indices),
        lateral_sq_errors_nm2=tuple(lateral_sq_errors_nm2),
        axial_sq_errors_nm2=tuple(axial_sq_errors_nm2),
        unmatched_prediction_indices=tuple(
            index for index in range(int(pred.xyz_px_nm.shape[0])) if index not in matched_prediction_set
        ),
        unmatched_target_indices=tuple(
            index for index in range(int(target.xyz_px_nm.shape[0])) if index not in matched_target_set
        ),
    )


def _empty_emitters(*, device: torch.device) -> LegacyEmitterSet:
    return LegacyEmitterSet(
        batch_index=torch.empty((0,), dtype=torch.long, device="cpu"),
        probability=torch.empty((0,), dtype=torch.float32, device="cpu"),
        xyz_px_nm=torch.empty((0, 3), dtype=torch.float32, device="cpu"),
        photons=torch.empty((0,), dtype=torch.float32, device="cpu"),
        sigma_xy_px=torch.empty((0, 2), dtype=torch.float32, device="cpu"),
        sigma_z_nm=torch.empty((0,), dtype=torch.float32, device="cpu"),
        sigma_photons=torch.empty((0,), dtype=torch.float32, device="cpu"),
    )


__all__ = [
    "LITELOC_ADJACENT_THRESHOLD",
    "LITELOC_CANDIDATE_THRESHOLD",
    "LITELOC_EVAL_THRESHOLD",
    "LITELOC_FORMAL_INFER_THRESHOLD",
    "LegacyEmitterSet",
    "LegacyEvalMetrics",
    "LegacyLocalizationMatches",
    "decode_liteloc_eval_emitters",
    "decode_liteloc_formal_infer_emitters",
    "decode_legacy_targets",
    "evaluate_legacy_localizations",
    "liteloc_spatial_integration_probability",
    "match_legacy_localizations",
]
