from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from neptune_v03.localization.smlm_output import SMLMOutputChannels


@dataclass(frozen=True)
class LegacyEmitterSet:
    batch_index: torch.Tensor
    probability: torch.Tensor
    xyz_px_nm: torch.Tensor
    photons: torch.Tensor
    sigma_xy_px: torch.Tensor


@dataclass(frozen=True)
class LegacyEvalMetrics:
    precision: float
    recall: float
    jaccard: float
    rmse_lat: float
    rmse_ax: float
    predicted_emitters: float
    target_emitters: float

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "jaccard": self.jaccard,
            "rmse_lat": self.rmse_lat,
            "rmse_ax": self.rmse_ax,
            "predicted_emitters": self.predicted_emitters,
            "target_emitters": self.target_emitters,
        }


def spatial_integration_probability(
    p: torch.Tensor,
    *,
    raw_th: float = 0.3,
    split_th: float = 0.6,
    aggregation: str = "norm_sum",
) -> torch.Tensor:
    if p.ndim != 3:
        raise ValueError(f"p must have shape (N,H,W), got {tuple(p.shape)}")
    aggregate = _aggregation_fn(aggregation)
    filt = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=p.dtype,
        device=p.device,
    ).view(1, 1, 3, 3)
    conv = torch.nn.functional.conv2d(p.unsqueeze(1), filt, padding=1)
    p_clip = torch.where(p > float(raw_th), p, torch.zeros_like(p))
    pool = torch.nn.functional.max_pool2d(p_clip.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    max_mask1 = torch.eq(p.unsqueeze(1), pool)
    p_ps1 = max_mask1.to(dtype=p.dtype) * conv

    p_copy = p.unsqueeze(1).clone()
    p_copy *= 1.0 - max_mask1.to(dtype=p.dtype)
    max_mask2 = torch.where(p_copy > float(split_th), torch.ones_like(p_copy), torch.zeros_like(p_copy))
    p_ps2 = max_mask2 * conv
    return aggregate(p_ps1, p_ps2).squeeze(1)


def decode_legacy_smlm_emitters(
    y_out: torch.Tensor,
    *,
    raw_th: float = 0.3,
    split_th: float = 0.6,
    accept_th: float = 0.5,
    photon_scale: float | None = None,
    z_scale: float | None = None,
    aggregation: str = "norm_sum",
    max_emitters: int | None = None,
) -> LegacyEmitterSet:
    if y_out.ndim != 4 or int(y_out.shape[1]) != SMLMOutputChannels.count:
        raise ValueError(f"expected SMLM output shape (N,10,H,W), got {tuple(y_out.shape)}")
    out = y_out.detach()
    p = spatial_integration_probability(out[:, SMLMOutputChannels.p], raw_th=raw_th, split_th=split_th, aggregation=aggregation)
    batch_size, height, width = int(p.shape[0]), int(p.shape[1]), int(p.shape[2])
    rows, cols = torch.meshgrid(
        torch.arange(height, dtype=out.dtype, device=out.device),
        torch.arange(width, dtype=out.dtype, device=out.device),
        indexing="ij",
    )
    active = p >= float(accept_th)
    if max_emitters is not None:
        active = _topk_mask(p, active, max_emitters=int(max_emitters))
    if not bool(active.any()):
        return _empty_emitters(device=out.device)

    batch_ix = active.nonzero(as_tuple=False)[:, 0]
    row_ix = active.nonzero(as_tuple=False)[:, 1]
    col_ix = active.nonzero(as_tuple=False)[:, 2]
    z = out[batch_ix, SMLMOutputChannels.z_mu, row_ix, col_ix]
    photons = out[batch_ix, SMLMOutputChannels.photons_mu, row_ix, col_ix]
    if z_scale is not None:
        scale = abs(float(z_scale))
        scale_nm = scale * 1000.0 if scale <= 10.0 else scale
        z = z * scale_nm
    if photon_scale is not None:
        photons = photons * float(photon_scale)
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
    if int(match_dims) not in {2, 3}:
        raise ValueError(f"match_dims must be 2 or 3, got {match_dims!r}")
    true_positive = 0
    lateral_sq_errors: list[float] = []
    axial_sq_errors: list[float] = []
    pred_count = int(pred.xyz_px_nm.shape[0])
    target_count = int(target.xyz_px_nm.shape[0])
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
            true_positive += 1
            matched_predictions.add(int(pi))
            matched_targets.add(int(ti))
            lateral_sq_errors.append(float(lat_sq))
            axial_sq_errors.append(float(ax_sq))
    false_positive = pred_count - true_positive
    false_negative = target_count - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    jaccard = true_positive / max(true_positive + false_positive + false_negative, 1)
    rmse_lat = (sum(lateral_sq_errors) / max(len(lateral_sq_errors), 1)) ** 0.5
    rmse_ax = (sum(axial_sq_errors) / max(len(axial_sq_errors), 1)) ** 0.5
    return LegacyEvalMetrics(
        precision=float(precision),
        recall=float(recall),
        jaccard=float(jaccard),
        rmse_lat=float(rmse_lat),
        rmse_ax=float(rmse_ax),
        predicted_emitters=float(pred_count),
        target_emitters=float(target_count),
    )


def _aggregation_fn(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    key = str(name or "norm_sum")
    if key == "sum":
        return torch.add
    if key == "max":
        return torch.max
    if key == "norm_sum":
        return lambda left, right: torch.clamp(torch.add(left, right), 0.0, 1.0)
    raise ValueError(f"unsupported spatial integration aggregation: {name!r}")


def _topk_mask(p: torch.Tensor, active: torch.Tensor, *, max_emitters: int) -> torch.Tensor:
    if int(max_emitters) <= 0:
        return torch.zeros_like(active)
    limited = torch.zeros_like(active)
    for batch_idx in range(int(p.shape[0])):
        values = torch.where(active[batch_idx], p[batch_idx], torch.full_like(p[batch_idx], -torch.inf))
        keep = min(int(max_emitters), int(active[batch_idx].sum().item()))
        if keep <= 0:
            continue
        indices = torch.topk(values.reshape(-1), k=keep).indices
        limited[batch_idx].reshape(-1)[indices] = True
    return limited


def _empty_emitters(*, device: torch.device) -> LegacyEmitterSet:
    return LegacyEmitterSet(
        batch_index=torch.empty((0,), dtype=torch.long, device="cpu"),
        probability=torch.empty((0,), dtype=torch.float32, device="cpu"),
        xyz_px_nm=torch.empty((0, 3), dtype=torch.float32, device="cpu"),
        photons=torch.empty((0,), dtype=torch.float32, device="cpu"),
        sigma_xy_px=torch.empty((0, 2), dtype=torch.float32, device="cpu"),
    )


__all__ = [
    "LegacyEmitterSet",
    "LegacyEvalMetrics",
    "decode_legacy_smlm_emitters",
    "decode_legacy_targets",
    "evaluate_legacy_localizations",
    "spatial_integration_probability",
]
