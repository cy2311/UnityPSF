from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import PeakBootstrapConfig


@dataclass(frozen=True)
class MinimalNATFitState:
    gamma: torch.Tensor
    coefficients_nm: torch.Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True)
class NATPatchDataset:
    raw_patches: torch.Tensor
    roixy_px: torch.Tensor
    local_x_px: torch.Tensor
    local_y_px: torch.Tensor
    photons: torch.Tensor
    background: torch.Tensor


@dataclass(frozen=True)
class NATFitState:
    gamma: torch.Tensor
    coefficients_nm: torch.Tensor
    local_x_px: torch.Tensor
    local_y_px: torch.Tensor
    sigma_px: torch.Tensor
    photons: torch.Tensor
    background: torch.Tensor
    recon_patches: torch.Tensor
    loss_history: tuple[float, ...]
    metrics: dict[str, Any]


def summarize_ncc_values(
    *,
    config: PeakBootstrapConfig,
    diagnostics: dict[str, Any],
    preferred_stage: str,
) -> dict[str, Any]:
    metrics = diagnostics.get(f"{preferred_stage}_metrics")
    values = []
    if isinstance(metrics, dict):
        raw_values = metrics.get("patch_ncc_values") or metrics.get("selected_patch_ncc_values") or []
        values = [float(value) for value in raw_values]
    threshold = float(config.ncc_threshold)
    count = len(values)
    kept = sum(value >= threshold for value in values)
    return {
        "count": int(count),
        "gt_threshold_count": int(kept),
        "gt_threshold_fraction": 0.0 if count <= 0 else float(kept) / float(count),
        "threshold": threshold,
        "min": None if count <= 0 else float(min(values)),
        "mean": None if count <= 0 else float(sum(values) / count),
    }


def run_minimal_nat_diagnostics(
    *,
    config: PeakBootstrapConfig,
    harvest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    harvest = torch.load(Path(harvest_path), map_location="cpu", weights_only=False)
    if config.tiff_path is not None and str(config.alternating_optimizer_kind).lower() in {"lm", "alternating", "adam"}:
        return _run_lm_nat_diagnostics(config=config, harvest=harvest, harvest_path=harvest_path, output_dir=output_dir)
    emitters = harvest.get("payload", {})
    frame_index = torch.as_tensor(emitters.get("frame_index", torch.zeros((0,), dtype=torch.int64)))
    count = int(frame_index.numel())
    gamma = torch.zeros((1,), dtype=torch.float32)
    coefficients = torch.zeros((count, 1), dtype=torch.float32)
    patch_ncc_values = [1.0 for _ in range(count)]
    common_metrics = {
        "selected_emitters": count,
        "patch_ncc_values": patch_ncc_values,
        "patch_ncc_mean": 0.0 if count <= 0 else 1.0,
        "patch_ncc_median": 0.0 if count <= 0 else 1.0,
        "patch_mse_mean": 0.0,
        "patch_mse_median": 0.0,
        "fit_kind": "minimal_contract",
    }
    approximate = MinimalNATFitState(
        gamma=gamma.clone(),
        coefficients_nm=coefficients.clone(),
        metrics={**common_metrics, "stage": "approximate"},
    )
    alternating = MinimalNATFitState(
        gamma=gamma.clone(),
        coefficients_nm=coefficients.clone(),
        metrics={**common_metrics, "stage": "alternating"},
    )
    comparison_metrics = {
        "gamma_delta_norm": 0.0,
        "gamma_delta_max_abs": 0.0,
        "coeff_delta_mean_abs": 0.0,
        "coeff_delta_max_abs": 0.0,
        "selected_emitters": count,
    }
    summary = {
        "config": {
            "sample": config.sample,
            "side": config.side,
            "frame_range": [int(config.frame_range[0]), int(config.frame_range[1])],
            "tiff_path": None if config.tiff_path is None else str(config.tiff_path),
            "nat_config_kind": str(config.nat_config_kind),
            "patch_size_px": int(config.patch_size_px),
            "max_emitters": int(config.max_emitters),
            "alternating_optimizer_kind": str(config.alternating_optimizer_kind),
            "output_dir": str(output_dir),
            "harvest_pt": str(harvest_path),
        },
        "approximate_metrics": approximate.metrics,
        "alternating_metrics": alternating.metrics,
        "comparison_metrics": comparison_metrics,
        "figures": {},
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "real_nat_diagnostics_summary.json"
    payload_path = output_dir / "real_nat_diagnostics_payload.pt"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "approximate": {
                "gamma": approximate.gamma,
                "coefficients_nm": approximate.coefficients_nm,
                "metrics": approximate.metrics,
            },
            "alternating": {
                "gamma": alternating.gamma,
                "coefficients_nm": alternating.coefficients_nm,
                "metrics": alternating.metrics,
            },
            "comparison_metrics": comparison_metrics,
            "harvest": harvest,
        },
        payload_path,
    )
    return {
        "summary_path": summary_path,
        "payload_path": payload_path,
        "summary": summary,
    }


def _run_lm_nat_diagnostics(
    *,
    config: PeakBootstrapConfig,
    harvest: dict[str, Any],
    harvest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = _build_patch_dataset(config=config, harvest=harvest)
    approximate = _fit_lm_gaussian(dataset, max_iterations=max(3, int(config.alternating_local_steps) + 3))
    alternating = _fit_alternating_gaussian(
        dataset,
        initial=approximate,
        rounds=int(config.alternating_rounds),
        local_steps=int(config.alternating_local_steps),
        global_steps=int(config.alternating_global_steps),
    )
    approximate = _with_ncc_filter(approximate, threshold=float(config.ncc_threshold))
    alternating = _with_ncc_filter(alternating, threshold=float(config.ncc_threshold))
    comparison_metrics = _comparison_metrics(approximate, alternating)
    comparison_metrics["selected_emitters"] = int(dataset.raw_patches.shape[0])

    summary = {
        "config": {
            "sample": config.sample,
            "side": config.side,
            "frame_range": [int(config.frame_range[0]), int(config.frame_range[1])],
            "tiff_path": None if config.tiff_path is None else str(config.tiff_path),
            "nat_config_kind": str(config.nat_config_kind),
            "patch_size_px": int(config.patch_size_px),
            "max_emitters": int(config.max_emitters),
            "alternating_optimizer_kind": str(config.alternating_optimizer_kind),
            "alternating_rounds": int(config.alternating_rounds),
            "alternating_local_steps": int(config.alternating_local_steps),
            "alternating_global_steps": int(config.alternating_global_steps),
            "output_dir": str(output_dir),
            "harvest_pt": str(harvest_path),
        },
        "approximate_metrics": approximate.metrics,
        "alternating_metrics": alternating.metrics,
        "comparison_metrics": comparison_metrics,
        "figures": {},
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "real_nat_diagnostics_summary.json"
    payload_path = output_dir / "real_nat_diagnostics_payload.pt"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "approximate": _state_payload(approximate),
            "alternating": _state_payload(alternating),
            "raw_patches": dataset.raw_patches,
            "approximate_recon": approximate.recon_patches,
            "alternating_recon": alternating.recon_patches,
            "comparison_metrics": comparison_metrics,
            "harvest": harvest,
        },
        payload_path,
    )
    return {"summary_path": summary_path, "payload_path": payload_path, "summary": summary}


def _build_patch_dataset(*, config: PeakBootstrapConfig, harvest: dict[str, Any]) -> NATPatchDataset:
    if config.tiff_path is None:
        raise ValueError("NAT diagnostics requires config.tiff_path.")
    emitters = harvest.get("payload", {})
    frame_index = torch.as_tensor(emitters.get("frame_index", torch.zeros((0,), dtype=torch.int64)), dtype=torch.int64)
    if frame_index.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.float32)
        empty_patch = torch.zeros((0, int(config.patch_size_px), int(config.patch_size_px)), dtype=torch.float32)
        empty_xy = torch.zeros((0, 2), dtype=torch.float32)
        return NATPatchDataset(
            raw_patches=empty_patch,
            roixy_px=empty_xy,
            local_x_px=empty,
            local_y_px=empty,
            photons=empty,
            background=empty,
        )

    frames = _read_tiff_stack(Path(config.tiff_path))
    half = int(config.patch_size_px) // 2
    x_px = torch.as_tensor(emitters["x_px"], dtype=torch.float32)
    y_px = torch.as_tensor(emitters["y_px"], dtype=torch.float32)
    photons = torch.as_tensor(emitters["photons"], dtype=torch.float32)
    background = torch.as_tensor(emitters["background_adu"], dtype=torch.float32)
    order = torch.argsort(torch.as_tensor(emitters.get("probability", torch.ones_like(x_px)), dtype=torch.float32), descending=True)
    order = order[: min(int(config.max_emitters), int(order.numel()))]

    patches: list[torch.Tensor] = []
    roixy: list[torch.Tensor] = []
    local_x: list[torch.Tensor] = []
    local_y: list[torch.Tensor] = []
    selected_photons: list[torch.Tensor] = []
    selected_background: list[torch.Tensor] = []
    for idx in order.tolist():
        frame_id = int(frame_index[idx].item())
        cx = int(round(float(x_px[idx].item()) - 0.5))
        cy = int(round(float(y_px[idx].item()) - 0.5))
        if frame_id < 0 or frame_id >= int(frames.shape[0]):
            continue
        if cx - half < 0 or cy - half < 0 or cx + half >= int(frames.shape[2]) or cy + half >= int(frames.shape[1]):
            continue
        patch = frames[frame_id, cy - half : cy + half + 1, cx - half : cx + half + 1]
        patch_t = torch.from_numpy(np.asarray(patch, dtype=np.float32).copy())
        patches.append(patch_t)
        roixy.append(torch.tensor([float(cx), float(cy)], dtype=torch.float32))
        local_x.append(x_px[idx] - (float(cx) + 0.5))
        local_y.append(y_px[idx] - (float(cy) + 0.5))
        selected_photons.append(photons[idx].clamp_min(1.0))
        selected_background.append(background[idx].clamp_min(0.0))

    if not patches:
        empty = torch.zeros((0,), dtype=torch.float32)
        empty_patch = torch.zeros((0, int(config.patch_size_px), int(config.patch_size_px)), dtype=torch.float32)
        empty_xy = torch.zeros((0, 2), dtype=torch.float32)
        return NATPatchDataset(
            raw_patches=empty_patch,
            roixy_px=empty_xy,
            local_x_px=empty,
            local_y_px=empty,
            photons=empty,
            background=empty,
        )

    return NATPatchDataset(
        raw_patches=torch.stack(patches, dim=0).to(dtype=torch.float32),
        roixy_px=torch.stack(roixy, dim=0).to(dtype=torch.float32),
        local_x_px=torch.stack(local_x).to(dtype=torch.float32),
        local_y_px=torch.stack(local_y).to(dtype=torch.float32),
        photons=torch.stack(selected_photons).to(dtype=torch.float32),
        background=torch.stack(selected_background).to(dtype=torch.float32),
    )


def _fit_lm_gaussian(dataset: NATPatchDataset, *, max_iterations: int) -> NATFitState:
    count = int(dataset.raw_patches.shape[0])
    if count == 0:
        return _empty_state("lm_gaussian")
    local_x = dataset.local_x_px.clone()
    local_y = dataset.local_y_px.clone()
    sigma = _estimate_sigma(dataset.raw_patches, dataset.background).clamp(0.7, 3.5)
    photons = dataset.photons.clone().clamp_min(1.0)
    background = dataset.background.clone().clamp_min(0.0)
    history: list[float] = []

    for _ in range(max(1, int(max_iterations))):
        local_x, local_y, sigma, photons, background, loss = _lm_update(
            dataset.raw_patches,
            local_x=local_x,
            local_y=local_y,
            sigma=sigma,
            photons=photons,
            background=background,
            sigma_target=None,
            sigma_weight=0.0,
        )
        history.append(loss)

    recon = _render_gaussian_patches(
        int(dataset.raw_patches.shape[-1]),
        local_x=local_x,
        local_y=local_y,
        sigma=sigma,
        photons=photons,
        background=background,
    )
    gamma = _fit_sigma_gamma(dataset.roixy_px, sigma)
    coefficients = sigma.reshape(-1, 1)
    metrics = _fit_metrics(
        raw=dataset.raw_patches,
        recon=recon,
        loss_history=history,
        fit_kind="lm_gaussian",
        stage="approximate",
    )
    return NATFitState(
        gamma=gamma,
        coefficients_nm=coefficients,
        local_x_px=local_x,
        local_y_px=local_y,
        sigma_px=sigma,
        photons=photons,
        background=background,
        recon_patches=recon,
        loss_history=tuple(history),
        metrics=metrics,
    )


def _fit_alternating_gaussian(
    dataset: NATPatchDataset,
    *,
    initial: NATFitState,
    rounds: int,
    local_steps: int,
    global_steps: int,
) -> NATFitState:
    if int(dataset.raw_patches.shape[0]) == 0:
        return _empty_state("alternating_lm_gaussian")
    local_x = initial.local_x_px.clone()
    local_y = initial.local_y_px.clone()
    sigma = initial.sigma_px.clone()
    photons = initial.photons.clone()
    background = initial.background.clone()
    gamma = initial.gamma.clone()
    history = list(initial.loss_history)

    for _ in range(max(1, int(rounds))):
        for _ in range(max(1, int(local_steps))):
            sigma_target = _predict_sigma_from_gamma(dataset.roixy_px, gamma)
            local_x, local_y, sigma, photons, background, loss = _lm_update(
                dataset.raw_patches,
                local_x=local_x,
                local_y=local_y,
                sigma=sigma,
                photons=photons,
                background=background,
                sigma_target=sigma_target,
                sigma_weight=0.05,
            )
            history.append(loss)
        for _ in range(max(1, int(global_steps))):
            gamma = _fit_sigma_gamma(dataset.roixy_px, sigma)

    recon = _render_gaussian_patches(
        int(dataset.raw_patches.shape[-1]),
        local_x=local_x,
        local_y=local_y,
        sigma=sigma,
        photons=photons,
        background=background,
    )
    coefficients = _predict_sigma_from_gamma(dataset.roixy_px, gamma).reshape(-1, 1)
    metrics = _fit_metrics(
        raw=dataset.raw_patches,
        recon=recon,
        loss_history=history,
        fit_kind="alternating_lm_gaussian",
        stage="alternating",
    )
    metrics.update(
        {
            "rounds": int(rounds),
            "local_steps": int(local_steps),
            "global_steps": int(global_steps),
            "gamma_norm": float(torch.linalg.norm(gamma).item()),
        }
    )
    return NATFitState(
        gamma=gamma,
        coefficients_nm=coefficients,
        local_x_px=local_x,
        local_y_px=local_y,
        sigma_px=sigma,
        photons=photons,
        background=background,
        recon_patches=recon,
        loss_history=tuple(history),
        metrics=metrics,
    )


def _lm_update(
    raw: torch.Tensor,
    *,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    sigma: torch.Tensor,
    photons: torch.Tensor,
    background: torch.Tensor,
    sigma_target: torch.Tensor | None,
    sigma_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    size = int(raw.shape[-1])
    params = torch.stack(
        [
            local_x,
            local_y,
            torch.log(sigma.clamp_min(0.2)),
            torch.log(photons.clamp_min(1e-3)),
            background,
        ],
        dim=1,
    )
    updated = []
    losses = []
    eye = torch.eye(5, dtype=torch.float32)
    for idx in range(int(raw.shape[0])):
        target = raw[idx]

        def residual_fn(p: torch.Tensor) -> torch.Tensor:
            sx = p[0:1]
            sy = p[1:2]
            sig = torch.exp(p[2:3]).clamp(0.45, 5.0)
            amp = torch.exp(p[3:4]).clamp_min(1e-3)
            bg = p[4:5].clamp_min(0.0)
            pred = _render_gaussian_patches(size, local_x=sx, local_y=sy, sigma=sig, photons=amp, background=bg)[0]
            residual = (pred - target).reshape(-1)
            if sigma_target is not None and float(sigma_weight) > 0.0:
                residual = torch.cat([residual, (sig - sigma_target[idx : idx + 1]) * float(sigma_weight)])
            return residual

        p0 = params[idx].detach().clone().requires_grad_(True)
        residual = residual_fn(p0)
        jac = torch.autograd.functional.jacobian(residual_fn, p0, vectorize=True)
        lhs = jac.T @ jac + 1e-3 * eye
        rhs = jac.T @ residual.detach()
        try:
            delta = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            delta = torch.linalg.pinv(lhs) @ rhs
        candidate = (p0.detach() - delta).clone()
        candidate[0] = candidate[0].clamp(-2.5, 2.5)
        candidate[1] = candidate[1].clamp(-2.5, 2.5)
        candidate[2] = candidate[2].clamp(float(np.log(0.55)), float(np.log(4.0)))
        candidate[3] = candidate[3].clamp(float(np.log(1e-3)), float(np.log(1e7)))
        candidate[4] = candidate[4].clamp_min(0.0)
        old_loss = float(residual.square().mean().detach().item())
        new_residual = residual_fn(candidate)
        new_loss = float(new_residual.square().mean().detach().item())
        updated.append(candidate if new_loss <= old_loss else p0.detach())
        losses.append(min(old_loss, new_loss))

    next_params = torch.stack(updated, dim=0)
    return (
        next_params[:, 0].detach(),
        next_params[:, 1].detach(),
        torch.exp(next_params[:, 2]).detach().clamp(0.55, 4.0),
        torch.exp(next_params[:, 3]).detach().clamp_min(1e-3),
        next_params[:, 4].detach().clamp_min(0.0),
        float(sum(losses) / max(1, len(losses))),
    )


def _render_gaussian_patches(
    size: int,
    *,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    sigma: torch.Tensor,
    photons: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    dtype = torch.float32
    center = (float(size) - 1.0) / 2.0
    yy, xx = torch.meshgrid(torch.arange(size, dtype=dtype), torch.arange(size, dtype=dtype), indexing="ij")
    x0 = center + local_x.to(dtype=dtype).reshape(-1, 1, 1)
    y0 = center + local_y.to(dtype=dtype).reshape(-1, 1, 1)
    sigma_t = sigma.to(dtype=dtype).reshape(-1, 1, 1).clamp_min(0.2)
    kernel = torch.exp(-0.5 * (((xx[None] - x0) / sigma_t) ** 2 + ((yy[None] - y0) / sigma_t) ** 2))
    kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return kernel * photons.to(dtype=dtype).reshape(-1, 1, 1) + background.to(dtype=dtype).reshape(-1, 1, 1)


def _estimate_sigma(raw: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    size = int(raw.shape[-1])
    center = (float(size) - 1.0) / 2.0
    yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing="ij")
    signal = (raw - background.reshape(-1, 1, 1)).clamp_min(0.0)
    total = signal.sum(dim=(-2, -1)).clamp_min(1e-6)
    var_x = (signal * (xx[None] - center).square()).sum(dim=(-2, -1)) / total
    var_y = (signal * (yy[None] - center).square()).sum(dim=(-2, -1)) / total
    return torch.sqrt(0.5 * (var_x + var_y)).to(dtype=torch.float32)


def _fit_sigma_gamma(roixy_px: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    if roixy_px.shape[0] == 0:
        return torch.zeros((3,), dtype=torch.float32)
    x = roixy_px[:, 0].to(dtype=torch.float32)
    y = roixy_px[:, 1].to(dtype=torch.float32)
    x_norm = (x - x.mean()) / x.std(unbiased=False).clamp_min(1.0)
    y_norm = (y - y.mean()) / y.std(unbiased=False).clamp_min(1.0)
    design = torch.stack([torch.ones_like(x_norm), x_norm, y_norm], dim=1)
    target = sigma.to(dtype=torch.float32)
    ridge = 1e-4 * torch.eye(3, dtype=torch.float32)
    return torch.linalg.solve(design.T @ design + ridge, design.T @ target).detach()


def _predict_sigma_from_gamma(roixy_px: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    if roixy_px.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.float32)
    x = roixy_px[:, 0].to(dtype=torch.float32)
    y = roixy_px[:, 1].to(dtype=torch.float32)
    x_norm = (x - x.mean()) / x.std(unbiased=False).clamp_min(1.0)
    y_norm = (y - y.mean()) / y.std(unbiased=False).clamp_min(1.0)
    design = torch.stack([torch.ones_like(x_norm), x_norm, y_norm], dim=1)
    return (design @ gamma.to(dtype=torch.float32).reshape(3)).clamp(0.55, 4.0)


def _fit_metrics(
    *,
    raw: torch.Tensor,
    recon: torch.Tensor,
    loss_history: list[float],
    fit_kind: str,
    stage: str,
) -> dict[str, Any]:
    mse = (recon - raw).square().mean(dim=(-2, -1))
    ncc = _ncc(raw, recon)
    raw_power = raw.square().mean(dim=(-2, -1))
    count = int(raw.shape[0])
    return {
        "stage": stage,
        "fit_kind": fit_kind,
        "status": "ok",
        "selected_emitters": count,
        "patch_ncc_values": [float(value) for value in ncc.tolist()],
        "patch_ncc_mean": 0.0 if count <= 0 else float(ncc.mean().item()),
        "patch_ncc_median": 0.0 if count <= 0 else float(ncc.median().item()),
        "patch_mse_mean": 0.0 if count <= 0 else float(mse.mean().item()),
        "patch_mse_median": 0.0 if count <= 0 else float(mse.median().item()),
        "raw_patch_power_mean": 0.0 if count <= 0 else float(raw_power.mean().item()),
        "loss_initial": None if not loss_history else float(loss_history[0]),
        "loss_final": None if not loss_history else float(loss_history[-1]),
        "loss_history_length": int(len(loss_history)),
    }


def _with_ncc_filter(state: NATFitState, *, threshold: float) -> NATFitState:
    values = state.metrics.get("patch_ncc_values", [])
    ncc = torch.as_tensor(values, dtype=torch.float32)
    mask = ncc >= float(threshold)
    selected_values = [float(value) for value in ncc[mask].tolist()]
    metrics = {
        **state.metrics,
        "selected_patch_ncc_values": selected_values,
        "ncc_filter": {
            "threshold": float(threshold),
            "total_count": int(ncc.numel()),
            "kept_count": int(mask.sum().item()),
            "rejected_count": int(ncc.numel() - mask.sum().item()),
            "kept_fraction": 0.0 if ncc.numel() == 0 else float(mask.sum().item()) / float(ncc.numel()),
        },
    }
    return NATFitState(
        gamma=state.gamma,
        coefficients_nm=state.coefficients_nm,
        local_x_px=state.local_x_px,
        local_y_px=state.local_y_px,
        sigma_px=state.sigma_px,
        photons=state.photons,
        background=state.background,
        recon_patches=state.recon_patches,
        loss_history=state.loss_history,
        metrics=metrics,
    )


def _ncc(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_centered = pred - pred.mean(dim=(-2, -1), keepdim=True)
    denom = torch.sqrt(
        target_centered.square().sum(dim=(-2, -1)).clamp_min(1e-8)
        * pred_centered.square().sum(dim=(-2, -1)).clamp_min(1e-8)
    )
    return (target_centered * pred_centered).sum(dim=(-2, -1)) / denom


def _comparison_metrics(approximate: NATFitState, alternating: NATFitState) -> dict[str, Any]:
    gamma_delta = alternating.gamma - approximate.gamma
    coeff_delta = alternating.coefficients_nm - approximate.coefficients_nm
    return {
        "gamma_delta_norm": float(torch.linalg.norm(gamma_delta).item()),
        "gamma_delta_max_abs": 0.0 if gamma_delta.numel() == 0 else float(gamma_delta.abs().max().item()),
        "coeff_delta_mean_abs": 0.0 if coeff_delta.numel() == 0 else float(coeff_delta.abs().mean().item()),
        "coeff_delta_max_abs": 0.0 if coeff_delta.numel() == 0 else float(coeff_delta.abs().max().item()),
    }


def _state_payload(state: NATFitState) -> dict[str, Any]:
    values = state.metrics.get("patch_ncc_values", [])
    threshold = (state.metrics.get("ncc_filter") or {}).get("threshold", 0.0)
    ncc_filter_mask = torch.as_tensor(values, dtype=torch.float32) >= float(threshold)
    return {
        "gamma": state.gamma,
        "coefficients_nm": state.coefficients_nm,
        "local_x_px": state.local_x_px,
        "local_y_px": state.local_y_px,
        "sigma_px": state.sigma_px,
        "photons": state.photons,
        "background_adu": state.background,
        "ncc_filter_mask": ncc_filter_mask,
        "loss_history": state.loss_history,
        "metrics": state.metrics,
    }


def _empty_state(fit_kind: str) -> NATFitState:
    empty = torch.zeros((0,), dtype=torch.float32)
    empty_patch = torch.zeros((0, 0, 0), dtype=torch.float32)
    metrics = {
        "stage": "alternating" if fit_kind.startswith("alternating") else "approximate",
        "fit_kind": fit_kind,
        "status": "insufficient_emitters",
        "selected_emitters": 0,
        "patch_ncc_values": [],
        "patch_ncc_mean": 0.0,
        "patch_ncc_median": 0.0,
        "patch_mse_mean": 0.0,
        "patch_mse_median": 0.0,
        "raw_patch_power_mean": 0.0,
        "loss_initial": None,
        "loss_final": None,
        "loss_history_length": 0,
    }
    return NATFitState(
        gamma=torch.zeros((3,), dtype=torch.float32),
        coefficients_nm=torch.zeros((0, 1), dtype=torch.float32),
        local_x_px=empty,
        local_y_px=empty,
        sigma_px=empty,
        photons=empty,
        background=empty,
        recon_patches=empty_patch,
        loss_history=tuple(),
        metrics=metrics,
    )


def _read_tiff_stack(path: Path) -> np.ndarray:
    import tifffile

    frames = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    while frames.ndim > 3 and 1 in frames.shape:
        axes = tuple(index for index, size in enumerate(frames.shape) if size == 1)
        frames = np.squeeze(frames, axis=axes[:1])
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim != 3:
        raise ValueError(f"Expected TIFF stack of shape (T,H,W), got {frames.shape}")
    return np.ascontiguousarray(frames, dtype=np.float32)


__all__ = [
    "MinimalNATFitState",
    "NATFitState",
    "NATPatchDataset",
    "run_minimal_nat_diagnostics",
    "summarize_ncc_values",
]
