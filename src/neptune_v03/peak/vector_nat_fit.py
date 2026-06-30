from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neptune_v03.optics import build_named_nat_config, evaluate_zernike_from_roi_positions_torch, get_fov_coordinates_torch
from neptune_v03.optics.vector_psf import VectorPSFParams, build_vector_psf_context, noll_to_nm, render_vector_psf_bank, render_vector_psf_stack

from .contract import PeakBootstrapConfig
from .nat_optimizer import fit_nat_alternating, initialize_local_from_spots, invert_nat_order1_torch


@dataclass(frozen=True)
class VectorNATPatchDataset:
    patches_adu: torch.Tensor
    roixy_px: torch.Tensor
    local_x_nm: torch.Tensor
    local_y_nm: torch.Tensor
    z_nm: torch.Tensor
    photons_init: torch.Tensor
    background_adu: torch.Tensor
    selection_metrics: dict[str, Any]


@dataclass(frozen=True)
class VectorNATFitState:
    gamma: torch.Tensor
    coefficients_nm: torch.Tensor
    roixy_px: torch.Tensor
    local_x_nm: torch.Tensor
    local_y_nm: torch.Tensor
    z_nm: torch.Tensor
    photons: torch.Tensor
    background_adu: torch.Tensor
    recon_patches: torch.Tensor
    loss_history: tuple[float, ...]
    metrics: dict[str, Any]


class _VectorPSFAdapter:
    def __init__(
        self,
        *,
        nat_config,
        patch_size_px: int,
        device: torch.device,
        pixel_size_x_nm: float = 101.11,
        pixel_size_y_nm: float = 98.83,
        wavelength_nm: float = 660.0,
    ) -> None:
        self.device = device
        self.psf_size = int(patch_size_px)
        self.wavelength_nm = float(wavelength_nm)
        self.ctx = build_vector_psf_context(
            NA=1.4,
            wavelength_nm=self.wavelength_nm,
            pixel_size_nm_x=float(pixel_size_x_nm),
            pixel_size_nm_y=float(pixel_size_y_nm),
            noll_indices=nat_aberration_noll_indices(nat_config),
            params=VectorPSFParams(
                npupil=128,
                psf_size=int(patch_size_px),
                refmed=1.518,
                refcov=1.518,
                refimm=1.518,
                objstage0=0.0,
                otf_rescale_xy=(0.0, 0.0),
                batch_size=64,
            ),
            device=device,
        )

    def simulate(
        self,
        x_nm: torch.Tensor,
        y_nm: torch.Tensor,
        z_nm: torch.Tensor,
        photons: torch.Tensor,
        zernike_coefs: torch.Tensor,
    ) -> torch.Tensor:
        del x_nm, y_nm
        count = int(z_nm.reshape(-1).numel())
        if count == 0:
            return torch.zeros((0, self.psf_size, self.psf_size), device=self.device, dtype=torch.float32)
        coeffs_nm = zernike_coefs.to(device=self.device, dtype=torch.float32)
        if coeffs_nm.ndim == 1:
            coeffs_nm = coeffs_nm.unsqueeze(0).expand(count, -1)
        coeffs_rad = coeffs_nm * (2.0 * math.pi / max(self.wavelength_nm, 1e-6)) * self.ctx.normfac[None, :]
        psf = render_vector_psf_bank(
            self.ctx,
            coeffs_rad,
            z_nm.to(device=self.device, dtype=torch.float32).reshape(-1) * 1e-9,
            out_size=self.psf_size,
            batch_size=self.ctx.batch_size,
            return_torch=True,
        ).clamp_min(0.0)
        psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        return psf * photons.to(device=self.device, dtype=torch.float32).reshape(-1, 1, 1)


def run_vector_nat_diagnostics(
    *,
    config: PeakBootstrapConfig,
    harvest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    harvest = torch.load(Path(harvest_path), map_location="cpu", weights_only=False)
    pixel_size_x_nm = 101.11
    pixel_size_y_nm = 98.83
    nat_config = build_named_nat_config(
        str(config.nat_config_kind),
        img_size_x=_width_from_config(config),
        img_size_y=_height_from_config(config),
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    dataset = _build_patch_dataset(config=config, harvest=harvest)
    fixed_coefficients, fixed_mask = _fixed_coefficients_and_mask(config, nat_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    approximate = _fit_approximate_vector(
        dataset,
        nat_config=nat_config,
        fixed_coefficients_nm=fixed_coefficients,
        fixed_mode_mask=fixed_mask,
        device=device,
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    gamma_init = _initial_gamma(config=config, nat_config=nat_config, approximate=approximate)
    alternating = _fit_alternating_vector(
        dataset,
        config=config,
        nat_config=nat_config,
        gamma_init=gamma_init,
        fixed_coefficients_nm=fixed_coefficients,
        fixed_mode_mask=fixed_mask,
        device=device,
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    approximate = _with_ncc_filter(approximate, threshold=float(config.ncc_threshold))
    alternating = _with_ncc_filter(alternating, threshold=float(config.ncc_threshold))
    comparison_metrics = _comparison_metrics(approximate, alternating)
    comparison_metrics.update(
        {
            "selected_emitters": int(dataset.patches_adu.shape[0]),
            "freeze_defocus_zero_gauge": bool(config.freeze_defocus_zero_gauge),
            "vectorfit_astig_gauge": bool(config.vectorfit_astig_gauge),
            "vectorfit_astig_anchor_nm": config.vectorfit_astig_anchor_nm,
            "vectorfit_astig_anchor_mode": str(config.vectorfit_astig_anchor_mode),
            "vectorfit_phasor_z_init": bool(config.vectorfit_phasor_z_init),
            "implementation": "neptune_v03_native_vector_nat",
        }
    )
    summary = {
        "config": {
            **config.to_json_dict(),
            "image_width_px": int(_width_from_config(config)),
            "image_height_px": int(_height_from_config(config)),
            "pixel_size_x_nm": float(pixel_size_x_nm),
            "pixel_size_y_nm": float(pixel_size_y_nm),
            "output_dir": str(output_dir),
            "harvest_pt": str(harvest_path),
            "implementation": "neptune_v03_native_vector_nat",
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
            "raw_patches": dataset.patches_adu,
            "approximate_recon": approximate.recon_patches,
            "alternating_recon": alternating.recon_patches,
            "comparison_metrics": comparison_metrics,
            "harvest": harvest,
        },
        payload_path,
    )
    _write_vector_psf_smoke(output_dir / "vector_psf_smoke.json", coeffs=alternating.coefficients_nm)
    return {"summary_path": summary_path, "payload_path": payload_path, "summary": summary}


def nat_aberration_noll_indices(config) -> tuple[int, ...]:
    mode_to_noll: dict[tuple[int, int], int] = {}
    for index in range(1, 128):
        mode_to_noll[noll_to_nm(index)] = index
    return tuple(mode_to_noll[mode] for mode in config.aberrations)


def render_astigmatic_vector_psf_smoke(coefficients_nm: torch.Tensor) -> dict[str, Any]:
    ctx = _build_vector_context(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), patch_size_px=25)
    coeff = coefficients_nm.reshape(-1).to(dtype=torch.float32)
    coeff_rad = coeff.to(device=ctx.device) * (2.0 * math.pi / 660.0) * ctx.normfac
    z_nm = torch.tensor([-600.0, 0.0, 600.0], dtype=torch.float32, device=ctx.device)
    stack = render_vector_psf_stack(ctx, coeff_rad, z_nm * 1e-9, out_size=25, batch_size=3).detach().cpu()
    return {
        "orientation_moments": [_orientation_moment(frame) for frame in stack],
        "plane_max": [float(frame.max().item()) for frame in stack],
    }


def _fit_approximate_vector(
    dataset: VectorNATPatchDataset,
    *,
    nat_config,
    fixed_coefficients_nm: torch.Tensor | None,
    fixed_mode_mask: torch.Tensor | None,
    device: torch.device,
    pixel_size_x_nm: float,
    pixel_size_y_nm: float,
) -> VectorNATFitState:
    count = int(dataset.patches_adu.shape[0])
    if count < 3:
        return _empty_state("vector_psf_approximate", nat_config=nat_config, dataset=dataset)
    local_init = initialize_local_from_spots(
        dataset.patches_adu.to(device=device, dtype=torch.float32),
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    coeffs = torch.zeros((count, len(nat_config.aberrations)), dtype=torch.float32)
    coeffs = _apply_fixed_coefficients(coeffs, fixed_coefficients_nm, fixed_mode_mask)
    coeffs_for_gamma = _subtract_fixed_coefficients(coeffs, fixed_coefficients_nm, fixed_mode_mask)
    normalized, _ = get_fov_coordinates_torch(
        dataset.roixy_px,
        img_size_x=nat_config.img_size_x,
        img_size_y=nat_config.img_size_y,
        pixel_size_x_nm=nat_config.pixel_size_x_nm,
        pixel_size_y_nm=nat_config.pixel_size_y_nm,
        local_x_nm=dataset.local_x_nm,
        local_y_nm=dataset.local_y_nm,
    )
    gamma = invert_nat_order1_torch(normalized[:, 0], normalized[:, 1], coeffs_for_gamma, nat_config, ridge=1e-4).detach().cpu()
    psf = _VectorPSFAdapter(
        nat_config=nat_config,
        patch_size_px=int(dataset.patches_adu.shape[-1]),
        device=device,
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    recon = (
        psf.simulate(
            local_init.x_nm,
            local_init.y_nm,
            dataset.z_nm.to(device=device, dtype=torch.float32),
            dataset.photons_init.to(device=device, dtype=torch.float32),
            zernike_coefs=coeffs.to(device=device),
        )
        + dataset.background_adu.to(device=device, dtype=torch.float32).reshape(-1, 1, 1)
    ).detach().cpu()
    metrics = _fit_metrics(
        raw=dataset.patches_adu,
        recon=recon,
        loss_history=[],
        fit_kind="vector_psf_approximate",
        stage="approximate",
        extra={**dataset.selection_metrics, "implementation": "neptune_v03_native_vector_nat"},
    )
    return VectorNATFitState(
        gamma=gamma,
        coefficients_nm=coeffs.detach().cpu(),
        roixy_px=dataset.roixy_px,
        local_x_nm=local_init.x_nm.detach().cpu(),
        local_y_nm=local_init.y_nm.detach().cpu(),
        z_nm=dataset.z_nm,
        photons=dataset.photons_init,
        background_adu=dataset.background_adu,
        recon_patches=recon,
        loss_history=tuple(),
        metrics=metrics,
    )


def _fit_alternating_vector(
    dataset: VectorNATPatchDataset,
    *,
    config: PeakBootstrapConfig,
    nat_config,
    gamma_init: torch.Tensor,
    fixed_coefficients_nm: torch.Tensor | None,
    fixed_mode_mask: torch.Tensor | None,
    device: torch.device,
    pixel_size_x_nm: float,
    pixel_size_y_nm: float,
) -> VectorNATFitState:
    count = int(dataset.patches_adu.shape[0])
    if count < 3:
        return _empty_state("vector_psf_alternating", nat_config=nat_config, dataset=dataset)
    psf = _VectorPSFAdapter(
        nat_config=nat_config,
        patch_size_px=int(dataset.patches_adu.shape[-1]),
        device=device,
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    gamma_train_mask = _gamma_train_mask_from_fixed_modes(nat_config, fixed_mode_mask)
    gamma_fixed_values = None if gamma_train_mask is None else torch.zeros((len(nat_config.gammas),), dtype=torch.float32, device=device)
    gamma_positive_index = None
    gamma_symmetry_vector = None
    if bool(config.vectorfit_astig_gauge):
        gamma_positive_index = _astig_positive_gamma_index(nat_config)
        gamma_symmetry_vector = _gamma_symmetry_vector(nat_config).to(device=device)

    fixed_device = None if fixed_coefficients_nm is None else fixed_coefficients_nm.to(device=device, dtype=torch.float32)

    def renderer(
        x_nm: torch.Tensor,
        y_nm: torch.Tensor,
        z_nm: torch.Tensor,
        photons: torch.Tensor,
        zernike_coefs_nm: torch.Tensor,
    ) -> torch.Tensor:
        coeffs = zernike_coefs_nm.to(device=device, dtype=torch.float32)
        if fixed_device is not None:
            coeffs = coeffs + fixed_device.reshape(1, -1)
        return psf.simulate(x_nm, y_nm, z_nm, photons, zernike_coefs=coeffs)

    init = initialize_local_from_spots(
        dataset.patches_adu.to(device=device, dtype=torch.float32),
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    init_z = dataset.z_nm.to(device=device, dtype=torch.float32).clamp(-600.0, 600.0)
    if bool(config.vectorfit_phasor_z_init):
        coeffs_init, _, _ = evaluate_zernike_from_roi_positions_torch(
            dataset.roixy_px.to(device=device, dtype=torch.float32),
            gamma_init.to(device=device, dtype=torch.float32),
            nat_config,
            local_x_nm=init.x_nm,
            local_y_nm=init.y_nm,
            dtype=torch.float32,
            device=device,
        )
        if fixed_device is not None:
            coeffs_init = coeffs_init + fixed_device.reshape(1, -1)
        init_z = _astig_phasor_z_initial_nm(
            dataset.patches_adu.to(device=device, dtype=torch.float32),
            local_x_nm=init.x_nm,
            local_y_nm=init.y_nm,
            zernike_coefficients_nm=coeffs_init,
            pixel_size_x_nm=pixel_size_x_nm,
            pixel_size_y_nm=pixel_size_y_nm,
            wavelength_nm=660.0,
            z_min_nm=-600.0,
            z_max_nm=600.0,
        )
    local_init = type(init)(
        x_nm=dataset.local_x_nm.to(device=device, dtype=torch.float32),
        y_nm=dataset.local_y_nm.to(device=device, dtype=torch.float32),
        z_nm=init_z,
        photons=dataset.photons_init.to(device=device, dtype=torch.float32).clamp_min(1.0),
        background=dataset.background_adu.to(device=device, dtype=torch.float32).clamp_min(1e-3),
    )
    snapshot = fit_nat_alternating(
        spots=dataset.patches_adu.to(device=device, dtype=torch.float32),
        roixy_px=dataset.roixy_px.to(device=device, dtype=torch.float32),
        renderer=renderer,
        config=nat_config,
        gamma_init=gamma_init.to(device=device, dtype=torch.float32),
        local_init=local_init,
        rounds=int(config.alternating_rounds),
        local_steps=int(config.alternating_local_steps),
        global_steps=int(config.alternating_global_steps),
        local_warmup_rounds=int(config.alternating_local_warmup_rounds),
        local_warmup_steps=int(config.alternating_local_warmup_steps),
        optimizer_kind=str(config.alternating_optimizer_kind),
        local_lr=1e-2,
        gamma_lr=2e-3,
        gamma_l2=0.0,
        gamma_train_mask=gamma_train_mask,
        gamma_fixed_values=gamma_fixed_values,
        gamma_positive_index=gamma_positive_index,
        gamma_symmetry_vector=gamma_symmetry_vector,
        min_photons=dataset.photons_init.to(device=device, dtype=torch.float32).clamp_min(1.0) * 0.05,
        min_background=dataset.background_adu.to(device=device, dtype=torch.float32).clamp_min(1e-3) * 0.05,
        z_min_nm=-600.0,
        z_max_nm=600.0,
        x_bound_nm=float(pixel_size_x_nm) * 2.5,
        y_bound_nm=float(pixel_size_y_nm) * 2.5,
    )
    coeffs, _, _ = evaluate_zernike_from_roi_positions_torch(
        dataset.roixy_px,
        snapshot.gamma.detach().cpu(),
        nat_config,
        local_x_nm=snapshot.x_nm.detach().cpu(),
        local_y_nm=snapshot.y_nm.detach().cpu(),
    )
    coeffs = _apply_fixed_coefficients(coeffs, fixed_coefficients_nm, fixed_mode_mask)
    recon = (
        renderer(
            snapshot.x_nm.detach(),
            snapshot.y_nm.detach(),
            snapshot.z_nm.detach(),
            snapshot.photons.detach(),
            coeffs.to(device=device, dtype=torch.float32) if fixed_device is None else _subtract_fixed_coefficients(coeffs, fixed_coefficients_nm, fixed_mode_mask).to(device=device),
        )
        + snapshot.background.detach().reshape(-1, 1, 1)
    ).detach().cpu()
    metrics = _fit_metrics(
        raw=dataset.patches_adu,
        recon=recon,
        loss_history=list(snapshot.loss_history),
        fit_kind="vector_psf_alternating_poisson_nll",
        stage="alternating",
        extra={
            **dataset.selection_metrics,
            "rounds": int(config.alternating_rounds),
            "local_steps": int(config.alternating_local_steps),
            "global_steps": int(config.alternating_global_steps),
            "gamma_norm": float(torch.linalg.norm(snapshot.gamma.detach().cpu()).item()),
            "fixed_mode_count": 0 if fixed_mode_mask is None else int(fixed_mode_mask.sum().item()),
            "implementation": "neptune_v03_native_vector_nat",
        },
    )
    return VectorNATFitState(
        gamma=snapshot.gamma.detach().cpu(),
        coefficients_nm=coeffs.detach().cpu(),
        roixy_px=dataset.roixy_px,
        local_x_nm=snapshot.x_nm.detach().cpu(),
        local_y_nm=snapshot.y_nm.detach().cpu(),
        z_nm=snapshot.z_nm.detach().cpu(),
        photons=snapshot.photons.detach().cpu(),
        background_adu=snapshot.background.detach().cpu(),
        recon_patches=recon,
        loss_history=tuple(float(v) for v in snapshot.loss_history),
        metrics=metrics,
    )


def _build_patch_dataset(*, config: PeakBootstrapConfig, harvest: dict[str, Any]) -> VectorNATPatchDataset:
    if config.tiff_path is None:
        raise ValueError("Vector NAT fitting requires config.tiff_path.")
    emitters = harvest.get("payload", {})
    frame_index = torch.as_tensor(emitters.get("frame_index", torch.zeros((0,), dtype=torch.int64)), dtype=torch.int64)
    frames = _read_tiff_stack(Path(config.tiff_path))
    half = int(config.patch_size_px) // 2
    order = torch.argsort(torch.as_tensor(emitters.get("probability", torch.zeros_like(frame_index, dtype=torch.float32)), dtype=torch.float32), descending=True)
    order = order[: min(int(config.max_emitters), int(order.numel()))]
    x_px = torch.as_tensor(emitters.get("x_px", torch.zeros((0,))), dtype=torch.float32)
    y_px = torch.as_tensor(emitters.get("y_px", torch.zeros((0,))), dtype=torch.float32)
    local_x_nm = torch.as_tensor(emitters.get("local_x_nm", torch.zeros_like(x_px)), dtype=torch.float32)
    local_y_nm = torch.as_tensor(emitters.get("local_y_nm", torch.zeros_like(y_px)), dtype=torch.float32)
    photons = torch.as_tensor(emitters.get("photons", torch.ones_like(x_px)), dtype=torch.float32)
    background = torch.as_tensor(emitters.get("background_adu", torch.zeros_like(x_px)), dtype=torch.float32)
    z_nm = torch.as_tensor(emitters.get("z_um", torch.zeros_like(x_px)), dtype=torch.float32) * 1000.0
    patches: list[torch.Tensor] = []
    roixy: list[torch.Tensor] = []
    keep_indices: list[int] = []
    for idx in order.tolist():
        frame_id = int(frame_index[idx].item())
        cx = int(round(float(x_px[idx].item()) - 0.5))
        cy = int(round(float(y_px[idx].item()) - 0.5))
        if frame_id < 0 or frame_id >= int(frames.shape[0]):
            continue
        if cx - half < 0 or cy - half < 0 or cx + half >= int(frames.shape[2]) or cy + half >= int(frames.shape[1]):
            continue
        patch = torch.from_numpy(np.asarray(frames[frame_id, cy - half : cy + half + 1, cx - half : cx + half + 1], dtype=np.float32).copy())
        keep, _ = _patch_quality_metrics(patch, config)
        if not keep:
            continue
        patches.append(patch)
        roixy.append(torch.tensor([float(cx) - float(config.crop_x0), float(cy) - float(config.crop_y0)], dtype=torch.float32))
        keep_indices.append(int(idx))
        if len(keep_indices) >= int(config.max_emitters):
            break
    if not keep_indices:
        empty = torch.zeros((0,), dtype=torch.float32)
        return VectorNATPatchDataset(
            patches_adu=torch.zeros((0, int(config.patch_size_px), int(config.patch_size_px)), dtype=torch.float32),
            roixy_px=torch.zeros((0, 2), dtype=torch.float32),
            local_x_nm=empty,
            local_y_nm=empty,
            z_nm=empty,
            photons_init=empty,
            background_adu=empty,
            selection_metrics={"selected_emitters": 0, "status": "no_valid_emitters"},
        )
    selected = torch.as_tensor(keep_indices, dtype=torch.long)
    return VectorNATPatchDataset(
        patches_adu=torch.stack(patches).to(dtype=torch.float32),
        roixy_px=torch.stack(roixy).to(dtype=torch.float32),
        local_x_nm=local_x_nm[selected].to(dtype=torch.float32),
        local_y_nm=local_y_nm[selected].to(dtype=torch.float32),
        z_nm=z_nm[selected].to(dtype=torch.float32).clamp(-600.0, 600.0),
        photons_init=photons[selected].to(dtype=torch.float32).clamp_min(1.0),
        background_adu=background[selected].to(dtype=torch.float32).clamp_min(1e-3),
        selection_metrics={
            "selected_emitters": int(selected.numel()),
            "candidate_emitters": int(frame_index.numel()),
            "max_emitters": int(config.max_emitters),
        },
    )


def _patch_quality_metrics(patch: torch.Tensor, config: PeakBootstrapConfig) -> tuple[bool, dict[str, float]]:
    values = patch.to(dtype=torch.float32)
    h, w = values.shape
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    cy = (h - 1.0) / 2.0
    cx = (w - 1.0) / 2.0
    radius = torch.sqrt((yy - cy).square() + (xx - cx).square())
    border = torch.cat([values[0], values[-1], values[1:-1, 0], values[1:-1, -1]])
    signal = (values - border.median()).clamp_min(0.0)
    peak_flat = int(torch.argmax(signal).item())
    py = peak_flat // w
    px = peak_flat % w
    peak_distance = float(((float(py) - cy) ** 2 + (float(px) - cx) ** 2) ** 0.5)
    center = signal[radius <= float(config.max_patch_peak_distance_px)]
    outside = signal[radius > 3.5]
    center_peak = float(center.max().item()) if int(center.numel()) else 0.0
    outside_peak = float(outside.max().item()) if int(outside.numel()) else 0.0
    secondary_fraction = outside_peak / max(center_peak, 1e-6)
    signal_sum = float(signal.sum().item())
    keep = (
        peak_distance <= float(config.max_patch_peak_distance_px)
        and signal_sum > float(config.min_signal_sum_norm)
        and center_peak > float(config.min_center_peak_norm)
        and secondary_fraction <= float(config.max_secondary_peak_fraction)
    )
    return keep, {"peak_distance_px": peak_distance, "secondary_fraction": secondary_fraction}


def _initial_gamma(*, config: PeakBootstrapConfig, nat_config, approximate: VectorNATFitState) -> torch.Tensor:
    gamma = approximate.gamma.clone()
    if gamma.numel() != len(nat_config.gammas):
        gamma = torch.zeros((len(nat_config.gammas),), dtype=torch.float32)
    if bool(config.vectorfit_astig_gauge):
        anchor = config.vectorfit_astig_anchor_nm
        if anchor is None:
            anchor = 150.0 * 660.0 * 1e-3
        gamma[_astig_positive_gamma_index(nat_config)] = abs(float(anchor))
    return gamma


def _fixed_coefficients_and_mask(config: PeakBootstrapConfig, nat_config) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    coeffs = torch.zeros((len(nat_config.aberrations),), dtype=torch.float32)
    mask = torch.zeros((len(nat_config.aberrations),), dtype=torch.bool)
    mode_to_index = {mode: idx for idx, mode in enumerate(nat_config.aberrations)}
    if bool(config.freeze_defocus_zero_gauge) and (2, 0) in mode_to_index:
        mask[mode_to_index[(2, 0)]] = True
        coeffs[mode_to_index[(2, 0)]] = 0.0
    if not bool(mask.any().item()):
        return None, None
    return coeffs, mask


def _apply_fixed_coefficients(coeffs: torch.Tensor, fixed: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor:
    if fixed is None or mask is None:
        return coeffs
    result = coeffs.clone()
    result[..., mask] = fixed[mask].to(device=result.device, dtype=result.dtype)
    return result


def _subtract_fixed_coefficients(coeffs: torch.Tensor, fixed: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor:
    if fixed is None or mask is None:
        return coeffs
    result = coeffs.clone()
    result[..., mask] = result[..., mask] - fixed[mask].to(device=result.device, dtype=result.dtype)
    return result


def _gamma_train_mask_from_fixed_modes(nat_config, fixed_mode_mask: torch.Tensor | None) -> torch.Tensor | None:
    if fixed_mode_mask is None:
        return None
    mode_to_index = {mode: idx for idx, mode in enumerate(nat_config.aberrations)}
    train = torch.ones((len(nat_config.gammas),), dtype=torch.bool)
    for gamma_index, gamma in enumerate(nat_config.gammas):
        for component in gamma.components:
            if bool(fixed_mode_mask[mode_to_index[(component.n, component.m)]].item()):
                train[gamma_index] = False
                break
    return train


def _astig_positive_gamma_index(nat_config) -> int:
    for index, gamma in enumerate(nat_config.gammas):
        if gamma.components and gamma.components[0].px == 0 and gamma.components[0].py == 0 and gamma.components[0].n == 2 and gamma.components[0].m == 2:
            return int(index)
    raise ValueError("NAT config has no astigmatism gamma anchor.")


def _gamma_symmetry_vector(nat_config) -> torch.Tensor:
    values = torch.ones((len(nat_config.gammas),), dtype=torch.float32)
    for index, gamma in enumerate(nat_config.gammas):
        if gamma.components and int(gamma.components[0].m) % 2 == 0:
            values[index] = -1.0
    return values


def _astig_phasor_z_initial_nm(
    spots: torch.Tensor,
    *,
    local_x_nm: torch.Tensor,
    local_y_nm: torch.Tensor,
    zernike_coefficients_nm: torch.Tensor,
    pixel_size_x_nm: float,
    pixel_size_y_nm: float,
    wavelength_nm: float,
    z_min_nm: float,
    z_max_nm: float,
) -> torch.Tensor:
    device = spots.device
    dtype = spots.dtype
    count, height, width = spots.shape
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=device, dtype=dtype) - (height - 1.0) / 2.0) * float(pixel_size_y_nm),
        (torch.arange(width, device=device, dtype=dtype) - (width - 1.0) / 2.0) * float(pixel_size_x_nm),
        indexing="ij",
    )
    x0 = local_x_nm.to(device=device, dtype=dtype).reshape(count)
    y0 = local_y_nm.to(device=device, dtype=dtype).reshape(count)
    coeffs = zernike_coefficients_nm.to(device=device, dtype=dtype)
    mode_to_index = {mode: index for index, mode in enumerate(build_named_nat_config("order1").aberrations)}
    astig_hv = coeffs[:, mode_to_index[(2, 2)]] / float(wavelength_nm)
    astig_diag = coeffs[:, mode_to_index[(2, -2)]] / float(wavelength_nm)
    astig_rms = torch.sqrt(astig_hv.square() + astig_diag.square()).clamp_min(1e-6)
    astig_angle = torch.atan2(astig_diag, astig_hv) / 2.0
    fxy = spots.to(device=device, dtype=dtype)
    xx0 = xx[None, :, :] - x0[:, None, None]
    yy0 = yy[None, :, :] - y0[:, None, None]
    qx1 = 2.0 * torch.pi / (float(width) * float(pixel_size_x_nm))
    qy1 = 2.0 * torch.pi / (float(height) * float(pixel_size_y_nm))
    phasora_xy = (torch.sin(qx1 * xx0) * torch.sin(qy1 * yy0) * fxy).sum(dim=(-2, -1))
    phasora_xx = 2.0 * (torch.cos(qx1 * xx0) * fxy).sum(dim=(-2, -1))
    phasora_yy = 2.0 * (torch.cos(qy1 * yy0) * fxy).sum(dim=(-2, -1))
    delt_a = torch.cos(2.0 * astig_angle) * (phasora_yy - phasora_xx) + torch.sin(2.0 * astig_angle) * 2.0 * phasora_xy
    ac = phasora_xx + phasora_yy
    dof = float(wavelength_nm) / 1.518
    s_curve_length = (2.0 * 1.518 * float(wavelength_nm) / (1.4**2)) * (8.0**0.5) * astig_rms
    gamma_fac = float(dof) / s_curve_length.clamp_min(1e-6)
    discr = ac.square() - (1.0 + gamma_fac.square()) * delt_a.square()
    z0_safe = s_curve_length * (1.0 + gamma_fac.square()) * delt_a / (ac + torch.sqrt(discr.clamp_min(1e-6))).clamp_min(1e-6)
    z0_fallback = s_curve_length * (1.0 + gamma_fac.square()) * delt_a / ac.clamp_min(1e-6)
    return torch.where(discr > 0.0, z0_safe, z0_fallback).clamp(float(z_min_nm), float(z_max_nm)).to(dtype=torch.float32)


def _fit_metrics(
    *,
    raw: torch.Tensor,
    recon: torch.Tensor,
    loss_history: list[float],
    fit_kind: str,
    stage: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    raw_f = raw.to(dtype=torch.float32)
    recon_f = recon.to(dtype=torch.float32)
    ncc = _ncc(raw_f, recon_f)
    mse = (recon_f - raw_f).square().mean(dim=(-2, -1))
    count = int(raw_f.shape[0])
    return {
        "stage": stage,
        "fit_kind": fit_kind,
        "status": "ok",
        "selected_emitters": count,
        "patch_ncc_values": [float(v) for v in ncc.tolist()],
        "patch_ncc_mean": 0.0 if count <= 0 else float(ncc.mean().item()),
        "patch_ncc_median": 0.0 if count <= 0 else float(ncc.median().item()),
        "patch_mse_mean": 0.0 if count <= 0 else float(mse.mean().item()),
        "patch_mse_median": 0.0 if count <= 0 else float(mse.median().item()),
        "raw_patch_power_mean": 0.0 if count <= 0 else float(raw_f.square().mean(dim=(-2, -1)).mean().item()),
        "loss_initial": None if not loss_history else float(loss_history[0]),
        "loss_final": None if not loss_history else float(loss_history[-1]),
        "loss_history_length": int(len(loss_history)),
        **extra,
    }


def _with_ncc_filter(state: VectorNATFitState, *, threshold: float) -> VectorNATFitState:
    ncc = torch.as_tensor(state.metrics.get("patch_ncc_values", []), dtype=torch.float32)
    mask = ncc >= float(threshold)
    metrics = {
        **state.metrics,
        "selected_patch_ncc_values": [float(v) for v in ncc[mask].tolist()],
        "ncc_filter": {
            "threshold": float(threshold),
            "total_count": int(ncc.numel()),
            "kept_count": int(mask.sum().item()),
            "rejected_count": int(ncc.numel() - mask.sum().item()),
            "kept_fraction": 0.0 if ncc.numel() == 0 else float(mask.sum().item()) / float(ncc.numel()),
        },
    }
    return VectorNATFitState(**{**state.__dict__, "metrics": metrics})


def _ncc(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if int(target.shape[0]) == 0:
        return torch.zeros((0,), dtype=torch.float32)
    target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_centered = pred - pred.mean(dim=(-2, -1), keepdim=True)
    denom = torch.sqrt(target_centered.square().sum(dim=(-2, -1)).clamp_min(1e-8) * pred_centered.square().sum(dim=(-2, -1)).clamp_min(1e-8))
    return (target_centered * pred_centered).sum(dim=(-2, -1)) / denom


def _comparison_metrics(approximate: VectorNATFitState, alternating: VectorNATFitState) -> dict[str, Any]:
    gamma_delta = alternating.gamma - approximate.gamma
    coeff_delta = alternating.coefficients_nm - approximate.coefficients_nm
    return {
        "gamma_delta_norm": float(torch.linalg.norm(gamma_delta).item()),
        "gamma_delta_max_abs": 0.0 if gamma_delta.numel() == 0 else float(gamma_delta.abs().max().item()),
        "coeff_delta_mean_abs": 0.0 if coeff_delta.numel() == 0 else float(coeff_delta.abs().mean().item()),
        "coeff_delta_max_abs": 0.0 if coeff_delta.numel() == 0 else float(coeff_delta.abs().max().item()),
    }


def _state_payload(state: VectorNATFitState) -> dict[str, Any]:
    values = state.metrics.get("patch_ncc_values", [])
    threshold = (state.metrics.get("ncc_filter") or {}).get("threshold", 0.0)
    return {
        "gamma": state.gamma,
        "coefficients_nm": state.coefficients_nm,
        "roixy_px": state.roixy_px,
        "local_x_nm": state.local_x_nm,
        "local_y_nm": state.local_y_nm,
        "z_nm": state.z_nm,
        "photons": state.photons,
        "background_adu": state.background_adu,
        "ncc_filter_mask": torch.as_tensor(values, dtype=torch.float32) >= float(threshold),
        "loss_history": state.loss_history,
        "metrics": state.metrics,
    }


def _empty_state(fit_kind: str, *, nat_config, dataset: VectorNATPatchDataset) -> VectorNATFitState:
    empty = torch.zeros((0,), dtype=torch.float32)
    metrics = {
        "stage": "alternating" if "alternating" in fit_kind else "approximate",
        "fit_kind": fit_kind,
        "status": "insufficient_emitters",
        "selected_emitters": int(dataset.patches_adu.shape[0]),
        "patch_ncc_values": [],
        "patch_ncc_mean": 0.0,
        "patch_ncc_median": 0.0,
        "patch_mse_mean": 0.0,
        "patch_mse_median": 0.0,
        "loss_history_length": 0,
        "implementation": "neptune_v03_native_vector_nat",
    }
    return VectorNATFitState(
        gamma=torch.zeros((len(nat_config.gammas),), dtype=torch.float32),
        coefficients_nm=torch.zeros((0, len(nat_config.aberrations)), dtype=torch.float32),
        roixy_px=dataset.roixy_px,
        local_x_nm=empty,
        local_y_nm=empty,
        z_nm=empty,
        photons=empty,
        background_adu=empty,
        recon_patches=dataset.patches_adu.clone(),
        loss_history=tuple(),
        metrics=metrics,
    )


def _write_vector_psf_smoke(path: Path, *, coeffs: torch.Tensor) -> None:
    if coeffs.numel() == 0:
        payload = {"status": "empty"}
    else:
        payload = {"status": "ok", **render_astigmatic_vector_psf_smoke(coeffs[0])}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _orientation_moment(frame: torch.Tensor) -> float:
    image = frame.to(dtype=torch.float32)
    h, w = image.shape
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    total = image.sum().clamp_min(1e-8)
    x0 = (image * xx).sum() / total
    y0 = (image * yy).sum() / total
    mu_xx = (image * (xx - x0).square()).sum() / total
    mu_yy = (image * (yy - y0).square()).sum() / total
    return float((mu_xx - mu_yy).item())


def _width_from_config(config: PeakBootstrapConfig) -> int:
    if config.crop_x1 is not None:
        return int(config.crop_x1) - int(config.crop_x0)
    return 600


def _height_from_config(config: PeakBootstrapConfig) -> int:
    if config.crop_y1 is not None:
        return int(config.crop_y1) - int(config.crop_y0)
    return 1200


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


def _build_vector_context(device: torch.device, *, patch_size_px: int):
    mode_to_noll = {noll_to_nm(index): index for index in range(1, 64)}
    noll_indices = [mode_to_noll[mode] for mode in build_named_nat_config("order1").aberrations]
    return build_vector_psf_context(
        NA=1.4,
        wavelength_nm=660.0,
        pixel_size_nm_x=101.11,
        pixel_size_nm_y=98.83,
        noll_indices=noll_indices,
        params=VectorPSFParams(
            npupil=128,
            psf_size=int(patch_size_px),
            refmed=1.518,
            refcov=1.518,
            refimm=1.518,
            objstage0=0.0,
            otf_rescale_xy=(0.0, 0.0),
            batch_size=16,
        ),
        device=device,
    )


__all__ = [
    "VectorNATFitState",
    "nat_aberration_noll_indices",
    "render_astigmatic_vector_psf_smoke",
    "run_vector_nat_diagnostics",
]
