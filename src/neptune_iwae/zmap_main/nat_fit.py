from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import math

import numpy as np
import torch
import tifffile

from .optics.vector_psf_utils import VectorPSFParams, build_vector_psf_context, render_vector_psf_bank
from .optics.zernike import noll_to_nm

from .harvest import HarvestedEmitters
from .nat_field import NATFieldConfig, build_named_nat_config, default_order1_config
from .nat_optimizer import NATLocalInit, fit_nat_alternating, initialize_local_from_spots
from .nat_torch import get_fov_coordinates_torch, invert_nat_order1_torch
from .nat_torch import evaluate_zernike_from_roi_positions_torch


HERE = Path(__file__).resolve().parent


def nat_aberration_noll_indices(config: NATFieldConfig) -> tuple[int, ...]:
    mode_to_noll: dict[tuple[int, int], int] = {}
    for noll_index in range(1, 128):
        mode_to_noll[noll_to_nm(noll_index)] = noll_index
    try:
        return tuple(mode_to_noll[mode] for mode in config.aberrations)
    except KeyError as exc:
        raise ValueError(f"Unsupported Zernike mode for vector PSF rendering: {exc.args[0]}") from exc


class _NeptunePSFAdapter:
    """Neptune-local vector PSF adapter used by NAT fitting."""

    def __init__(self, cfg: "NATSpotFitterConfig") -> None:
        self.device = torch.device(str(cfg.device))
        self.psf_size = int(cfg.patch_size_px)
        self.wavelength_nm = float(cfg.wavelength_nm)
        self.ctx = build_vector_psf_context(
            NA=float(cfg.NA),
            wavelength_nm=float(cfg.wavelength_nm),
            pixel_size_nm_x=float(cfg.pixel_size_x_nm),
            pixel_size_nm_y=float(cfg.pixel_size_y_nm),
            noll_indices=nat_aberration_noll_indices(cfg.nat_config),
            params=VectorPSFParams(
                npupil=int(cfg.npupil),
                psf_size=int(cfg.patch_size_px),
                refmed=float(cfg.refmed),
                refcov=float(cfg.refcov),
                refimm=float(cfg.refimm),
                objstage0=float(cfg.objstage0),
                otf_rescale_xy=(float(cfg.otf_rescale_xy[0]), float(cfg.otf_rescale_xy[1])),
                batch_size=64,
            ),
            device=self.device,
        )

    def simulate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        photons: torch.Tensor,
        objstage: torch.Tensor | None = None,
        zernike_coefs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del objstage
        x_nm = x.to(device=self.device, dtype=torch.float32).reshape(-1)
        y_nm = y.to(device=self.device, dtype=torch.float32).reshape(-1)
        z_nm = z.to(device=self.device, dtype=torch.float32).reshape(-1)
        photons_t = photons.to(device=self.device, dtype=torch.float32).reshape(-1)
        if x_nm.numel() == 0:
            return torch.zeros((0, self.psf_size, self.psf_size), device=self.device, dtype=torch.float32)

        if zernike_coefs is None:
            coeffs_nm = torch.zeros((x_nm.shape[0], len(self.ctx.noll_indices)), device=self.device, dtype=torch.float32)
        else:
            coeffs_nm = zernike_coefs.to(device=self.device, dtype=torch.float32)
            if coeffs_nm.ndim == 1:
                coeffs_nm = coeffs_nm.unsqueeze(0).expand(x_nm.shape[0], -1)

        coeffs_rad = coeffs_nm * (2.0 * math.pi / max(self.wavelength_nm, 1e-6)) * self.ctx.normfac[None, :]
        psf = render_vector_psf_bank(
            self.ctx,
            coeffs_rad,
            z_nm * 1e-9,
            out_size=self.psf_size,
            batch_size=self.ctx.batch_size,
            return_torch=True,
        )
        psf = psf.clamp_min(0.0)
        psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        return psf * photons_t[:, None, None]


def _load_pipeline_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonicalize_stack_shape(arr: np.ndarray) -> np.ndarray:
    frames = np.asarray(arr)
    while frames.ndim > 3 and 1 in frames.shape:
        squeeze_axes = tuple(ix for ix, dim in enumerate(frames.shape) if dim == 1)
        frames = np.squeeze(frames, axis=squeeze_axes[:1])
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim != 3:
        raise ValueError(f"Expected TIFF stack with shape (N,H,W), got {tuple(frames.shape)}")
    return frames


def _canonicalize_stack_dims(shape: tuple[int, ...]) -> tuple[int, int, int]:
    non_singleton = tuple(int(dim) for dim in shape if int(dim) != 1)
    if len(non_singleton) == 2:
        return 1, int(non_singleton[0]), int(non_singleton[1])
    if len(non_singleton) == 3:
        return int(non_singleton[0]), int(non_singleton[1]), int(non_singleton[2])
    raise ValueError(f"Unsupported TIFF stack shape: {shape}")


def _load_tiff_stack(path: Path) -> np.ndarray:
    try:
        return _canonicalize_stack_shape(np.asarray(tifffile.memmap(path)))
    except ValueError:
        return _canonicalize_stack_shape(tifffile.imread(path))


def _read_selected_tiff_frames(path: Path, frame_indices: torch.Tensor) -> dict[int, np.ndarray]:
    requested = sorted({int(value) for value in frame_indices.detach().cpu().tolist()})
    if not requested:
        return {}
    try:
        stack = _canonicalize_stack_shape(np.asarray(tifffile.memmap(path)))
        return {frame_ix: np.asarray(stack[frame_ix], dtype=np.float32) for frame_ix in requested}
    except ValueError:
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError(f"TIFF has no image series: {path}")
            loaded = _canonicalize_stack_shape(tif.series[0].asarray(key=requested))
        if loaded.shape[0] != len(requested):
            raise ValueError("Failed to load the requested TIFF frame subset.")
        return {
            frame_ix: np.asarray(loaded[idx], dtype=np.float32)
            for idx, frame_ix in enumerate(requested)
        }


@dataclass(frozen=True)
class NATSpotFitterConfig:
    pipeline_config_path: Path
    tiff_path: Path
    nat_config: NATFieldConfig
    nat_config_kind: str = "order1"
    device: str = "cpu"
    patch_size_px: int = 15
    max_emitters: int = 64
    fit_steps: int = 60
    fit_lr: float = 5e-2
    coeff_l2: float = 1e-4
    photon_l2: float = 1e-6
    background_l2: float = 1e-6
    ridge: float = 1e-4
    input_offset: float = 0.0
    input_scale: float = 1.0
    patch_adu_mode: str = "baseline_clamp"
    qe: float = 1.0
    e_per_adu: float = 1.0
    baseline_adu: float = 0.0
    pixel_size_x_nm: float = 100.0
    pixel_size_y_nm: float = 100.0
    z_min_nm: float = -600.0
    z_max_nm: float = 600.0
    npupil: int = 64
    otf_rescale_xy: tuple[float, float] = (0.0, 0.0)
    NA: float = 1.4
    wavelength_nm: float = 660.0
    refmed: float = 1.518
    refcov: float = 1.518
    refimm: float = 1.518
    objstage0: float = 0.0
    base_coefficients_nm: torch.Tensor | None = None
    fixed_coefficients_nm: torch.Tensor | None = None
    fixed_mode_mask: torch.Tensor | None = None
    min_neighbor_distance_px: float = 15.0
    global_projected_min_distance_px: float = 0.0
    selection_pool_multiplier: int = 50
    selection_mode: str = "probability_topk"
    spatial_balance_grid_px: int = 128
    spatial_balance_min_per_cell: int = 0
    spatial_balance_max_per_cell: int = 0
    max_patch_peak_distance_px: float = 2.5
    secondary_peak_exclusion_radius_px: float = 3.5
    max_secondary_peak_fraction: float = 0.45
    rank_spots_by_quality: bool = False
    min_center_peak_norm: float = 0.0
    min_signal_sum_norm: float = 0.0
    alternating_min_photon_fraction: float = 0.0
    alternating_min_background_fraction: float = 0.0
    roi_x_min_px: float = 0.0
    roi_x_max_px: float | None = None
    roi_y_min_px: float = 0.0
    roi_y_max_px: float | None = None
    roi_edge_margin_px: float = 0.0
    vectorfit_astig_gauge: bool = False
    vectorfit_astig_anchor_nm: float | None = None
    vectorfit_astig_anchor_mode: str = "init_only"
    vectorfit_phasor_z_init: bool = False
    alternating_optimizer_kind: str = "adam"
    emitters_preselected: bool = False

    @property
    def emit_scale(self) -> float:
        return float(self.qe) / max(float(self.e_per_adu), 1e-6) / max(float(self.input_scale), 1e-6)

    @property
    def emit_scale_adu(self) -> float:
        return float(self.qe) / max(float(self.e_per_adu), 1e-6)


@dataclass(frozen=True)
class NATPatchDataset:
    patches_norm: torch.Tensor
    patches_adu: torch.Tensor
    roixy_px: torch.Tensor
    local_x_nm: torch.Tensor
    local_y_nm: torch.Tensor
    z_nm: torch.Tensor
    photons_init: torch.Tensor
    background_init: torch.Tensor
    background_adu: torch.Tensor
    probability: torch.Tensor
    frame_index: torch.Tensor
    selected_indices: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    selection_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NATSpotFitResult:
    gamma: torch.Tensor
    coefficients_nm: torch.Tensor
    roixy_px: torch.Tensor
    local_x_nm: torch.Tensor
    local_y_nm: torch.Tensor
    z_nm: torch.Tensor
    photons: torch.Tensor
    background_norm: torch.Tensor
    background_adu: torch.Tensor
    loss_history: tuple[float, ...]
    metrics: dict[str, Any]


def build_nat_spot_fitter_config(
    *,
    pipeline_config_path: str | Path,
    tiff_path: str | Path | None = None,
    nat_config: NATFieldConfig | None = None,
    nat_config_kind: str = "order1",
    device: str = "cpu",
    patch_size_px: int | None = None,
    max_emitters: int = 64,
    fit_steps: int = 60,
    fit_lr: float = 5e-2,
    coeff_l2: float = 1e-4,
    photon_l2: float = 1e-6,
    background_l2: float = 1e-6,
    ridge: float = 1e-4,
    base_coefficients_nm: torch.Tensor | None = None,
    fixed_coefficients_nm: torch.Tensor | None = None,
    fixed_mode_mask: torch.Tensor | None = None,
    freeze_initial_astig_standard: bool = False,
    freeze_defocus_zero_gauge: bool = False,
    vectorfit_astig_gauge: bool = False,
    vectorfit_astig_anchor_nm: float | None = None,
    vectorfit_astig_anchor_mode: str = "init_only",
    vectorfit_phasor_z_init: bool = False,
    alternating_optimizer_kind: str = "adam",
    emitters_preselected: bool = False,
    min_neighbor_distance_px: float | None = None,
    global_projected_min_distance_px: float = 0.0,
    selection_mode: str = "probability_topk",
    spatial_balance_grid_px: int = 128,
    spatial_balance_min_per_cell: int = 0,
    spatial_balance_max_per_cell: int = 0,
    max_patch_peak_distance_px: float = 2.5,
    secondary_peak_exclusion_radius_px: float = 3.5,
    max_secondary_peak_fraction: float = 0.45,
    rank_spots_by_quality: bool = False,
    min_center_peak_norm: float = 0.0,
    min_signal_sum_norm: float = 0.0,
    alternating_min_photon_fraction: float = 0.0,
    alternating_min_background_fraction: float = 0.0,
    roi_x_min_px: float = 0.0,
    roi_x_max_px: float | None = None,
    roi_y_min_px: float = 0.0,
    roi_y_max_px: float | None = None,
    roi_edge_margin_px: float = 0.0,
    input_offset: float | None = None,
    input_scale: float | None = None,
    patch_adu_mode: str = "baseline_clamp",
    baseline_adu: float | None = None,
) -> NATSpotFitterConfig:
    pipeline_path = Path(pipeline_config_path)
    cfg = _load_pipeline_json(pipeline_path)
    optical = cfg.get("optical") or {}
    camera = cfg.get("camera") or {}
    scaling = ((cfg.get("train") or {}).get("scaling") or {})
    emitter = ((cfg.get("simulation") or {}).get("emitter") or {})
    vector = ((((cfg.get("simulation") or {}).get("psf") or {}).get("vector")) or {})
    source_tiff = tiff_path if tiff_path is not None else ((cfg.get("phase_retrieval") or {}).get("input") or {}).get("tiff_path")
    if source_tiff is None:
        raise ValueError("No TIFF path provided and pipeline config has no phase_retrieval.input.tiff_path.")
    psf_size = int(vector.get("psf_size", 15))
    patch = psf_size if patch_size_px is None else int(patch_size_px)
    if patch % 2 == 0:
        patch += 1
    patch = max(7, patch)
    roi_width_px = None if roi_x_max_px is None else float(roi_x_max_px) - float(roi_x_min_px)
    roi_height_px = None if roi_y_max_px is None else float(roi_y_max_px) - float(roi_y_min_px)
    nat_cfg = build_named_nat_config(
        str(nat_config_kind),
        img_size_x=int(round(roi_width_px)) if roi_width_px is not None else int((emitter.get("field_extent") or [[0, 1024], [0, 1024]])[0][1]),
        img_size_y=int(round(roi_height_px)) if roi_height_px is not None else int((emitter.get("field_extent") or [[0, 1024], [0, 1024]])[1][1]),
        pixel_size_x_nm=float(optical.get("pixel_size_nm_x", 100.0)),
        pixel_size_y_nm=float(optical.get("pixel_size_nm_y", 100.0)),
    ) if nat_config is None else nat_config
    fixed_coeff_nm_value = None if fixed_coefficients_nm is None else torch.as_tensor(fixed_coefficients_nm, dtype=torch.float32).clone()
    fixed_mask_value = None if fixed_mode_mask is None else torch.as_tensor(fixed_mode_mask, dtype=torch.bool).clone()
    if bool(freeze_initial_astig_standard):
        astig_fixed = initial_astig_standard_coefficients_nm(nat_cfg)
        astig_mask = initial_astig_standard_mode_mask(nat_cfg)
        if fixed_coeff_nm_value is None:
            fixed_coeff_nm_value = astig_fixed
        else:
            fixed_coeff_nm_value = fixed_coeff_nm_value.reshape(-1).clone()
            fixed_coeff_nm_value[astig_mask] = astig_fixed[astig_mask]
        if fixed_mask_value is None:
            fixed_mask_value = astig_mask
        else:
            fixed_mask_value = fixed_mask_value.reshape(-1).clone()
            fixed_mask_value[astig_mask] = True
    if bool(freeze_defocus_zero_gauge):
        defocus_fixed = initial_defocus_zero_coefficients_nm(nat_cfg)
        defocus_mask = initial_defocus_zero_mode_mask(nat_cfg)
        if fixed_coeff_nm_value is None:
            fixed_coeff_nm_value = defocus_fixed
        else:
            fixed_coeff_nm_value = fixed_coeff_nm_value.reshape(-1).clone()
            fixed_coeff_nm_value[defocus_mask] = defocus_fixed[defocus_mask]
        if fixed_mask_value is None:
            fixed_mask_value = defocus_mask
        else:
            fixed_mask_value = fixed_mask_value.reshape(-1).clone()
            fixed_mask_value[defocus_mask] = True
    z_range = emitter.get("z_range", [-0.6, 0.6])
    return NATSpotFitterConfig(
        pipeline_config_path=pipeline_path,
        tiff_path=Path(source_tiff),
        nat_config=nat_cfg,
        nat_config_kind=str(nat_config_kind),
        device=str(device),
        patch_size_px=patch,
        max_emitters=int(max_emitters),
        fit_steps=int(fit_steps),
        fit_lr=float(fit_lr),
        coeff_l2=float(coeff_l2),
        photon_l2=float(photon_l2),
        background_l2=float(background_l2),
        ridge=float(ridge),
        input_offset=float(scaling.get("input_offset", 0.0)) if input_offset is None else float(input_offset),
        input_scale=max(float(scaling.get("input_scale", 1.0)) if input_scale is None else float(input_scale), 1e-6),
        patch_adu_mode=str(patch_adu_mode),
        qe=float(camera.get("qe", 1.0)),
        e_per_adu=float(camera.get("e_per_adu", 1.0)),
        baseline_adu=float(camera.get("baseline", 0.0)) if baseline_adu is None else float(baseline_adu),
        pixel_size_x_nm=float(optical.get("pixel_size_nm_x", 100.0)),
        pixel_size_y_nm=float(optical.get("pixel_size_nm_y", 100.0)),
        z_min_nm=float(z_range[0]) * 1000.0,
        z_max_nm=float(z_range[1]) * 1000.0,
        npupil=int(vector.get("npupil", 64)),
        otf_rescale_xy=tuple(float(v) for v in vector.get("otf_rescale_xy", [0.0, 0.0])),
        NA=float(optical.get("NA", 1.4)),
        wavelength_nm=float(optical.get("wavelength_nm", 660.0)),
        refmed=float(vector.get("refmed", optical.get("n_medium", 1.518))),
        refcov=float(vector.get("refcov", optical.get("n_medium", 1.518))),
        refimm=float(vector.get("refimm", optical.get("n_medium", 1.518))),
        objstage0=float(vector.get("objstage0", 0.0)),
        base_coefficients_nm=None if base_coefficients_nm is None else torch.as_tensor(base_coefficients_nm, dtype=torch.float32).clone(),
        fixed_coefficients_nm=fixed_coeff_nm_value,
        fixed_mode_mask=fixed_mask_value,
        min_neighbor_distance_px=float(patch if min_neighbor_distance_px is None else min_neighbor_distance_px),
        global_projected_min_distance_px=max(float(global_projected_min_distance_px), 0.0),
        max_patch_peak_distance_px=float(max_patch_peak_distance_px),
        selection_mode=str(selection_mode),
        spatial_balance_grid_px=max(int(spatial_balance_grid_px), 1),
        spatial_balance_min_per_cell=max(int(spatial_balance_min_per_cell), 0),
        spatial_balance_max_per_cell=max(int(spatial_balance_max_per_cell), 0),
        secondary_peak_exclusion_radius_px=float(secondary_peak_exclusion_radius_px),
        max_secondary_peak_fraction=float(max_secondary_peak_fraction),
        rank_spots_by_quality=bool(rank_spots_by_quality),
        min_center_peak_norm=float(min_center_peak_norm),
        min_signal_sum_norm=float(min_signal_sum_norm),
        alternating_min_photon_fraction=float(alternating_min_photon_fraction),
        alternating_min_background_fraction=float(alternating_min_background_fraction),
        roi_x_min_px=float(roi_x_min_px),
        roi_x_max_px=None if roi_x_max_px is None else float(roi_x_max_px),
        roi_y_min_px=float(roi_y_min_px),
        roi_y_max_px=None if roi_y_max_px is None else float(roi_y_max_px),
        roi_edge_margin_px=float(roi_edge_margin_px),
        vectorfit_astig_gauge=bool(vectorfit_astig_gauge),
        vectorfit_astig_anchor_nm=None if vectorfit_astig_anchor_nm is None else float(vectorfit_astig_anchor_nm),
        vectorfit_astig_anchor_mode=_normalize_vectorfit_astig_anchor_mode(vectorfit_astig_anchor_mode),
        vectorfit_phasor_z_init=bool(vectorfit_phasor_z_init),
        alternating_optimizer_kind=str(alternating_optimizer_kind),
        emitters_preselected=bool(emitters_preselected),
    )


def _mode_index_or_raise(config: NATFieldConfig, mode: tuple[int, int]) -> int:
    try:
        return list(config.aberrations).index(mode)
    except ValueError as exc:
        raise ValueError(f"Required aberration mode {mode} missing from NAT config.") from exc


def initial_astig_standard_mode_mask(config: NATFieldConfig | None = None) -> torch.Tensor:
    cfg = default_order1_config() if config is None else config
    mask = torch.zeros((len(cfg.aberrations),), dtype=torch.bool)
    mask[_mode_index_or_raise(cfg, (2, 2))] = True
    mask[_mode_index_or_raise(cfg, (2, -2))] = True
    return mask


def initial_astig_standard_coefficients_nm(config: NATFieldConfig | None = None) -> torch.Tensor:
    cfg = default_order1_config() if config is None else config
    coeffs = torch.zeros((len(cfg.aberrations),), dtype=torch.float32)
    coeffs[_mode_index_or_raise(cfg, (2, 2))] = -68.4301840
    coeffs[_mode_index_or_raise(cfg, (2, -2))] = 7.1589142
    return coeffs


def initial_defocus_zero_mode_mask(config: NATFieldConfig | None = None) -> torch.Tensor:
    cfg = default_order1_config() if config is None else config
    mask = torch.zeros((len(cfg.aberrations),), dtype=torch.bool)
    mask[_mode_index_or_raise(cfg, (2, 0))] = True
    return mask


def initial_defocus_zero_coefficients_nm(config: NATFieldConfig | None = None) -> torch.Tensor:
    cfg = default_order1_config() if config is None else config
    return torch.zeros((len(cfg.aberrations),), dtype=torch.float32)


def vectorfit_astig_positive_gamma_index(config: NATFieldConfig | None = None) -> int:
    cfg = default_order1_config() if config is None else config
    for index, gamma in enumerate(cfg.gammas):
        if not gamma.components:
            continue
        component = gamma.components[0]
        if component.px == 0 and component.py == 0 and component.n == 2 and component.m == 2:
            return int(index)
    raise ValueError("NAT config has no Vectorfit-compatible astigmatism gamma anchor.")


def _normalize_vectorfit_astig_anchor_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "init": "init_only",
        "initialization": "init_only",
        "warmup_release": "init_only",
        "release": "init_only",
        "fixed": "hard_fixed",
        "pin": "hard_fixed",
        "pinned": "hard_fixed",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"init_only", "hard_fixed"}:
        raise ValueError("vectorfit_astig_anchor_mode must be 'init_only' or 'hard_fixed'.")
    return normalized


def vectorfit_astig_gamma_init(config: NATFieldConfig | None = None, *, wavelength_nm: float = 660.0) -> torch.Tensor:
    cfg = default_order1_config() if config is None else config
    gamma = torch.zeros((len(cfg.gammas),), dtype=torch.float32)
    gamma[vectorfit_astig_positive_gamma_index(cfg)] = float(150.0 * float(wavelength_nm) * 1e-3)
    return gamma


def vectorfit_gamma_symmetry_vector(config: NATFieldConfig | None = None) -> torch.Tensor:
    cfg = default_order1_config() if config is None else config
    values = torch.ones((len(cfg.gammas),), dtype=torch.float32)
    for index, gamma in enumerate(cfg.gammas):
        if not gamma.components:
            continue
        if int(gamma.components[0].m) % 2 == 0:
            values[index] = -1.0
    return values


def astig_phasor_z_initial_nm(
    spots: torch.Tensor,
    *,
    local_x_nm: torch.Tensor,
    local_y_nm: torch.Tensor,
    zernike_coefficients_nm: torch.Tensor,
    pixel_size_x_nm: float,
    pixel_size_y_nm: float,
    wavelength_nm: float,
    NA: float,
    refmed: float,
    z_min_nm: float,
    z_max_nm: float,
) -> torch.Tensor:
    if spots.ndim != 3:
        raise ValueError("spots must have shape (N,H,W).")
    device = spots.device
    dtype = spots.dtype
    count, height, width = spots.shape
    x0 = local_x_nm.to(device=device, dtype=dtype).reshape(-1)
    y0 = local_y_nm.to(device=device, dtype=dtype).reshape(-1)
    coeffs = zernike_coefficients_nm.to(device=device, dtype=dtype)
    if x0.shape[0] != count or y0.shape[0] != count or coeffs.shape[0] != count:
        raise ValueError("local positions and zernike coefficients must match spot count.")

    yy, xx = torch.meshgrid(
        (torch.arange(height, device=device, dtype=dtype) - (height - 1.0) / 2.0) * float(pixel_size_y_nm),
        (torch.arange(width, device=device, dtype=dtype) - (width - 1.0) / 2.0) * float(pixel_size_x_nm),
        indexing="ij",
    )
    qx1 = 2.0 * torch.pi / (float(width) * float(pixel_size_x_nm))
    qy1 = 2.0 * torch.pi / (float(height) * float(pixel_size_y_nm))
    xx0 = xx[None, :, :] - x0[:, None, None]
    yy0 = yy[None, :, :] - y0[:, None, None]
    fxy = spots.to(device=device, dtype=dtype)

    mode_to_index = {mode: index for index, mode in enumerate(default_order1_config().aberrations)}
    astig_hv = coeffs[:, mode_to_index[(2, 2)]] / float(wavelength_nm)
    astig_diag = coeffs[:, mode_to_index[(2, -2)]] / float(wavelength_nm)
    astig_rms = torch.sqrt(astig_hv.square() + astig_diag.square()).clamp_min(1e-6)
    astig_angle = torch.atan2(astig_diag, astig_hv) / 2.0

    sinfacxa = torch.sin(qx1 * xx0)
    sinfacya = torch.sin(qy1 * yy0)
    cosfacxa = torch.cos(qx1 * xx0)
    cosfacya = torch.cos(qy1 * yy0)
    phasora_xy = (sinfacxa * sinfacya * fxy).sum(dim=(-2, -1))
    phasora_xx = 2.0 * (cosfacxa * fxy).sum(dim=(-2, -1))
    phasora_yy = 2.0 * (cosfacya * fxy).sum(dim=(-2, -1))
    delt_a = torch.cos(2.0 * astig_angle) * (phasora_yy - phasora_xx) + torch.sin(2.0 * astig_angle) * 2.0 * phasora_xy
    ac = phasora_xx + phasora_yy

    if float(NA) < float(refmed):
        dof = float(wavelength_nm) / (float(refmed) - (float(refmed) ** 2 - float(NA) ** 2) ** 0.5)
    else:
        dof = float(wavelength_nm) / float(refmed)
    s_curve_length = (2.0 * float(refmed) * float(wavelength_nm) / (float(NA) ** 2)) * (8.0 ** 0.5) * astig_rms
    gamma_fac = float(dof) / s_curve_length.clamp_min(1e-6)
    discr = ac.square() - (1.0 + gamma_fac.square()) * delt_a.square()
    z0_safe = s_curve_length * (1.0 + gamma_fac.square()) * delt_a / (
        ac + torch.sqrt(discr.clamp_min(1e-6))
    ).clamp_min(1e-6)
    z0_fallback = s_curve_length * (1.0 + gamma_fac.square()) * delt_a / ac.clamp_min(1e-6)
    z0 = torch.where(discr > 0.0, z0_safe, z0_fallback)
    return z0.clamp(float(z_min_nm), float(z_max_nm)).to(dtype=torch.float32)


def _fixed_coefficients_and_mask(
    cfg: NATSpotFitterConfig,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if cfg.fixed_coefficients_nm is None:
        return None, None
    coeffs = torch.as_tensor(cfg.fixed_coefficients_nm, dtype=torch.float32).reshape(-1)
    if coeffs.shape[0] != len(cfg.nat_config.aberrations):
        raise ValueError("fixed_coefficients_nm must have shape (num_aberrations,).")
    mask = (
        torch.as_tensor(cfg.fixed_mode_mask, dtype=torch.bool).reshape(-1)
        if cfg.fixed_mode_mask is not None
        else torch.ones_like(coeffs, dtype=torch.bool)
    )
    if mask.shape[0] != coeffs.shape[0]:
        raise ValueError("fixed_mode_mask must have shape (num_aberrations,).")
    if device is not None:
        coeffs = coeffs.to(device=device)
        mask = mask.to(device=device)
    return coeffs, mask


def _apply_fixed_coefficients_(
    coeffs: torch.Tensor,
    fixed_coefficients_nm: torch.Tensor | None,
    fixed_mode_mask: torch.Tensor | None,
) -> torch.Tensor:
    if fixed_coefficients_nm is None or fixed_mode_mask is None:
        return coeffs
    coeffs[..., fixed_mode_mask] = fixed_coefficients_nm[fixed_mode_mask]
    return coeffs


def _subtract_fixed_coefficients(
    coeffs: torch.Tensor,
    fixed_coefficients_nm: torch.Tensor | None,
    fixed_mode_mask: torch.Tensor | None,
) -> torch.Tensor:
    if fixed_coefficients_nm is None or fixed_mode_mask is None:
        return coeffs
    result = coeffs.clone()
    result[..., fixed_mode_mask] = result[..., fixed_mode_mask] - fixed_coefficients_nm[fixed_mode_mask]
    return result


def _gamma_train_mask_from_fixed_modes(
    config: NATFieldConfig,
    fixed_mode_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if fixed_mode_mask is None:
        return None
    fixed_mask = fixed_mode_mask.to(dtype=torch.bool).reshape(-1)
    mode_to_index = {mode: idx for idx, mode in enumerate(config.aberrations)}
    train = torch.ones((len(config.gammas),), dtype=torch.bool)
    for gamma_index, gamma in enumerate(config.gammas):
        for component in gamma.components:
            mode_index = mode_to_index[(component.n, component.m)]
            if bool(fixed_mask[mode_index].item()):
                train[gamma_index] = False
                break
    return train


def _select_emitters_for_fit(emitters: HarvestedEmitters, cfg: NATSpotFitterConfig, height: int, width: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(emitters) == 0:
        return torch.zeros((0,), dtype=torch.long), {"total_emitters": 0, "base_valid_count": 0, "isolated_count": 0}
    half = cfg.patch_size_px // 2
    cx = torch.round(emitters.x_px - 0.5).to(dtype=torch.int64)
    cy = torch.round(emitters.y_px - 0.5).to(dtype=torch.int64)
    roi_x_min = float(cfg.roi_x_min_px) + float(cfg.roi_edge_margin_px)
    roi_y_min = float(cfg.roi_y_min_px) + float(cfg.roi_edge_margin_px)
    roi_x_max = float(width if cfg.roi_x_max_px is None else cfg.roi_x_max_px) - float(cfg.roi_edge_margin_px)
    roi_y_max = float(height if cfg.roi_y_max_px is None else cfg.roi_y_max_px) - float(cfg.roi_edge_margin_px)
    keep = (
        torch.isfinite(emitters.x_px)
        & torch.isfinite(emitters.y_px)
        & torch.isfinite(emitters.photons)
        & torch.isfinite(emitters.local_x_nm)
        & torch.isfinite(emitters.local_y_nm)
        & torch.isfinite(emitters.z_um)
        & (cx - half >= 0)
        & (cy - half >= 0)
        & (cx + half < width)
        & (cy + half < height)
        & (emitters.x_px >= roi_x_min)
        & (emitters.x_px < roi_x_max)
        & (emitters.y_px >= roi_y_min)
        & (emitters.y_px < roi_y_max)
    )
    idx = keep.nonzero(as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return idx, {
            "total_emitters": len(emitters),
            "base_valid_count": 0,
            "isolated_count": 0,
            "roi_x_min_px": float(cfg.roi_x_min_px),
            "roi_x_max_px": None if cfg.roi_x_max_px is None else float(cfg.roi_x_max_px),
            "roi_y_min_px": float(cfg.roi_y_min_px),
            "roi_y_max_px": None if cfg.roi_y_max_px is None else float(cfg.roi_y_max_px),
            "roi_edge_margin_px": float(cfg.roi_edge_margin_px),
        }
    order = torch.argsort(emitters.probability[idx], descending=True)
    idx = idx[order]
    pool_size = min(int(idx.numel()), max(int(cfg.max_emitters), int(cfg.max_emitters) * max(1, int(cfg.selection_pool_multiplier))))
    pool = idx[:pool_size]
    selection_metrics: dict[str, Any] = {"selection_mode": str(cfg.selection_mode)}
    if str(cfg.selection_mode).lower() == "spatial_balanced" and pool.numel() > 0:
        grid = max(int(cfg.spatial_balance_grid_px), 1)
        min_per_cell = max(int(cfg.spatial_balance_min_per_cell), 0)
        max_per_cell = max(int(cfg.spatial_balance_max_per_cell), 0)
        roi_width = max(float(width if cfg.roi_x_max_px is None else cfg.roi_x_max_px) - float(cfg.roi_x_min_px), 0.0)
        roi_height = max(float(height if cfg.roi_y_max_px is None else cfg.roi_y_max_px) - float(cfg.roi_y_min_px), 0.0)
        target_cell_x_count = max(int(math.ceil(roi_width / float(grid))), 1)
        target_cell_y_count = max(int(math.ceil(roi_height / float(grid))), 1)
        target_cells = {
            (cell_x, cell_y)
            for cell_y in range(target_cell_y_count)
            for cell_x in range(target_cell_x_count)
        }
        pool_x = emitters.x_px[pool]
        pool_y = emitters.y_px[pool]
        cell_to_indices: dict[tuple[int, int], list[int]] = {}
        for local_ix, global_ix in enumerate(pool.tolist()):
            cell_x = int(max(float(pool_x[local_ix].item()) - float(cfg.roi_x_min_px), 0.0) // grid)
            cell_y = int(max(float(pool_y[local_ix].item()) - float(cfg.roi_y_min_px), 0.0) // grid)
            cell_to_indices.setdefault((cell_x, cell_y), []).append(global_ix)
        pool_counts = {cell: len(cell_to_indices.get(cell, [])) for cell in target_cells}
        underfilled_cells = [
            f"{cell[0]},{cell[1]}"
            for cell, count in sorted(pool_counts.items(), key=lambda item: (item[0][1], item[0][0]))
            if count < min_per_cell
        ]
        round_robin: list[int] = []
        max_depth = max((len(values) for values in cell_to_indices.values()), default=0)
        if max_per_cell > 0:
            max_depth = min(max_depth, max_per_cell)
        ordered_cells = sorted(cell_to_indices.items(), key=lambda item: (-len(item[1]), item[0][1], item[0][0]))
        first_pass_depth = min(min_per_cell, max_depth) if min_per_cell > 0 else 0
        for depth in range(first_pass_depth):
            for _, values in ordered_cells:
                if depth < len(values):
                    round_robin.append(int(values[depth]))
        for depth in range(first_pass_depth, max_depth):
            for _, values in ordered_cells:
                if depth < len(values):
                    round_robin.append(int(values[depth]))
        pool = torch.tensor(round_robin, dtype=torch.long)
        selection_metrics = {
            "spatial_balance_grid_px": int(grid),
            "spatial_balance_min_per_cell": int(min_per_cell),
            "spatial_balance_max_per_cell": int(max_per_cell),
            "spatial_balance_nonempty_cell_count": int(len(cell_to_indices)),
            "spatial_balance_target_cell_x_count": int(target_cell_x_count),
            "spatial_balance_target_cell_y_count": int(target_cell_y_count),
            "spatial_balance_target_cell_count": int(len(target_cells)),
            "spatial_balance_empty_cell_count": int(sum(1 for count in pool_counts.values() if count == 0)),
            "spatial_balance_underfilled_cell_count": int(len(underfilled_cells)),
            "spatial_balance_underfilled_cells": underfilled_cells,
            "spatial_balance_target_emitters": int(cfg.max_emitters),
            **selection_metrics,
        }
    if float(cfg.min_neighbor_distance_px) <= 0.0:
        if float(cfg.global_projected_min_distance_px) > 0.0:
            selected, projected_metrics = _apply_global_projected_min_distance(emitters, pool, cfg)
        else:
            selected = pool[: int(cfg.max_emitters)]
            selected, projected_metrics = _apply_global_projected_min_distance(emitters, selected, cfg)
        if str(cfg.selection_mode).lower() == "spatial_balanced" and selected.numel() > 0:
            grid = max(int(cfg.spatial_balance_grid_px), 1)
            sel_x = emitters.x_px[selected]
            sel_y = emitters.y_px[selected]
            selected_cells = {
                (
                    int(max(float(sel_x[i].item()) - float(cfg.roi_x_min_px), 0.0) // grid),
                    int(max(float(sel_y[i].item()) - float(cfg.roi_y_min_px), 0.0) // grid),
                )
                for i in range(selected.numel())
            }
            selection_metrics = {
                **selection_metrics,
                "spatial_balance_selected_cell_count": int(len(selected_cells)),
            }
        return selected, {
            "total_emitters": len(emitters),
            "base_valid_count": int(idx.numel()),
            "pool_count": int(pool.numel()),
            "isolated_count": int(selected.numel()),
            "max_emitters_after_image_quality": int(cfg.max_emitters),
            "min_neighbor_distance_px": float(cfg.min_neighbor_distance_px),
            **projected_metrics,
            "roi_x_min_px": float(cfg.roi_x_min_px),
            "roi_x_max_px": None if cfg.roi_x_max_px is None else float(cfg.roi_x_max_px),
            "roi_y_min_px": float(cfg.roi_y_min_px),
            "roi_y_max_px": None if cfg.roi_y_max_px is None else float(cfg.roi_y_max_px),
            "roi_edge_margin_px": float(cfg.roi_edge_margin_px),
            **selection_metrics,
        }

    all_x = emitters.x_px.to(dtype=torch.float32)
    all_y = emitters.y_px.to(dtype=torch.float32)
    all_frame = emitters.frame_index.to(dtype=torch.int64)
    min_dist_sq = float(cfg.min_neighbor_distance_px) ** 2
    pool_x = all_x[pool]
    pool_y = all_y[pool]
    pool_frame = all_frame[pool]
    isolated_mask = torch.zeros((pool.numel(),), dtype=torch.bool)
    for frame_id in torch.unique(pool_frame).tolist():
        pool_mask = pool_frame == int(frame_id)
        frame_mask = all_frame == int(frame_id)
        frame_x = all_x[frame_mask]
        frame_y = all_y[frame_mask]
        dx = pool_x[pool_mask][:, None] - frame_x[None, :]
        dy = pool_y[pool_mask][:, None] - frame_y[None, :]
        dist_sq = dx.square() + dy.square()
        neighbor = (dist_sq > 1e-6) & (dist_sq < min_dist_sq)
        isolated_mask[pool_mask] = ~neighbor.any(dim=1)
    selected = pool[isolated_mask]
    if str(cfg.selection_mode).lower() == "spatial_balanced" and selected.numel() > 0:
        grid = max(int(cfg.spatial_balance_grid_px), 1)
        sel_x = emitters.x_px[selected]
        sel_y = emitters.y_px[selected]
        selected_cells = {
            (
                int(max(float(sel_x[i].item()) - float(cfg.roi_x_min_px), 0.0) // grid),
                int(max(float(sel_y[i].item()) - float(cfg.roi_y_min_px), 0.0) // grid),
            )
            for i in range(selected.numel())
        }
        selection_metrics = {
            **selection_metrics,
            "spatial_balance_selected_cell_count": int(len(selected_cells)),
        }
    selected, projected_metrics = _apply_global_projected_min_distance(emitters, selected, cfg)
    return selected, {
        "total_emitters": len(emitters),
        "base_valid_count": int(idx.numel()),
        "pool_count": int(pool.numel()),
        "isolated_count": int(selected.numel()),
        "max_emitters_after_image_quality": int(cfg.max_emitters),
        "min_neighbor_distance_px": float(cfg.min_neighbor_distance_px),
        **projected_metrics,
        "roi_x_min_px": float(cfg.roi_x_min_px),
        "roi_x_max_px": None if cfg.roi_x_max_px is None else float(cfg.roi_x_max_px),
        "roi_y_min_px": float(cfg.roi_y_min_px),
        "roi_y_max_px": None if cfg.roi_y_max_px is None else float(cfg.roi_y_max_px),
        "roi_edge_margin_px": float(cfg.roi_edge_margin_px),
        **selection_metrics,
    }


def _spatial_balance_target_cells(cfg: NATSpotFitterConfig, height: int, width: int) -> tuple[set[tuple[int, int]], int, int]:
    grid = max(int(cfg.spatial_balance_grid_px), 1)
    roi_width = max(float(width if cfg.roi_x_max_px is None else cfg.roi_x_max_px) - float(cfg.roi_x_min_px), 0.0)
    roi_height = max(float(height if cfg.roi_y_max_px is None else cfg.roi_y_max_px) - float(cfg.roi_y_min_px), 0.0)
    target_cell_x_count = max(int(math.ceil(roi_width / float(grid))), 1)
    target_cell_y_count = max(int(math.ceil(roi_height / float(grid))), 1)
    target_cells = {
        (cell_x, cell_y)
        for cell_y in range(target_cell_y_count)
        for cell_x in range(target_cell_x_count)
    }
    return target_cells, target_cell_x_count, target_cell_y_count


def _apply_global_projected_min_distance(
    emitters: HarvestedEmitters,
    selected: torch.Tensor,
    cfg: NATSpotFitterConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    min_distance = float(cfg.global_projected_min_distance_px)
    metrics: dict[str, Any] = {
        "global_projected_min_distance_px": float(min_distance),
        "global_projected_distance_before_count": int(selected.numel()),
    }
    if min_distance <= 0.0 or selected.numel() <= 1:
        metrics["global_projected_distance_after_count"] = int(selected.numel())
        return selected, metrics

    min_dist_sq = min_distance * min_distance
    kept: list[int] = []
    kept_xy: list[tuple[float, float]] = []
    for global_ix in selected.tolist():
        x = float(emitters.x_px[int(global_ix)].item())
        y = float(emitters.y_px[int(global_ix)].item())
        if all((x - prev_x) ** 2 + (y - prev_y) ** 2 >= min_dist_sq for prev_x, prev_y in kept_xy):
            kept.append(int(global_ix))
            kept_xy.append((x, y))

    filtered = torch.tensor(kept, dtype=torch.long)
    metrics["global_projected_distance_after_count"] = int(filtered.numel())
    metrics["global_projected_distance_rejected_count"] = int(selected.numel() - filtered.numel())
    return filtered, metrics


def _spatial_balance_final_metrics(points: list[torch.Tensor], cfg: NATSpotFitterConfig, height: int, width: int) -> dict[str, Any]:
    grid = max(int(cfg.spatial_balance_grid_px), 1)
    min_per_cell = max(int(cfg.spatial_balance_min_per_cell), 0)
    max_per_cell = max(int(cfg.spatial_balance_max_per_cell), 0)
    target_cells, target_cell_x_count, target_cell_y_count = _spatial_balance_target_cells(cfg, height, width)
    cell_counts: dict[tuple[int, int], int] = {cell: 0 for cell in target_cells}
    for point in points:
        cell_x = int(max(float(point[0].item()), 0.0) // grid)
        cell_y = int(max(float(point[1].item()), 0.0) // grid)
        cell = (cell_x, cell_y)
        cell_counts[cell] = int(cell_counts.get(cell, 0) + 1)
    counts = list(cell_counts.values())
    underfilled_cells = [
        f"{cell[0]},{cell[1]}"
        for cell, count in sorted(cell_counts.items(), key=lambda item: (item[0][1], item[0][0]))
        if count < min_per_cell
    ] if min_per_cell > 0 else []
    return {
        "spatial_balance_selected_counts": {
            f"{cell[0]},{cell[1]}": int(count)
            for cell, count in sorted(cell_counts.items(), key=lambda item: (item[0][1], item[0][0]))
            if count > 0
        },
        "spatial_balance_selected_cell_count": int(sum(1 for count in counts if count > 0)),
        "spatial_balance_target_cell_x_count": int(target_cell_x_count),
        "spatial_balance_target_cell_y_count": int(target_cell_y_count),
        "spatial_balance_target_cell_count": int(len(target_cells)),
        "spatial_balance_min_selected_count": int(min(counts)) if counts else 0,
        "spatial_balance_max_selected_count": int(max(counts)) if counts else 0,
        "spatial_balance_max_per_cell": int(max_per_cell),
        "spatial_balance_overfilled_cell_count": int(sum(1 for count in counts if max_per_cell > 0 and count > max_per_cell)),
        "spatial_balance_underfilled_cell_count": int(len(underfilled_cells)),
        "spatial_balance_empty_cell_count": int(sum(1 for count in counts if count == 0)),
        "spatial_balance_underfilled_cells": underfilled_cells,
    }


def _spatial_balance_final_order(points: list[torch.Tensor], cfg: NATSpotFitterConfig) -> tuple[list[int], dict[str, Any]]:
    if str(cfg.selection_mode).lower() != "spatial_balanced" or not points:
        return list(range(len(points))), {"spatial_balance_final_rebalance_applied": False}
    grid = max(int(cfg.spatial_balance_grid_px), 1)
    max_per_cell = max(int(cfg.spatial_balance_max_per_cell), 0)
    cell_to_positions: dict[tuple[int, int], list[int]] = {}
    for pos, point in enumerate(points):
        cell = (
            int(max(float(point[0].item()), 0.0) // grid),
            int(max(float(point[1].item()), 0.0) // grid),
        )
        cell_to_positions.setdefault(cell, []).append(int(pos))
    if len(cell_to_positions) <= 1:
        return list(range(len(points))), {
            "spatial_balance_final_rebalance_applied": False,
            "spatial_balance_final_rebalance_cell_count": int(len(cell_to_positions)),
        }

    ordered: list[int] = []
    max_depth = max((len(values) for values in cell_to_positions.values()), default=0)
    if max_per_cell > 0:
        max_depth = min(max_depth, max_per_cell)
    for depth in range(max_depth):
        for _, positions in sorted(cell_to_positions.items(), key=lambda item: (item[0][1], item[0][0])):
            if depth < len(positions):
                ordered.append(int(positions[depth]))
    return ordered, {
        "spatial_balance_final_rebalance_applied": True,
        "spatial_balance_final_rebalance_cell_count": int(len(cell_to_positions)),
        "spatial_balance_final_rebalance_max_depth": int(max_depth),
        "spatial_balance_final_rebalance_input_count": int(len(points)),
        "spatial_balance_final_rebalance_output_count": int(len(ordered)),
    }


def _patch_quality_metrics(patch: torch.Tensor, cfg: NATSpotFitterConfig) -> tuple[bool, dict[str, float]]:
    values = patch.to(dtype=torch.float32)
    size_y, size_x = int(values.shape[0]), int(values.shape[1])
    center_y = (size_y - 1.0) / 2.0
    center_x = (size_x - 1.0) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(size_y, dtype=torch.float32),
        torch.arange(size_x, dtype=torch.float32),
        indexing="ij",
    )
    radius = torch.sqrt((yy - center_y).square() + (xx - center_x).square())
    border = torch.cat([values[0], values[-1], values[1:-1, 0], values[1:-1, -1]])
    signal = (values - border.median()).clamp_min(0.0)
    peak_flat = int(torch.argmax(signal).item())
    peak_y = peak_flat // size_x
    peak_x = peak_flat % size_x
    peak_distance = float(((float(peak_y) - center_y) ** 2 + (float(peak_x) - center_x) ** 2) ** 0.5)
    center_mask = radius <= float(cfg.max_patch_peak_distance_px)
    outside_mask = radius > float(cfg.secondary_peak_exclusion_radius_px)
    center_peak = float(signal[center_mask].max().item()) if bool(center_mask.any().item()) else 0.0
    outside_peak = float(signal[outside_mask].max().item()) if bool(outside_mask.any().item()) else 0.0
    secondary_fraction = outside_peak / max(center_peak, 1e-6)
    signal_sum = float(signal.sum().item())
    keep = (
        peak_distance <= float(cfg.max_patch_peak_distance_px)
        and center_peak > max(0.0, float(cfg.min_center_peak_norm))
        and signal_sum > max(0.0, float(cfg.min_signal_sum_norm))
        and secondary_fraction <= float(cfg.max_secondary_peak_fraction)
    )
    quality_score = center_peak + 0.05 * signal_sum - 0.5 * outside_peak - 0.25 * peak_distance
    return keep, {
        "peak_distance_px": peak_distance,
        "center_peak": center_peak,
        "outside_peak": outside_peak,
        "secondary_fraction": float(secondary_fraction),
        "signal_sum": signal_sum,
        "quality_score": float(quality_score),
    }


def _empty_patch_dataset(cfg: NATSpotFitterConfig, selection_metrics: dict[str, Any]) -> NATPatchDataset:
    empty = torch.zeros((0,), dtype=torch.float32)
    empty_patch = torch.zeros((0, cfg.patch_size_px, cfg.patch_size_px), dtype=torch.float32)
    empty_xy = torch.zeros((0, 2), dtype=torch.float32)
    return NATPatchDataset(
        patches_norm=empty_patch,
        patches_adu=empty_patch,
        roixy_px=empty_xy,
        local_x_nm=empty,
        local_y_nm=empty,
        z_nm=empty,
        photons_init=empty,
        background_init=empty,
        background_adu=empty,
        probability=empty,
        frame_index=torch.zeros((0,), dtype=torch.int64),
        selected_indices=torch.zeros((0,), dtype=torch.long),
        selection_metrics=selection_metrics,
    )


def _subset_harvested_emitters(emitters: HarvestedEmitters, indices: torch.Tensor) -> HarvestedEmitters:
    selected = indices.to(dtype=torch.long).reshape(-1)
    return HarvestedEmitters(
        frame_index=emitters.frame_index[selected].clone(),
        probability=emitters.probability[selected].clone(),
        x_px=emitters.x_px[selected].clone(),
        y_px=emitters.y_px[selected].clone(),
        z_um=emitters.z_um[selected].clone(),
        photons=emitters.photons[selected].clone(),
        x_sigma_px=emitters.x_sigma_px[selected].clone(),
        y_sigma_px=emitters.y_sigma_px[selected].clone(),
        z_sigma_um=emitters.z_sigma_um[selected].clone(),
        photon_sigma=emitters.photon_sigma[selected].clone(),
        background_adu=emitters.background_adu[selected].clone(),
        local_x_nm=emitters.local_x_nm[selected].clone(),
        local_y_nm=emitters.local_y_nm[selected].clone(),
    )


def _select_preselected_emitters(
    emitters: HarvestedEmitters,
    cfg: NATSpotFitterConfig,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    half = cfg.patch_size_px // 2
    cx = torch.round(emitters.x_px.to(dtype=torch.float32) - 0.5).to(dtype=torch.int64)
    cy = torch.round(emitters.y_px.to(dtype=torch.float32) - 0.5).to(dtype=torch.int64)
    roi_x_min = float(cfg.roi_x_min_px) + float(cfg.roi_edge_margin_px)
    roi_x_max = float(width if cfg.roi_x_max_px is None else cfg.roi_x_max_px) - float(cfg.roi_edge_margin_px)
    roi_y_min = float(cfg.roi_y_min_px) + float(cfg.roi_edge_margin_px)
    roi_y_max = float(height if cfg.roi_y_max_px is None else cfg.roi_y_max_px) - float(cfg.roi_edge_margin_px)
    keep = (
        torch.isfinite(emitters.x_px)
        & torch.isfinite(emitters.y_px)
        & torch.isfinite(emitters.photons)
        & torch.isfinite(emitters.local_x_nm)
        & torch.isfinite(emitters.local_y_nm)
        & torch.isfinite(emitters.z_um)
        & (cx - half >= 0)
        & (cy - half >= 0)
        & (cx + half < width)
        & (cy + half < height)
        & (emitters.x_px >= roi_x_min)
        & (emitters.x_px < roi_x_max)
        & (emitters.y_px >= roi_y_min)
        & (emitters.y_px < roi_y_max)
    )
    selected = keep.nonzero(as_tuple=False).reshape(-1)
    return selected, {
        "selection_mode": "preselected",
        "emitters_preselected": True,
        "total_emitters": len(emitters),
        "base_valid_count": int(selected.numel()),
        "pool_count": int(selected.numel()),
        "isolated_count": int(selected.numel()),
        "max_emitters_after_image_quality": int(selected.numel()),
        "min_neighbor_distance_px": float(cfg.min_neighbor_distance_px),
        "global_projected_min_distance_px": float(cfg.global_projected_min_distance_px),
        "global_projected_distance_before_count": int(selected.numel()),
        "global_projected_distance_after_count": int(selected.numel()),
        "global_projected_distance_rejected_count": 0,
        "roi_x_min_px": float(cfg.roi_x_min_px),
        "roi_x_max_px": None if cfg.roi_x_max_px is None else float(cfg.roi_x_max_px),
        "roi_y_min_px": float(cfg.roi_y_min_px),
        "roi_y_max_px": None if cfg.roi_y_max_px is None else float(cfg.roi_y_max_px),
        "roi_edge_margin_px": float(cfg.roi_edge_margin_px),
    }


def _extract_patch_dataset(emitters: HarvestedEmitters, cfg: NATSpotFitterConfig) -> NATPatchDataset:
    full_stack: np.ndarray | None = None
    try:
        full_stack = _canonicalize_stack_shape(np.asarray(tifffile.memmap(cfg.tiff_path)))
        height = int(full_stack.shape[1])
        width = int(full_stack.shape[2])
        frame_lookup: dict[int, np.ndarray] | None = None
    except ValueError:
        with tifffile.TiffFile(cfg.tiff_path) as tif:
            if not tif.series:
                raise ValueError(f"TIFF has no image series: {cfg.tiff_path}")
            shape = tif.series[0].shape
            canonical_shape = _canonicalize_stack_dims(tuple(int(dim) for dim in shape))
            height = int(canonical_shape[1])
            width = int(canonical_shape[2])
        frame_lookup = None
    if bool(cfg.emitters_preselected):
        selected, selection_metrics = _select_preselected_emitters(emitters, cfg, height=height, width=width)
    else:
        selected, selection_metrics = _select_emitters_for_fit(emitters, cfg, height, width)
    if selected.numel() == 0:
        return _empty_patch_dataset(cfg, selection_metrics)

    half = cfg.patch_size_px // 2
    if full_stack is None:
        frame_lookup = _read_selected_tiff_frames(cfg.tiff_path, emitters.frame_index[selected])
    patches: list[torch.Tensor] = []
    patches_adu: list[torch.Tensor] = []
    roixy: list[torch.Tensor] = []
    background_init: list[float] = []
    background_adu: list[float] = []
    kept_indices: list[int] = []
    peak_distances: list[float] = []
    secondary_fractions: list[float] = []
    center_peaks: list[float] = []
    signal_sums: list[float] = []
    quality_scores: list[float] = []
    for idx in selected.tolist():
        frame_ix = int(emitters.frame_index[idx].item())
        cx = int(round(float(emitters.x_px[idx].item() - 0.5)))
        cy = int(round(float(emitters.y_px[idx].item() - 0.5)))
        frame = (
            np.asarray(full_stack[frame_ix], dtype=np.float32)
            if full_stack is not None
            else frame_lookup[frame_ix]
        )
        patch = frame[cy - half : cy + half + 1, cx - half : cx + half + 1].astype(np.float32, copy=False)
        patch_t = torch.from_numpy(np.array(patch, copy=True))
        patch_norm = (patch_t - float(cfg.input_offset)) / float(cfg.input_scale)
        if str(cfg.patch_adu_mode) == "raw":
            patch_adu = patch_t
        elif str(cfg.patch_adu_mode) == "border_subtract":
            border = torch.cat([patch_t[0], patch_t[-1], patch_t[1:-1, 0], patch_t[1:-1, -1]])
            patch_adu = patch_t - border.median()
        else:
            patch_adu = (patch_t - float(cfg.baseline_adu)).clamp_min(0.0)
        if bool(cfg.emitters_preselected):
            keep_patch = True
            patch_metrics = {
                "peak_distance_px": 0.0,
                "secondary_fraction": 0.0,
                "center_peak": 0.0,
                "signal_sum": 0.0,
                "quality_score": 0.0,
            }
        else:
            keep_patch, patch_metrics = _patch_quality_metrics(patch_norm, cfg)
            if not keep_patch:
                continue
        patches.append(patch_norm)
        patches_adu.append(patch_adu)
        roixy.append(
            torch.tensor(
                [float(cx) - float(cfg.roi_x_min_px), float(cy) - float(cfg.roi_y_min_px)],
                dtype=torch.float32,
            )
        )
        bg_adu = float(emitters.background_adu[idx].item())
        bg_init = (bg_adu - float(cfg.input_offset)) / float(cfg.input_scale)
        background_init.append(bg_init)
        background_adu.append(bg_adu)
        kept_indices.append(int(idx))
        peak_distances.append(float(patch_metrics["peak_distance_px"]))
        secondary_fractions.append(float(patch_metrics["secondary_fraction"]))
        center_peaks.append(float(patch_metrics["center_peak"]))
        signal_sums.append(float(patch_metrics["signal_sum"]))
        quality_scores.append(float(patch_metrics["quality_score"]))

    selection_metrics = {
        **selection_metrics,
        "image_quality_count": len(kept_indices),
        "rejected_by_image_quality": int(selected.numel()) - len(kept_indices),
        "emitters_preselected": bool(cfg.emitters_preselected),
        "max_patch_peak_distance_px": float(cfg.max_patch_peak_distance_px),
        "secondary_peak_exclusion_radius_px": float(cfg.secondary_peak_exclusion_radius_px),
        "max_secondary_peak_fraction": float(cfg.max_secondary_peak_fraction),
        "rank_spots_by_quality": bool(cfg.rank_spots_by_quality),
        "min_center_peak_norm": float(cfg.min_center_peak_norm),
        "min_signal_sum_norm": float(cfg.min_signal_sum_norm),
        "peak_distance_px_mean": None if not peak_distances else float(np.mean(peak_distances)),
        "secondary_fraction_mean": None if not secondary_fractions else float(np.mean(secondary_fractions)),
        "center_peak_norm_mean": None if not center_peaks else float(np.mean(center_peaks)),
        "signal_sum_norm_mean": None if not signal_sums else float(np.mean(signal_sums)),
        "quality_score_mean": None if not quality_scores else float(np.mean(quality_scores)),
    }
    if not kept_indices:
        return _empty_patch_dataset(cfg, selection_metrics)

    kept_total = len(kept_indices)
    if bool(cfg.rank_spots_by_quality) and kept_total > 0 and not bool(cfg.emitters_preselected):
        order = sorted(range(kept_total), key=lambda i: quality_scores[i], reverse=True)
        kept_indices = [kept_indices[i] for i in order]
        patches = [patches[i] for i in order]
        patches_adu = [patches_adu[i] for i in order]
        roixy = [roixy[i] for i in order]
        background_init = [background_init[i] for i in order]
        background_adu = [background_adu[i] for i in order]
        peak_distances = [peak_distances[i] for i in order]
        secondary_fractions = [secondary_fractions[i] for i in order]
        center_peaks = [center_peaks[i] for i in order]
        signal_sums = [signal_sums[i] for i in order]
        quality_scores = [quality_scores[i] for i in order]

    final_rebalance_metrics: dict[str, Any] = {"spatial_balance_final_rebalance_applied": False}
    if str(cfg.selection_mode).lower() == "spatial_balanced" and kept_total > 0 and not bool(cfg.emitters_preselected):
        order, final_rebalance_metrics = _spatial_balance_final_order(roixy, cfg)
        kept_indices = [kept_indices[i] for i in order]
        patches = [patches[i] for i in order]
        patches_adu = [patches_adu[i] for i in order]
        roixy = [roixy[i] for i in order]
        background_init = [background_init[i] for i in order]
        background_adu = [background_adu[i] for i in order]
        peak_distances = [peak_distances[i] for i in order]
        secondary_fractions = [secondary_fractions[i] for i in order]
        center_peaks = [center_peaks[i] for i in order]
        signal_sums = [signal_sums[i] for i in order]
        quality_scores = [quality_scores[i] for i in order]
        kept_total = len(kept_indices)
    keep_limit = kept_total if bool(cfg.emitters_preselected) else min(kept_total, int(cfg.max_emitters))
    kept_indices = kept_indices[:keep_limit]
    patches = patches[:keep_limit]
    patches_adu = patches_adu[:keep_limit]
    roixy = roixy[:keep_limit]
    background_init = background_init[:keep_limit]
    background_adu = background_adu[:keep_limit]
    selected_kept = torch.tensor(kept_indices, dtype=torch.long)
    spatial_balance_metrics: dict[str, Any] = {
        "spatial_balance_selected_counts": None,
        "spatial_balance_min_selected_count": None,
        "spatial_balance_underfilled_cell_count": None,
    }
    if str(cfg.selection_mode).lower() == "spatial_balanced" and roixy:
        spatial_balance_metrics = _spatial_balance_final_metrics(roixy[:keep_limit], cfg, height=height, width=width)
    selection_metrics = {
        **selection_metrics,
        "image_quality_count_before_final_cap": int(kept_total),
        "final_selected_count": int(keep_limit),
        "final_center_peak_norm_mean": None if not center_peaks[:keep_limit] else float(np.mean(center_peaks[:keep_limit])),
        "final_signal_sum_norm_mean": None if not signal_sums[:keep_limit] else float(np.mean(signal_sums[:keep_limit])),
        "final_secondary_fraction_mean": None if not secondary_fractions[:keep_limit] else float(np.mean(secondary_fractions[:keep_limit])),
        "final_quality_score_mean": None if not quality_scores[:keep_limit] else float(np.mean(quality_scores[:keep_limit])),
        **final_rebalance_metrics,
        **spatial_balance_metrics,
    }

    return NATPatchDataset(
        patches_norm=torch.stack(patches, dim=0).to(dtype=torch.float32),
        patches_adu=torch.stack(patches_adu, dim=0).to(dtype=torch.float32),
        roixy_px=torch.stack(roixy, dim=0).to(dtype=torch.float32),
        local_x_nm=emitters.local_x_nm[selected_kept].to(dtype=torch.float32),
        local_y_nm=emitters.local_y_nm[selected_kept].to(dtype=torch.float32),
        z_nm=(emitters.z_um[selected_kept] * 1000.0).to(dtype=torch.float32),
        photons_init=emitters.photons[selected_kept].clamp_min(1.0).to(dtype=torch.float32),
        background_init=torch.tensor(background_init, dtype=torch.float32),
        background_adu=torch.tensor(background_adu, dtype=torch.float32),
        probability=emitters.probability[selected_kept].to(dtype=torch.float32),
        frame_index=emitters.frame_index[selected_kept].to(dtype=torch.int64),
        selected_indices=selected_kept,
        selection_metrics=selection_metrics,
    )


def build_patch_dataset_from_emitters(emitters: HarvestedEmitters, cfg: NATSpotFitterConfig) -> NATPatchDataset:
    return _extract_patch_dataset(emitters, cfg)


def materialize_selected_harvested_emitters(
    emitters: HarvestedEmitters,
    cfg: NATSpotFitterConfig,
) -> tuple[HarvestedEmitters, dict[str, Any]]:
    dataset = _extract_patch_dataset(emitters, cfg)
    selected_emitters = _subset_harvested_emitters(emitters, dataset.selected_indices)
    metrics = {
        "raw_emitters": int(len(emitters)),
        "selected_emitters": int(len(selected_emitters)),
        "selected_indices_count": int(dataset.selected_indices.numel()),
        **dataset.selection_metrics,
    }
    return selected_emitters, metrics


def _build_fourier_psf(cfg: NATSpotFitterConfig):
    return _NeptunePSFAdapter(cfg)


def fit_nat_from_harvested_emitters(
    emitters: HarvestedEmitters,
    cfg: NATSpotFitterConfig,
) -> NATSpotFitResult:
    dataset = _extract_patch_dataset(emitters, cfg)
    if dataset.patches_norm.shape[0] < 3:
        gamma = torch.zeros((len(cfg.nat_config.gammas),), dtype=torch.float32)
        return NATSpotFitResult(
            gamma=gamma,
            coefficients_nm=torch.zeros((0, len(cfg.nat_config.aberrations)), dtype=torch.float32),
            roixy_px=dataset.roixy_px,
            local_x_nm=dataset.local_x_nm,
            local_y_nm=dataset.local_y_nm,
            z_nm=dataset.z_nm,
            photons=dataset.photons_init,
            background_norm=dataset.background_init,
            background_adu=dataset.background_adu,
            loss_history=tuple(),
            metrics={
                "status": "insufficient_emitters",
                "selected_emitters": int(dataset.patches_norm.shape[0]),
                "required_emitters": 3,
            },
        )

    psf = _build_fourier_psf(cfg)
    device = torch.device(cfg.device)
    obs = dataset.patches_norm.to(device=device, dtype=torch.float32)
    fixed_coefficients_nm, fixed_mode_mask = _fixed_coefficients_and_mask(cfg, device=device)
    if cfg.base_coefficients_nm is None:
        coeffs_init = torch.zeros((obs.shape[0], len(cfg.nat_config.aberrations)), device=device, dtype=torch.float32)
    else:
        base = torch.as_tensor(cfg.base_coefficients_nm, dtype=torch.float32, device=device).reshape(1, -1)
        if base.shape[1] != len(cfg.nat_config.aberrations):
            raise ValueError("base_coefficients_nm width must match nat_config.aberrations.")
        coeffs_init = base.expand(obs.shape[0], -1).clone()
    _apply_fixed_coefficients_(coeffs_init, fixed_coefficients_nm, fixed_mode_mask)
    coeffs_nm = torch.nn.Parameter(coeffs_init)
    log_photons = torch.log(dataset.photons_init.to(device=device, dtype=torch.float32).clamp_min(1.0))
    log_photons = torch.nn.Parameter(log_photons)
    background = torch.nn.Parameter(dataset.background_init.to(device=device, dtype=torch.float32).clone())
    optimizer = torch.optim.Adam([coeffs_nm, log_photons, background], lr=float(cfg.fit_lr))

    local_x_nm = dataset.local_x_nm.to(device=device, dtype=torch.float32)
    local_y_nm = dataset.local_y_nm.to(device=device, dtype=torch.float32)
    z_nm = dataset.z_nm.to(device=device, dtype=torch.float32).clamp(float(cfg.z_min_nm), float(cfg.z_max_nm))
    loss_history: list[float] = []

    for _ in range(max(1, int(cfg.fit_steps))):
        _apply_fixed_coefficients_(coeffs_nm.data, fixed_coefficients_nm, fixed_mode_mask)
        photons = torch.exp(log_photons)
        signal = psf.simulate(local_x_nm, local_y_nm, z_nm, photons, zernike_coefs=coeffs_nm)
        expected = float(cfg.emit_scale) * signal + background[:, None, None]
        recon = torch.mean((expected - obs) ** 2)
        reg = float(cfg.coeff_l2) * torch.mean(coeffs_nm.square())
        reg = reg + float(cfg.photon_l2) * torch.mean((photons - dataset.photons_init.to(device=device)) ** 2)
        reg = reg + float(cfg.background_l2) * torch.mean((background - dataset.background_init.to(device=device)) ** 2)
        loss = recon + reg
        optimizer.zero_grad()
        loss.backward()
        if fixed_mode_mask is not None and coeffs_nm.grad is not None:
            coeffs_nm.grad[:, fixed_mode_mask] = 0.0
        optimizer.step()
        _apply_fixed_coefficients_(coeffs_nm.data, fixed_coefficients_nm, fixed_mode_mask)
        loss_history.append(float(loss.detach().cpu().item()))

    coeffs_fit = coeffs_nm.detach().cpu()
    coeffs_for_gamma = _subtract_fixed_coefficients(
        coeffs_fit,
        None if fixed_coefficients_nm is None else fixed_coefficients_nm.detach().cpu(),
        None if fixed_mode_mask is None else fixed_mode_mask.detach().cpu(),
    )
    gamma = invert_nat_order1_torch(
        *get_fov_coordinates_torch(
            dataset.roixy_px,
            img_size_x=cfg.nat_config.img_size_x,
            img_size_y=cfg.nat_config.img_size_y,
            pixel_size_x_nm=cfg.nat_config.pixel_size_x_nm,
            pixel_size_y_nm=cfg.nat_config.pixel_size_y_nm,
            local_x_nm=dataset.local_x_nm,
            local_y_nm=dataset.local_y_nm,
        )[0].T,
        coeffs_for_gamma,
        config=cfg.nat_config,
        ridge=float(cfg.ridge),
        dtype=torch.float32,
        device="cpu",
    ).detach().cpu()

    metrics = {
        "status": "ok",
        "selected_emitters": int(dataset.patches_norm.shape[0]),
        "fit_steps": int(cfg.fit_steps),
        "patch_size_px": int(cfg.patch_size_px),
        "loss_initial": float(loss_history[0]),
        "loss_final": float(loss_history[-1]),
        "coeff_norm_mean": float(torch.linalg.norm(coeffs_fit, dim=1).mean().item()),
        "gamma_norm": float(torch.linalg.norm(gamma).item()),
        "fixed_mode_count": 0 if fixed_mode_mask is None else int(fixed_mode_mask.sum().item()),
        "fixed_coefficients_nm": [] if fixed_coefficients_nm is None else [float(v) for v in fixed_coefficients_nm.detach().cpu().tolist()],
        "alternating_min_photon_fraction": float(cfg.alternating_min_photon_fraction),
        "alternating_min_background_fraction": float(cfg.alternating_min_background_fraction),
        **{f"selection_{key}": value for key, value in dataset.selection_metrics.items()},
    }
    return NATSpotFitResult(
        gamma=gamma,
        coefficients_nm=coeffs_fit,
        roixy_px=dataset.roixy_px.detach().cpu(),
        local_x_nm=dataset.local_x_nm.detach().cpu(),
        local_y_nm=dataset.local_y_nm.detach().cpu(),
        z_nm=dataset.z_nm.detach().cpu(),
        photons=torch.exp(log_photons.detach()).cpu(),
        background_norm=background.detach().cpu(),
        background_adu=(background.detach().cpu() * float(cfg.input_scale) + float(cfg.input_offset)),
        loss_history=tuple(loss_history),
        metrics=metrics,
    )


def fit_nat_alternating_from_harvested_emitters(
    emitters: HarvestedEmitters,
    cfg: NATSpotFitterConfig,
    *,
    gamma_init: torch.Tensor | None = None,
    approximate_init: NATSpotFitResult | None = None,
    rounds: int = 4,
    local_steps: int = 20,
    global_steps: int = 20,
    local_warmup_rounds: int = 0,
    local_warmup_steps: int = 0,
    local_lm_chunk_size: int | None = None,
    local_lr: float = 1e-2,
    local_z_lr_multiplier: float = 1.0,
    gamma_lr: float = 2e-3,
    gamma_l2: float = 0.0,
    progress_callback: Callable[[dict[str, int | float | str]], None] | None = None,
) -> NATSpotFitResult:
    dataset = _extract_patch_dataset(emitters, cfg)
    if dataset.patches_norm.shape[0] < 3:
        gamma = torch.zeros((len(cfg.nat_config.gammas),), dtype=torch.float32)
        return NATSpotFitResult(
            gamma=gamma,
            coefficients_nm=torch.zeros((0, len(cfg.nat_config.aberrations)), dtype=torch.float32),
            roixy_px=dataset.roixy_px,
            local_x_nm=dataset.local_x_nm,
            local_y_nm=dataset.local_y_nm,
            z_nm=dataset.z_nm,
            photons=dataset.photons_init,
            background_norm=dataset.background_init,
            background_adu=dataset.background_adu,
            loss_history=tuple(),
            metrics={
                "status": "insufficient_emitters",
                "method": "alternating",
                "selected_emitters": int(dataset.patches_norm.shape[0]),
                "required_emitters": 3,
            },
        )

    psf = _build_fourier_psf(cfg)
    device = torch.device(cfg.device)
    fixed_coefficients_nm, fixed_mode_mask = _fixed_coefficients_and_mask(cfg, device=device)
    gamma_train_mask = _gamma_train_mask_from_fixed_modes(cfg.nat_config, fixed_mode_mask)
    gamma_fixed_values = None if gamma_train_mask is None else torch.zeros((len(cfg.nat_config.gammas),), dtype=torch.float32, device=device)

    def renderer(
        x_nm: torch.Tensor,
        y_nm: torch.Tensor,
        z_nm: torch.Tensor,
        photons: torch.Tensor,
        zernike_coefs_nm: torch.Tensor,
    ) -> torch.Tensor:
        zernike_total = (
            zernike_coefs_nm.to(device=device, dtype=torch.float32)
            if fixed_coefficients_nm is None
            else zernike_coefs_nm.to(device=device, dtype=torch.float32) + fixed_coefficients_nm.reshape(1, -1)
        )
        signal = psf.simulate(
            x_nm.to(device=device, dtype=torch.float32),
            y_nm.to(device=device, dtype=torch.float32),
            z_nm.to(device=device, dtype=torch.float32),
            photons.to(device=device, dtype=torch.float32),
            zernike_coefs=zernike_total,
        )
        return float(cfg.emit_scale_adu) * signal

    warm_start = approximate_init
    if warm_start is not None:
        warm_roixy = warm_start.roixy_px.to(dtype=torch.float32)
        dataset_roixy = dataset.roixy_px.to(dtype=torch.float32)
        if warm_roixy.shape != dataset_roixy.shape or not torch.allclose(warm_roixy, dataset_roixy, atol=1e-3, rtol=0.0):
            warm_start = None

    gamma_seed = gamma_init
    if gamma_seed is None and warm_start is not None:
        gamma_seed = warm_start.gamma
    initial_gamma = (
        torch.zeros(len(cfg.nat_config.gammas), dtype=torch.float32, device=device)
        if gamma_seed is None
        else torch.as_tensor(gamma_seed, dtype=torch.float32, device=device).reshape(-1).detach().clone()
    )
    if initial_gamma.shape[0] != len(cfg.nat_config.gammas):
        raise ValueError("gamma_init length must match nat_config.gammas.")
    gamma_positive_index = None
    gamma_symmetry_vector = None
    vectorfit_anchor_nm = None
    vectorfit_anchor_mode = _normalize_vectorfit_astig_anchor_mode(cfg.vectorfit_astig_anchor_mode)
    if bool(cfg.vectorfit_astig_gauge):
        gamma_positive_index = vectorfit_astig_positive_gamma_index(cfg.nat_config)
        gamma_symmetry_vector = vectorfit_gamma_symmetry_vector(cfg.nat_config).to(device=device)
        vectorfit_anchor_nm = (
            float(cfg.vectorfit_astig_anchor_nm)
            if cfg.vectorfit_astig_anchor_nm is not None
            else float(150.0 * float(cfg.wavelength_nm) * 1e-3)
        )
        initial_gamma = initial_gamma.clone()
        anchor_index = int(gamma_positive_index)
        initial_gamma[anchor_index] = abs(float(vectorfit_anchor_nm))
        if vectorfit_anchor_mode == "hard_fixed":
            if gamma_train_mask is None:
                gamma_train_mask = torch.ones((len(cfg.nat_config.gammas),), dtype=torch.bool, device=device)
            else:
                gamma_train_mask = gamma_train_mask.to(device=device, dtype=torch.bool).reshape(-1).clone()
            if gamma_fixed_values is None:
                gamma_fixed_values = torch.zeros((len(cfg.nat_config.gammas),), dtype=torch.float32, device=device)
            else:
                gamma_fixed_values = gamma_fixed_values.to(device=device, dtype=torch.float32).reshape(-1).clone()
            gamma_train_mask[anchor_index] = False
            gamma_fixed_values[anchor_index] = abs(float(vectorfit_anchor_nm))
    if gamma_train_mask is not None and gamma_fixed_values is not None:
        initial_gamma = initial_gamma.clone()
        initial_gamma[~gamma_train_mask] = gamma_fixed_values[~gamma_train_mask]

    background_floor = initialize_local_from_spots(
        dataset.patches_adu.to(device=device, dtype=torch.float32),
        pixel_size_x_nm=float(cfg.pixel_size_x_nm),
        pixel_size_y_nm=float(cfg.pixel_size_y_nm),
    ).background.clamp_min(1e-3)
    init_local_x_nm = dataset.local_x_nm.to(device=device, dtype=torch.float32)
    init_local_y_nm = dataset.local_y_nm.to(device=device, dtype=torch.float32)
    init_photons = dataset.photons_init.to(device=device, dtype=torch.float32).clamp_min(1.0)
    if float(cfg.alternating_min_photon_fraction) > 0.0:
        init_photons = init_photons.clamp_min(
            dataset.photons_init.to(device=device, dtype=torch.float32).clamp_min(1.0)
            * float(cfg.alternating_min_photon_fraction)
        )
    init_background = background_floor
    init_z_nm = dataset.z_nm.to(device=device, dtype=torch.float32).clamp(float(cfg.z_min_nm), float(cfg.z_max_nm))
    if warm_start is not None:
        init_local_x_nm = warm_start.local_x_nm.to(device=device, dtype=torch.float32).reshape_as(init_local_x_nm)
        init_local_y_nm = warm_start.local_y_nm.to(device=device, dtype=torch.float32).reshape_as(init_local_y_nm)
        init_photons = warm_start.photons.to(device=device, dtype=torch.float32).reshape_as(init_photons).clamp_min(1.0)
        init_background = warm_start.background_adu.to(device=device, dtype=torch.float32).reshape_as(background_floor).clamp_min(1e-3)
        if not bool(cfg.vectorfit_phasor_z_init):
            init_z_nm = warm_start.z_nm.to(device=device, dtype=torch.float32).reshape_as(init_z_nm).clamp(
                float(cfg.z_min_nm),
                float(cfg.z_max_nm),
            )
    if float(cfg.alternating_min_background_fraction) > 0.0:
        init_background = torch.maximum(
            init_background,
            dataset.background_adu.to(device=device, dtype=torch.float32).clamp_min(1e-3)
            * float(cfg.alternating_min_background_fraction),
        )
    used_phasor_z_init = False
    if bool(cfg.vectorfit_phasor_z_init):
        init_coeffs_nm, _, _ = evaluate_zernike_from_roi_positions_torch(
            dataset.roixy_px.to(device=device, dtype=torch.float32),
            initial_gamma.to(device=device, dtype=torch.float32),
            cfg.nat_config,
            local_x_nm=init_local_x_nm,
            local_y_nm=init_local_y_nm,
            dtype=torch.float32,
            device=device,
        )
        if fixed_coefficients_nm is not None:
            init_coeffs_nm = init_coeffs_nm + fixed_coefficients_nm.reshape(1, -1)
        init_z_nm = astig_phasor_z_initial_nm(
            dataset.patches_adu.to(device=device, dtype=torch.float32),
            local_x_nm=init_local_x_nm,
            local_y_nm=init_local_y_nm,
            zernike_coefficients_nm=init_coeffs_nm,
            pixel_size_x_nm=float(cfg.pixel_size_x_nm),
            pixel_size_y_nm=float(cfg.pixel_size_y_nm),
            wavelength_nm=float(cfg.wavelength_nm),
            NA=float(cfg.NA),
            refmed=float(cfg.refmed),
            z_min_nm=float(cfg.z_min_nm),
            z_max_nm=float(cfg.z_max_nm),
        ).to(device=device, dtype=torch.float32)
        used_phasor_z_init = True
    local_init_counts = NATLocalInit(
        x_nm=init_local_x_nm,
        y_nm=init_local_y_nm,
        z_nm=init_z_nm,
        photons=init_photons,
        background=init_background.clamp_min(1e-3),
    )

    snapshot = fit_nat_alternating(
        spots=dataset.patches_adu.to(device=device, dtype=torch.float32),
        roixy_px=dataset.roixy_px.to(device=device, dtype=torch.float32),
        renderer=renderer,
        config=cfg.nat_config,
        gamma_init=initial_gamma,
        local_init=local_init_counts,
        rounds=int(rounds),
        local_steps=int(local_steps),
        global_steps=int(global_steps),
        local_warmup_rounds=int(local_warmup_rounds),
        local_warmup_steps=int(local_warmup_steps),
        optimizer_kind=str(cfg.alternating_optimizer_kind),
        local_lr=float(local_lr),
        local_z_lr_multiplier=float(local_z_lr_multiplier),
        gamma_lr=float(gamma_lr),
        gamma_l2=float(gamma_l2),
        gamma_train_mask=gamma_train_mask,
        gamma_fixed_values=gamma_fixed_values,
        gamma_positive_index=gamma_positive_index,
        gamma_symmetry_vector=gamma_symmetry_vector,
        z_min_nm=float(cfg.z_min_nm),
        z_max_nm=float(cfg.z_max_nm),
        x_bound_nm=float(cfg.patch_size_px) * float(cfg.pixel_size_x_nm) / 2.0,
        y_bound_nm=float(cfg.patch_size_px) * float(cfg.pixel_size_y_nm) / 2.0,
        local_lm_chunk_size=local_lm_chunk_size,
        min_photons=(
            dataset.photons_init.to(device=device, dtype=torch.float32).clamp_min(1.0)
            * float(cfg.alternating_min_photon_fraction)
            if float(cfg.alternating_min_photon_fraction) > 0.0
            else None
        ),
        min_background=(
            dataset.background_adu.to(device=device, dtype=torch.float32).clamp_min(1e-3)
            * float(cfg.alternating_min_background_fraction)
            if float(cfg.alternating_min_background_fraction) > 0.0
            else None
        ),
        progress_callback=progress_callback,
    )

    gamma = snapshot.gamma.detach().cpu().to(dtype=torch.float32)
    coeffs_residual_nm, _, _ = evaluate_zernike_from_roi_positions_torch(
        dataset.roixy_px,
        gamma,
        cfg.nat_config,
        local_x_nm=snapshot.x_nm.detach().cpu(),
        local_y_nm=snapshot.y_nm.detach().cpu(),
    )
    coeffs_nm = _apply_fixed_coefficients_(
        coeffs_residual_nm.detach().cpu().clone(),
        None if fixed_coefficients_nm is None else fixed_coefficients_nm.detach().cpu(),
        None if fixed_mode_mask is None else fixed_mode_mask.detach().cpu(),
    )
    loss_history = tuple(float(value) for value in snapshot.loss_history)
    metrics = {
        "status": "ok",
        "method": "alternating",
        "selected_emitters": int(dataset.patches_norm.shape[0]),
        "rounds": int(rounds),
        "local_steps": int(local_steps),
        "global_steps": int(global_steps),
        "patch_size_px": int(cfg.patch_size_px),
        "loss_initial": None if not loss_history else float(loss_history[0]),
        "loss_final": None if not loss_history else float(loss_history[-1]),
        "coeff_norm_mean": float(torch.linalg.norm(coeffs_nm.detach().cpu(), dim=1).mean().item()),
        "gamma_norm": float(torch.linalg.norm(gamma).item()),
        "fixed_mode_count": 0 if fixed_mode_mask is None else int(fixed_mode_mask.sum().item()),
        "fixed_coefficients_nm": [] if fixed_coefficients_nm is None else [float(v) for v in fixed_coefficients_nm.detach().cpu().tolist()],
        "vectorfit_astig_gauge": bool(cfg.vectorfit_astig_gauge),
        "vectorfit_astig_positive_gamma_index": None if gamma_positive_index is None else int(gamma_positive_index),
        "vectorfit_astig_anchor_nm": vectorfit_anchor_nm,
        "vectorfit_astig_anchor_mode": vectorfit_anchor_mode,
        "vectorfit_phasor_z_init": bool(used_phasor_z_init),
        "warm_start_approximate_used": bool(warm_start is not None),
        "alternating_optimizer_kind": str(cfg.alternating_optimizer_kind),
        "local_warmup_rounds": int(local_warmup_rounds),
        "local_warmup_steps": int(local_warmup_steps) if int(local_warmup_steps) > 0 else int(local_steps),
        "local_z_lr_multiplier": float(local_z_lr_multiplier),
        "z_min_nm": float(cfg.z_min_nm),
        "z_max_nm": float(cfg.z_max_nm),
        "alternating_min_photon_fraction": float(cfg.alternating_min_photon_fraction),
        "alternating_min_background_fraction": float(cfg.alternating_min_background_fraction),
        **{f"selection_{key}": value for key, value in dataset.selection_metrics.items()},
    }
    return NATSpotFitResult(
        gamma=gamma,
        coefficients_nm=coeffs_nm.detach().cpu().to(dtype=torch.float32),
        roixy_px=dataset.roixy_px.detach().cpu(),
        local_x_nm=snapshot.x_nm.detach().cpu().to(dtype=torch.float32),
        local_y_nm=snapshot.y_nm.detach().cpu().to(dtype=torch.float32),
        z_nm=snapshot.z_nm.detach().cpu().to(dtype=torch.float32),
        photons=snapshot.photons.detach().cpu().to(dtype=torch.float32),
        background_norm=((snapshot.background.detach().cpu() - float(cfg.input_offset)) / float(cfg.input_scale)).to(dtype=torch.float32),
        background_adu=snapshot.background.detach().cpu().to(dtype=torch.float32),
        loss_history=loss_history,
        metrics=metrics,
    )


__all__ = [
    "NATPatchDataset",
    "NATSpotFitResult",
    "NATSpotFitterConfig",
    "nat_aberration_noll_indices",
    "build_nat_spot_fitter_config",
    "initial_defocus_zero_coefficients_nm",
    "initial_defocus_zero_mode_mask",
    "initial_astig_standard_coefficients_nm",
    "initial_astig_standard_mode_mask",
    "vectorfit_astig_gamma_init",
    "vectorfit_astig_positive_gamma_index",
    "vectorfit_gamma_symmetry_vector",
    "astig_phasor_z_initial_nm",
    "build_patch_dataset_from_emitters",
    "materialize_selected_harvested_emitters",
    "fit_nat_alternating_from_harvested_emitters",
    "fit_nat_from_harvested_emitters",
]
