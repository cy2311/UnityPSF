"""Online provider configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import torch


@dataclass(frozen=True)
class OnlineBatchProviderConfig:
    batch_size: int
    channels: int = 3
    height: int = 128
    width: int = 128
    emitters_per_sample: int = 8
    seed: int = 0
    steps_per_epoch: int = 1
    background: float = 0.0
    signal: float = 1.0
    simulation_backend: str = "cuda_gemm"
    psf_type: str = "vector"
    pixel_size_nm_x: float = 101.11
    pixel_size_nm_y: float = 98.83
    wavelength_nm: float = 660.0
    na: float = 1.4
    refmed: float = 1.518
    refcov: float = 1.518
    refimm: float = 1.518
    objstage0: float = 0.0
    otf_rescale_xy: tuple[float, float] = (0.0, 0.0)
    npupil: int = 128
    vector_psf_size: int = 51
    vector_batch_size: int = 96
    zemit0: float | None = None
    emitter_density_um2: float | None = None
    lifetime_avg: float = 1.0
    warmup_frames: float = 6.0
    photon_range: tuple[float, float] | None = None
    photon_mean: float | None = None
    photon_sigma: float | None = None
    background_range: tuple[float, float] | None = None
    background_scale: float = 1.0
    z_range: tuple[float, float] | None = None
    conditioning_mode: str = "channels"
    nat_simulation_mode: str = "tile_center"
    nat_grid_size: int | tuple[int, int] = 32
    nat_grid_z_steps: int = 41
    append_domain_onehot: bool = False
    condition_feature_dim: int = 0
    condition_dim: int = 0
    condition_fields: tuple[str, ...] = ()
    conditioning_profile: str = "circular_2d_50nm"
    domain_count: int = 2
    domain_balance_mode: str = "fixed"
    dual_domain_coeff_maps: tuple[Mapping[str, str], ...] = ()
    fixed_astig_anchor_nm: float | None = None
    circular_psf_radius_nm: float | None = None
    pupil_carrier_complex: torch.Tensor | None = None
    batch_strategy: str = "triplet"
    sequence_window_chunks: int = 1
    sequence_count: int = 64
    camera_qe: float = 0.9
    camera_spurious_charge: float = 0.002
    camera_em_gain: float = 1.0
    camera_baseline: float = 398.6
    camera_e_per_adu: float = 1.020784562122306
    camera_readout: bool = False
    pxyz_target_order: str = "legacy_iwae"
    photon_scale: float | None = None
    z_scale: float | None = None
    z_scale_unit: str | None = None
    simulation_output_device: str = "cpu"
    cached_window_order: str = "auto"
    cached_window_max_gpu_sequences: int = 0
    cached_window_cache_frames: int = 0
    cached_window_cache_storage: str = "cpu"
    cached_window_prefetch_batches: int = 1
    ppg_map_path: str | None = None
    field_origin_sampling_mode: str = "grid"
    field_origin_stride_px: int = 40
    empirical_psf_path: str | None = None
    empirical_psf_channel: str | None = None
    empirical_psf_focus_index: int | None = None
    camera_read_sigma: float = 0.0
    unified_normalization_enabled: bool = False
    train_background_adu: float | None = None
    normalization_background_percentile: float = 50.0
