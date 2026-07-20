from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from neptune_v03.localization.smlm_targets import V03_PXYZ_TARGET_ORDER, V03_PXYZ_TARGET_UNITS
from neptune_v03.localization.training_adapter import LocalizationTrainBatch
from neptune_v03.optics.vector_psf import VectorPSFParams, build_vector_psf_context, noll_to_nm, render_vector_psf_bank
from neptune_v03.runtime.profiling import time_block


@dataclass(frozen=True)
class LocalizationSimulatorConfig:
    batch_size: int
    frames_per_sample: int = 3
    height: int = 128
    width: int = 128
    emitters_per_sample: int = 8
    seed: int = 0
    photons: float = 1.0
    background: float = 0.0
    psf_type: str = "vector"
    pixel_size_nm_x: float = 101.11
    pixel_size_nm_y: float = 98.83
    wavelength_nm: float = 660.0
    na: float = 1.4
    npupil: int = 128
    vector_psf_size: int = 51
    vector_batch_size: int = 96
    refmed: float = 1.518
    refcov: float = 1.518
    refimm: float = 1.518
    objstage0: float = 0.0
    zemit0: float | None = None
    otf_rescale_xy: tuple[float, float] = (0.0, 0.0)
    photon_range: tuple[float, float] | None = None
    photon_mean: float | None = None
    photon_sigma: float | None = None
    background_range: tuple[float, float] | None = None
    background_scale: float = 1.0
    z_range: tuple[float, float] | None = None
    emitter_density_um2: float | None = None
    lifetime_avg: float = 1.0
    warmup_frames: float = 6.0
    field_origin_xy: tuple[int, int] = (0, 0)
    coeff_maps_nm: torch.Tensor | None = None
    coeff_mode_order: tuple[tuple[int, int], ...] = ()
    output_device: str = "cpu"


def simulate_localization_batch(
    config: LocalizationSimulatorConfig,
    *,
    epoch: int,
    step: int,
    source: str = "native_simulator",
) -> LocalizationTrainBatch:
    if str(config.psf_type).strip().lower() != "vector":
        raise ValueError("localization simulator requires psf_type='vector'")
    batch_size = int(config.batch_size)
    channels = int(config.frames_per_sample)
    height = int(config.height)
    width = int(config.width)
    max_emitters = _max_emitters_per_sample(config)
    if min(batch_size, channels, height, width, max_emitters) <= 0:
        raise ValueError("simulator dimensions and emitter count must be positive")

    generator = torch.Generator().manual_seed(int(config.seed))
    counts = _sample_counts(config, batch_size=batch_size, generator=generator)
    xs = torch.zeros((batch_size, max_emitters), dtype=torch.float32)
    ys = torch.zeros((batch_size, max_emitters), dtype=torch.float32)
    photons = torch.zeros((batch_size, max_emitters), dtype=torch.float32)
    z = torch.zeros((batch_size, max_emitters), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_emitters), dtype=torch.bool)
    for batch_idx, count in enumerate(counts.tolist()):
        count_i = int(count)
        if count_i <= 0:
            continue
        mask[batch_idx, :count_i] = True
        xs[batch_idx, :count_i] = torch.rand((count_i,), generator=generator) * float(width - 1)
        ys[batch_idx, :count_i] = torch.rand((count_i,), generator=generator) * float(height - 1)
        photons[batch_idx, :count_i] = _sample_photons(config, emitters=count_i, generator=generator)
        z[batch_idx, :count_i] = _sample_range(config.z_range, shape=(count_i,), fallback=0.0, generator=generator)
    background = _sample_range(config.background_range, shape=(batch_size,), fallback=float(config.background), generator=generator)

    vector_renderer = _build_vector_renderer(config)
    device = vector_renderer.device
    xs = xs.to(device=device)
    ys = ys.to(device=device)
    photons = photons.to(device=device)
    z = z.to(device=device)
    mask = mask.to(device=device)
    background = background.to(device=device)

    detect = torch.zeros((batch_size, height, width), dtype=torch.float32, device=device)
    model_input = background.view(batch_size, 1, 1, 1).expand(batch_size, channels, height, width).clone()
    for batch_idx in range(batch_size):
        count = int(counts[batch_idx].item())
        if count <= 0:
            continue
        rows = torch.round(ys[batch_idx, :count]).to(dtype=torch.long).clamp_(0, height - 1)
        cols = torch.round(xs[batch_idx, :count]).to(dtype=torch.long).clamp_(0, width - 1)
        detect[batch_idx, rows, cols] = 1.0
        rendered = vector_renderer.render_many(
            height=height,
            width=width,
            xs=xs[batch_idx, :count],
            ys=ys[batch_idx, :count],
            z_um=z[batch_idx, :count],
            photons=photons[batch_idx, :count],
        )
        for channel_idx in range(channels):
            model_input[batch_idx, channel_idx] += rendered

    background_target = background / max(float(config.background_scale), 1e-12)

    batch = LocalizationTrainBatch(
        model_input=model_input,
        detect_tar=detect,
        bkg_tar=background_target.view(batch_size, 1, 1).expand(batch_size, height, width).clone(),
        pxyz_tar=torch.stack([xs, ys, z, photons], dim=-1).to(dtype=torch.float32),
        mask_tar=mask,
        metadata={
            "epoch": int(epoch),
            "step": int(step),
            "seed": int(config.seed),
            "source": source,
            "pxyz_target_order": V03_PXYZ_TARGET_ORDER,
            "pxyz_target_units": V03_PXYZ_TARGET_UNITS,
            "z_range": _range_metadata(config.z_range),
            "photon_range": _range_metadata(config.photon_range),
            "background_range": _range_metadata(config.background_range),
            "background_scale": float(config.background_scale),
            "photon_mean": None if config.photon_mean is None else float(config.photon_mean),
            "photon_sigma": None if config.photon_sigma is None else float(config.photon_sigma),
            "psf_type": str(config.psf_type).lower(),
            "emitter_density_um2": None if config.emitter_density_um2 is None else float(config.emitter_density_um2),
            "target_active_emitters_per_frame": _target_active_emitters_per_frame(config),
            "emitter_counts": [int(v) for v in counts.tolist()],
            "field_origin_xy": [int(config.field_origin_xy[0]), int(config.field_origin_xy[1])],
        },
    )
    return _localization_batch_to_output_device(batch, output_device=str(config.output_device), renderer_device=device)


def _sample_photons(
    config: LocalizationSimulatorConfig,
    *,
    emitters: int,
    generator: torch.Generator,
) -> torch.Tensor:
    shape = (int(emitters),)
    if config.photon_mean is not None and config.photon_sigma is not None:
        photons = torch.normal(
            mean=float(config.photon_mean),
            std=float(config.photon_sigma),
            size=shape,
            generator=generator,
        )
        if config.photon_range is not None:
            lo, hi = _range_tuple(config.photon_range)
            photons = photons.clamp(min=lo, max=hi)
        return photons.to(dtype=torch.float32)
    return _sample_range(config.photon_range, shape=shape, fallback=float(config.photons), generator=generator)


def _target_active_emitters_per_frame(config: LocalizationSimulatorConfig) -> float | None:
    if config.emitter_density_um2 is None:
        return None
    area_um2 = (
        float(config.width)
        * float(config.pixel_size_nm_x)
        / 1000.0
        * float(config.height)
        * float(config.pixel_size_nm_y)
        / 1000.0
    )
    return float(config.emitter_density_um2) * area_um2


def _max_emitters_per_sample(config: LocalizationSimulatorConfig) -> int:
    if config.emitter_density_um2 is None:
        return int(config.emitters_per_sample)
    target_active = _target_active_emitters_per_frame(config)
    assert target_active is not None
    return max(1, int(math.ceil(target_active)))


def _sample_counts(config: LocalizationSimulatorConfig, *, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    if config.emitter_density_um2 is None:
        return torch.full((int(batch_size),), int(config.emitters_per_sample), dtype=torch.long)
    target_active = _target_active_emitters_per_frame(config)
    assert target_active is not None
    counts = torch.poisson(torch.full((int(batch_size),), float(target_active), dtype=torch.float32), generator=generator)
    return counts.to(dtype=torch.long).clamp_(min=1, max=_max_emitters_per_sample(config))


class _VectorEmitterRenderer:
    def __init__(self, config: LocalizationSimulatorConfig) -> None:
        if config.coeff_maps_nm is None:
            raise ValueError("vector simulator requires coeff_maps_nm")
        self.config = config
        self.maps_nm = config.coeff_maps_nm.to(dtype=torch.float32).contiguous()
        self.mode_order = tuple((int(n), int(m)) for n, m in config.coeff_mode_order)
        if int(self.maps_nm.shape[0]) != len(self.mode_order):
            raise ValueError("coeff_maps_nm channel count must match coeff_mode_order")
        noll_indices = _noll_indices_for_modes(self.mode_order)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.maps_nm = self.maps_nm.to(device=self.device)
        self.ctx = build_vector_psf_context(
            NA=float(config.na),
            wavelength_nm=float(config.wavelength_nm),
            pixel_size_nm_x=float(config.pixel_size_nm_x),
            pixel_size_nm_y=float(config.pixel_size_nm_y),
            noll_indices=noll_indices,
            params=VectorPSFParams(
                npupil=int(config.npupil),
                psf_size=int(config.vector_psf_size),
                refmed=float(config.refmed),
                refcov=float(config.refcov),
                refimm=float(config.refimm),
                objstage0=float(config.objstage0),
                otf_rescale_xy=tuple(float(v) for v in config.otf_rescale_xy),
                zemit0=None if config.zemit0 is None else float(config.zemit0),
                batch_size=int(config.vector_batch_size),
            ),
            device=device,
        )

    def render_many(
        self,
        *,
        height: int,
        width: int,
        xs: torch.Tensor,
        ys: torch.Tensor,
        z_um: torch.Tensor,
        photons: torch.Tensor,
        field_origin_xy: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        patches, center_x, center_y = self._render_patches(
            xs=xs,
            ys=ys,
            z_um=z_um,
            photons=photons,
            field_origin_xy=field_origin_xy,
        )
        return _place_patches(
            patches.detach(),
            height=height,
            width=width,
            center_x=center_x.detach(),
            center_y=center_y.detach(),
        )

    def render_frames(
        self,
        *,
        frame_count: int,
        height: int,
        width: int,
        frame_indices: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        z_um: torch.Tensor,
        photons: torch.Tensor,
        field_origin_xy: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        patches, center_x, center_y = self._render_patches(
            xs=xs,
            ys=ys,
            z_um=z_um,
            photons=photons,
            field_origin_xy=field_origin_xy,
        )
        return _place_patches_frames(
            patches.detach(),
            frame_count=int(frame_count),
            height=height,
            width=width,
            frame_indices=frame_indices.to(device=self.device, dtype=torch.long).reshape(-1),
            center_x=center_x.detach(),
            center_y=center_y.detach(),
        )

    def _render_patches(
        self,
        *,
        xs: torch.Tensor,
        ys: torch.Tensor,
        z_um: torch.Tensor,
        photons: torch.Tensor,
        field_origin_xy: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs_t = xs.to(device=self.device, dtype=torch.float32).reshape(-1)
        ys_t = ys.to(device=self.device, dtype=torch.float32).reshape(-1)
        z_t = z_um.to(device=self.device, dtype=torch.float32).reshape(-1)
        photons_t = photons.to(device=self.device, dtype=torch.float32).reshape(-1)
        origin = self.config.field_origin_xy if field_origin_xy is None else field_origin_xy
        cols = torch.round(xs_t + int(origin[0])).to(dtype=torch.long).clamp_(0, self.maps_nm.shape[2] - 1)
        rows = torch.round(ys_t + int(origin[1])).to(dtype=torch.long).clamp_(0, self.maps_nm.shape[1] - 1)
        coeffs_nm = self.maps_nm[:, rows, cols].transpose(0, 1).contiguous()
        coeffs_rad = coeffs_nm * (2.0 * math.pi / max(float(self.config.wavelength_nm), 1e-6)) * self.ctx.normfac[None, :]
        with time_block("render_vector_patches"):
            patches = render_vector_psf_bank(
                self.ctx,
                coeffs_rad,
                z_t * 1e-6,
                out_size=int(self.config.vector_psf_size),
                batch_size=int(self.config.vector_batch_size),
                return_torch=True,
            )
        center_x = torch.floor(xs_t).to(dtype=torch.long)
        center_y = torch.floor(ys_t).to(dtype=torch.long)
        local_x = xs_t - (center_x.to(dtype=torch.float32) + 0.5)
        local_y = ys_t - (center_y.to(dtype=torch.float32) + 0.5)
        with time_block("fourier_shift_patches"):
            patches = _fourier_shift_patches(patches, shift_x_px=local_x, shift_y_px=local_y).clamp_min(0.0)
        patches = patches / patches.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        patches = patches * photons_t[:, None, None]
        return patches, center_x, center_y


def _build_vector_renderer(config: LocalizationSimulatorConfig) -> _VectorEmitterRenderer:
    return _VectorEmitterRenderer(config)


def _localization_batch_to_output_device(
    batch: LocalizationTrainBatch,
    *,
    output_device: str,
    renderer_device: torch.device,
) -> LocalizationTrainBatch:
    device = _resolve_output_device(output_device, renderer_device=renderer_device)
    return LocalizationTrainBatch(
        model_input=_move_model_input(batch.model_input, device),
        detect_tar=batch.detect_tar.to(device=device),
        bkg_tar=batch.bkg_tar.to(device=device),
        pxyz_tar=batch.pxyz_tar.to(device=device),
        mask_tar=batch.mask_tar.to(device=device),
        metadata=batch.metadata,
    )


def _resolve_output_device(output_device: str, *, renderer_device: torch.device) -> torch.device:
    key = str(output_device or "cpu").strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key == "renderer":
        return renderer_device
    raise ValueError("output_device must be 'cpu' or 'renderer'")


def _move_model_input(model_input: torch.Tensor | tuple[torch.Tensor, ...], device: torch.device) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(model_input, tuple):
        return tuple(item.to(device=device) for item in model_input)
    return model_input.to(device=device)


def _noll_indices_for_modes(mode_order: tuple[tuple[int, int], ...]) -> list[int]:
    mode_to_noll = {noll_to_nm(index): index for index in range(1, 128)}
    return [mode_to_noll[(int(n), int(m))] for n, m in mode_order]


def _fourier_shift_patches(patches: torch.Tensor, *, shift_x_px: torch.Tensor, shift_y_px: torch.Tensor) -> torch.Tensor:
    if patches.numel() == 0:
        return patches
    shift_x = shift_x_px.to(device=patches.device, dtype=patches.dtype).reshape(-1)
    shift_y = shift_y_px.to(device=patches.device, dtype=patches.dtype).reshape(-1)
    if float(torch.max(torch.abs(torch.cat((shift_x, shift_y)))).detach().cpu().item()) < 1e-8:
        return patches
    height, width = int(patches.shape[-2]), int(patches.shape[-1])
    freq_y = torch.fft.fftfreq(height, d=1.0, device=patches.device, dtype=patches.dtype)
    freq_x = torch.fft.fftfreq(width, d=1.0, device=patches.device, dtype=patches.dtype)
    phase_arg = -2.0 * math.pi * (shift_y[:, None, None] * freq_y[None, :, None] + shift_x[:, None, None] * freq_x[None, None, :])
    phase = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    return torch.fft.ifft2(torch.fft.fft2(patches) * phase).real


def _place_patches(patches: torch.Tensor, *, height: int, width: int, center_x: torch.Tensor, center_y: torch.Tensor) -> torch.Tensor:
    frame = torch.zeros((int(height), int(width)), dtype=patches.dtype, device=patches.device)
    if patches.numel() == 0:
        return frame
    psf_size = int(patches.shape[-1])
    radius = psf_size // 2
    patch_y = torch.arange(psf_size, device=patches.device).view(1, psf_size, 1)
    patch_x = torch.arange(psf_size, device=patches.device).view(1, 1, psf_size)
    image_y = center_y.to(device=patches.device, dtype=torch.long).view(-1, 1, 1) + patch_y - radius
    image_x = center_x.to(device=patches.device, dtype=torch.long).view(-1, 1, 1) + patch_x - radius
    valid = (image_y >= 0) & (image_y < int(height)) & (image_x >= 0) & (image_x < int(width))
    image_y = image_y.expand_as(patches).clamp(0, int(height) - 1)
    image_x = image_x.expand_as(patches).clamp(0, int(width) - 1)
    flat = (image_y * int(width) + image_x)[valid]
    frame.reshape(-1).scatter_add_(0, flat.reshape(-1), patches[valid].reshape(-1))
    return frame


def _place_patches_frames(
    patches: torch.Tensor,
    *,
    frame_count: int,
    height: int,
    width: int,
    frame_indices: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
) -> torch.Tensor:
    with time_block("place_patches_frames"):
        frames = torch.zeros((int(frame_count), int(height), int(width)), dtype=patches.dtype, device=patches.device)
        if patches.numel() == 0:
            return frames
        psf_size = int(patches.shape[-1])
        radius = psf_size // 2
        patch_y = torch.arange(psf_size, device=patches.device).view(1, psf_size, 1)
        patch_x = torch.arange(psf_size, device=patches.device).view(1, 1, psf_size)
        image_y = center_y.to(device=patches.device, dtype=torch.long).view(-1, 1, 1) + patch_y - radius
        image_x = center_x.to(device=patches.device, dtype=torch.long).view(-1, 1, 1) + patch_x - radius
        valid = (image_y >= 0) & (image_y < int(height)) & (image_x >= 0) & (image_x < int(width))
        image_y = image_y.expand_as(patches).clamp(0, int(height) - 1)
        image_x = image_x.expand_as(patches).clamp(0, int(width) - 1)
        frame = frame_indices.to(device=patches.device, dtype=torch.long).view(-1, 1, 1).expand_as(patches)
        flat = (frame * int(height) * int(width) + image_y * int(width) + image_x)[valid]
        frames.reshape(-1).scatter_add_(0, flat.reshape(-1), patches[valid].reshape(-1))
        return frames


def _sample_range(
    value: tuple[float, float] | None,
    *,
    shape: tuple[int, ...],
    fallback: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if value is None:
        return torch.full(shape, float(fallback), dtype=torch.float32)
    lo, hi = _range_tuple(value)
    if hi == lo:
        return torch.full(shape, lo, dtype=torch.float32)
    return torch.rand(shape, generator=generator, dtype=torch.float32) * (hi - lo) + lo


def _range_tuple(value: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(value[0]), float(value[1])
    if hi < lo:
        raise ValueError("range max must be greater than or equal to range min")
    return lo, hi


def _range_metadata(value: tuple[float, float] | None) -> list[float] | None:
    if value is None:
        return None
    lo, hi = _range_tuple(value)
    return [lo, hi]
