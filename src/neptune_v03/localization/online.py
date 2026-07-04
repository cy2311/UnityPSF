from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
import math
import os
from typing import Mapping

import numpy as np
import torch

from neptune_v03.field_origin import build_sliding_window_origin_bank
from neptune_v03.localization.conditioning import (
    ConditioningProviderStore,
    FullResZernikeConditioning,
    _load_conditioning_providers,
    build_default_conditioning_maps,
    condition_feature_order,
)
from neptune_v03.localization.smlm_targets import LEGACY_IWAE_PXYZ_TARGET_ORDER, V03_PXYZ_TARGET_ORDER
from neptune_v03.localization.simulator import (
    LocalizationSimulatorConfig,
    _build_vector_renderer,
    _fourier_shift_patches,
    _max_emitters_per_sample,
    _place_patches_frames,
    _sample_counts,
    _sample_photons,
    _sample_range,
    simulate_localization_batch,
)
from neptune_v03.localization.training_adapter import LocalizationTrainBatch, to_training_batch
from neptune_v03.runtime.profiling import time_block
from neptune_v03.training.loop import TrainingBatch

try:  # pragma: no cover - optional CUDA runtime dependency.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional CUDA runtime dependency.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _triton_lut_bilinear_shift_kernel(inp, sx, sy, out, patch_size: tl.constexpr, block: tl.constexpr):
        patch_id = tl.program_id(0)
        offsets = tl.arange(0, block)
        total = patch_size * patch_size
        mask = offsets < total
        y = offsets // patch_size
        x = offsets - y * patch_size
        dx = tl.load(sx + patch_id)
        dy = tl.load(sy + patch_id)
        src_x = x.to(tl.float32) - dx
        src_y = y.to(tl.float32) - dy
        x0f = tl.floor(src_x)
        y0f = tl.floor(src_y)
        x0 = x0f.to(tl.int32)
        y0 = y0f.to(tl.int32)
        wx = src_x - x0f
        wy = src_y - y0f
        x1 = x0 + 1
        y1 = y0 + 1
        base = patch_id * total
        m00 = mask & (x0 >= 0) & (x0 < patch_size) & (y0 >= 0) & (y0 < patch_size)
        m01 = mask & (x1 >= 0) & (x1 < patch_size) & (y0 >= 0) & (y0 < patch_size)
        m10 = mask & (x0 >= 0) & (x0 < patch_size) & (y1 >= 0) & (y1 < patch_size)
        m11 = mask & (x1 >= 0) & (x1 < patch_size) & (y1 >= 0) & (y1 < patch_size)
        v00 = tl.load(inp + base + y0 * patch_size + x0, mask=m00, other=0.0)
        v01 = tl.load(inp + base + y0 * patch_size + x1, mask=m01, other=0.0)
        v10 = tl.load(inp + base + y1 * patch_size + x0, mask=m10, other=0.0)
        v11 = tl.load(inp + base + y1 * patch_size + x1, mask=m11, other=0.0)
        value = (1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v01 + (1.0 - wx) * wy * v10 + wx * wy * v11
        tl.store(out + base + offsets, value, mask=mask)

    @triton.jit
    def _triton_place_patches_frames_kernel(
        patches,
        frame_indices,
        center_x,
        center_y,
        out,
        patch_size: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        block: tl.constexpr,
    ):
        patch_id = tl.program_id(0)
        offsets = tl.arange(0, block)
        total = patch_size * patch_size
        mask = offsets < total
        py = offsets // patch_size
        px = offsets - py * patch_size
        radius = patch_size // 2
        frame = tl.load(frame_indices + patch_id)
        cx = tl.load(center_x + patch_id)
        cy = tl.load(center_y + patch_id)
        iy = cy + py - radius
        ix = cx + px - radius
        valid = mask & (iy >= 0) & (iy < height) & (ix >= 0) & (ix < width)
        value = tl.load(patches + patch_id * total + offsets, mask=mask, other=0.0)
        out_idx = frame * height * width + iy * width + ix
        tl.atomic_add(out + out_idx, value, sem="relaxed", mask=valid)


class _VectorRendererCache:
    def __init__(self) -> None:
        self._items: dict[tuple[object, ...], object] = {}
        self.hits = 0
        self.misses = 0
        self.generation = 0

    def get(self, config: LocalizationSimulatorConfig, *, physical_model_version: int | None = None):
        key = _renderer_cache_key(config, physical_model_version=physical_model_version)
        if key in self._items:
            self.hits += 1
            return self._items[key]
        renderer = _build_vector_renderer(config)
        self._items[key] = renderer
        self.misses += 1
        return renderer

    def clear(self) -> None:
        self._items.clear()
        self.generation += 1


@dataclass(frozen=True)
class _LUTPatchBank:
    patches: torch.Tensor
    tile_x: torch.Tensor
    tile_y: torch.Tensor
    z_bins: torch.Tensor
    subpixel_bins: torch.Tensor
    coordinate_mode: str = "local"


class _LUTPatchBankCache:
    def __init__(self, renderer_cache: _VectorRendererCache) -> None:
        self._renderer_cache = renderer_cache
        self._items: dict[tuple[object, ...], _LUTPatchBank] = {}
        self.hits = 0
        self.misses = 0
        self.generation = 0

    def get(
        self,
        config: OnlineBatchProviderConfig,
        sim_config: LocalizationSimulatorConfig,
        *,
        physical_model_version: int | None = None,
    ) -> _LUTPatchBank:
        key = _lut_patch_bank_key(config, sim_config, physical_model_version=physical_model_version)
        if key in self._items:
            self.hits += 1
            return self._items[key]
        renderer = self._renderer_cache.get(sim_config, physical_model_version=physical_model_version)
        with time_block("lut_bank_build"):
            bank = _build_lut_patch_bank(config, sim_config, renderer)
        self._items[key] = bank
        self.misses += 1
        return bank

    def clear(self) -> None:
        self._items.clear()
        self.generation += 1


def _renderer_cache_key(
    config: LocalizationSimulatorConfig,
    *,
    physical_model_version: int | None = None,
) -> tuple[object, ...]:
    maps = config.coeff_maps_nm
    maps_key = None if maps is None else (int(maps.data_ptr()), tuple(int(v) for v in maps.shape), str(maps.dtype))
    return (
        None if physical_model_version is None else int(physical_model_version),
        maps_key,
        tuple((int(n), int(m)) for n, m in config.coeff_mode_order),
        float(config.na),
        float(config.wavelength_nm),
        float(config.pixel_size_nm_x),
        float(config.pixel_size_nm_y),
        int(config.npupil),
        int(config.vector_psf_size),
        int(config.vector_batch_size),
        float(config.refmed),
        float(config.refcov),
        float(config.refimm),
        float(config.objstage0),
        tuple(float(v) for v in config.otf_rescale_xy),
    )


def _lut_patch_bank_key(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    physical_model_version: int | None = None,
) -> tuple[object, ...]:
    maps = sim_config.coeff_maps_nm
    field_origin = tuple(int(v) for v in sim_config.field_origin_xy)
    field_mode = _resolve_lut_field_mode(config)
    if field_mode == "global_field":
        if maps is None:
            raise ValueError("global-field LUT requires coeff_maps_nm")
        field_origin = ("global_field", int(maps.shape[2]), int(maps.shape[1]))
    elif maps is not None:
        field_origin = _effective_lut_field_origin(
            field_origin,
            roi_width=int(config.width),
            roi_height=int(config.height),
            map_width=int(maps.shape[2]),
            map_height=int(maps.shape[1]),
        )
    return (
        _renderer_cache_key(sim_config, physical_model_version=physical_model_version),
        int(config.height),
        int(config.width),
        field_origin,
        int(config.lut_field_stride),
        int(config.lut_z_steps),
        int(config.lut_subpixel_bins),
        field_mode,
        _resolve_lut_storage_dtype(config),
        _range_key(config.z_range),
    )


def _range_key(value: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    return float(value[0]), float(value[1])


def _effective_lut_field_origin(
    field_origin_xy: tuple[int, int],
    *,
    roi_width: int,
    roi_height: int,
    map_width: int,
    map_height: int,
) -> tuple[int, int]:
    x0, y0 = (int(v) for v in field_origin_xy)
    max_x0 = max(0, int(map_width) - int(roi_width))
    max_y0 = max(0, int(map_height) - int(roi_height))
    return (max(0, min(x0, max_x0)), max(0, min(y0, max_y0)))


def _build_lut_patch_bank(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    renderer,
) -> _LUTPatchBank:
    maps = sim_config.coeff_maps_nm
    if maps is None:
        raise ValueError("LUT patch bank requires coeff_maps_nm")
    device = renderer.device
    full_h = int(maps.shape[1])
    full_w = int(maps.shape[2])
    roi_h = int(config.height)
    roi_w = int(config.width)
    field_mode = _resolve_lut_field_mode(config)
    if field_mode == "global_field":
        origin_x = origin_y = 0
        extent_w = full_w
        extent_h = full_h
        coordinate_mode = "global"
    else:
        origin_x, origin_y = _effective_lut_field_origin(
            tuple(int(v) for v in sim_config.field_origin_xy),
            roi_width=roi_w,
            roi_height=roi_h,
            map_width=full_w,
            map_height=full_h,
        )
        if origin_x < 0 or origin_y < 0 or origin_x + roi_w > full_w or origin_y + roi_h > full_h:
            raise ValueError(
                "LUT patch bank local ROI exceeds conditioning map bounds: "
                f"origin={(origin_x, origin_y)} roi={(roi_w, roi_h)} map={(full_w, full_h)}"
            )
        extent_w = roi_w
        extent_h = roi_h
        coordinate_mode = "local"
    stride = max(1, int(config.lut_field_stride))
    tile_x = torch.arange(0, extent_w, stride, dtype=torch.float32, device=device)
    tile_y = torch.arange(0, extent_h, stride, dtype=torch.float32, device=device)
    if int(tile_x[-1].item()) != extent_w - 1:
        tile_x = torch.cat((tile_x, torch.tensor([float(extent_w - 1)], device=device)))
    if int(tile_y[-1].item()) != extent_h - 1:
        tile_y = torch.cat((tile_y, torch.tensor([float(extent_h - 1)], device=device)))
    z_lo, z_hi = config.z_range if config.z_range is not None else (-0.6, 0.6)
    z_bins = torch.linspace(float(z_lo), float(z_hi), steps=max(1, int(config.lut_z_steps)), device=device)
    sub_count = max(1, int(config.lut_subpixel_bins))
    if sub_count == 1:
        subpixel_bins = torch.zeros((1,), dtype=torch.float32, device=device)
    else:
        subpixel_bins = torch.linspace(
            -0.5 + 0.5 / float(sub_count),
            0.5 - 0.5 / float(sub_count),
            steps=sub_count,
            device=device,
        )

    shape = (
        int(tile_y.numel()),
        int(tile_x.numel()),
        int(z_bins.numel()),
        int(subpixel_bins.numel()),
        int(subpixel_bins.numel()),
        int(sim_config.vector_psf_size),
        int(sim_config.vector_psf_size),
    )
    grid_y, grid_x, grid_z, grid_sy, grid_sx = torch.meshgrid(tile_y, tile_x, z_bins, subpixel_bins, subpixel_bins, indexing="ij")
    xs = (grid_x.reshape(-1) + 0.5 + grid_sx.reshape(-1)).clamp(0.0, float(extent_w - 1))
    ys = (grid_y.reshape(-1) + 0.5 + grid_sy.reshape(-1)).clamp(0.0, float(extent_h - 1))
    z = grid_z.reshape(-1)
    storage_dtype = _lut_storage_torch_dtype(config, device=device)
    patches = torch.empty(shape, dtype=storage_dtype, device=device)
    flat_bank = patches.reshape(-1, int(sim_config.vector_psf_size), int(sim_config.vector_psf_size))
    chunk = max(1, int(sim_config.vector_batch_size))
    photons_buffer = torch.ones((chunk,), dtype=torch.float32, device=device)
    with time_block("lut_build_render_loop"):
        for start in range(0, int(xs.numel()), chunk):
            end = min(int(xs.numel()), start + chunk)
            rendered, _, _ = renderer._render_patches(
                xs=xs[start:end],
                ys=ys[start:end],
                z_um=z[start:end],
                photons=photons_buffer[: end - start],
                field_origin_xy=(origin_x, origin_y),
            )
            with time_block("lut_build_pack_store"):
                flat_bank[start:end].copy_(rendered.to(dtype=storage_dtype))
    return _LUTPatchBank(
        patches=patches.detach(),
        tile_x=tile_x,
        tile_y=tile_y,
        z_bins=z_bins,
        subpixel_bins=subpixel_bins,
        coordinate_mode=coordinate_mode,
    )


def _lookup_lut_patches(
    bank: _LUTPatchBank,
    *,
    xs: torch.Tensor,
    ys: torch.Tensor,
    z_um: torch.Tensor,
    photons: torch.Tensor,
    field_origin_xy: tuple[int, int],
) -> torch.Tensor:
    with time_block("lut_lookup"):
        local_x_input = xs.to(device=bank.patches.device, dtype=torch.float32).reshape(-1)
        local_y_input = ys.to(device=bank.patches.device, dtype=torch.float32).reshape(-1)
        if str(bank.coordinate_mode) == "global":
            origin_x, origin_y = (int(v) for v in field_origin_xy)
            local_x_input = local_x_input + float(origin_x)
            local_y_input = local_y_input + float(origin_y)
        z = z_um.to(device=bank.patches.device, dtype=torch.float32).reshape(-1)
        phot = photons.to(device=bank.patches.device, dtype=torch.float32).reshape(-1)
        tile_x_idx = torch.argmin(torch.abs(local_x_input[:, None] - bank.tile_x[None, :]), dim=1)
        tile_y_idx = torch.argmin(torch.abs(local_y_input[:, None] - bank.tile_y[None, :]), dim=1)
        z_idx = torch.argmin(torch.abs(z[:, None] - bank.z_bins[None, :]), dim=1)
        local_x = local_x_input - (torch.floor(local_x_input) + 0.5)
        local_y = local_y_input - (torch.floor(local_y_input) + 0.5)
        sub_x_idx = torch.argmin(torch.abs(local_x[:, None] - bank.subpixel_bins[None, :]), dim=1)
        sub_y_idx = torch.argmin(torch.abs(local_y[:, None] - bank.subpixel_bins[None, :]), dim=1)
        patches = bank.patches[tile_y_idx, tile_x_idx, z_idx, sub_y_idx, sub_x_idx].to(dtype=torch.float32)
        return patches * phot[:, None, None]


def _apply_lut_subpixel_shift(
    patches: torch.Tensor,
    *,
    xs: torch.Tensor,
    ys: torch.Tensor,
    subpixel_bins: torch.Tensor | None = None,
    chunk_size: int,
    backend: str | None = None,
) -> torch.Tensor:
    if patches.numel() == 0:
        return patches
    local_x = xs.to(device=patches.device, dtype=torch.float32).reshape(-1) - (
        torch.floor(xs.to(device=patches.device, dtype=torch.float32).reshape(-1)) + 0.5
    )
    local_y = ys.to(device=patches.device, dtype=torch.float32).reshape(-1) - (
        torch.floor(ys.to(device=patches.device, dtype=torch.float32).reshape(-1)) + 0.5
    )
    if subpixel_bins is not None:
        bins = subpixel_bins.to(device=patches.device, dtype=torch.float32).reshape(-1)
        local_x = local_x - bins[torch.argmin(torch.abs(local_x[:, None] - bins[None, :]), dim=1)]
        local_y = local_y - bins[torch.argmin(torch.abs(local_y[:, None] - bins[None, :]), dim=1)]
    if float(torch.max(torch.abs(torch.cat((local_x, local_y)))).detach().cpu().item()) < 1e-8:
        return patches

    shift_backend = _resolve_lut_subpixel_shift_backend(backend)
    shifted_chunks: list[torch.Tensor] = []
    chunk = max(1, int(chunk_size))
    for start in range(0, int(patches.shape[0]), chunk):
        end = min(int(patches.shape[0]), start + chunk)
        original = patches[start:end]
        original_sum = original.sum(dim=(-2, -1), keepdim=True)
        shifted = _shift_lut_patch_chunk(
            original,
            shift_x_px=local_x[start:end],
            shift_y_px=local_y[start:end],
            backend=shift_backend,
        ).clamp_min(0.0)
        shifted_sum = shifted.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        shifted_chunks.append(shifted * (original_sum / shifted_sum))
    return torch.cat(shifted_chunks, dim=0)


def _resolve_lut_subpixel_shift_backend(value: str | None = None) -> str:
    raw = str(value or os.environ.get("NEPTUNE_V03_LUT_SHIFT_BACKEND", "fourier")).strip().lower()
    aliases = {
        "fft": "fourier",
        "fourier_shift": "fourier",
        "triton": "triton_bilinear",
        "bilinear_triton": "triton_bilinear",
    }
    raw = aliases.get(raw, raw)
    if raw not in {"fourier", "triton_bilinear"}:
        raise ValueError("NEPTUNE_V03_LUT_SHIFT_BACKEND must be 'fourier' or 'triton_bilinear'")
    if raw == "triton_bilinear" and not _TRITON_AVAILABLE:
        raise RuntimeError("NEPTUNE_V03_LUT_SHIFT_BACKEND=triton_bilinear requested, but Triton is unavailable")
    return raw


def _shift_lut_patch_chunk(
    patches: torch.Tensor,
    *,
    shift_x_px: torch.Tensor,
    shift_y_px: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    if backend == "fourier":
        with time_block("fourier_shift_patches"):
            return _fourier_shift_patches(
                patches,
                shift_x_px=shift_x_px,
                shift_y_px=shift_y_px,
            )
    if backend == "triton_bilinear":
        with time_block("triton_shift_patches"):
            return _triton_bilinear_shift_patches(
                patches,
                shift_x_px=shift_x_px,
                shift_y_px=shift_y_px,
            )
    raise ValueError(f"unsupported LUT subpixel shift backend: {backend}")


def _triton_bilinear_shift_patches(
    patches: torch.Tensor,
    *,
    shift_x_px: torch.Tensor,
    shift_y_px: torch.Tensor,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable")
    if not patches.is_cuda:
        raise RuntimeError("Triton LUT subpixel shift requires CUDA patches")
    if patches.dtype != torch.float32:
        patches = patches.to(dtype=torch.float32)
    assert triton is not None
    patch_size = int(patches.shape[-1])
    total = patch_size * patch_size
    block = triton.next_power_of_2(total)
    out = torch.empty_like(patches)
    _triton_lut_bilinear_shift_kernel[(int(patches.shape[0]),)](
        patches.contiguous(),
        shift_x_px.to(device=patches.device, dtype=torch.float32).contiguous(),
        shift_y_px.to(device=patches.device, dtype=torch.float32).contiguous(),
        out,
        patch_size,
        block,
    )
    return out


def _resolve_projection_backend(value: str | None = None) -> str:
    raw = str(value or os.environ.get("NEPTUNE_V03_PROJECTION_BACKEND", "formal")).strip().lower()
    aliases = {
        "default": "formal",
        "scatter": "formal",
        "scatter_add": "formal",
        "torch": "formal",
        "triton": "triton_fused",
        "triton_atomic": "triton_fused",
        "fused": "triton_fused",
    }
    raw = aliases.get(raw, raw)
    if raw not in {"formal", "triton_fused"}:
        raise ValueError("NEPTUNE_V03_PROJECTION_BACKEND must be 'formal' or 'triton_fused'")
    if raw == "triton_fused" and not _TRITON_AVAILABLE:
        raise RuntimeError("NEPTUNE_V03_PROJECTION_BACKEND=triton_fused requested, but Triton is unavailable")
    return raw


def _project_patches_to_frames(
    patches: torch.Tensor,
    *,
    frame_count: int,
    height: int,
    width: int,
    frame_indices: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    backend: str | None = None,
) -> torch.Tensor:
    projection_backend = _resolve_projection_backend(backend)
    if projection_backend == "formal":
        return _place_patches_frames(
            patches,
            frame_count=int(frame_count),
            height=int(height),
            width=int(width),
            frame_indices=frame_indices,
            center_x=center_x,
            center_y=center_y,
        )
    if projection_backend == "triton_fused":
        with time_block("triton_project_patches_to_frames"):
            return _triton_project_patches_to_frames(
                patches,
                frame_count=int(frame_count),
                height=int(height),
                width=int(width),
                frame_indices=frame_indices,
                center_x=center_x,
                center_y=center_y,
            )
    raise AssertionError(f"Unhandled projection backend {projection_backend!r}")


def _triton_project_patches_to_frames(
    patches: torch.Tensor,
    *,
    frame_count: int,
    height: int,
    width: int,
    frame_indices: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton projection requested, but Triton is unavailable")
    if not patches.is_cuda:
        raise RuntimeError("Triton projection requires CUDA patches")
    if patches.dtype != torch.float32:
        raise RuntimeError("Triton projection requires float32 patches")
    assert triton is not None
    patch_size = int(patches.shape[-1])
    total = patch_size * patch_size
    block = triton.next_power_of_2(total)
    out = torch.zeros((int(frame_count), int(height), int(width)), dtype=patches.dtype, device=patches.device)
    _triton_place_patches_frames_kernel[(int(patches.shape[0]),)](
        patches.contiguous(),
        frame_indices.to(device=patches.device, dtype=torch.long).contiguous(),
        center_x.to(device=patches.device, dtype=torch.long).contiguous(),
        center_y.to(device=patches.device, dtype=torch.long).contiguous(),
        out,
        patch_size,
        int(height),
        int(width),
        block,
    )
    return out


def _lut_patch_count(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    cache: _LUTPatchBankCache | None,
    *,
    physical_model_version: int | None = None,
) -> int:
    if cache is None:
        return 0
    bank = cache._items.get(_lut_patch_bank_key(config, sim_config, physical_model_version=physical_model_version))
    if bank is None:
        return 0
    return int(bank.patches.shape[0] * bank.patches.shape[1] * bank.patches.shape[2] * bank.patches.shape[3] * bank.patches.shape[4])


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
    simulation_backend: str = "native"
    psf_type: str = "vector"
    pixel_size_nm_x: float = 101.11
    pixel_size_nm_y: float = 98.83
    wavelength_nm: float = 660.0
    na: float = 1.4
    npupil: int = 128
    vector_psf_size: int = 51
    vector_batch_size: int = 96
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
    domain_count: int = 2
    domain_balance_mode: str = "fixed"
    dual_domain_coeff_maps: tuple[Mapping[str, str], ...] = ()
    batch_strategy: str = "triplet"
    sequence_window_chunks: int = 1
    sequence_count: int = 64
    camera_qe: float = 0.9
    camera_spurious_charge: float = 0.002
    camera_baseline: float = 398.6
    camera_e_per_adu: float = 1.020784562122306
    pxyz_target_order: str = "legacy_iwae"
    photon_scale: float | None = None
    z_scale: float | None = None
    simulation_output_device: str = "cpu"
    cached_window_order: str = "auto"
    cached_window_max_gpu_sequences: int = 0
    lut_field_stride: int = 16
    lut_z_steps: int = 41
    lut_subpixel_bins: int = 1
    lut_field_mode: str = "roi_origin"
    lut_storage_dtype: str = "fp32"
    field_origin_sampling_mode: str = "grid"
    field_origin_stride_px: int = 40


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _cached_window_precompute_enabled() -> bool:
    return _env_flag("NEPTUNE_V03_CACHED_WINDOW_PRECOMPUTE", default=True)


def _lut_epoch_prewarm_enabled() -> bool:
    return _env_flag("NEPTUNE_V03_LUT_EPOCH_PREWARM", default=True)


def _resolve_lut_field_mode(config: OnlineBatchProviderConfig) -> str:
    raw = str(config.lut_field_mode or os.environ.get("NEPTUNE_V03_LUT_FIELD_MODE", "roi_origin")).strip().lower()
    aliases = {
        "roi": "roi_origin",
        "roi_bank": "roi_origin",
        "roi_origin_bank": "roi_origin",
        "global": "global_field",
        "full_field": "global_field",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"roi_origin", "global_field"}:
        raise ValueError("lut_field_mode must be 'roi_origin' or 'global_field'")
    return mode


def _resolve_lut_storage_dtype(config: OnlineBatchProviderConfig) -> str:
    raw = str(config.lut_storage_dtype or os.environ.get("NEPTUNE_V03_LUT_STORAGE_DTYPE", "fp32")).strip().lower()
    aliases = {"float16": "fp16", "half": "fp16", "float32": "fp32", "single": "fp32"}
    dtype = aliases.get(raw, raw)
    if dtype not in {"fp16", "fp32"}:
        raise ValueError("lut_storage_dtype must be 'fp16' or 'fp32'")
    return dtype


def _lut_storage_torch_dtype(config: OnlineBatchProviderConfig, *, device: torch.device) -> torch.dtype:
    del device
    return torch.float16 if _resolve_lut_storage_dtype(config) == "fp16" else torch.float32


def _resolve_field_origin_sampling_mode(config: OnlineBatchProviderConfig) -> str:
    raw = str(config.field_origin_sampling_mode or "grid").strip().lower()
    aliases = {
        "fixed_grid": "grid",
        "cached_grid": "grid",
        "random_step": "per_step_random",
        "step_random": "per_step_random",
        "sliding": "sliding_window",
        "fd_deeploc": "sliding_window",
        "fd-deeploc": "sliding_window",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"grid", "per_step_random", "sliding_window"}:
        raise ValueError("field_origin_sampling_mode must be 'grid', 'per_step_random', or 'sliding_window'")
    return mode


def _validate_online_lut_modes(config: OnlineBatchProviderConfig) -> None:
    if (
        str(config.simulation_backend).strip().lower() == "lut"
        and _resolve_field_origin_sampling_mode(config) == "per_step_random"
        and _resolve_lut_field_mode(config) != "global_field"
    ):
        raise ValueError("field_origin_sampling_mode='per_step_random' requires lut_field_mode='global_field'")


def build_online_batch_provider(config: OnlineBatchProviderConfig, *, condition_store: ConditioningProviderStore | None = None):
    _validate_online_lut_modes(config)
    if config.dual_domain_coeff_maps and int(config.domain_count) != len(config.dual_domain_coeff_maps):
        raise ValueError("domain_count must match dual_domain_coeff_maps length")
    if condition_store is None and config.dual_domain_coeff_maps:
        condition_store = ConditioningProviderStore.from_coeff_maps(config.dual_domain_coeff_maps)
    condition_providers = None if condition_store is not None else _load_condition_providers(config)
    default_condition_providers = (
        (None, build_default_conditioning_maps(width=int(config.width), height=int(config.height))),
    )
    renderer_cache = _VectorRendererCache()
    lut_patch_bank_cache = _LUTPatchBankCache(renderer_cache)
    lifecycle = _PhysicalVersionedLUTLifecycle(
        config,
        renderer_cache=renderer_cache,
        lut_patch_bank_cache=lut_patch_bank_cache,
    )
    if condition_store is not None:
        initial_version, initial_providers = condition_store.snapshot()
        lifecycle.activate(
            version=initial_version,
            condition_providers=initial_providers or default_condition_providers,
        )
        condition_store.add_update_listener(
            lambda version, providers: lifecycle.activate(
                version=version,
                condition_providers=providers or default_condition_providers,
            )
        )
    else:
        lifecycle.activate(version=None, condition_providers=condition_providers or default_condition_providers)

    def condition_state() -> tuple[int | None, tuple[tuple[str | None, FullResZernikeConditioning], ...], int]:
        condition_store_version = None
        active_condition_providers = condition_providers
        if condition_store is not None:
            condition_store_version, active_condition_providers = condition_store.snapshot()
        if active_condition_providers is None:
            active_condition_providers = default_condition_providers
        lifecycle.activate(version=condition_store_version, condition_providers=active_condition_providers)
        return condition_store_version, active_condition_providers, lifecycle.cache_generation

    def provider(epoch: int) -> list[TrainingBatch]:
        if int(config.steps_per_epoch) <= 0:
            raise ValueError("steps_per_epoch must be positive")
        condition_store_version, active_condition_providers, cache_generation = condition_state()

        if str(config.batch_strategy) == "cached_window":
            return _TrainingBatchSequence(
                _build_cached_window_epoch_batches(
                    config,
                    epoch=int(epoch),
                    condition_providers=active_condition_providers,
                    condition_store_version=condition_store_version,
                    cache_generation=cache_generation,
                    condition_state_getter=condition_state,
                    renderer_cache=renderer_cache,
                    lut_patch_bank_cache=lut_patch_bank_cache,
                    lut_lifecycle=lifecycle,
                )
            )

        batches: list[TrainingBatch] = []
        base_seed = int(config.seed) + int(epoch)
        for step_idx in range(int(config.steps_per_epoch)):
            seed = base_seed + step_idx
            loc_batch = _build_native_online_batch(
                config,
                epoch=int(epoch),
                seed=seed,
                step=step_idx + 1,
                global_step=(int(epoch) - 1) * int(config.steps_per_epoch) + step_idx,
                condition_providers=active_condition_providers,
                condition_store_version=condition_store_version,
                cache_generation=cache_generation,
                renderer_cache=renderer_cache,
                lut_patch_bank_cache=lut_patch_bank_cache,
            )
            batches.append(to_training_batch(loc_batch))
        return batches

    return provider


class _PhysicalVersionedLUTLifecycle:
    def __init__(
        self,
        config: OnlineBatchProviderConfig,
        *,
        renderer_cache: _VectorRendererCache,
        lut_patch_bank_cache: _LUTPatchBankCache,
    ) -> None:
        self.config = config
        self.renderer_cache = renderer_cache
        self.lut_patch_bank_cache = lut_patch_bank_cache
        self.active_version: int | None = None
        self.cache_generation = 0
        self.prewarm_count = 0
        self._prewarmed_schedule_keys: set[tuple[object, ...]] = set()

    def activate(
        self,
        *,
        version: int | None,
        condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...],
    ) -> None:
        version_key = None if version is None else int(version)
        if version_key == self.active_version:
            return
        if self.active_version is not None or version_key is not None:
            self.renderer_cache.clear()
            self.lut_patch_bank_cache.clear()
            self.cache_generation += 1
            self._prewarmed_schedule_keys.clear()
        self.active_version = version_key
        self._prewarm(condition_providers)

    def _prewarm(self, condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...]) -> None:
        if str(self.config.simulation_backend).strip().lower() != "lut":
            return
        if str(self.config.psf_type).strip().lower() != "vector":
            return
        for domain_index, (_domain_name, condition_provider) in enumerate(condition_providers):
            if _resolve_lut_field_mode(self.config) == "global_field":
                field_origin = (0, 0)
            else:
                roi_width = _condition_roi_extent(
                    int(self.config.width),
                    int(self.config.nat_grid_size[0] if isinstance(self.config.nat_grid_size, tuple) else self.config.nat_grid_size),
                )
                roi_height = _condition_roi_extent(
                    int(self.config.height),
                    int(self.config.nat_grid_size[1] if isinstance(self.config.nat_grid_size, tuple) else self.config.nat_grid_size),
                )
                field_origin = _condition_origin(
                    self.config,
                    sample_index=0,
                    global_step=int(domain_index),
                    roi_width=roi_width,
                    roi_height=roi_height,
                )
            sim_config = _simulator_config(
                self.config,
                seed=int(self.config.seed),
                field_origin=field_origin,
                frames_per_sample=1,
                condition_provider=condition_provider,
            )
            self.lut_patch_bank_cache.get(
                self.config,
                sim_config,
                physical_model_version=self.active_version,
            )
            self.prewarm_count += 1

    def prewarm_cached_window_schedule(
        self,
        *,
        condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...],
    ) -> None:
        if not _lut_epoch_prewarm_enabled():
            return
        if str(self.config.simulation_backend).strip().lower() != "lut":
            return
        if str(self.config.psf_type).strip().lower() != "vector":
            return
        schedule_key = (
            "cached_window_schedule",
            None if self.active_version is None else int(self.active_version),
            tuple(
                (
                    None if name is None else str(name),
                    int(provider.full_maps_nm.data_ptr()),
                    tuple(int(v) for v in provider.full_maps_nm.shape),
                )
                for name, provider in condition_providers
            ),
            int(self.config.height),
            int(self.config.width),
            tuple(int(v) for v in (self.config.nat_grid_size if isinstance(self.config.nat_grid_size, tuple) else (self.config.nat_grid_size, self.config.nat_grid_size))),
            int(self.config.lut_field_stride),
            int(self.config.lut_z_steps),
            int(self.config.lut_subpixel_bins),
            _resolve_lut_field_mode(self.config),
            _resolve_lut_storage_dtype(self.config),
            _resolve_field_origin_sampling_mode(self.config),
            int(self.config.field_origin_stride_px),
            _range_key(self.config.z_range),
        )
        if schedule_key in self._prewarmed_schedule_keys:
            return
        with time_block("lut_physical_version_prewarm"):
            if _resolve_lut_field_mode(self.config) == "global_field":
                for _domain_index, (_domain_name, condition_provider) in enumerate(condition_providers):
                    sim_config = _simulator_config(
                        self.config,
                        seed=int(self.config.seed),
                        field_origin=(0, 0),
                        frames_per_sample=1,
                        condition_provider=condition_provider,
                    )
                    self.lut_patch_bank_cache.get(
                        self.config,
                        sim_config,
                        physical_model_version=self.active_version,
                    )
                    self.prewarm_count += 1
                self._prewarmed_schedule_keys.add(schedule_key)
                return
            for _domain_index, (_domain_name, condition_provider) in enumerate(condition_providers):
                full_h, full_w = condition_provider.image_shape_hw
                roi_w = min(int(self.config.width), int(full_w))
                roi_h = min(int(self.config.height), int(full_h))
                max_x0 = max(0, int(full_w) - roi_w)
                max_y0 = max(0, int(full_h) - roi_h)
                if _resolve_field_origin_sampling_mode(self.config) == "sliding_window":
                    origin_iter = build_sliding_window_origin_bank(
                        field_width_px=int(full_w),
                        field_height_px=int(full_h),
                        roi_width_px=roi_w,
                        roi_height_px=roi_h,
                        stride_px=int(self.config.field_origin_stride_px),
                    )
                else:
                    grid = self.config.nat_grid_size
                    if isinstance(grid, tuple):
                        grid_x = max(1, int(grid[0]))
                        grid_y = max(1, int(grid[1] if len(grid) > 1 else grid[0]))
                    else:
                        grid_x = grid_y = max(1, int(grid))
                    origin_iter = tuple(
                        (
                            int(0 if grid_x <= 1 else round(float(col) * float(max_x0) / float(grid_x - 1))),
                            int(0 if grid_y <= 1 else round(float(row) * float(max_y0) / float(grid_y - 1))),
                        )
                        for row in range(grid_y)
                        for col in range(grid_x)
                    )
                seen_origins: set[tuple[int, int]] = set()
                for field_origin in origin_iter:
                    effective_origin = _effective_lut_field_origin(
                        field_origin,
                        roi_width=int(self.config.width),
                        roi_height=int(self.config.height),
                        map_width=int(full_w),
                        map_height=int(full_h),
                    )
                    if effective_origin in seen_origins:
                        continue
                    seen_origins.add(effective_origin)
                    sim_config = _simulator_config(
                        self.config,
                        seed=int(self.config.seed),
                        field_origin=effective_origin,
                        frames_per_sample=1,
                        condition_provider=condition_provider,
                    )
                    self.lut_patch_bank_cache.get(
                        self.config,
                        sim_config,
                        physical_model_version=self.active_version,
                    )
                    self.prewarm_count += 1
        self._prewarmed_schedule_keys.add(schedule_key)


def _build_native_online_batch(
    config: OnlineBatchProviderConfig,
    *,
    epoch: int,
    seed: int,
    step: int,
    global_step: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
    cache_generation: int = 0,
    renderer_cache: _VectorRendererCache | None = None,
    lut_patch_bank_cache: _LUTPatchBankCache | None = None,
) -> LocalizationTrainBatch:
    strategy = str(config.batch_strategy)
    if strategy not in {"triplet", "sequence_window"}:
        raise ValueError(f"unsupported online batch_strategy: {strategy}")
    domain_index = _domain_index(config, global_step=global_step)
    condition_provider, _, _ = _condition_provider(
        config,
        domain_index=domain_index,
        condition_providers=condition_providers,
    )
    roi_width = _condition_roi_extent(int(config.width), int(config.nat_grid_size[0] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    roi_height = _condition_roi_extent(int(config.height), int(config.nat_grid_size[1] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    field_origin = _condition_origin(
        config,
        sample_index=0,
        global_step=global_step,
        roi_width=roi_width,
        roi_height=roi_height,
    )
    sim_config = _simulator_config(
        config,
        seed=int(seed),
        field_origin=field_origin,
        frames_per_sample=int(config.channels),
        condition_provider=condition_provider,
    )
    if str(config.simulation_backend).strip().lower() == "lut":
        loc_batch = _build_lut_like_sequence_batch(
            config,
            sim_config,
            epoch=int(epoch),
            step=int(step),
            physical_model_version=condition_store_version,
            cache_generation=cache_generation,
            renderer_cache=renderer_cache,
            lut_patch_bank_cache=lut_patch_bank_cache,
        )
    elif strategy == "sequence_window":
        loc_batch = _build_sequence_window_batch(config, sim_config, epoch=int(epoch), step=int(step))
    else:
        loc_batch = simulate_localization_batch(
            sim_config,
            epoch=int(epoch),
            step=int(step),
            source="native_online",
        )
    loc_batch.metadata.update(
        {
            "batch_strategy": strategy,
            "frames_per_sample": int(config.channels),
            "sequence_window_chunks": int(config.sequence_window_chunks),
            **({} if condition_store_version is None else {"physical_model_version": int(condition_store_version)}),
            "lut_cache_generation": int(cache_generation),
        }
    )
    target_order = loc_batch.metadata.get("pxyz_target_order")
    expected_target_order = _pxyz_target_order_tuple(config)
    if tuple(target_order or ()) != expected_target_order:
        pxyz_tar, target_order = _finalize_pxyz_targets(config, loc_batch.pxyz_tar)
        loc_batch = LocalizationTrainBatch(
            model_input=loc_batch.model_input,
            detect_tar=loc_batch.detect_tar,
            bkg_tar=loc_batch.bkg_tar,
            pxyz_tar=pxyz_tar,
            mask_tar=loc_batch.mask_tar,
            metadata={**loc_batch.metadata, "pxyz_target_order": target_order},
        )
    if str(config.conditioning_mode) == "film":
        loc_batch = _attach_film_conditioning(
            config,
            loc_batch,
            global_step=global_step,
            domain_index=domain_index,
            condition_providers=condition_providers,
            condition_store_version=condition_store_version,
        )
    return loc_batch


def _attach_film_conditioning(
    config: OnlineBatchProviderConfig,
    loc_batch: LocalizationTrainBatch,
    *,
    global_step: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    domain_index: int | None = None,
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    image = loc_batch.model_input
    if not isinstance(image, torch.Tensor):
        raise TypeError("film conditioning expects tensor image input before conditioning is attached")
    feature_dim = _condition_feature_dim(config)
    domain_count = int(config.domain_count)
    selected_domain = _domain_index(config, global_step=global_step) if domain_index is None else int(domain_index)
    condition, condition_source, domain_name = _build_condition_features(
        config,
        global_step=global_step,
        domain_index=selected_domain,
        image=image,
        condition_providers=condition_providers,
    )
    if bool(config.append_domain_onehot):
        onehot = torch.zeros((int(image.shape[0]), domain_count), dtype=image.dtype, device=image.device)
        onehot[:, selected_domain] = 1.0
        condition = torch.cat([condition, onehot], dim=1)
        domain_onehot_slice = (feature_dim, feature_dim + domain_count)
    else:
        domain_onehot_slice = None
    return LocalizationTrainBatch(
        model_input=(image, condition),
        detect_tar=loc_batch.detect_tar,
        bkg_tar=loc_batch.bkg_tar,
        pxyz_tar=loc_batch.pxyz_tar,
        mask_tar=loc_batch.mask_tar,
        metadata={
            **loc_batch.metadata,
            "conditioning_mode": "film",
            "condition_source": condition_source,
            "condition_dim": int(condition.shape[1]),
            "condition_feature_dim": feature_dim,
            "condition_feature_order": condition_feature_order(
                include_domain_onehot=bool(config.append_domain_onehot),
                domain_count=domain_count,
            ),
            "condition_domain_onehot_slice": domain_onehot_slice,
            "domain_count": domain_count,
            "domain_index": selected_domain,
            **({} if condition_store_version is None else {"condition_store_version": int(condition_store_version)}),
            **({} if domain_name is None else {"domain_name": domain_name}),
        },
    )


class _TrainingBatchSequence:
    def __init__(self, loc_batches) -> None:
        self._loc_batches = loc_batches

    def __len__(self) -> int:
        return len(self._loc_batches)

    def __iter__(self):
        for loc_batch in self._loc_batches:
            yield to_training_batch(loc_batch)

    def __getitem__(self, index: int) -> TrainingBatch:
        return to_training_batch(self._loc_batches[int(index)])


class _CachedWindowEpochBatches:
    def __init__(
        self,
        config: OnlineBatchProviderConfig,
        *,
        epoch: int,
        condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
        condition_store_version: int | None = None,
        cache_generation: int = 0,
        condition_state_getter: Callable[
            [], tuple[int | None, tuple[tuple[str | None, FullResZernikeConditioning], ...], int]
        ]
        | None = None,
        renderer_cache: _VectorRendererCache | None = None,
        lut_patch_bank_cache: _LUTPatchBankCache | None = None,
        lut_lifecycle: _PhysicalVersionedLUTLifecycle | None = None,
    ) -> None:
        self.config = config
        self.epoch = int(epoch)
        self.condition_providers = condition_providers
        self.condition_store_version = condition_store_version
        self.cache_generation = int(cache_generation)
        self.condition_state_getter = condition_state_getter
        self.renderer_cache = renderer_cache
        self.lut_patch_bank_cache = lut_patch_bank_cache
        self.lut_lifecycle = lut_lifecycle
        self.field_origin_sampling_mode = _resolve_field_origin_sampling_mode(config)
        self.window_size = int(config.channels)
        if self.window_size != 3:
            raise ValueError("cached_window currently expects channels/window_size=3")
        self.batch_size = int(config.batch_size)
        self.center_samples = int(config.steps_per_epoch) * self.batch_size
        if self.field_origin_sampling_mode == "per_step_random":
            self.sequence_count = max(1, int(config.steps_per_epoch))
        else:
            self.sequence_count = max(1, int(config.sequence_count))
        self.centers_per_sequence = int(math.ceil(float(self.center_samples) / float(self.sequence_count)))
        self.frames_per_sequence = self.centers_per_sequence + self.window_size - 1
        self.base_seed = int(config.seed) + int(epoch)
        self.global_step_base = (int(epoch) - 1) * int(config.steps_per_epoch)
        self._sequence_cache: OrderedDict[int, dict[str, object]] = OrderedDict()
        self.cached_window_order = _resolve_cached_window_order(config)
        if self.field_origin_sampling_mode == "per_step_random" and self.cached_window_order != "sequence_block":
            raise ValueError("field_origin_sampling_mode='per_step_random' requires cached_window_order='sequence_block' or 'auto' with renderer output")
        self.max_cached_sequences = _resolve_cached_window_max_cached_sequences(config)
        self.sequence_cache_evictions = 0
        self._reported_metric_sequence_indices: set[int] = set()
        self.center_records: list[tuple[int, int, int]] = []
        for sequence_idx in range(self.sequence_count):
            for center_offset in range(self.centers_per_sequence):
                if len(self.center_records) >= self.center_samples:
                    break
                center_idx = center_offset + self.window_size // 2
                self.center_records.append((sequence_idx, center_idx, len(self.center_records)))
        rng = np.random.default_rng(self.base_seed)
        self.order = self._build_order(rng)
        self._prewarm_epoch_lut_patch_banks()

    def __len__(self) -> int:
        return int(self.config.steps_per_epoch)

    def __iter__(self):
        for batch_idx in range(len(self)):
            yield self[batch_idx]

    def __getitem__(self, batch_idx: int) -> LocalizationTrainBatch:
        self._refresh_condition_state()
        batch_idx = int(batch_idx)
        selected = [
            self.center_records[int(idx)]
            for idx in self.order[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        ]
        sequence_indices = sorted({int(item[0]) for item in selected})
        sequences = [self._sequence(index) for index in sequence_indices]
        sequence_by_index = {int(index): sequence for index, sequence in zip(sequence_indices, sequences, strict=True)}
        metric_sequence_indices = set(sequence_indices) - self._reported_metric_sequence_indices
        ordered_sequences = [
            _sequence_with_batch_metric_visibility(
                sequence_by_index[int(sequence_idx)],
                include_metrics=int(sequence_idx) in metric_sequence_indices,
            )
            for sequence_idx, _, _ in selected
        ]
        remapped_selected = [(sample_idx, center_idx, global_center_idx) for sample_idx, (_, center_idx, global_center_idx) in enumerate(selected)]
        loc_batch = _slice_cached_window_batch(
            self.config,
            sequences=ordered_sequences,
            selected=remapped_selected,
            epoch=self.epoch,
            batch_idx=batch_idx,
            center_samples=self.center_samples,
            sequence_count=self.sequence_count,
            frames_per_sequence=self.frames_per_sequence,
            condition_providers=self.condition_providers,
            condition_store_version=self.condition_store_version,
        )
        loc_batch.metadata.update(
            {
                "cached_window_order": self.cached_window_order,
                "cached_window_max_cached_sequences": int(self.max_cached_sequences),
                "cached_window_cached_sequence_count": int(len(self._sequence_cache)),
                "cached_window_sequence_cache_evictions": int(self.sequence_cache_evictions),
                "field_origin_sampling_mode": self.field_origin_sampling_mode,
            }
        )
        self._reported_metric_sequence_indices.update(sequence_indices)
        return loc_batch

    def _build_order(self, rng: np.random.Generator) -> np.ndarray:
        if self.cached_window_order == "shuffle":
            order = np.arange(len(self.center_records), dtype=np.int64)
            rng.shuffle(order)
            return order
        if self.cached_window_order != "sequence_block":
            raise ValueError(f"unsupported cached_window_order: {self.cached_window_order}")
        blocks: list[int] = []
        sequence_order = np.arange(self.sequence_count, dtype=np.int64)
        rng.shuffle(sequence_order)
        records_by_sequence: list[list[int]] = [[] for _ in range(self.sequence_count)]
        for record_index, (sequence_idx, _center_idx, _global_center_idx) in enumerate(self.center_records):
            records_by_sequence[int(sequence_idx)].append(int(record_index))
        for sequence_idx in sequence_order.tolist():
            sequence_records = np.asarray(records_by_sequence[int(sequence_idx)], dtype=np.int64)
            rng.shuffle(sequence_records)
            blocks.extend(int(item) for item in sequence_records.tolist())
        return np.asarray(blocks, dtype=np.int64)

    def _refresh_condition_state(self) -> None:
        if self.condition_state_getter is None:
            return
        version, providers, cache_generation = self.condition_state_getter()
        version_key = None if version is None else int(version)
        cache_generation_i = int(cache_generation)
        if (
            version_key == self.condition_store_version
            and cache_generation_i == self.cache_generation
            and providers is self.condition_providers
        ):
            return
        self.condition_store_version = version_key
        self.condition_providers = providers
        self.cache_generation = cache_generation_i
        self._sequence_cache.clear()
        self._reported_metric_sequence_indices.clear()
        self._prewarm_epoch_lut_patch_banks()

    def _sequence(self, sequence_idx: int) -> dict[str, object]:
        sequence_idx = int(sequence_idx)
        if sequence_idx in self._sequence_cache:
            payload = self._sequence_cache.pop(sequence_idx)
            self._sequence_cache[sequence_idx] = payload
            return payload
        config = self.config
        self._evict_sequence_cache_to_fit(new_items=1)
        domain_index = _domain_index(config, global_step=self.global_step_base + sequence_idx)
        condition_provider, condition_source, _ = _condition_provider(
            config,
            domain_index=domain_index,
            condition_providers=self.condition_providers,
        )
        field_origin_index = None
        field_origin_bank_size = None
        if self.field_origin_sampling_mode == "per_step_random":
            field_origin = _random_sequence_condition_origin(
                config,
                condition_provider=condition_provider,
                epoch=self.epoch,
                sequence_idx=sequence_idx,
                domain_index=domain_index,
            )
        elif self.field_origin_sampling_mode == "sliding_window":
            field_origin, field_origin_index, field_origin_bank_size = _sliding_window_sequence_condition_origin(
                config,
                condition_provider=condition_provider,
                epoch=self.epoch,
                sequence_idx=sequence_idx,
                sequence_count=self.sequence_count,
            )
        else:
            field_origin = _cached_sequence_condition_origin(
                config,
                condition_provider=condition_provider,
                epoch=self.epoch,
                sequence_idx=sequence_idx,
                sequence_count=self.sequence_count,
            )
        sim_config = _simulator_config(
            config,
            seed=self.base_seed + sequence_idx * 10_000,
            field_origin=field_origin,
            frames_per_sample=1,
            condition_provider=condition_provider,
        )
        if str(config.simulation_backend).strip().lower() == "lut":
            sequence = _simulate_lut_like_sequence(
                config,
                sim_config,
                frame_count=self.frames_per_sequence,
                physical_model_version=self.condition_store_version,
                cache_generation=self.cache_generation,
                renderer_cache=self.renderer_cache,
                lut_patch_bank_cache=self.lut_patch_bank_cache,
            )
        else:
            sequence = _simulate_native_sequence(
                config,
                sim_config,
                epoch=self.epoch,
                sequence_idx=sequence_idx,
                frame_count=self.frames_per_sequence,
                physical_model_version=self.condition_store_version,
                cache_generation=self.cache_generation,
                renderer_cache=self.renderer_cache,
            )
        payload = {
            **sequence,
            "sequence_idx": sequence_idx,
            "domain_index": domain_index,
            "domain_name": (
                None
                if not self.condition_providers
                else self.condition_providers[int(domain_index) % len(self.condition_providers)][0]
            ),
            "field_origin": field_origin,
            "field_origin_index": field_origin_index,
            "field_origin_bank_size": field_origin_bank_size,
            "global_step": self.global_step_base + sequence_idx,
        }
        if _cached_window_precompute_enabled():
            _precompute_cached_sequence_payload(
                config,
                payload,
                condition_provider=condition_provider,
                condition_source=condition_source,
            )
        self._sequence_cache[sequence_idx] = payload
        self._evict_sequence_cache_over_limit()
        return payload

    def _prewarm_epoch_lut_patch_banks(self) -> None:
        if self.lut_lifecycle is None:
            return
        self.lut_lifecycle.prewarm_cached_window_schedule(
            condition_providers=self.condition_providers or (),
        )

    def _evict_sequence_cache_to_fit(self, *, new_items: int) -> None:
        if self.max_cached_sequences <= 0:
            return
        while len(self._sequence_cache) + int(new_items) > int(self.max_cached_sequences):
            self._sequence_cache.popitem(last=False)
            self.sequence_cache_evictions += 1

    def _evict_sequence_cache_over_limit(self) -> None:
        if self.max_cached_sequences <= 0:
            return
        while len(self._sequence_cache) > int(self.max_cached_sequences):
            self._sequence_cache.popitem(last=False)
            self.sequence_cache_evictions += 1


def _resolve_cached_window_order(config: OnlineBatchProviderConfig) -> str:
    raw = str(config.cached_window_order or "auto").strip().lower()
    if raw == "auto":
        if str(config.simulation_output_device).strip().lower() == "renderer":
            return "sequence_block"
        return "shuffle"
    if raw in {"shuffle", "sequence_block"}:
        return raw
    raise ValueError("cached_window_order must be 'auto', 'shuffle', or 'sequence_block'")


def _resolve_cached_window_max_cached_sequences(config: OnlineBatchProviderConfig) -> int:
    value = int(config.cached_window_max_gpu_sequences)
    if value > 0:
        return value
    if str(config.simulation_output_device).strip().lower() == "renderer":
        return 4
    return 0


def _sequence_with_batch_metric_visibility(sequence: dict[str, object], *, include_metrics: bool) -> dict[str, object]:
    if include_metrics:
        return sequence
    metadata = dict(sequence.get("metadata", {}))
    for key in (
        "renderer_cache_hits",
        "renderer_cache_misses",
        "render_many_calls",
        "render_frames_calls",
        "lut_patch_bank_hits",
        "lut_patch_bank_misses",
    ):
        metadata[key] = 0
    return {**sequence, "metadata": metadata}


def _build_cached_window_epoch_batches(
    config: OnlineBatchProviderConfig,
    *,
    epoch: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
    cache_generation: int = 0,
    condition_state_getter: Callable[
        [], tuple[int | None, tuple[tuple[str | None, FullResZernikeConditioning], ...], int]
    ]
    | None = None,
    renderer_cache: _VectorRendererCache | None = None,
    lut_patch_bank_cache: _LUTPatchBankCache | None = None,
    lut_lifecycle: _PhysicalVersionedLUTLifecycle | None = None,
) -> _CachedWindowEpochBatches:
    return _CachedWindowEpochBatches(
        config,
        epoch=int(epoch),
        condition_providers=condition_providers,
        condition_store_version=condition_store_version,
        cache_generation=cache_generation,
        condition_state_getter=condition_state_getter,
        renderer_cache=renderer_cache,
        lut_patch_bank_cache=lut_patch_bank_cache,
        lut_lifecycle=lut_lifecycle,
    )


def _precompute_cached_sequence_payload(
    config: OnlineBatchProviderConfig,
    payload: dict[str, object],
    *,
    condition_provider: FullResZernikeConditioning,
    condition_source: str,
) -> None:
    window_size = int(config.channels)
    frames = payload.get("frames")
    pxyz = payload.get("pxyz")
    mask = payload.get("mask")
    bkg = payload.get("bkg")
    if not isinstance(frames, torch.Tensor) or not isinstance(pxyz, torch.Tensor) or not isinstance(mask, torch.Tensor) or not isinstance(bkg, torch.Tensor):
        return
    center_count = max(0, int(frames.shape[0]) - window_size + 1)
    if center_count <= 0:
        return
    center_start = window_size // 2
    window_frames = []
    detect_targets = []
    bkg_targets = []
    finalized_targets = []
    finalized_masks = []
    target_order: tuple[str, ...] | None = None
    with time_block("cached_sequence_precompute_windows"):
        for center_offset in range(center_count):
            center_idx = center_start + center_offset
            start = int(center_idx) - window_size // 2
            end = start + window_size
            target = pxyz[int(center_idx)]
            active = mask[int(center_idx)]
            window_frames.append(frames[start:end])
            detect_targets.append(_detect_from_v03_targets(target, active, height=int(config.height), width=int(config.width)))
            bkg_targets.append(bkg[int(center_idx)])
            final_target, target_order = _finalize_pxyz_targets(config, target[active])
            finalized_targets.append(final_target)
            finalized_masks.append(torch.ones((int(final_target.shape[0]),), dtype=torch.bool, device=final_target.device))
        payload["precomputed_center_start"] = int(center_start)
        payload["precomputed_window_frames"] = torch.stack(window_frames, dim=0).contiguous()
        payload["precomputed_detect"] = torch.stack(detect_targets, dim=0).contiguous()
        payload["precomputed_bkg"] = torch.stack(bkg_targets, dim=0).contiguous()
        payload["precomputed_pxyz"] = tuple(finalized_targets)
        payload["precomputed_mask"] = tuple(finalized_masks)
        payload["precomputed_target_order"] = target_order or _pxyz_target_order_tuple(config)
    if str(config.conditioning_mode) == "film":
        with time_block("cached_sequence_precompute_condition"):
            field_origin = tuple(int(v) for v in payload["field_origin"])
            image_device = frames.device
            image_dtype = frames.dtype
            vector = condition_provider.condition_vector_from_xy(
                x0=int(field_origin[0]),
                y0=int(field_origin[1]),
                height=int(config.height),
                width=int(config.width),
                device=image_device,
                dtype=image_dtype,
            )
            payload["precomputed_condition_vector"] = _match_feature_dim(
                vector,
                feature_dim=_condition_feature_dim(config),
            ).contiguous()
            payload["precomputed_condition_source"] = str(condition_source)


def _simulate_native_sequence(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    epoch: int,
    sequence_idx: int,
    frame_count: int,
    physical_model_version: int | None = None,
    cache_generation: int = 0,
    renderer_cache: _VectorRendererCache | None = None,
) -> dict[str, object]:
    sequence_config = replace(sim_config, batch_size=int(frame_count), frames_per_sample=1)
    cache_hits_before = 0 if renderer_cache is None else int(renderer_cache.hits)
    cache_misses_before = 0 if renderer_cache is None else int(renderer_cache.misses)
    sequence = _simulate_native_sequence_batched(
        config,
        sequence_config,
        physical_model_version=physical_model_version,
        renderer_cache=renderer_cache,
    )
    cache_hits_after = 0 if renderer_cache is None else int(renderer_cache.hits)
    cache_misses_after = 0 if renderer_cache is None else int(renderer_cache.misses)
    sequence.metadata.update(
        {
            "epoch": int(epoch),
            "step": int(sequence_idx) + 1,
            "seed": int(sequence_config.seed),
            "source": "native_cached_window_batched_sequence_frames",
            "sequence_source": "native_cached_window_batched_sequence_frames",
            "renderer_cache_hits": cache_hits_after - cache_hits_before,
            "renderer_cache_misses": cache_misses_after - cache_misses_before,
            "render_many_calls": 0,
            "render_frames_calls": int(sequence.metadata.get("render_frames_calls", 0)),
            **({} if physical_model_version is None else {"physical_model_version": int(physical_model_version)}),
            "lut_cache_generation": int(cache_generation),
        }
    )
    return {
        "frames": sequence.model_input[:, 0],
        "pxyz": sequence.pxyz_tar,
        "mask": sequence.mask_tar,
        "bkg": sequence.bkg_tar,
        "counts": [int(item) for item in sequence.metadata.get("emitter_counts", [])],
        "metadata": dict(sequence.metadata),
        "sequence_source": "native_cached_window_batched_sequence_frames",
    }


def _simulate_native_sequence_batched(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    physical_model_version: int | None = None,
    renderer_cache: _VectorRendererCache | None = None,
) -> LocalizationTrainBatch:
    frame_count = int(sim_config.batch_size)
    height = int(sim_config.height)
    width = int(sim_config.width)
    max_emitters = _max_emitters_per_sample(sim_config)
    if min(frame_count, height, width, max_emitters) <= 0:
        raise ValueError("native sequence dimensions and emitter count must be positive")

    generator = torch.Generator().manual_seed(int(sim_config.seed))
    counts = _sample_counts(sim_config, batch_size=frame_count, generator=generator)
    xs = torch.zeros((frame_count, max_emitters), dtype=torch.float32)
    ys = torch.zeros((frame_count, max_emitters), dtype=torch.float32)
    photons = torch.zeros((frame_count, max_emitters), dtype=torch.float32)
    z = torch.zeros((frame_count, max_emitters), dtype=torch.float32)
    mask = torch.zeros((frame_count, max_emitters), dtype=torch.bool)
    for frame_idx, count in enumerate(counts.tolist()):
        count_i = int(count)
        if count_i <= 0:
            continue
        mask[frame_idx, :count_i] = True
        xs[frame_idx, :count_i] = torch.rand((count_i,), generator=generator) * float(width - 1)
        ys[frame_idx, :count_i] = torch.rand((count_i,), generator=generator) * float(height - 1)
        photons[frame_idx, :count_i] = _sample_photons(sim_config, emitters=count_i, generator=generator)
        z[frame_idx, :count_i] = _sample_range(sim_config.z_range, shape=(count_i,), fallback=0.0, generator=generator)
    background = _sample_range(
        sim_config.background_range,
        shape=(frame_count,),
        fallback=float(sim_config.background),
        generator=generator,
    )

    if renderer_cache is None:
        renderer = _build_vector_renderer(sim_config)
    else:
        renderer = renderer_cache.get(sim_config, physical_model_version=physical_model_version)
    device = renderer.device
    xs = xs.to(device=device)
    ys = ys.to(device=device)
    photons = photons.to(device=device)
    z = z.to(device=device)
    mask = mask.to(device=device)
    background = background.to(device=device)

    frames = background.view(frame_count, 1, 1).expand(frame_count, height, width).clone()
    render_frames_calls = 0
    active = mask.reshape(-1)
    if bool(active.any()):
        all_frame_indices = torch.arange(frame_count, device=device, dtype=torch.long).view(frame_count, 1).expand(frame_count, max_emitters)
        flat_frame_indices = all_frame_indices.reshape(-1)[active]
        flat_xs = xs.reshape(-1)[active]
        flat_ys = ys.reshape(-1)[active]
        flat_z = z.reshape(-1)[active]
        flat_photons = photons.reshape(-1)[active]
        frames += renderer.render_frames(
            frame_count=frame_count,
            height=height,
            width=width,
            frame_indices=flat_frame_indices,
            xs=flat_xs,
            ys=flat_ys,
            z_um=flat_z,
            photons=flat_photons,
            field_origin_xy=tuple(int(v) for v in sim_config.field_origin_xy),
        )
        render_frames_calls = 1

    bkg_tar = background.view(frame_count, 1, 1).expand(frame_count, height, width).clone() / max(float(config.background_scale), 1e-12)
    batch = LocalizationTrainBatch(
        model_input=frames[:, None].to(dtype=torch.float32).contiguous(),
        detect_tar=torch.zeros((frame_count, height, width), dtype=torch.float32, device=device),
        bkg_tar=bkg_tar.to(dtype=torch.float32).contiguous(),
        pxyz_tar=torch.stack([xs, ys, z, photons], dim=-1).to(dtype=torch.float32).contiguous(),
        mask_tar=mask.contiguous(),
        metadata={
            "seed": int(sim_config.seed),
            "pxyz_target_order": V03_PXYZ_TARGET_ORDER,
            "background_scale": float(config.background_scale),
            "psf_type": str(sim_config.psf_type).lower(),
            "emitter_counts": [int(v) for v in counts.tolist()],
            "render_frames_calls": render_frames_calls,
        },
    )
    if str(config.simulation_output_device).strip().lower() == "cpu":
        return LocalizationTrainBatch(
            model_input=batch.model_input.cpu(),
            detect_tar=batch.detect_tar.cpu(),
            bkg_tar=batch.bkg_tar.cpu(),
            pxyz_tar=batch.pxyz_tar.cpu(),
            mask_tar=batch.mask_tar.cpu(),
            metadata=batch.metadata,
        )
    if str(config.simulation_output_device).strip().lower() == "renderer":
        return batch
    raise ValueError("simulation_output_device must be 'cpu' or 'renderer'")


def _simulate_lut_like_sequence(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    frame_count: int,
    physical_model_version: int | None = None,
    cache_generation: int = 0,
    renderer_cache: _VectorRendererCache | None = None,
    lut_patch_bank_cache: _LUTPatchBankCache | None = None,
) -> dict[str, object]:
    cache_hits_before = 0 if renderer_cache is None else int(renderer_cache.hits)
    cache_misses_before = 0 if renderer_cache is None else int(renderer_cache.misses)
    bank_hits_before = 0 if lut_patch_bank_cache is None else int(lut_patch_bank_cache.hits)
    bank_misses_before = 0 if lut_patch_bank_cache is None else int(lut_patch_bank_cache.misses)
    frame_photons, pxyz_by_frame, mask_by_frame, bkg_by_frame, counts = _simulate_lut_like_sequence_frames(
        config,
        sim_config,
        frame_count=int(frame_count),
        physical_model_version=physical_model_version,
        renderer_cache=renderer_cache,
        lut_patch_bank_cache=lut_patch_bank_cache,
    )
    cache_hits_after = 0 if renderer_cache is None else int(renderer_cache.hits)
    cache_misses_after = 0 if renderer_cache is None else int(renderer_cache.misses)
    bank_hits_after = 0 if lut_patch_bank_cache is None else int(lut_patch_bank_cache.hits)
    bank_misses_after = 0 if lut_patch_bank_cache is None else int(lut_patch_bank_cache.misses)
    with time_block("poisson_camera_noise"):
        raw_adu = torch.poisson(
            frame_photons * float(config.camera_qe) + float(config.camera_spurious_charge)
        ) / max(float(config.camera_e_per_adu), 1e-12) + float(config.camera_baseline)
    return {
        "frames": raw_adu.to(dtype=torch.float32),
        "pxyz": pxyz_by_frame,
        "mask": mask_by_frame,
        "bkg": bkg_by_frame / max(float(config.background_scale), 1e-12),
        "counts": [int(item) for item in counts],
        "metadata": {
            "simulation_backend": "lut",
            "simulation_backend_effective": "lut_global_field" if _resolve_lut_field_mode(config) == "global_field" else "lut_patch_bank",
            "lut_field_mode": _resolve_lut_field_mode(config),
            "lut_storage_dtype": _resolve_lut_storage_dtype(config),
            "lut_shift_backend": _resolve_lut_subpixel_shift_backend(),
            "projection_backend": _resolve_projection_backend(),
            "input_domain": "raw_adu",
            "frame_emitter_policy": "center_frame_only",
            "background_scale": float(config.background_scale),
            "psf_type": str(config.psf_type).lower(),
            "renderer_cache_hits": cache_hits_after - cache_hits_before,
            "renderer_cache_misses": cache_misses_after - cache_misses_before,
            "lut_patch_bank_hits": bank_hits_after - bank_hits_before,
            "lut_patch_bank_misses": bank_misses_after - bank_misses_before,
            "lut_patch_count": _lut_patch_count(
                config,
                sim_config,
                lut_patch_bank_cache,
                physical_model_version=physical_model_version,
            ),
            **({} if physical_model_version is None else {"physical_model_version": int(physical_model_version)}),
            **({} if physical_model_version is None else {"lut_patch_bank_version": int(physical_model_version)}),
            "lut_cache_generation": int(cache_generation),
            "render_many_calls": 0,
        },
        "sequence_source": "native_lut_like_cached_window_sequence_frames",
    }


def _slice_cached_window_batch(
    config: OnlineBatchProviderConfig,
    *,
    sequences: list[dict[str, object]],
    selected: list[tuple[int, int, int]],
    epoch: int,
    batch_idx: int,
    center_samples: int,
    sequence_count: int,
    frames_per_sequence: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    window_size = int(config.channels)
    frames = []
    detect = []
    bkg = []
    pxyz = []
    mask = []
    counts = []
    target_indices = []
    sequence_indices = []
    frame_indices = []
    global_center_indices = []
    domain_indices = []
    domain_names = []
    condition_origins = []
    condition_origin_indices = []
    condition_origin_bank_sizes = []
    precomputed_condition_vectors = []
    precomputed_condition_sources = []
    sequence_global_steps = []
    with time_block("cached_window_slice_loop"):
        for sequence_idx, center_idx, global_center_idx in selected:
            sequence = sequences[int(sequence_idx)]
            start = int(center_idx) - window_size // 2
            end = start + window_size
            sequence_frames = sequence["frames"]
            sequence_pxyz = sequence["pxyz"]
            sequence_mask = sequence["mask"]
            sequence_bkg = sequence["bkg"]
            if not isinstance(sequence_frames, torch.Tensor):
                raise TypeError("cached sequence frames must be a tensor")
            if not isinstance(sequence_pxyz, torch.Tensor) or not isinstance(sequence_mask, torch.Tensor) or not isinstance(sequence_bkg, torch.Tensor):
                raise TypeError("cached sequence targets must be tensors")
            precomputed_offset = int(center_idx) - int(sequence.get("precomputed_center_start", -10**9))
            precomputed_windows = sequence.get("precomputed_window_frames")
            precomputed_detect = sequence.get("precomputed_detect")
            precomputed_bkg = sequence.get("precomputed_bkg")
            precomputed_pxyz = sequence.get("precomputed_pxyz")
            precomputed_mask = sequence.get("precomputed_mask")
            if (
                isinstance(precomputed_windows, torch.Tensor)
                and isinstance(precomputed_detect, torch.Tensor)
                and isinstance(precomputed_bkg, torch.Tensor)
                and isinstance(precomputed_pxyz, tuple)
                and isinstance(precomputed_mask, tuple)
                and 0 <= precomputed_offset < int(precomputed_windows.shape[0])
            ):
                frames.append(precomputed_windows[precomputed_offset])
                detect.append(precomputed_detect[precomputed_offset])
                bkg.append(precomputed_bkg[precomputed_offset])
                pxyz.append(precomputed_pxyz[precomputed_offset])
                mask.append(precomputed_mask[precomputed_offset])
                target_order = tuple(sequence.get("precomputed_target_order", _pxyz_target_order_tuple(config)))
                active_count = int(precomputed_mask[precomputed_offset].sum().item())
            else:
                target = sequence_pxyz[int(center_idx)]
                active = sequence_mask[int(center_idx)]
                frames.append(sequence_frames[start:end])
                detect.append(_detect_from_v03_targets(target, active, height=int(config.height), width=int(config.width)))
                bkg.append(sequence_bkg[int(center_idx)])
                final_target, target_order = _finalize_pxyz_targets(config, target[active])
                pxyz.append(final_target)
                mask.append(torch.ones((int(final_target.shape[0]),), dtype=torch.bool, device=final_target.device))
                active_count = int(active.sum().item())
            sequence_counts = sequence["counts"]
            counts.append(int(sequence_counts[int(center_idx)] if int(center_idx) < len(sequence_counts) else active_count))
            target_indices.append(int(center_idx))
            sequence_indices.append(int(sequence.get("sequence_idx", sequence_idx)))
            frame_indices.append(tuple(range(start, end)))
            global_center_indices.append(int(global_center_idx))
            domain_indices.append(int(sequence["domain_index"]))
            domain_names.append(None if sequence.get("domain_name") is None else str(sequence["domain_name"]))
            condition_origins.append(tuple(int(item) for item in sequence["field_origin"]))
            condition_origin_indices.append(
                None if sequence.get("field_origin_index") is None else int(sequence["field_origin_index"])
            )
            condition_origin_bank_sizes.append(
                None if sequence.get("field_origin_bank_size") is None else int(sequence["field_origin_bank_size"])
            )
            if isinstance(sequence.get("precomputed_condition_vector"), torch.Tensor):
                precomputed_condition_vectors.append(sequence["precomputed_condition_vector"])
                precomputed_condition_sources.append(str(sequence.get("precomputed_condition_source", "unknown")))
            sequence_global_steps.append(int(sequence["global_step"]))

    unique_sequences: list[dict[str, object]] = []
    seen_sequence_keys: set[object] = set()
    with time_block("cached_window_unique_sequence_aggregate"):
        for sequence in sequences:
            key: object = sequence.get("sequence_idx", id(sequence))
            if key in seen_sequence_keys:
                continue
            seen_sequence_keys.add(key)
            unique_sequences.append(sequence)

        base_metadata = dict(sequences[0].get("metadata", {})) if sequences else {}
        aggregate_metadata = {
            "renderer_cache_hits": sum(int(dict(sequence.get("metadata", {})).get("renderer_cache_hits", 0)) for sequence in unique_sequences),
            "renderer_cache_misses": sum(int(dict(sequence.get("metadata", {})).get("renderer_cache_misses", 0)) for sequence in unique_sequences),
            "render_many_calls": sum(int(dict(sequence.get("metadata", {})).get("render_many_calls", 0)) for sequence in unique_sequences),
            "render_frames_calls": sum(int(dict(sequence.get("metadata", {})).get("render_frames_calls", 0)) for sequence in unique_sequences),
            "lut_patch_bank_hits": sum(int(dict(sequence.get("metadata", {})).get("lut_patch_bank_hits", 0)) for sequence in unique_sequences),
            "lut_patch_bank_misses": sum(int(dict(sequence.get("metadata", {})).get("lut_patch_bank_misses", 0)) for sequence in unique_sequences),
            "lut_patch_count": max(int(dict(sequence.get("metadata", {})).get("lut_patch_count", 0)) for sequence in unique_sequences) if unique_sequences else 0,
        }
    with time_block("cached_window_pad_pxyz_targets"):
        padded_pxyz, padded_mask = _pad_pxyz_targets(pxyz, mask)
    with time_block("cached_window_stack_tensors"):
        model_input = torch.stack(frames, dim=0).to(dtype=torch.float32).contiguous()
        detect_tar = torch.stack(detect, dim=0).contiguous()
        bkg_tar = torch.stack(bkg, dim=0).contiguous()
        pxyz_tar = padded_pxyz.contiguous()
        mask_tar = padded_mask.contiguous()
    loc_batch = LocalizationTrainBatch(
        model_input=model_input,
        detect_tar=detect_tar,
        bkg_tar=bkg_tar,
        pxyz_tar=pxyz_tar,
        mask_tar=mask_tar,
        metadata={
            **base_metadata,
            **aggregate_metadata,
            "epoch": int(epoch),
            "step": int(batch_idx) + 1,
            "seed": int(config.seed) + int(epoch),
            "source": "native_online",
            "batch_strategy": "cached_window",
            "sequence_source": str(sequences[0]["sequence_source"]) if sequences else "cached_window_sequence_frames",
            "frames_per_sample": window_size,
            "sequence_window_chunks": int(config.sequence_window_chunks),
            "sequence_count": int(sequence_count),
            "center_samples": int(center_samples),
            "sequence_simulated_frames": int(frames_per_sequence) * int(sequence_count),
            "frames_per_sequence": int(frames_per_sequence),
            "loader_batch_index": int(batch_idx),
            "shuffle": True,
            "center_target_frame_indices": target_indices,
            "window_sequence_indices": sequence_indices,
            "window_sequence_domain_indices": domain_indices,
            "window_sequence_domain_names": domain_names,
            "window_condition_origins": condition_origins,
            "window_condition_origin_indices": condition_origin_indices,
            "window_condition_origin_bank_sizes": condition_origin_bank_sizes,
            "window_sequence_global_steps": sequence_global_steps,
            "window_frame_indices": frame_indices,
            "window_global_center_indices": global_center_indices,
            "emitter_counts": counts,
            "pxyz_target_order": target_order,
        },
    )
    if str(config.conditioning_mode) == "film":
        with time_block("cached_window_attach_film_conditioning"):
            if len(precomputed_condition_vectors) == len(frames):
                loc_batch = _attach_cached_window_precomputed_film_conditioning(
                    config,
                    loc_batch,
                    condition_vectors=precomputed_condition_vectors,
                    condition_sources=precomputed_condition_sources,
                    condition_store_version=condition_store_version,
                )
            else:
                loc_batch = _attach_cached_window_film_conditioning(
                    config,
                    loc_batch,
                    condition_providers=condition_providers,
                    condition_store_version=condition_store_version,
                )
    return loc_batch


def _pad_pxyz_targets(targets: list[torch.Tensor], masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not targets:
        return torch.zeros((0, 0, 4), dtype=torch.float32), torch.zeros((0, 0), dtype=torch.bool)
    max_count = max(int(target.shape[0]) for target in targets)
    dtype = targets[0].dtype
    device = targets[0].device
    padded_targets = torch.zeros((len(targets), max_count, 4), dtype=dtype, device=device)
    padded_masks = torch.zeros((len(targets), max_count), dtype=torch.bool, device=device)
    for sample_idx, (target, mask) in enumerate(zip(targets, masks, strict=True)):
        count = int(target.shape[0])
        if count <= 0:
            continue
        padded_targets[sample_idx, :count] = target
        padded_masks[sample_idx, :count] = mask.to(dtype=torch.bool, device=device)
    return padded_targets, padded_masks


def _attach_cached_window_film_conditioning(
    config: OnlineBatchProviderConfig,
    loc_batch: LocalizationTrainBatch,
    *,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    image = loc_batch.model_input
    if not isinstance(image, torch.Tensor):
        raise TypeError("cached-window film conditioning expects tensor image input before conditioning is attached")
    feature_dim = _condition_feature_dim(config)
    domain_count = int(config.domain_count)
    domain_indices = [int(item) for item in loc_batch.metadata["window_sequence_domain_indices"]]
    condition_origins = [tuple(int(v) for v in item) for item in loc_batch.metadata["window_condition_origins"]]
    roi_width = int(config.width)
    roi_height = int(config.height)
    vectors = []
    condition_sources = []
    with time_block("cached_window_condition_vector_loop"):
        for sample_idx, domain_index in enumerate(domain_indices):
            provider, condition_source, _ = _condition_provider(
                config,
                domain_index=domain_index,
                condition_providers=condition_providers,
            )
            x0, y0 = condition_origins[sample_idx]
            vector = provider.condition_vector_from_xy(
                x0=x0,
                y0=y0,
                height=roi_height,
                width=roi_width,
                device=image.device,
                dtype=image.dtype,
            )
            vectors.append(_match_feature_dim(vector, feature_dim=feature_dim))
            condition_sources.append(condition_source)
    with time_block("cached_window_condition_stack_onehot"):
        condition = torch.stack(vectors, dim=0).contiguous()
        if bool(config.append_domain_onehot):
            onehot = torch.zeros((int(image.shape[0]), domain_count), dtype=image.dtype, device=image.device)
            for sample_idx, domain_index in enumerate(domain_indices):
                onehot[sample_idx, int(domain_index) % domain_count] = 1.0
            condition = torch.cat([condition, onehot], dim=1)
            domain_onehot_slice = (feature_dim, feature_dim + domain_count)
        else:
            domain_onehot_slice = None
    source_set = tuple(sorted(set(condition_sources)))
    condition_source = source_set[0] if len(source_set) == 1 else source_set
    return LocalizationTrainBatch(
        model_input=(image, condition),
        detect_tar=loc_batch.detect_tar,
        bkg_tar=loc_batch.bkg_tar,
        pxyz_tar=loc_batch.pxyz_tar,
        mask_tar=loc_batch.mask_tar,
        metadata={
            **loc_batch.metadata,
            "conditioning_mode": "film",
            "condition_source": condition_source,
            "condition_dim": int(condition.shape[1]),
            "condition_feature_dim": feature_dim,
            "condition_feature_order": condition_feature_order(
                include_domain_onehot=bool(config.append_domain_onehot),
                domain_count=domain_count,
            ),
            "condition_domain_onehot_slice": domain_onehot_slice,
            "domain_count": domain_count,
            "domain_index": domain_indices,
            "domain_name": loc_batch.metadata["window_sequence_domain_names"],
            **({} if condition_store_version is None else {"condition_store_version": int(condition_store_version)}),
        },
    )


def _attach_cached_window_precomputed_film_conditioning(
    config: OnlineBatchProviderConfig,
    loc_batch: LocalizationTrainBatch,
    *,
    condition_vectors: list[torch.Tensor],
    condition_sources: list[str],
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    image = loc_batch.model_input
    if not isinstance(image, torch.Tensor):
        raise TypeError("cached-window film conditioning expects tensor image input before conditioning is attached")
    feature_dim = _condition_feature_dim(config)
    domain_count = int(config.domain_count)
    domain_indices = [int(item) for item in loc_batch.metadata["window_sequence_domain_indices"]]
    with time_block("cached_window_condition_stack_onehot"):
        vectors = [
            _match_feature_dim(vector.to(device=image.device, dtype=image.dtype), feature_dim=feature_dim)
            for vector in condition_vectors
        ]
        condition = torch.stack(vectors, dim=0).contiguous()
        if bool(config.append_domain_onehot):
            onehot = torch.zeros((int(image.shape[0]), domain_count), dtype=image.dtype, device=image.device)
            for sample_idx, domain_index in enumerate(domain_indices):
                onehot[sample_idx, int(domain_index) % domain_count] = 1.0
            condition = torch.cat([condition, onehot], dim=1)
            domain_onehot_slice = (feature_dim, feature_dim + domain_count)
        else:
            domain_onehot_slice = None
    source_set = tuple(sorted(set(condition_sources)))
    condition_source = source_set[0] if len(source_set) == 1 else source_set
    return LocalizationTrainBatch(
        model_input=(image, condition),
        detect_tar=loc_batch.detect_tar,
        bkg_tar=loc_batch.bkg_tar,
        pxyz_tar=loc_batch.pxyz_tar,
        mask_tar=loc_batch.mask_tar,
        metadata={
            **loc_batch.metadata,
            "conditioning_mode": "film",
            "condition_source": condition_source,
            "condition_dim": int(condition.shape[1]),
            "condition_feature_dim": feature_dim,
            "condition_feature_order": condition_feature_order(
                include_domain_onehot=bool(config.append_domain_onehot),
                domain_count=domain_count,
            ),
            "condition_domain_onehot_slice": domain_onehot_slice,
            "domain_count": domain_count,
            "domain_index": domain_indices,
            "domain_name": loc_batch.metadata["window_sequence_domain_names"],
            **({} if condition_store_version is None else {"condition_store_version": int(condition_store_version)}),
        },
    )


def _finalize_pxyz_targets(config: OnlineBatchProviderConfig, target: torch.Tensor) -> tuple[torch.Tensor, tuple[str, ...]]:
    order = _normalize_pxyz_target_order(config.pxyz_target_order)
    if order == "legacy_iwae":
        converted = torch.stack((target[..., 3], target[..., 0], target[..., 1], target[..., 2]), dim=-1)
        if config.photon_scale is not None:
            converted[..., 0] = converted[..., 0] / max(float(config.photon_scale), 1e-12)
        if config.z_scale is not None:
            converted[..., 3] = converted[..., 3] / max(float(config.z_scale), 1e-12)
        return converted, LEGACY_IWAE_PXYZ_TARGET_ORDER
    return target, V03_PXYZ_TARGET_ORDER


def _pxyz_target_order_tuple(config: OnlineBatchProviderConfig) -> tuple[str, ...]:
    order = _normalize_pxyz_target_order(config.pxyz_target_order)
    if order == "legacy_iwae":
        return LEGACY_IWAE_PXYZ_TARGET_ORDER
    return V03_PXYZ_TARGET_ORDER


def _normalize_pxyz_target_order(value: str) -> str:
    key = str(value or "legacy_iwae").strip().lower()
    if key in {"legacy_iwae", "iwae", "old", "phot_xyz", "phot,x,y,z", "photons_x_y_z"}:
        return "legacy_iwae"
    if key in {"v03", "xyzph", "x,y,z,phot", "x_y_z_photons"}:
        return "v03"
    raise ValueError(f"unsupported online pxyz_target_order: {value!r}")


def _simulator_config(
    config: OnlineBatchProviderConfig,
    *,
    seed: int,
    field_origin: tuple[int, int],
    frames_per_sample: int,
    condition_provider: FullResZernikeConditioning,
) -> LocalizationSimulatorConfig:
    return LocalizationSimulatorConfig(
        batch_size=int(config.batch_size),
        frames_per_sample=int(frames_per_sample),
        height=int(config.height),
        width=int(config.width),
        emitters_per_sample=int(config.emitters_per_sample),
        seed=int(seed),
        photons=float(config.signal),
        background=float(config.background),
        psf_type=str(config.psf_type),
        pixel_size_nm_x=float(config.pixel_size_nm_x),
        pixel_size_nm_y=float(config.pixel_size_nm_y),
        wavelength_nm=float(config.wavelength_nm),
        na=float(config.na),
        npupil=int(config.npupil),
        vector_psf_size=int(config.vector_psf_size),
        vector_batch_size=int(config.vector_batch_size),
        emitter_density_um2=config.emitter_density_um2,
        lifetime_avg=float(config.lifetime_avg),
        warmup_frames=float(config.warmup_frames),
        photon_range=config.photon_range,
        photon_mean=config.photon_mean,
        photon_sigma=config.photon_sigma,
        background_range=config.background_range,
        background_scale=float(config.background_scale),
        z_range=config.z_range,
        field_origin_xy=field_origin,
        coeff_maps_nm=condition_provider.full_maps_nm if str(config.psf_type).lower() == "vector" else None,
        coeff_mode_order=tuple(condition_provider.mode_order) if str(config.psf_type).lower() == "vector" else (),
        output_device=str(config.simulation_output_device),
    )


def _build_sequence_window_batch(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    epoch: int,
    step: int,
) -> LocalizationTrainBatch:
    window_size = int(config.channels)
    if window_size != 3:
        raise ValueError("sequence_window currently expects channels/window_size=3")
    center_offset = window_size // 2
    center_count = int(config.batch_size)
    sequence_frames = center_count + window_size - 1
    sequence = simulate_localization_batch(
        replace(sim_config, batch_size=sequence_frames, frames_per_sample=1),
        epoch=int(epoch),
        step=int(step),
        source="native_online_sequence_frames",
    )
    frame_images = sequence.model_input[:, 0]
    frames = []
    detect = []
    bkg = []
    pxyz = []
    mask = []
    target_indices = []
    for start in range(center_count):
        target_idx = int(start) + center_offset
        frames.append(frame_images[start : start + window_size])
        detect.append(sequence.detect_tar[target_idx])
        bkg.append(sequence.bkg_tar[target_idx])
        target, target_order = _finalize_pxyz_targets(config, sequence.pxyz_tar[target_idx])
        pxyz.append(target)
        mask.append(sequence.mask_tar[target_idx])
        target_indices.append(target_idx)
    metadata = {
        **sequence.metadata,
        "source": "native_online",
        "sequence_source": "native_online_sequence_frames",
        "frames_per_sample": window_size,
        "sequence_simulated_frames": sequence_frames,
        "center_target_frame_indices": target_indices,
        "emitter_counts": [int(sequence.metadata["emitter_counts"][idx]) for idx in target_indices],
        "pxyz_target_order": target_order,
    }
    return LocalizationTrainBatch(
        model_input=torch.stack(frames, dim=0).contiguous(),
        detect_tar=torch.stack(detect, dim=0).contiguous(),
        bkg_tar=torch.stack(bkg, dim=0).contiguous(),
        pxyz_tar=torch.stack(pxyz, dim=0).contiguous(),
        mask_tar=torch.stack(mask, dim=0).contiguous(),
        metadata=metadata,
    )


def _build_lut_like_sequence_batch(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    epoch: int,
    step: int,
    physical_model_version: int | None = None,
    cache_generation: int = 0,
    renderer_cache: _VectorRendererCache | None = None,
    lut_patch_bank_cache: _LUTPatchBankCache | None = None,
) -> LocalizationTrainBatch:
    window_size = int(config.channels)
    if window_size != 3:
        raise ValueError("lut sequence generation currently expects channels/window_size=3")
    center_count = int(config.batch_size)
    sequence_frames = center_count + window_size - 1
    frame_photons, pxyz_by_frame, mask_by_frame, bkg_by_frame, counts = _simulate_lut_like_sequence_frames(
        config,
        sim_config,
        frame_count=sequence_frames,
        physical_model_version=physical_model_version,
        renderer_cache=renderer_cache,
        lut_patch_bank_cache=lut_patch_bank_cache,
    )
    raw_adu = torch.poisson(
        frame_photons * float(config.camera_qe) + float(config.camera_spurious_charge)
    ) / max(float(config.camera_e_per_adu), 1e-12) + float(config.camera_baseline)
    frames = []
    detect = []
    bkg = []
    pxyz = []
    mask = []
    target_indices = []
    for start in range(center_count):
        target_idx = int(start) + window_size // 2
        frames.append(raw_adu[start : start + window_size])
        target = pxyz_by_frame[target_idx]
        active = mask_by_frame[target_idx]
        detect.append(_detect_from_v03_targets(target, active, height=int(config.height), width=int(config.width)))
        bkg.append(bkg_by_frame[target_idx] / max(float(config.background_scale), 1e-12))
        final_target, target_order = _finalize_pxyz_targets(config, target)
        pxyz.append(final_target)
        mask.append(active)
        target_indices.append(target_idx)
    return LocalizationTrainBatch(
        model_input=torch.stack(frames, dim=0).to(dtype=torch.float32).contiguous(),
        detect_tar=torch.stack(detect, dim=0).contiguous(),
        bkg_tar=torch.stack(bkg, dim=0).contiguous(),
        pxyz_tar=torch.stack(pxyz, dim=0).contiguous(),
        mask_tar=torch.stack(mask, dim=0).contiguous(),
        metadata={
            "epoch": int(epoch),
            "step": int(step),
            "seed": int(config.seed),
            "source": "native_online",
            "sequence_source": "native_lut_like_sequence_frames",
            "simulation_backend": "lut",
            "simulation_backend_effective": "lut_patch_bank",
            "lut_shift_backend": _resolve_lut_subpixel_shift_backend(),
            "projection_backend": _resolve_projection_backend(),
            "input_domain": "raw_adu",
            "frame_emitter_policy": "center_frame_only",
            "frames_per_sample": window_size,
            "sequence_simulated_frames": sequence_frames,
            "center_target_frame_indices": target_indices,
            "emitter_counts": [int(counts[idx]) for idx in target_indices],
            "background_scale": float(config.background_scale),
            "psf_type": str(config.psf_type).lower(),
            "pxyz_target_order": target_order,
            **({} if physical_model_version is None else {"physical_model_version": int(physical_model_version)}),
            **({} if physical_model_version is None else {"lut_patch_bank_version": int(physical_model_version)}),
            "lut_cache_generation": int(cache_generation),
            "render_many_calls": 0,
        },
    )


def _simulate_lut_like_sequence_frames(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    frame_count: int,
    physical_model_version: int | None = None,
    renderer_cache: _VectorRendererCache | None = None,
    lut_patch_bank_cache: _LUTPatchBankCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    height = int(config.height)
    width = int(config.width)
    with time_block("lut_sequence_sample_emitters"):
        rng = np.random.default_rng(int(sim_config.seed))
        target_active = _target_active_emitters(config)
        pool = max(
            1,
            int(math.ceil(target_active * (int(frame_count) + float(config.warmup_frames)) / (1.0 + max(float(config.lifetime_avg), 1e-6)))),
        )
        xy = rng.uniform(0.0, [float(width - 1), float(height - 1)], size=(pool, 2)).astype(np.float32)
        z_lo, z_hi = config.z_range if config.z_range is not None else (-0.6, 0.6)
        z = rng.uniform(float(z_lo), float(z_hi), size=pool).astype(np.float32)
        photons = _sample_lut_photons(config, rng=rng, count=pool)
        t0 = rng.uniform(-0.5 * float(config.warmup_frames), float(frame_count) + 0.5 * float(config.warmup_frames), size=pool)
        t1 = t0 + rng.exponential(max(float(config.lifetime_avg), 1e-6), size=pool)
        bg_values = _sample_lut_background(config, rng=rng, count=int(frame_count))
    if str(config.psf_type).strip().lower() != "vector":
        raise ValueError("online LUT simulation requires psf_type='vector'")
    if renderer_cache is None:
        renderer_cache = _VectorRendererCache()
    if lut_patch_bank_cache is None:
        lut_patch_bank_cache = _LUTPatchBankCache(renderer_cache)
    with time_block("lut_sequence_get_bank"):
        lut_bank = lut_patch_bank_cache.get(
            config,
            sim_config,
            physical_model_version=physical_model_version,
        )
    render_device = lut_bank.patches.device
    with time_block("lut_sequence_frame_background_init"):
        frames = (
            torch.as_tensor(bg_values[:, None, None], dtype=torch.float32, device=render_device)
            .expand(int(frame_count), height, width)
            .clone()
        )
    with time_block("lut_sequence_events_by_frame"):
        events_by_frame: list[list[tuple[int, float]]] = [[] for _ in range(int(frame_count))]
        for frame_idx in range(int(frame_count)):
            overlap = np.minimum(t1, float(frame_idx + 1)) - np.maximum(t0, float(frame_idx))
            active = np.nonzero(overlap > 0)[0]
            for emitter_idx in active:
                events_by_frame[frame_idx].append((int(emitter_idx), float(overlap[emitter_idx])))

    with time_block("lut_sequence_target_alloc"):
        max_count = max(1, max(len(items) for items in events_by_frame))
        pxyz = torch.zeros((int(frame_count), max_count, 4), dtype=torch.float32, device=render_device)
        mask = torch.zeros((int(frame_count), max_count), dtype=torch.bool, device=render_device)
    counts: list[int] = []
    flat_frame_indices: list[int] = []
    flat_emitter_indices: list[int] = []
    flat_overlaps: list[float] = []
    with time_block("lut_sequence_flat_event_arrays"):
        for frame_idx, items in enumerate(events_by_frame):
            counts.append(len(items))
            if not items:
                continue
            flat_frame_indices.extend([int(frame_idx)] * len(items))
            flat_emitter_indices.extend(int(item[0]) for item in items)
            flat_overlaps.extend(float(item[1]) for item in items)
    if flat_emitter_indices:
        with time_block("lut_sequence_event_tensors_to_device"):
            emitter_idx = np.asarray(flat_emitter_indices, dtype=np.int64)
            overlaps = np.asarray(flat_overlaps, dtype=np.float32)
            frame_indices = torch.as_tensor(flat_frame_indices, dtype=torch.long, device=render_device)
            xs = torch.as_tensor(xy[emitter_idx, 0], dtype=torch.float32, device=render_device)
            ys = torch.as_tensor(xy[emitter_idx, 1], dtype=torch.float32, device=render_device)
            z_t = torch.as_tensor(z[emitter_idx], dtype=torch.float32, device=render_device)
            photons_t = torch.as_tensor(photons[emitter_idx] * overlaps, dtype=torch.float32, device=render_device)
        patches = _lookup_lut_patches(
            lut_bank,
            xs=xs,
            ys=ys,
            z_um=z_t,
            photons=photons_t,
            field_origin_xy=tuple(int(v) for v in sim_config.field_origin_xy),
        )
        patches = _apply_lut_subpixel_shift(
            patches,
            xs=xs,
            ys=ys,
            subpixel_bins=lut_bank.subpixel_bins,
            chunk_size=int(config.vector_batch_size),
        )
        with time_block("lut_sequence_center_indices"):
            center_x = torch.floor(xs).to(dtype=torch.long)
            center_y = torch.floor(ys).to(dtype=torch.long)
        frames += _project_patches_to_frames(
            patches,
            frame_count=int(frame_count),
            height=height,
            width=width,
            frame_indices=frame_indices,
            center_x=center_x,
            center_y=center_y,
        )
        with time_block("lut_sequence_pxyz_mask_fill"):
            cursor = 0
            for frame_idx, count in enumerate(counts):
                if count <= 0:
                    continue
                frame_slice = slice(cursor, cursor + int(count))
                pxyz[frame_idx, :count, 0] = xs[frame_slice]
                pxyz[frame_idx, :count, 1] = ys[frame_slice]
                pxyz[frame_idx, :count, 2] = z_t[frame_slice]
                pxyz[frame_idx, :count, 3] = photons_t[frame_slice]
                mask[frame_idx, :count] = True
                cursor += int(count)
    with time_block("lut_sequence_bkg_target_init"):
        bkg = (
            torch.as_tensor(bg_values[:, None, None], dtype=torch.float32, device=render_device)
            .expand(int(frame_count), height, width)
            .clone()
        )
    with time_block("lut_sequence_output_device_transfer"):
        if str(config.simulation_output_device).strip().lower() == "cpu":
            frames = frames.cpu()
            pxyz = pxyz.cpu()
            mask = mask.cpu()
            bkg = bkg.cpu()
        elif str(config.simulation_output_device).strip().lower() != "renderer":
            raise ValueError("simulation_output_device must be 'cpu' or 'renderer'")
    return frames, pxyz, mask, bkg, counts


def _target_active_emitters(config: OnlineBatchProviderConfig) -> float:
    if config.emitter_density_um2 is None:
        return float(config.emitters_per_sample)
    area_um2 = (
        float(config.width)
        * float(config.pixel_size_nm_x)
        / 1000.0
        * float(config.height)
        * float(config.pixel_size_nm_y)
        / 1000.0
    )
    return float(config.emitter_density_um2) * area_um2


def _sample_lut_photons(config: OnlineBatchProviderConfig, *, rng: np.random.Generator, count: int) -> np.ndarray:
    if config.photon_mean is not None and config.photon_sigma is not None:
        values = rng.normal(float(config.photon_mean), float(config.photon_sigma), size=int(count)).astype(np.float32)
    else:
        lo, hi = config.photon_range if config.photon_range is not None else (float(config.signal), float(config.signal))
        values = rng.uniform(float(lo), float(hi), size=int(count)).astype(np.float32)
    if config.photon_range is not None:
        lo, hi = config.photon_range
        values = np.clip(values, float(lo), float(hi))
    return values.astype(np.float32, copy=False)


def _sample_lut_background(config: OnlineBatchProviderConfig, *, rng: np.random.Generator, count: int) -> np.ndarray:
    if config.background_range is None:
        return np.full((int(count),), float(config.background), dtype=np.float32)
    lo, hi = config.background_range
    return rng.uniform(float(lo), float(hi), size=int(count)).astype(np.float32)


def _detect_from_v03_targets(target: torch.Tensor, mask: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    detect = torch.zeros((int(height), int(width)), dtype=torch.float32)
    if not bool(mask.any()):
        return detect
    active = target[mask]
    cols = torch.round(active[:, 0]).to(dtype=torch.long).clamp_(0, int(width) - 1)
    rows = torch.round(active[:, 1]).to(dtype=torch.long).clamp_(0, int(height) - 1)
    detect[rows, cols] = 1.0
    return detect


def _condition_feature_dim(config: OnlineBatchProviderConfig) -> int:
    if int(config.condition_feature_dim) > 0:
        return int(config.condition_feature_dim)
    if int(config.condition_dim) > 0:
        domain_terms = int(config.domain_count) if bool(config.append_domain_onehot) else 0
        return max(0, int(config.condition_dim) - domain_terms)
    return 8


def _domain_index(config: OnlineBatchProviderConfig, *, global_step: int) -> int:
    domain_count = int(config.domain_count)
    if domain_count <= 0:
        raise ValueError("domain_count must be positive")
    if str(config.domain_balance_mode) == "alternate_step":
        return int(global_step) % domain_count
    return 0


def _cached_sequence_condition_origin(
    config: OnlineBatchProviderConfig,
    *,
    condition_provider: FullResZernikeConditioning,
    epoch: int,
    sequence_idx: int,
    sequence_count: int,
) -> tuple[int, int]:
    full_h, full_w = condition_provider.image_shape_hw
    roi_w = min(int(config.width), int(full_w))
    roi_h = min(int(config.height), int(full_h))
    max_x0 = max(0, int(full_w) - roi_w)
    max_y0 = max(0, int(full_h) - roi_h)
    grid = config.nat_grid_size
    if isinstance(grid, tuple):
        grid_x = max(1, int(grid[0]))
        grid_y = max(1, int(grid[1] if len(grid) > 1 else grid[0]))
    else:
        grid_x = grid_y = max(1, int(grid))
    cell_count = max(1, grid_x * grid_y)
    cell = (int(epoch) - 1) * int(sequence_count) + int(sequence_idx)
    cell = int(cell) % cell_count
    col = cell % grid_x
    row = cell // grid_x
    x0 = 0 if grid_x <= 1 else round(float(col) * float(max_x0) / float(grid_x - 1))
    y0 = 0 if grid_y <= 1 else round(float(row) * float(max_y0) / float(grid_y - 1))
    return int(x0), int(y0)


def _sliding_window_sequence_condition_origin(
    config: OnlineBatchProviderConfig,
    *,
    condition_provider: FullResZernikeConditioning,
    epoch: int,
    sequence_idx: int,
    sequence_count: int,
) -> tuple[tuple[int, int], int, int]:
    full_h, full_w = condition_provider.image_shape_hw
    roi_w = min(int(config.width), int(full_w))
    roi_h = min(int(config.height), int(full_h))
    origins = build_sliding_window_origin_bank(
        field_width_px=int(full_w),
        field_height_px=int(full_h),
        roi_width_px=roi_w,
        roi_height_px=roi_h,
        stride_px=int(config.field_origin_stride_px),
    )
    order = np.arange(len(origins), dtype=np.int64)
    seed = (int(config.seed) + 1_000_003 * int(epoch)) % (2**63 - 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    offset = ((int(epoch) - 1) * int(sequence_count) + int(sequence_idx)) % len(origins)
    origin_index = int(order[int(offset)])
    return tuple(int(v) for v in origins[origin_index]), origin_index, int(len(origins))


def _random_sequence_condition_origin(
    config: OnlineBatchProviderConfig,
    *,
    condition_provider: FullResZernikeConditioning,
    epoch: int,
    sequence_idx: int,
    domain_index: int,
) -> tuple[int, int]:
    full_h, full_w = condition_provider.image_shape_hw
    roi_w = min(int(config.width), int(full_w))
    roi_h = min(int(config.height), int(full_h))
    max_x0 = max(0, int(full_w) - roi_w)
    max_y0 = max(0, int(full_h) - roi_h)
    seed = (
        int(config.seed)
        + 1_000_003 * int(epoch)
        + 97_531 * int(sequence_idx)
        + 12_289 * int(domain_index)
    ) % (2**63 - 1)
    rng = np.random.default_rng(seed)
    x0 = int(rng.integers(0, max_x0 + 1)) if max_x0 > 0 else 0
    y0 = int(rng.integers(0, max_y0 + 1)) if max_y0 > 0 else 0
    return x0, y0


def _build_condition_features(
    config: OnlineBatchProviderConfig,
    *,
    global_step: int,
    domain_index: int,
    image: torch.Tensor,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
) -> tuple[torch.Tensor, str, str | None]:
    feature_dim = _condition_feature_dim(config)
    provider, condition_source, domain_name = _condition_provider(
        config,
        domain_index=domain_index,
        condition_providers=condition_providers,
    )
    roi_width = _condition_roi_extent(int(config.width), int(config.nat_grid_size[0] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    roi_height = _condition_roi_extent(int(config.height), int(config.nat_grid_size[1] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    vectors = []
    for sample_idx in range(int(config.batch_size)):
        x0, y0 = _condition_origin(config, sample_index=sample_idx, global_step=global_step, roi_width=roi_width, roi_height=roi_height)
        vector = provider.condition_vector_from_xy(
            x0=x0,
            y0=y0,
            height=roi_height,
            width=roi_width,
            device=image.device,
            dtype=image.dtype,
        )
        vectors.append(_match_feature_dim(vector, feature_dim=feature_dim))
    return torch.stack(vectors, dim=0).contiguous(), condition_source, domain_name


def _condition_provider(
    config: OnlineBatchProviderConfig,
    *,
    domain_index: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
) -> tuple[FullResZernikeConditioning, str, str | None]:
    if condition_providers:
        domain_name, provider = condition_providers[int(domain_index) % len(condition_providers)]
        source = "synthetic_nat" if domain_name is None else "dual_domain_coeff_maps"
        return provider, source, domain_name
    return build_default_conditioning_maps(width=int(config.width), height=int(config.height)), "synthetic_nat", None


def _load_condition_providers(
    config: OnlineBatchProviderConfig,
) -> tuple[tuple[str | None, FullResZernikeConditioning], ...] | None:
    return _load_conditioning_providers(config.dual_domain_coeff_maps)


def _condition_roi_extent(axis_size: int, grid_size: int) -> int:
    return max(1, min(int(axis_size), int(axis_size) // max(1, int(grid_size))))


def _condition_origin(
    config: OnlineBatchProviderConfig,
    *,
    sample_index: int,
    global_step: int,
    roi_width: int,
    roi_height: int,
) -> tuple[int, int]:
    grid = config.nat_grid_size
    if isinstance(grid, tuple):
        grid_x = max(1, int(grid[0]))
        grid_y = max(1, int(grid[1] if len(grid) > 1 else grid[0]))
    else:
        grid_x = grid_y = max(1, int(grid))
    cell = (int(global_step) * int(config.batch_size) + int(sample_index)) % (grid_x * grid_y)
    col = cell % grid_x
    row = cell // grid_x
    max_x0 = max(0, int(config.width) - int(roi_width))
    max_y0 = max(0, int(config.height) - int(roi_height))
    x0 = 0 if grid_x <= 1 else round(float(col) * float(max_x0) / float(grid_x - 1))
    y0 = 0 if grid_y <= 1 else round(float(row) * float(max_y0) / float(grid_y - 1))
    return int(x0), int(y0)


def _match_feature_dim(vector: torch.Tensor, *, feature_dim: int) -> torch.Tensor:
    if int(feature_dim) == int(vector.shape[0]):
        return vector
    if int(feature_dim) < int(vector.shape[0]):
        return vector[: int(feature_dim)].contiguous()
    out = torch.zeros((int(feature_dim),), dtype=vector.dtype, device=vector.device)
    out[: int(vector.shape[0])] = vector
    return out
