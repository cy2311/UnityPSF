from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from unity_psf.localization.simulator import (
    LocalizationSimulatorConfig,
    _build_emitter_renderer,
    _fourier_shift_patches,
    _place_patches_frames,
)
from unity_psf.runtime.profiling import time_block
from unity_psf.runtime.environment import get_env

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
        renderer = _build_emitter_renderer(config)
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
    carrier = config.pupil_carrier_complex
    carrier_key = (
        None
        if carrier is None
        else (int(carrier.data_ptr()), tuple(int(v) for v in carrier.shape), str(carrier.dtype))
    )
    return (
        None if physical_model_version is None else int(physical_model_version),
        str(config.psf_type).strip().lower(),
        config.empirical_psf_path,
        config.empirical_psf_channel,
        config.empirical_psf_focus_index,
        maps_key,
        carrier_key,
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
        None if config.zemit0 is None else float(config.zemit0),
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
        field_x = torch.floor(local_x_input).clamp(float(bank.tile_x[0].item()), float(bank.tile_x[-1].item()))
        field_y = torch.floor(local_y_input).clamp(float(bank.tile_y[0].item()), float(bank.tile_y[-1].item()))
        x0_idx, x1_idx, wx = _linear_lut_indices_and_weight(field_x, bank.tile_x)
        y0_idx, y1_idx, wy = _linear_lut_indices_and_weight(field_y, bank.tile_y)
        z0_idx, z1_idx, wz = _linear_lut_indices_and_weight(z, bank.z_bins)
        local_x = local_x_input - (torch.floor(local_x_input) + 0.5)
        local_y = local_y_input - (torch.floor(local_y_input) + 0.5)
        sub_x_idx = torch.argmin(torch.abs(local_x[:, None] - bank.subpixel_bins[None, :]), dim=1)
        sub_y_idx = torch.argmin(torch.abs(local_y[:, None] - bank.subpixel_bins[None, :]), dim=1)
        p000 = bank.patches[y0_idx, x0_idx, z0_idx, sub_y_idx, sub_x_idx].to(dtype=torch.float32)
        p001 = bank.patches[y0_idx, x0_idx, z1_idx, sub_y_idx, sub_x_idx].to(dtype=torch.float32)
        p010 = bank.patches[y0_idx, x1_idx, z0_idx, sub_y_idx, sub_x_idx].to(dtype=torch.float32)
        p011 = bank.patches[y0_idx, x1_idx, z1_idx, sub_y_idx, sub_x_idx].to(dtype=torch.float32)
        p100 = bank.patches[y1_idx, x0_idx, z0_idx, sub_y_idx, sub_x_idx].to(dtype=torch.float32)
        p101 = bank.patches[y1_idx, x0_idx, z1_idx, sub_y_idx, sub_x_idx].to(dtype=torch.float32)
        p110 = bank.patches[y1_idx, x1_idx, z0_idx, sub_y_idx, sub_x_idx].to(dtype=torch.float32)
        p111 = bank.patches[y1_idx, x1_idx, z1_idx, sub_y_idx, sub_x_idx].to(dtype=torch.float32)
        wx = wx.view(-1, 1, 1)
        wy = wy.view(-1, 1, 1)
        wz = wz.view(-1, 1, 1)
        p00 = p000 * (1.0 - wz) + p001 * wz
        p01 = p010 * (1.0 - wz) + p011 * wz
        p10 = p100 * (1.0 - wz) + p101 * wz
        p11 = p110 * (1.0 - wz) + p111 * wz
        top = p00 * (1.0 - wx) + p01 * wx
        bottom = p10 * (1.0 - wx) + p11 * wx
        patches = top * (1.0 - wy) + bottom * wy
        return patches * phot[:, None, None]


def _linear_lut_indices_and_weight(values: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = grid.to(device=values.device, dtype=torch.float32).reshape(-1)
    values = values.to(device=grid.device, dtype=torch.float32).reshape(-1)
    if int(grid.numel()) <= 1:
        idx = torch.zeros_like(values, dtype=torch.long)
        return idx, idx, torch.zeros_like(values, dtype=torch.float32)
    v = values.clamp(float(grid[0].item()), float(grid[-1].item()))
    hi = torch.searchsorted(grid, v, right=False)
    hi = torch.clamp(hi, 0, int(grid.numel()) - 1)
    lo = torch.clamp(hi - 1, 0, int(grid.numel()) - 1)
    g0 = grid[lo]
    g1 = grid[hi]
    weight = torch.where(g1 > g0, (v - g0) / (g1 - g0), torch.zeros_like(v))
    return lo, hi, weight


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
    raw = str(value or get_env("UNITY_V04_LUT_SHIFT_BACKEND", "fourier")).strip().lower()
    aliases = {
        "fft": "fourier",
        "fourier_shift": "fourier",
        "triton": "triton_bilinear",
        "bilinear_triton": "triton_bilinear",
    }
    raw = aliases.get(raw, raw)
    if raw not in {"fourier", "triton_bilinear"}:
        raise ValueError("UNITY_V04_LUT_SHIFT_BACKEND must be 'fourier' or 'triton_bilinear'")
    if raw == "triton_bilinear" and not _TRITON_AVAILABLE:
        raise RuntimeError("UNITY_V04_LUT_SHIFT_BACKEND=triton_bilinear requested, but Triton is unavailable")
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
    raw = str(value or get_env("UNITY_V04_PROJECTION_BACKEND", "formal")).strip().lower()
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
        raise ValueError("UNITY_V04_PROJECTION_BACKEND must be 'formal' or 'triton_fused'")
    if raw == "triton_fused" and not _TRITON_AVAILABLE:
        raise RuntimeError("UNITY_V04_PROJECTION_BACKEND=triton_fused requested, but Triton is unavailable")
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


def _resolve_lut_field_mode(config) -> str:
    raw = str(config.lut_field_mode or get_env("UNITY_V04_LUT_FIELD_MODE", "roi_origin")).strip().lower()
    aliases = {"roi": "roi_origin", "roi_bank": "roi_origin", "roi_origin_bank": "roi_origin", "global": "global_field", "full_field": "global_field"}
    mode = aliases.get(raw, raw)
    if mode not in {"roi_origin", "global_field"}:
        raise ValueError("lut_field_mode must be 'roi_origin' or 'global_field'")
    return mode


def _resolve_lut_storage_dtype(config) -> str:
    raw = str(config.lut_storage_dtype or get_env("UNITY_V04_LUT_STORAGE_DTYPE", "fp32")).strip().lower()
    aliases = {"float16": "fp16", "half": "fp16", "float32": "fp32", "single": "fp32"}
    dtype = aliases.get(raw, raw)
    if dtype not in {"fp16", "fp32"}:
        raise ValueError("lut_storage_dtype must be 'fp16' or 'fp32'")
    return dtype


def _lut_storage_torch_dtype(config, *, device: torch.device) -> torch.dtype:
    del device
    return torch.float16 if _resolve_lut_storage_dtype(config) == "fp16" else torch.float32
