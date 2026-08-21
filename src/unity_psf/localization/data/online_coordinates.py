"""Conditioning provider selection and field-origin scheduling for online batches."""

from __future__ import annotations

import numpy as np
import torch

from unity_psf.field_origin import build_sliding_window_origin_bank
from unity_psf.localization.conditioning import (
    FullResZernikeConditioning,
    _load_conditioning_providers,
    build_astigmatism_anchor_conditioning_maps,
    build_default_conditioning_maps,
    build_zero_conditioning_maps,
)

from .online_conditioning import condition_feature_dim, condition_vector_for_config


def resolve_field_origin_sampling_mode(config) -> str:
    raw = str(config.field_origin_sampling_mode or "grid").strip().lower()
    aliases = {
        "fixed_grid": "grid",
        "cached_grid": "grid",
        "center_grid": "cell_center_grid",
        "centered_grid": "cell_center_grid",
        "random_step": "per_step_random",
        "step_random": "per_step_random",
        "sliding": "sliding_window",
        "fd_deeploc": "sliding_window",
        "fd-deeploc": "sliding_window",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"grid", "cell_center_grid", "per_step_random", "sliding_window"}:
        raise ValueError(
            "field_origin_sampling_mode must be 'grid', 'cell_center_grid', "
            "'per_step_random', or 'sliding_window'"
        )
    return mode


def cached_sequence_condition_origin(config, *, condition_provider, epoch: int, sequence_idx: int, sequence_count: int) -> tuple[int, int]:
    full_h, full_w = condition_provider.image_shape_hw
    roi_w, roi_h = min(int(config.width), int(full_w)), min(int(config.height), int(full_h))
    grid_x, grid_y = _grid_shape(config)
    cell = ((int(epoch) - 1) * int(sequence_count) + int(sequence_idx)) % max(1, grid_x * grid_y)
    mode = resolve_field_origin_sampling_mode(config)
    return (
        grid_axis_origin(cell_index=cell % grid_x, grid_size=grid_x, full_size=int(full_w), roi_size=roi_w, centered=mode == "cell_center_grid"),
        grid_axis_origin(cell_index=cell // grid_x, grid_size=grid_y, full_size=int(full_h), roi_size=roi_h, centered=mode == "cell_center_grid"),
    )


def sliding_window_sequence_condition_origin(config, *, condition_provider, epoch: int, sequence_idx: int, sequence_count: int) -> tuple[tuple[int, int], int, int]:
    full_h, full_w = condition_provider.image_shape_hw
    origins = build_sliding_window_origin_bank(
        field_width_px=int(full_w), field_height_px=int(full_h),
        roi_width_px=min(int(config.width), int(full_w)), roi_height_px=min(int(config.height), int(full_h)),
        stride_px=int(config.field_origin_stride_px),
    )
    order = np.arange(len(origins), dtype=np.int64)
    rng = np.random.default_rng((int(config.seed) + 1_000_003 * int(epoch)) % (2**63 - 1))
    rng.shuffle(order)
    origin_index = int(order[((int(epoch) - 1) * int(sequence_count) + int(sequence_idx)) % len(origins)])
    return tuple(int(v) for v in origins[origin_index]), origin_index, int(len(origins))


def random_sequence_condition_origin(config, *, condition_provider, epoch: int, sequence_idx: int, domain_index: int) -> tuple[int, int]:
    full_h, full_w = condition_provider.image_shape_hw
    max_x0 = max(0, int(full_w) - min(int(config.width), int(full_w)))
    max_y0 = max(0, int(full_h) - min(int(config.height), int(full_h)))
    seed = (int(config.seed) + 1_000_003 * int(epoch) + 97_531 * int(sequence_idx) + 12_289 * int(domain_index)) % (2**63 - 1)
    rng = np.random.default_rng(seed)
    return (int(rng.integers(0, max_x0 + 1)) if max_x0 else 0, int(rng.integers(0, max_y0 + 1)) if max_y0 else 0)


def build_condition_features(config, *, global_step: int, domain_index: int, image: torch.Tensor, condition_providers):
    provider, condition_source, domain_name = condition_provider(config, domain_index=domain_index, condition_providers=condition_providers)
    grid_x, grid_y = _grid_shape(config)
    roi_width, roi_height = condition_roi_extent(int(config.width), grid_x), condition_roi_extent(int(config.height), grid_y)
    vectors = []
    for sample_idx in range(int(config.batch_size)):
        x0, y0 = condition_origin(config, sample_index=sample_idx, global_step=global_step, roi_width=roi_width, roi_height=roi_height)
        vector = provider.condition_vector_from_xy(x0=x0, y0=y0, height=roi_height, width=roi_width, device=image.device, dtype=image.dtype)
        vectors.append(condition_vector_for_config(config, vector))
    return torch.stack(vectors, dim=0).contiguous(), condition_source, domain_name


def condition_provider(config, *, domain_index: int, condition_providers):
    if condition_providers:
        domain_name, provider = condition_providers[int(domain_index) % len(condition_providers)]
        return provider, conditioning_profile_source(config) if domain_name is None else "dual_domain_coeff_maps", domain_name
    provider = default_condition_provider(config)
    return provider, conditioning_profile_source(config), None


def conditioning_profile_source(config) -> str:
    profile = str(config.conditioning_profile).strip().lower().replace("-", "_")
    if profile in {"zero", "zero_aberration", "focal_2d"}:
        return "zero_profile"
    if profile.startswith("astigmatism_660nm"):
        return "astigmatism_anchor_profile"
    return "synthetic_nat"


def default_condition_provider(config) -> FullResZernikeConditioning:
    profile = str(config.conditioning_profile).strip().lower().replace("-", "_")
    if profile in {"default", "default_nat", "synthetic_nat"}:
        return build_default_conditioning_maps(width=int(config.width), height=int(config.height))
    if profile in {"zero", "zero_aberration", "focal_2d"}:
        return build_zero_conditioning_maps(width=int(config.width), height=int(config.height))
    return build_astigmatism_anchor_conditioning_maps(width=int(config.width), height=int(config.height), profile_name=profile)


def load_condition_providers(config):
    return _load_conditioning_providers(config.dual_domain_coeff_maps)


def condition_roi_extent(axis_size: int, grid_size: int) -> int:
    return max(1, min(int(axis_size), int(axis_size) // max(1, int(grid_size))))


def grid_axis_origin(*, cell_index: int, grid_size: int, full_size: int, roi_size: int, centered: bool) -> int:
    max_origin = max(0, int(full_size) - int(roi_size))
    if centered:
        cell_center = (float(cell_index) + 0.5) * float(full_size) / float(grid_size)
        return int(min(max_origin, max(0, round(cell_center - float(roi_size) / 2.0))))
    if int(grid_size) <= 1:
        return 0
    return int(round(float(cell_index) * float(max_origin) / float(grid_size - 1)))


def condition_origin(config, *, sample_index: int, global_step: int, roi_width: int, roi_height: int) -> tuple[int, int]:
    grid_x, grid_y = _grid_shape(config)
    cell = (int(global_step) * int(config.batch_size) + int(sample_index)) % (grid_x * grid_y)
    centered = resolve_field_origin_sampling_mode(config) == "cell_center_grid"
    return (
        grid_axis_origin(cell_index=cell % grid_x, grid_size=grid_x, full_size=int(config.width), roi_size=roi_width, centered=centered),
        grid_axis_origin(cell_index=cell // grid_x, grid_size=grid_y, full_size=int(config.height), roi_size=roi_height, centered=centered),
    )


def _grid_shape(config) -> tuple[int, int]:
    grid = config.nat_grid_size
    if isinstance(grid, tuple):
        return max(1, int(grid[0])), max(1, int(grid[1] if len(grid) > 1 else grid[0]))
    return max(1, int(grid)), max(1, int(grid))
