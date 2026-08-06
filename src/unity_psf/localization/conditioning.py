from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Mapping

import numpy as np
import torch

from unity_psf.optics import build_named_nat_config, full_roi_coeff_stack_torch, resolve_astigmatism_anchor_profile


INPUT_MODE_ORDER: tuple[tuple[int, int], ...] = (
    (2, 0),
    (3, 1),
    (3, -1),
    (4, 0),
    (3, -3),
    (3, 3),
)


def condition_feature_order(*, include_domain_onehot: bool = False, domain_count: int = 0) -> tuple[str, ...]:
    names = ["x_norm", "y_norm"]
    names.extend(f"zernike_nm_mean:n{int(n)}_m{int(m)}" for n, m in INPUT_MODE_ORDER)
    if include_domain_onehot:
        names.extend(f"domain_onehot:{idx}" for idx in range(int(domain_count)))
    return tuple(names)


@dataclass(frozen=True)
class FullResZernikeStats:
    mode_order: list[tuple[int, int]]
    input_mode_order: list[tuple[int, int]]
    input_channel_mean_nm: list[float]
    input_channel_std_nm: list[float]
    image_shape_hw: tuple[int, int]
    condition_vector_dim: int
    includes_normalized_xy: bool


class FullResZernikeConditioning:
    def __init__(self, *, full_maps_nm: torch.Tensor, mode_order: list[tuple[int, int]]) -> None:
        if full_maps_nm.ndim != 3:
            raise ValueError(f"Expected full_maps_nm as (C,H,W), got {tuple(full_maps_nm.shape)}")
        self.full_maps_nm = full_maps_nm.to(dtype=torch.float32).contiguous()
        self.mode_order = tuple((int(n), int(m)) for n, m in mode_order)
        self.mode_to_index = {mode: idx for idx, mode in enumerate(self.mode_order)}
        self.input_mode_order = tuple(mode for mode in INPUT_MODE_ORDER if mode in self.mode_to_index)
        if len(self.input_mode_order) != len(INPUT_MODE_ORDER):
            raise ValueError(f"Missing input modes from mode_order={self.mode_order!r}")
        self.input_indices = [self.mode_to_index[mode] for mode in self.input_mode_order]
        input_maps = self.full_maps_nm[self.input_indices]
        self.input_mean_nm = input_maps.mean(dim=(1, 2)).reshape(-1, 1, 1).contiguous()
        self.input_std_nm = input_maps.std(dim=(1, 2), unbiased=False).clamp_min(1e-6).reshape(-1, 1, 1).contiguous()

    @classmethod
    def from_npz(cls, path: str | Path) -> FullResZernikeConditioning:
        payload = np.load(Path(path))
        full_maps = torch.from_numpy(np.asarray(payload["zernike_maps_nm"], dtype=np.float32))
        mode_order = [tuple(int(v) for v in row.tolist()) for row in np.asarray(payload["mode_order"], dtype=np.int64)]
        return cls(full_maps_nm=full_maps, mode_order=mode_order)

    @property
    def image_shape_hw(self) -> tuple[int, int]:
        return int(self.full_maps_nm.shape[1]), int(self.full_maps_nm.shape[2])

    @property
    def num_input_channels(self) -> int:
        return len(self.input_indices)

    @property
    def condition_vector_dim(self) -> int:
        return 2 + self.num_input_channels

    def normalized_xy_from_xy(
        self,
        *,
        x0: int,
        y0: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        full_h, full_w = self.image_shape_hw
        cx = float(x0) + (float(width) - 1.0) / 2.0
        cy = float(y0) + (float(height) - 1.0) / 2.0
        x_norm = 0.0 if full_w <= 1 else (2.0 * cx / float(full_w - 1)) - 1.0
        y_norm = 0.0 if full_h <= 1 else (2.0 * cy / float(full_h - 1)) - 1.0
        xy = torch.tensor([x_norm, y_norm], dtype=dtype)
        if device is not None:
            xy = xy.to(device=device)
        return xy.contiguous()

    def input_patch_from_xy(
        self,
        *,
        x0: int,
        y0: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        x0_i = int(x0)
        y0_i = int(y0)
        h_i = int(height)
        w_i = int(width)
        full_h, full_w = self.image_shape_hw
        if x0_i < 0 or y0_i < 0 or x0_i + w_i > full_w or y0_i + h_i > full_h:
            raise ValueError(f"Conditioning patch {(x0_i, y0_i, h_i, w_i)} exceeds map {(full_h, full_w)}.")
        patch = self.full_maps_nm[self.input_indices, y0_i : y0_i + h_i, x0_i : x0_i + w_i]
        patch = (patch - self.input_mean_nm) / self.input_std_nm
        if device is not None:
            patch = patch.to(device=device, dtype=dtype)
        else:
            patch = patch.to(dtype=dtype)
        return patch.contiguous()

    def input_summary_vector_from_xy(
        self,
        *,
        x0: int,
        y0: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        patch = self.input_patch_from_xy(x0=x0, y0=y0, height=height, width=width, device=device, dtype=dtype)
        return patch.mean(dim=(1, 2)).contiguous()

    def condition_vector_from_xy(
        self,
        *,
        x0: int,
        y0: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        xy = self.normalized_xy_from_xy(x0=x0, y0=y0, height=height, width=width, device=device, dtype=dtype)
        coeff = self.input_summary_vector_from_xy(x0=x0, y0=y0, height=height, width=width, device=device, dtype=dtype)
        return torch.cat((xy, coeff), dim=0).contiguous()

    def stats(self) -> FullResZernikeStats:
        return FullResZernikeStats(
            mode_order=list(self.mode_order),
            input_mode_order=list(self.input_mode_order),
            input_channel_mean_nm=[float(v) for v in self.input_mean_nm.reshape(-1).tolist()],
            input_channel_std_nm=[float(v) for v in self.input_std_nm.reshape(-1).tolist()],
            image_shape_hw=self.image_shape_hw,
            condition_vector_dim=self.condition_vector_dim,
            includes_normalized_xy=True,
        )


class ConditioningProviderStore:
    def __init__(self, providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None = None) -> None:
        self._providers = providers
        self._version = 0
        self._listeners: list[
            Callable[[int, tuple[tuple[str | None, FullResZernikeConditioning], ...] | None], None]
        ] = []

    @classmethod
    def from_coeff_maps(cls, coeff_maps: tuple[Mapping[str, str], ...] | list[Mapping[str, str]]):
        return cls(_load_conditioning_providers(coeff_maps))

    @property
    def version(self) -> int:
        return int(self._version)

    def snapshot(self) -> tuple[int, tuple[tuple[str | None, FullResZernikeConditioning], ...] | None]:
        return self.version, self._providers

    def add_update_listener(
        self,
        listener: Callable[[int, tuple[tuple[str | None, FullResZernikeConditioning], ...] | None], None],
    ) -> None:
        self._listeners.append(listener)

    def update_from_coeff_maps(self, coeff_maps: tuple[Mapping[str, str], ...] | list[Mapping[str, str]]) -> None:
        self._providers = _load_conditioning_providers(coeff_maps)
        self._version += 1
        self._notify_update_listeners()

    def mark_updated(self) -> None:
        self._version += 1
        self._notify_update_listeners()

    def restore_version(self, version: int) -> None:
        """Restore a persisted provider generation after loading a checkpoint."""

        if isinstance(version, bool) or int(version) < 0:
            raise ValueError("conditioning provider version must be a non-negative integer")
        self._version = int(version)
        self._notify_update_listeners()

    def _notify_update_listeners(self) -> None:
        for listener in tuple(self._listeners):
            listener(self.version, self._providers)


def _load_conditioning_providers(
    coeff_maps: tuple[Mapping[str, str], ...] | list[Mapping[str, str]],
) -> tuple[tuple[str | None, FullResZernikeConditioning], ...] | None:
    if not coeff_maps:
        return None
    providers = []
    for idx, item in enumerate(coeff_maps):
        path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
        if path is None:
            raise ValueError("dual_domain_coeff_maps entries must include coeff_maps_npz, alternating_coeff_maps_npz, or path")
        providers.append((str(item.get("name", f"domain{idx}")), FullResZernikeConditioning.from_npz(str(path))))
    return tuple(providers)


def build_default_conditioning_maps(*, width: int, height: int) -> FullResZernikeConditioning:
    nat_config = build_named_nat_config("order1", img_size_x=int(width), img_size_y=int(height))
    gamma = torch.linspace(0.5, 1.2, steps=len(nat_config.gammas), dtype=torch.float32)
    stack = full_roi_coeff_stack_torch(gamma, nat_config)
    return FullResZernikeConditioning(full_maps_nm=stack.maps_nm, mode_order=stack.mode_order)


def build_zero_conditioning_maps(*, width: int, height: int) -> FullResZernikeConditioning:
    """Return a focal, aberration-free profile with the complete FiLM schema."""

    return FullResZernikeConditioning(
        full_maps_nm=torch.zeros((len(INPUT_MODE_ORDER), int(height), int(width)), dtype=torch.float32),
        mode_order=list(INPUT_MODE_ORDER),
    )


def build_astigmatism_anchor_conditioning_maps(
    *,
    width: int,
    height: int,
    profile_name: str | None = None,
) -> FullResZernikeConditioning:
    profile = resolve_astigmatism_anchor_profile(profile_name)
    nat_config = build_named_nat_config("order1", img_size_x=int(width), img_size_y=int(height))
    maps = torch.zeros((len(nat_config.aberrations), int(height), int(width)), dtype=torch.float32)
    astigmatism_index = nat_config.aberrations.index((2, 2))
    maps[astigmatism_index].fill_(float(profile.anchor_nm))
    return FullResZernikeConditioning(full_maps_nm=maps, mode_order=list(nat_config.aberrations))


__all__ = [
    "FullResZernikeConditioning",
    "FullResZernikeStats",
    "ConditioningProviderStore",
    "INPUT_MODE_ORDER",
    "build_astigmatism_anchor_conditioning_maps",
    "build_default_conditioning_maps",
    "build_zero_conditioning_maps",
    "condition_feature_order",
]
