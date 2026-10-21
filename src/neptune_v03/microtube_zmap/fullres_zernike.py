from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .optics.zernike import noll_to_nm


INPUT_MODE_ORDER: tuple[tuple[int, int], ...] = (
    (2, 0),
    (3, 1),
    (3, -1),
    (4, 0),
    (3, -3),
    (3, 3),
)


@dataclass(frozen=True)
class FullResZernikeStats:
    mode_order: list[tuple[int, int]]
    input_mode_order: list[tuple[int, int]]
    input_channel_mean_nm: list[float]
    input_channel_std_nm: list[float]
    image_shape_hw: tuple[int, int]
    condition_vector_dim: int
    includes_normalized_xy: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "mode_order": [[int(n), int(m)] for n, m in self.mode_order],
            "input_mode_order": [[int(n), int(m)] for n, m in self.input_mode_order],
        }


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
    def from_npz(cls, path: str | Path) -> "FullResZernikeConditioning":
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

    @staticmethod
    def _zernike_normfac(mode_order: list[tuple[int, int]]) -> torch.Tensor:
        values = []
        for n, m in mode_order:
            values.append(math.sqrt(2.0 * (float(n) + 1.0) / (1.0 + float(m == 0))))
        return torch.tensor(values, dtype=torch.float32)

    def phase_maps_for_vector_renderer(
        self,
        *,
        wavelength_nm: float = 660.0,
        noll_start: int = 4,
        noll_end: int = 12,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        noll_modes = [noll_to_nm(j) for j in range(int(noll_start), int(noll_end))]
        missing = [mode for mode in noll_modes if mode not in self.mode_to_index]
        if missing:
            raise ValueError(f"Missing Noll modes from full-res zernike maps: {missing!r}")
        indices = [self.mode_to_index[mode] for mode in noll_modes]
        maps_nm = self.full_maps_nm[indices]
        normfac = self._zernike_normfac(noll_modes).reshape(-1, 1, 1)
        maps_rad = maps_nm * (2.0 * math.pi / float(wavelength_nm)) * normfac
        return maps_rad.contiguous(), noll_modes

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

    def build_phase_payload(
        self,
        *,
        wavelength_nm: float = 660.0,
        noll_start: int = 4,
        noll_end: int = 12,
    ) -> dict[str, object]:
        height, width = self.image_shape_hw
        xs = torch.arange(width, dtype=torch.float32)
        ys = torch.arange(height, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        phase_maps, vector_mode_order = self.phase_maps_for_vector_renderer(
            wavelength_nm=float(wavelength_nm),
            noll_start=int(noll_start),
            noll_end=int(noll_end),
        )
        return {
            "phase_maps": phase_maps.detach().cpu().numpy(),
            "coords": coords.numpy(),
            "phase_units": "radian",
            "source_units": "nm",
            "source_mode_order": np.asarray(self.mode_order, dtype=np.int64),
            "mode_order": np.asarray(vector_mode_order, dtype=np.int64),
        }

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
        patch = self.input_patch_from_xy(
            x0=x0,
            y0=y0,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
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
        xy = self.normalized_xy_from_xy(
            x0=x0,
            y0=y0,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        coeff = self.input_summary_vector_from_xy(
            x0=x0,
            y0=y0,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
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

    def save_stats_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.stats().to_json_dict(), indent=2) + "\n", encoding="utf-8")


__all__ = ["FullResZernikeConditioning", "FullResZernikeStats", "INPUT_MODE_ORDER"]
