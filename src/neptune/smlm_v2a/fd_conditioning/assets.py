from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class FDChannelAsset:
    summary_path: Path
    phase_maps: torch.Tensor
    noll_indices: tuple[int, ...]
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    block_span_frames: int

    @property
    def field_shape(self) -> tuple[int, int]:
        return int(self.phase_maps.shape[-2]), int(self.phase_maps.shape[-1])

    @property
    def cell_width_px(self) -> int:
        _, width = self.field_shape
        return int(round(self.crop_width / max(1, width)))

    @property
    def cell_height_px(self) -> int:
        height, _ = self.field_shape
        return int(round(self.crop_height / max(1, height)))

    def conditioning_from_field_xy(
        self,
        field_xy: torch.Tensor | tuple[float, float] | list[float],
        *,
        out_size: tuple[int, int] = (40, 40),
    ) -> torch.Tensor:
        coords = torch.as_tensor(field_xy, dtype=torch.float32)
        if coords.numel() != 2:
            raise ValueError("field_xy must contain exactly two values.")
        height, width = self.field_shape
        x = int(torch.clamp(torch.round(coords[0]), 0, width - 1).item())
        y = int(torch.clamp(torch.round(coords[1]), 0, height - 1).item())
        coeffs = self.phase_maps[:, y, x].to(dtype=torch.float32)
        return coeffs[:, None, None].expand(-1, int(out_size[0]), int(out_size[1])).contiguous()

    def build_phase_payload(self) -> dict[str, object]:
        height, width = self.field_shape
        xs = torch.arange(width, dtype=torch.float32)
        ys = torch.arange(height, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        return {
            "phase_maps": self.phase_maps.detach().cpu().numpy(),
            "coords": coords.numpy(),
        }


def _load_csv_map(path: Path) -> torch.Tensor:
    rows: list[tuple[int, int, int, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    int(row["noll_index"]),
                    int(row["grid_y"]),
                    int(row["grid_x"]),
                    float(row["value"]),
                )
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    noll_values = sorted({item[0] for item in rows})
    noll_to_idx = {noll: idx for idx, noll in enumerate(noll_values)}
    height = max(item[1] for item in rows) + 1
    width = max(item[2] for item in rows) + 1
    out = torch.zeros((len(noll_values), height, width), dtype=torch.float32)
    for noll, y, x, value in rows:
        out[noll_to_idx[noll], y, x] = float(value)
    return out


def _load_block_maps(block_dir: Path) -> list[torch.Tensor]:
    maps: list[torch.Tensor] = []
    for block_path in sorted(block_dir.glob("block_*/field_zernike_map.pt")):
        maps.append(torch.load(block_path, map_location="cpu").to(dtype=torch.float32))
    if maps:
        return maps
    for block_path in sorted(block_dir.glob("block_*/field_zernike_map.csv")):
        maps.append(_load_csv_map(block_path))
    if not maps:
        raise FileNotFoundError(f"No field_zernike_map.pt or .csv files found under {block_dir}")
    return maps


def load_fd_channel_asset(summary_path: str | Path) -> FDChannelAsset:
    summary_path = Path(summary_path).resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    maps = _load_block_maps(summary_path.parent)
    stacked = torch.stack(maps, dim=0)
    mean_maps = stacked.mean(dim=0)
    noll_indices = tuple(int(v) for v in payload.get("noll_indices", []))
    return FDChannelAsset(
        summary_path=summary_path,
        phase_maps=mean_maps.contiguous(),
        noll_indices=noll_indices,
        crop_left=int(payload.get("crop_left", 0)),
        crop_top=int(payload.get("crop_top", 0)),
        crop_width=int(payload.get("crop_width", mean_maps.shape[-1])),
        crop_height=int(payload.get("crop_height", mean_maps.shape[-2])),
        block_span_frames=int(payload.get("block_span_frames", 0)),
    )
