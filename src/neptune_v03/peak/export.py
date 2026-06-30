from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neptune_v03.optics import build_named_nat_config, full_roi_coeff_stack_torch

from .contract import PeakBootstrapConfig


@dataclass(frozen=True)
class ExportNATZMapConfig:
    diagnostics_dir: Path
    output_dir: Path
    preferred_stage: str = "alternating"
    export_alternating: bool = True
    export_approximate: bool = False
    export_provisional_scalar_zmap: bool = True
    scalar_mode: str = "non_astig_rms_nm"


@dataclass(frozen=True)
class ExportNATZMapResult:
    config: ExportNATZMapConfig
    mode_order: list[tuple[int, int]]
    image_shape_hw: tuple[int, int]
    alternating_stack_npz: str | None
    approximate_stack_npz: str | None
    preferred_stack_npz: str | None
    provisional_scalar_zmap_npz: str | None
    figures: dict[str, str]
    output_dir: Path

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "config": {
                **asdict(self.config),
                "diagnostics_dir": str(self.config.diagnostics_dir),
                "output_dir": str(self.config.output_dir),
            },
            "mode_order": [[int(n), int(m)] for n, m in self.mode_order],
            "image_shape_hw": [int(self.image_shape_hw[0]), int(self.image_shape_hw[1])],
            "alternating_stack_npz": self.alternating_stack_npz,
            "approximate_stack_npz": self.approximate_stack_npz,
            "preferred_stack_npz": self.preferred_stack_npz,
            "provisional_scalar_zmap_npz": self.provisional_scalar_zmap_npz,
            "figures": self.figures,
            "output_dir": str(self.output_dir),
            "preferred_stage": self.config.preferred_stage,
        }


def run_export_nat_zmap(
    *,
    config: PeakBootstrapConfig,
    diagnostics_dir: Path,
    output_dir: Path,
    preferred_stage: str,
) -> dict[str, Any]:
    result = export_nat_zmap(
        ExportNATZMapConfig(
            diagnostics_dir=diagnostics_dir,
            output_dir=output_dir,
            preferred_stage=preferred_stage,
        ),
        peak_config=config,
    )
    summary_path = output_dir / "export_nat_zmap_summary.json"
    return {
        "summary_path": summary_path,
        "summary": result.to_serializable_dict(),
        "coeff_map_path": Path(result.alternating_stack_npz or result.preferred_stack_npz or result.approximate_stack_npz),
        "zmap_path": Path(result.provisional_scalar_zmap_npz),
    }


def export_nat_zmap(config: ExportNATZMapConfig, *, peak_config: PeakBootstrapConfig) -> ExportNATZMapResult:
    diagnostics_dir = Path(config.diagnostics_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((diagnostics_dir / "real_nat_diagnostics_summary.json").read_text(encoding="utf-8"))
    payload = torch.load(diagnostics_dir / "real_nat_diagnostics_payload.pt", map_location="cpu", weights_only=False)
    height, width = _image_shape_from_config(peak_config=peak_config, summary=summary)
    nat_config = _nat_config_from_summary(summary=summary, width=width, height=height)
    mode_order = list(nat_config.aberrations)

    alternating_stack_path: str | None = None
    approximate_stack_path: str | None = None
    preferred_stack_path: str | None = None
    provisional_scalar_path: str | None = None
    alternating_stack: np.ndarray | None = None

    if bool(config.export_alternating):
        alternating_stack = _evaluate_coeff_stack(payload["alternating"], nat_config=nat_config)
        path = output_dir / "alternating_full_roi_zernike_maps_nm.npz"
        _save_coeff_stack(path, stack_nm=alternating_stack, mode_order=mode_order)
        alternating_stack_path = str(path)

    if bool(config.export_approximate):
        approximate_stack = _evaluate_coeff_stack(payload["approximate"], nat_config=nat_config)
        path = output_dir / "approximate_full_roi_zernike_maps_nm.npz"
        _save_coeff_stack(path, stack_nm=approximate_stack, mode_order=mode_order)
        approximate_stack_path = str(path)

    preferred = str(config.preferred_stage).strip().lower()
    if preferred not in {"alternating", "approximate"}:
        preferred = "alternating"
    preferred_source = approximate_stack_path if preferred == "approximate" else alternating_stack_path
    if preferred_source is not None:
        preferred_path = output_dir / "preferred_full_roi_zernike_maps_nm.npz"
        shutil.copy2(Path(preferred_source), preferred_path)
        preferred_stack_path = str(preferred_path)

    if bool(config.export_provisional_scalar_zmap):
        if alternating_stack is None:
            alternating_stack = _evaluate_coeff_stack(payload["alternating"], nat_config=nat_config)
        scalar = _compute_scalar_map(alternating_stack, scalar_mode=str(config.scalar_mode), mode_order=mode_order)
        zmap_path = output_dir / f"provisional_{config.scalar_mode}.npz"
        np.savez_compressed(
            zmap_path,
            zmap_nm=scalar.astype(np.float32, copy=False),
            image_shape_hw=np.asarray([height, width], dtype=np.int64),
            value_range_nm=np.asarray([float(np.nanmin(scalar)), float(np.nanmax(scalar))], dtype=np.float32),
            source=np.asarray([f"nat_{config.scalar_mode}_provisional"]),
            continuous=np.asarray([True]),
            pixel_aligned=np.asarray([True]),
        )
        provisional_scalar_path = str(zmap_path)

    result = ExportNATZMapResult(
        config=config,
        mode_order=mode_order,
        image_shape_hw=(height, width),
        alternating_stack_npz=alternating_stack_path,
        approximate_stack_npz=approximate_stack_path,
        preferred_stack_npz=preferred_stack_path,
        provisional_scalar_zmap_npz=provisional_scalar_path,
        figures={},
        output_dir=output_dir,
    )
    (output_dir / "export_nat_zmap_summary.json").write_text(
        json.dumps(result.to_serializable_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _image_shape_from_config(*, peak_config: PeakBootstrapConfig, summary: dict[str, Any]) -> tuple[int, int]:
    cfg = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    width = None
    height = None
    if peak_config.crop_x1 is not None:
        width = int(peak_config.crop_x1) - int(peak_config.crop_x0)
    if peak_config.crop_y1 is not None:
        height = int(peak_config.crop_y1) - int(peak_config.crop_y0)
    if width is None:
        width = cfg.get("crop_x1") or cfg.get("image_width_px") or 600
    if height is None:
        height = cfg.get("crop_y1") or cfg.get("image_height_px") or 1200
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError(f"invalid peak export crop shape: height={height!r}, width={width!r}")
    return int(height), int(width)


def _nat_config_from_summary(*, summary: dict[str, Any], width: int, height: int):
    cfg = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    return build_named_nat_config(
        str(cfg.get("nat_config_kind", "order1")),
        img_size_x=int(width),
        img_size_y=int(height),
        pixel_size_x_nm=float(cfg.get("pixel_size_x_nm", 95.0)),
        pixel_size_y_nm=float(cfg.get("pixel_size_y_nm", 95.0)),
    )


def _evaluate_coeff_stack(stage: dict[str, Any], *, nat_config) -> np.ndarray:
    gamma = torch.as_tensor(_stage_get(stage, "gamma", torch.zeros((1,))), dtype=torch.float32).reshape(-1)
    if gamma.numel() != len(nat_config.gammas):
        expanded = torch.zeros((len(nat_config.gammas),), dtype=torch.float32)
        expanded[: min(gamma.numel(), expanded.numel())] = gamma[: min(gamma.numel(), expanded.numel())]
        gamma = expanded
    return full_roi_coeff_stack_torch(gamma, nat_config).maps_nm.detach().cpu().numpy().astype(np.float32, copy=False)


def _stage_get(stage: Any, name: str, default: Any) -> Any:
    if isinstance(stage, dict):
        return stage.get(name, default)
    return getattr(stage, name, default)


def _save_coeff_stack(path: Path, *, stack_nm: np.ndarray, mode_order: list[tuple[int, int]]) -> None:
    np.savez_compressed(
        path,
        zernike_maps_nm=stack_nm.astype(np.float32, copy=False),
        mode_order=np.asarray(mode_order, dtype=np.int64),
    )


def _compute_scalar_map(stack_nm: np.ndarray, *, scalar_mode: str, mode_order: list[tuple[int, int]]) -> np.ndarray:
    if scalar_mode == "all_mode_rms_nm":
        return np.sqrt(np.sum(stack_nm**2, axis=0, dtype=np.float32), dtype=np.float32)
    if scalar_mode == "non_astig_rms_nm":
        keep = [idx for idx, mode in enumerate(mode_order) if mode not in {(2, 2), (2, -2)}]
        if not keep:
            keep = list(range(int(stack_nm.shape[0])))
        return np.sqrt(np.sum(stack_nm[np.asarray(keep, dtype=np.int64)] ** 2, axis=0, dtype=np.float32), dtype=np.float32)
    raise ValueError(f"Unsupported scalar_mode: {scalar_mode!r}")


__all__ = [
    "ExportNATZMapConfig",
    "ExportNATZMapResult",
    "export_nat_zmap",
    "run_export_nat_zmap",
]
