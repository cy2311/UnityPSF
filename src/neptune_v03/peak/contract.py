from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from neptune_v03.runtime.artifacts import ArtifactRegistry
from neptune_v03.runtime.layout import RunLayout, write_stage_status


@dataclass(frozen=True)
class PeakBootstrapConfig:
    sample: str
    side: str
    frame_range: tuple[int, int]
    tiff_path: Path | None = None
    crop_x0: int = 0
    crop_x1: int | None = None
    crop_y0: int = 0
    crop_y1: int | None = None
    max_emitters: int = 1000
    max_candidates: int | None = None
    target_selected_emitters: int = 0
    min_distance_px: float = 15.0
    gaussian_sigma_px: float = 1.0
    threshold_sigma: float = 5.0
    patch_size_px: int = 15
    nat_config_kind: str = "order1"
    alternating_rounds: int = 3
    alternating_local_steps: int = 2
    alternating_global_steps: int = 2
    alternating_local_warmup_rounds: int = 0
    alternating_local_warmup_steps: int = 0
    alternating_optimizer_kind: str = "lm"
    global_projected_min_distance_px: float = 10.0
    spatial_balance_grid_px: int = 100
    spatial_balance_max_per_cell: int = 0
    max_patch_peak_distance_px: float = 2.5
    max_secondary_peak_fraction: float = 0.45
    min_center_peak_norm: float = 0.0
    min_signal_sum_norm: float = 0.0
    ncc_threshold: float = 0.7
    freeze_initial_astig_standard: bool = False
    freeze_defocus_zero_gauge: bool = True
    vectorfit_astig_gauge: bool = True
    vectorfit_astig_anchor_nm: float | None = None
    vectorfit_astig_anchor_mode: str = "init_only"
    vectorfit_phasor_z_init: bool = True
    include_fixed_astig_baseline: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_range"] = [int(self.frame_range[0]), int(self.frame_range[1])]
        payload["tiff_path"] = None if self.tiff_path is None else str(self.tiff_path)
        return payload


@dataclass(frozen=True)
class PeakBootstrapSummary:
    config: PeakBootstrapConfig
    candidate_count: int
    kept_count: int
    selected_emitters: int
    ncc_count: int
    ncc_gt_threshold_count: int
    preferred_export_stage: str
    paths: Mapping[str, Path]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self, run_dir: Path) -> dict[str, Any]:
        ncc_fraction = 0.0 if int(self.ncc_count) <= 0 else int(self.ncc_gt_threshold_count) / int(self.ncc_count)
        return {
            "stage": "peak",
            "config": self.config.to_json_dict(),
            "harvest_summary": {
                "candidate_count": int(self.candidate_count),
                "kept_count": int(self.kept_count),
            },
            "diagnostics_summary": {
                "selected_emitters": int(self.selected_emitters),
                "metrics": dict(self.metrics),
            },
            "ncc_summary": {
                "count": int(self.ncc_count),
                "gt_threshold_count": int(self.ncc_gt_threshold_count),
                "gt_threshold_fraction": float(ncc_fraction),
                "threshold": float(self.config.ncc_threshold),
            },
            "preferred_export_stage": str(self.preferred_export_stage),
            "paths": {name: _display_path(path, run_dir) for name, path in self.paths.items()},
        }


@dataclass(frozen=True)
class PeakBootstrapArtifacts:
    summary_path: Path
    artifacts_path: Path


def build_peak_bootstrap_artifacts(layout: RunLayout, summary: PeakBootstrapSummary) -> PeakBootstrapArtifacts:
    stage_dir = layout.stage_dir("peak")
    stage_dir.mkdir(parents=True, exist_ok=True)
    summary_path = stage_dir / "peak_nat_zmap_summary.json"
    payload = summary.to_json_dict(layout.run_dir)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    registry = ArtifactRegistry(layout)
    _register_peak_artifacts(registry, summary)
    registry.register(stage="peak", name="peak_nat_zmap_summary", kind="json", path=summary_path)
    artifacts_path = registry.write()

    write_stage_status(
        layout,
        "peak",
        "complete",
        {
            "summary_path": _display_path(summary_path, layout.run_dir),
            "kept_count": int(summary.kept_count),
            "selected_emitters": int(summary.selected_emitters),
            "preferred_export_stage": str(summary.preferred_export_stage),
        },
    )
    return PeakBootstrapArtifacts(summary_path=summary_path, artifacts_path=artifacts_path)


def _register_peak_artifacts(registry: ArtifactRegistry, summary: PeakBootstrapSummary) -> None:
    kinds = {
        "harvest": "torch",
        "harvest_summary": "json",
        "diagnostics_summary": "json",
        "export_summary": "json",
        "coeff_map": "npz",
        "zmap": "npz",
    }
    names = {
        "harvest": "harvest",
        "harvest_summary": "peak_harvest_summary",
        "diagnostics_summary": "real_nat_diagnostics_summary",
        "export_summary": "export_nat_zmap_summary",
        "coeff_map": "coeff_map",
        "zmap": "zmap",
    }
    for key in ("harvest", "harvest_summary", "diagnostics_summary", "export_summary", "coeff_map", "zmap"):
        path = summary.paths.get(key)
        if path is not None:
            registry.register(stage="peak", name=names[key], kind=kinds[key], path=path)


def _display_path(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(path)
