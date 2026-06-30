from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from neptune_v03.runtime.layout import RunLayout

from .backend import DefaultPeakBackend
from .contract import (
    PeakBootstrapArtifacts,
    PeakBootstrapConfig,
    PeakBootstrapSummary,
    build_peak_bootstrap_artifacts,
)


class PeakPipelineBackend(Protocol):
    def harvest_peaks(self, *, config: PeakBootstrapConfig, output_dir: Path) -> dict[str, Any]:
        ...

    def fit_nat_lm(
        self,
        *,
        config: PeakBootstrapConfig,
        harvest_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        ...

    def summarize_ncc(
        self,
        *,
        config: PeakBootstrapConfig,
        diagnostics: dict[str, Any],
        preferred_stage: str,
    ) -> dict[str, Any]:
        ...

    def export_coeff_maps(
        self,
        *,
        config: PeakBootstrapConfig,
        diagnostics_dir: Path,
        output_dir: Path,
        preferred_stage: str,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PeakPipelineResult:
    summary: PeakBootstrapSummary
    artifacts: PeakBootstrapArtifacts
    harvest: dict[str, Any]
    diagnostics: dict[str, Any]
    export: dict[str, Any]


@dataclass(frozen=True)
class CallablePeakBackend:
    harvest_peaks_fn: Callable[..., dict[str, Any]]
    fit_nat_lm_fn: Callable[..., dict[str, Any]]
    summarize_ncc_fn: Callable[..., dict[str, Any]]
    export_coeff_maps_fn: Callable[..., dict[str, Any]]

    def __init__(
        self,
        *,
        harvest_peaks: Callable[..., dict[str, Any]],
        fit_nat_lm: Callable[..., dict[str, Any]],
        summarize_ncc: Callable[..., dict[str, Any]],
        export_coeff_maps: Callable[..., dict[str, Any]],
    ) -> None:
        object.__setattr__(self, "harvest_peaks_fn", harvest_peaks)
        object.__setattr__(self, "fit_nat_lm_fn", fit_nat_lm)
        object.__setattr__(self, "summarize_ncc_fn", summarize_ncc)
        object.__setattr__(self, "export_coeff_maps_fn", export_coeff_maps)

    def harvest_peaks(self, *, config: PeakBootstrapConfig, output_dir: Path) -> dict[str, Any]:
        return self.harvest_peaks_fn(config=config, output_dir=output_dir)

    def fit_nat_lm(
        self,
        *,
        config: PeakBootstrapConfig,
        harvest_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        return self.fit_nat_lm_fn(config=config, harvest_path=harvest_path, output_dir=output_dir)

    def summarize_ncc(
        self,
        *,
        config: PeakBootstrapConfig,
        diagnostics: dict[str, Any],
        preferred_stage: str,
    ) -> dict[str, Any]:
        return self.summarize_ncc_fn(config=config, diagnostics=diagnostics, preferred_stage=preferred_stage)

    def export_coeff_maps(
        self,
        *,
        config: PeakBootstrapConfig,
        diagnostics_dir: Path,
        output_dir: Path,
        preferred_stage: str,
    ) -> dict[str, Any]:
        return self.export_coeff_maps_fn(
            config=config,
            diagnostics_dir=diagnostics_dir,
            output_dir=output_dir,
            preferred_stage=preferred_stage,
        )


class UnconfiguredPeakBackend:
    def harvest_peaks(self, *, config: PeakBootstrapConfig, output_dir: Path) -> dict[str, Any]:
        raise RuntimeError("No peak pipeline backend configured.")

    def fit_nat_lm(
        self,
        *,
        config: PeakBootstrapConfig,
        harvest_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        raise RuntimeError("No peak pipeline backend configured.")

    def summarize_ncc(
        self,
        *,
        config: PeakBootstrapConfig,
        diagnostics: dict[str, Any],
        preferred_stage: str,
    ) -> dict[str, Any]:
        raise RuntimeError("No peak pipeline backend configured.")

    def export_coeff_maps(
        self,
        *,
        config: PeakBootstrapConfig,
        diagnostics_dir: Path,
        output_dir: Path,
        preferred_stage: str,
    ) -> dict[str, Any]:
        raise RuntimeError("No peak pipeline backend configured.")


def run_peak_bootstrap_pipeline(
    *,
    layout: RunLayout,
    config: PeakBootstrapConfig,
    backend: PeakPipelineBackend | None = None,
) -> PeakPipelineResult:
    stage_dir = layout.stage_dir("peak")
    harvest_dir = stage_dir / "peak_harvest"
    diagnostics_dir = stage_dir / "real_nat_diagnostics"
    export_dir = stage_dir / "export_nat_zmap"
    runner = backend or DefaultPeakBackend()

    harvest = runner.harvest_peaks(config=config, output_dir=harvest_dir)
    harvest_path = Path(harvest["harvest_path"])
    diagnostics_result = runner.fit_nat_lm(
        config=config,
        harvest_path=harvest_path,
        output_dir=diagnostics_dir,
    )
    diagnostics_summary = dict(diagnostics_result["summary"])
    preferred_stage = _preferred_export_stage(diagnostics_summary)
    ncc = runner.summarize_ncc(
        config=config,
        diagnostics=diagnostics_summary,
        preferred_stage=preferred_stage,
    )
    export = runner.export_coeff_maps(
        config=config,
        diagnostics_dir=diagnostics_dir,
        output_dir=export_dir,
        preferred_stage=preferred_stage,
    )

    summary = PeakBootstrapSummary(
        config=config,
        candidate_count=int(harvest["candidate_count"]),
        kept_count=int(harvest["kept_count"]),
        selected_emitters=_selected_emitters(diagnostics_summary),
        ncc_count=int(ncc["count"]),
        ncc_gt_threshold_count=int(ncc["gt_threshold_count"]),
        preferred_export_stage=preferred_stage,
        paths={
            "harvest": harvest_path,
            "harvest_summary": Path(harvest["summary_path"]),
            "diagnostics_summary": Path(diagnostics_result["summary_path"]),
            "export_summary": Path(export["summary_path"]),
            "coeff_map": Path(export["coeff_map_path"]),
            "zmap": Path(export["zmap_path"]),
        },
        metrics=_metrics_for_stage(diagnostics_summary, preferred_stage),
    )
    artifacts = build_peak_bootstrap_artifacts(layout, summary)
    return PeakPipelineResult(
        summary=summary,
        artifacts=artifacts,
        harvest=harvest,
        diagnostics=diagnostics_result,
        export=export,
    )


def _preferred_export_stage(summary: dict[str, Any]) -> str:
    approximate = summary.get("approximate_metrics")
    alternating = summary.get("alternating_metrics")
    approx_ncc = _patch_ncc_mean(approximate)
    alt_ncc = _patch_ncc_mean(alternating)
    if approx_ncc is None:
        return "alternating"
    if alt_ncc is None:
        return "approximate"
    return "approximate" if float(approx_ncc) > float(alt_ncc) else "alternating"


def _patch_ncc_mean(metrics: Any) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("patch_ncc_mean_nonzero_raw")
    if value is not None:
        return float(value)
    value = metrics.get("patch_ncc_mean")
    if value is not None:
        return float(value)
    return None


def _selected_emitters(summary: dict[str, Any]) -> int:
    for section_name in ("comparison_metrics", "alternating_metrics", "approximate_metrics"):
        section = summary.get(section_name)
        if isinstance(section, dict) and section.get("selected_emitters") is not None:
            return int(section["selected_emitters"])
    return 0


def _metrics_for_stage(summary: dict[str, Any], stage: str) -> dict[str, Any]:
    metrics = summary.get(f"{stage}_metrics")
    return dict(metrics) if isinstance(metrics, dict) else {}
