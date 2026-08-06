from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import PeakBootstrapConfig
from .diagnostics import MinimalNATFitState, run_minimal_nat_diagnostics, summarize_ncc_values
from .export import run_export_nat_zmap
from .harvest import PeakCandidate, PeakHarvestResult, greedy_distance_filter, run_peak_harvest
from .vector_nat_fit import run_vector_nat_diagnostics


class DefaultPeakBackend:
    def harvest_peaks(self, *, config: PeakBootstrapConfig, output_dir: Path) -> dict[str, Any]:
        return run_peak_harvest(config=config, output_dir=output_dir)

    def fit_nat_lm(
        self,
        *,
        config: PeakBootstrapConfig,
        harvest_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        if _should_use_vector_nat(config):
            return run_vector_nat_diagnostics(config=config, harvest_path=harvest_path, output_dir=output_dir)
        return run_minimal_nat_diagnostics(config=config, harvest_path=harvest_path, output_dir=output_dir)

    def summarize_ncc(
        self,
        *,
        config: PeakBootstrapConfig,
        diagnostics: dict[str, Any],
        preferred_stage: str,
    ) -> dict[str, Any]:
        return summarize_ncc_values(config=config, diagnostics=diagnostics, preferred_stage=preferred_stage)

    def export_coeff_maps(
        self,
        *,
        config: PeakBootstrapConfig,
        diagnostics_dir: Path,
        output_dir: Path,
        preferred_stage: str,
    ) -> dict[str, Any]:
        return run_export_nat_zmap(
            config=config,
            diagnostics_dir=diagnostics_dir,
            output_dir=output_dir,
            preferred_stage=preferred_stage,
        )


__all__ = [
    "DefaultPeakBackend",
    "MinimalNATFitState",
    "PeakCandidate",
    "PeakHarvestResult",
    "greedy_distance_filter",
    "run_minimal_nat_diagnostics",
    "summarize_ncc_values",
]


def _should_use_vector_nat(config: PeakBootstrapConfig) -> bool:
    if config.tiff_path is None:
        return False
    if int(config.patch_size_px) < 15:
        return False
    if int(config.max_emitters) < 100:
        return False
    return True
