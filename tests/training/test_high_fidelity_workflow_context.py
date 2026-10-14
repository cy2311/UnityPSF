from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from unity_psf.training.high_fidelity.workflow_context import DomainPeakLayout


def test_domain_peak_layout_only_overrides_peak_stage(tmp_path: Path) -> None:
    base = SimpleNamespace(
        run_dir=tmp_path / "run",
        logs_dir=tmp_path / "logs",
        cache_dir=tmp_path / "cache",
        metadata_dir=tmp_path / "metadata",
        checkpoints_dir=tmp_path / "checkpoints",
        metrics_dir=tmp_path / "metrics",
        artifacts_dir=tmp_path / "artifacts",
        stage_dir=lambda stage: tmp_path / stage,
    )

    layout = DomainPeakLayout(base=base, domain_name="left/domain")

    assert layout.run_dir == base.run_dir
    assert layout.stage_dir("peak") == tmp_path / "peak" / "left_domain"
    assert layout.stage_dir("high_fidelity_localization") == tmp_path / "high_fidelity_localization"
