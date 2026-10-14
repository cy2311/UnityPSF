from __future__ import annotations

from pathlib import Path

import pytest

from unity_psf.config import resolve_config_reference


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_reference_is_independent_of_config_depth() -> None:
    source = PROJECT_ROOT / "configs/experiments/example.yaml"
    resolved = resolve_config_reference(
        "project://configs/modalities/emitter_2d/single_channel_smoke.yaml",
        source_path=source,
    )
    assert resolved == PROJECT_ROOT / "configs/modalities/emitter_2d/single_channel_smoke.yaml"


def test_relative_reference_stays_relative_to_source_config() -> None:
    source = PROJECT_ROOT / "configs/experiments/example.yaml"
    resolved = resolve_config_reference("local.yaml", source_path=source)
    assert resolved == PROJECT_ROOT / "configs/experiments/local.yaml"


def test_project_reference_requires_a_project_marker(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project root"):
        resolve_config_reference("project://configs/example.yaml", source_path=tmp_path / "example.yaml")
