from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    logs_dir: Path
    cache_dir: Path
    metadata_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    artifacts_dir: Path

    def stage_dir(self, stage: str) -> Path:
        _validate_relative_name(stage, "stage")
        return self.run_dir / "stages" / stage


def ensure_run_layout(root: Path | str, run_name: str, stage_names: Iterable[str] = ()) -> RunLayout:
    _validate_relative_name(run_name, "run_name")
    run_dir = Path(root) / run_name
    layout = RunLayout(
        run_dir=run_dir,
        logs_dir=run_dir / "logs",
        cache_dir=run_dir / "cache",
        metadata_dir=run_dir / "metadata",
        checkpoints_dir=run_dir / "checkpoints",
        metrics_dir=run_dir / "metrics",
        artifacts_dir=run_dir / "artifacts",
    )
    for directory in (
        layout.logs_dir,
        layout.cache_dir,
        layout.metadata_dir,
        layout.checkpoints_dir,
        layout.metrics_dir,
        layout.artifacts_dir,
        run_dir / "stages",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for stage in stage_names:
        layout.stage_dir(stage).mkdir(parents=True, exist_ok=True)
    return layout


def write_run_manifest(layout: RunLayout, payload: dict[str, Any]) -> Path:
    path = layout.metadata_dir / "run_manifest.json"
    _write_json(path, payload)
    return path


def write_stage_status(
    layout: RunLayout,
    stage: str,
    status: str,
    payload: dict[str, Any] | None = None,
) -> Path:
    path = layout.metadata_dir / "stage_status.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {}
    state[stage] = {"status": status, "payload": payload or {}}
    _write_json(path, state)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_relative_name(value: str, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a relative path inside the run root")
