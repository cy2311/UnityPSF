from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .layout import RunLayout


@dataclass(frozen=True)
class ArtifactRecord:
    stage: str
    name: str
    kind: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self, run_dir: Path) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "name": self.name,
            "kind": self.kind,
            "path": _display_path(self.path, run_dir),
            "metadata": self.metadata,
        }


class ArtifactRegistry:
    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout
        self._records: list[ArtifactRecord] = []

    def register(
        self,
        *,
        stage: str,
        name: str,
        kind: str,
        path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            stage=stage,
            name=name,
            kind=kind,
            path=Path(path),
            metadata=metadata or {},
        )
        self._records.append(record)
        return record

    def write(self) -> Path:
        path = self._layout.metadata_dir / "artifacts.json"
        payload = {"artifacts": [record.to_json(self._layout.run_dir) for record in self._records]}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _display_path(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(path)
