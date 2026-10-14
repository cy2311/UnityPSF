"""Filesystem artifacts shared by modality Expert Parallel workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from unity_psf.contracts import PSFModality
from unity_psf.training.checkpoint_io import atomic_torch_save as write_atomic_torch_payload


def rank_status_path(layout, rank: int) -> Path:
    return layout.metadata_dir / f"rank_{rank}.json"


def validation_batch_path(layout, modality: PSFModality | str) -> Path:
    return layout.artifacts_dir / f"{PSFModality.parse(modality).value}_validation_batches.pt"


def shard_path(layout, modality: PSFModality | str) -> Path:
    return layout.checkpoints_dir / f"{PSFModality.parse(modality).value}.resume.ckpt"


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_torch_payload(payload, path)
    return path
