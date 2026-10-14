"""Rank status coordination for modality Expert Parallel."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from unity_psf.contracts import PSFModality, load_modality_joint_checkpoint

from .expert_parallel_artifacts import (
    rank_status_path,
    shard_path as modality_shard_path,
    validation_batch_path,
)

def read_completed_rank_statuses(
    layout,
    *,
    assignments: Sequence[PSFModality],
    attempt_id: str,
    contract_hash: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    if timeout_seconds <= 0:
        raise ValueError("coordination timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    while True:
        statuses: list[dict[str, Any] | None] = [None] * len(assignments)
        pending: list[str] = []
        for rank, modality in enumerate(assignments):
            path = rank_status_path(layout, rank)
            if not path.is_file():
                pending.append(f"{modality.value}:missing")
                continue
            status = json.loads(path.read_text(encoding="utf-8"))
            if (
                status.get("attempt_id") != attempt_id
                or status.get("resolved_training_contract_hash") != contract_hash
            ):
                pending.append(f"{modality.value}:stale-attempt")
                continue
            if status.get("rank") != rank or status.get("owned_modality") != modality.value:
                raise RuntimeError(f"rank {rank} status identity does not match its assignment")
            if status.get("status") == "failed":
                raise RuntimeError(
                    f"modality rank {rank} ({modality.value}) failed: {status.get('error', 'unknown error')}"
                )
            if status.get("status") != "complete":
                pending.append(f"{modality.value}:{status.get('status', 'invalid')}")
                continue
            shard_path = Path(status.get("modality_checkpoint", "")).resolve()
            validation_path = Path(status.get("validation_batches", "")).resolve()
            if shard_path != modality_shard_path(layout, modality).resolve():
                raise RuntimeError(f"{modality.value} shard path escapes the current run")
            if validation_path != validation_batch_path(layout, modality).resolve():
                raise RuntimeError(f"{modality.value} validation path escapes the current run")
            if not shard_path.is_file():
                pending.append(f"{modality.value}:missing-checkpoint")
                continue
            if not validation_path.is_file():
                pending.append(f"{modality.value}:missing-validation")
                continue
            shard = load_modality_joint_checkpoint(shard_path)
            shard_provenance = shard.get("provenance", {})
            if (
                not isinstance(shard_provenance, Mapping)
                or shard_provenance.get("resolved_training_contract_hash") != contract_hash
            ):
                raise RuntimeError(f"{modality.value} shard provenance does not match the current contract")
            statuses[rank] = status
        if all(status is not None for status in statuses):
            return [status for status in statuses if status is not None]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for completed modality ranks: {pending}")
        time.sleep(0.2)
