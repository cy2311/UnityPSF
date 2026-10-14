"""Canonical release contract for the current UnityPSF training path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .modality_joint_checkpoint import (
    MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION,
    load_modality_joint_checkpoint,
)
from .training import RESOLVED_TRAINING_CONTRACT_SCHEMA, ResolvedTrainingContract


CANONICAL_TRAINING_ENTRYPOINT = "unity_psf.cli.train_selection"
CANONICAL_CHECKPOINT_SCHEMA = MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION
CANONICAL_LAUNCH_ENV = "UNITYPSF_CANONICAL_LAUNCH"
CANONICAL_CONTRACT_HASH_ENV = "UNITYPSF_CANONICAL_CONTRACT_HASH"


def validate_canonical_launch(contract_hash: str) -> None:
    """Reject direct execution of the internal trainer.

    The CLI materializes the resolved contract before launching torchrun and
    passes its hash through the child environment. A missing or stale marker
    means the internal executor was bypassed or is using another run.
    """

    if os.environ.get(CANONICAL_LAUNCH_ENV) != CANONICAL_TRAINING_ENTRYPOINT:
        raise RuntimeError(
            "formal training must be launched through unity-psf-train; "
            "the internal executor is not a public entrypoint"
        )
    if os.environ.get(CANONICAL_CONTRACT_HASH_ENV) != contract_hash:
        raise RuntimeError(
            "canonical launcher contract hash is missing or does not match the resolved contract"
        )


def load_canonical_release_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a release checkpoint written by the canonical training path."""

    payload = load_modality_joint_checkpoint(path)
    if payload["metadata"].get("checkpoint_role") != "release":
        raise ValueError("canonical release checkpoint must have checkpoint_role='release'")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("training_entrypoint") != CANONICAL_TRAINING_ENTRYPOINT:
        raise ValueError(
            "canonical release checkpoint must declare the canonical training entrypoint"
        )
    if provenance.get("resolved_training_contract_schema") != RESOLVED_TRAINING_CONTRACT_SCHEMA:
        raise ValueError(
            "canonical release checkpoint must declare the resolved training contract schema"
        )
    contract_hash = provenance.get("resolved_training_contract_hash")
    if not isinstance(contract_hash, str) or len(contract_hash) != 64:
        raise ValueError(
            "canonical release checkpoint must declare the resolved training contract hash"
        )
    snapshot = provenance.get("resolved_training_contract")
    if not isinstance(snapshot, Mapping):
        raise ValueError("canonical release checkpoint must embed the resolved training contract")
    contract = ResolvedTrainingContract.from_dict(snapshot)
    if contract.contract_hash != contract_hash:
        raise ValueError("canonical release checkpoint contract snapshot does not match its hash")
    return payload


__all__ = [
    "CANONICAL_CHECKPOINT_SCHEMA",
    "CANONICAL_CONTRACT_HASH_ENV",
    "CANONICAL_LAUNCH_ENV",
    "CANONICAL_TRAINING_ENTRYPOINT",
    "load_canonical_release_checkpoint",
    "validate_canonical_launch",
]
