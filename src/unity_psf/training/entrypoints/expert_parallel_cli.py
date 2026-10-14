"""Canonical launch validation and modality rank dispatch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import torch

from unity_psf.contracts import JointExpertKey
from unity_psf.contracts.canonical import validate_canonical_launch
from unity_psf.contracts.training import RESOLVED_TRAINING_CONTRACT_SCHEMA
from unity_psf.runtime import ensure_run_layout
from unity_psf.training import atomic_write_json

from .canonical_contract import (
    instance_specs,
    load_joint_config,
    load_resolved_training_contract,
    modality_groups,
)
from .expert_parallel_artifacts import rank_status_path
from .expert_parallel_coordination import read_completed_rank_statuses
from .expert_parallel_release import publish_joint_release
from .expert_parallel_worker import train_owned_modality


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one shared multichannel PSF modality expert per torchrun rank."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--coordination-timeout-seconds", type=float, default=3600.0)
    return parser


def distributed_identity() -> tuple[int, int, int]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ValueError(f"Expert Parallel must be launched by torchrun; missing {missing}")
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])


def attempt_id() -> str:
    value = os.environ.get("UNITYPSF_ATTEMPT_ID") or os.environ.get("TORCHELASTIC_RUN_ID")
    if value is None or not value.strip():
        raise ValueError("torchrun attempt identity is missing")
    return value.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rank, world_size, local_rank = distributed_identity()
    joint_config = load_joint_config(args.config, expected_execution="expert_parallel")
    if joint_config.get("rank_assignment") != "one_modality_per_rank":
        raise ValueError("modality Expert Parallel requires rank_assignment: one_modality_per_rank")
    resolved_contract = load_resolved_training_contract(args.config, joint_config)
    if resolved_contract.run_id != args.run_id:
        raise ValueError("run-id does not match the resolved training contract")
    validate_canonical_launch(resolved_contract.contract_hash)
    expected_run_dir = (args.run_root.expanduser().resolve() / args.run_id).resolve()
    if Path(resolved_contract.run_dir).expanduser().resolve() != expected_run_dir:
        raise ValueError("run-root and run-id do not match the resolved training contract")
    specs = instance_specs(joint_config)
    current_attempt = attempt_id()
    contract_hash = resolved_contract.contract_hash
    grouped = modality_groups(specs)
    assignments = tuple(grouped)
    if world_size != len(assignments):
        raise ValueError(f"configured Expert Parallel requires exactly {len(assignments)} ranks")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    epochs = int(joint_config.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("joint training epochs must be positive")
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu_smoke:
        raise RuntimeError("formal Expert Parallel training requires CUDA; CPU fallback is forbidden")
    device = f"cuda:{local_rank}" if cuda_available else "cpu"
    if cuda_available:
        torch.cuda.set_device(local_rank)

    layout = ensure_run_layout(
        args.run_root,
        args.run_id,
        stage_names=("expert_parallel_training",),
    )
    modality = assignments[rank]
    try:
        train_owned_modality(
            args,
            layout=layout,
            modality=modality,
            entries=grouped[modality],
            epochs=epochs,
            rank=rank,
            local_rank=local_rank,
            device=device,
            attempt_id=current_attempt,
            contract_hash=contract_hash,
            resolved_contract=resolved_contract,
        )
    except Exception as exc:
        atomic_write_json(
            rank_status_path(layout, rank),
            {
                "rank": rank,
                "local_rank": local_rank,
                "status": "failed",
                "attempt_id": current_attempt,
                "resolved_training_contract_hash": contract_hash,
                "resolved_training_contract_schema": RESOLVED_TRAINING_CONTRACT_SCHEMA,
                "owned_modality": modality.value,
                "channels": [JointExpertKey.parse(key).channel_id for key, _ in grouped[modality]],
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    if rank != 0:
        return 0
    statuses = read_completed_rank_statuses(
        layout,
        assignments=assignments,
        attempt_id=current_attempt,
        contract_hash=contract_hash,
        timeout_seconds=args.coordination_timeout_seconds,
    )
    publish_joint_release(
        args,
        layout=layout,
        assignments=assignments,
        rank_statuses=statuses,
        world_size=world_size,
        attempt_id=current_attempt,
        contract_hash=contract_hash,
        resolved_contract=resolved_contract,
    )
    return 0
