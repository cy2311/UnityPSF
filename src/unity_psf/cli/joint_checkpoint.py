"""Inspect, verify, and assemble a single-file UnityPSF joint checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from unity_psf.contracts import (
    CheckpointMetadata,
    JointCheckpointMetadata,
    MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION,
    PSFModality,
    assemble_joint_checkpoint,
    detect_joint_checkpoint_schema,
    load_checkpoint,
    load_joint_checkpoint,
    load_modality_joint_checkpoint,
    save_joint_checkpoint_payload,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage one-file UnityPSF joint checkpoints.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Print model and expert inventory as JSON.")
    inspect_parser.add_argument("checkpoint", type=Path)
    verify_parser = commands.add_parser("verify", help="Validate schema and every nested integrity hash.")
    verify_parser.add_argument("checkpoint", type=Path)
    assemble_parser = commands.add_parser("assemble", help="Assemble v2 instance checkpoints into one release file.")
    assemble_parser.add_argument("instance_checkpoints", nargs="+", type=Path)
    assemble_parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(path: Path) -> dict[str, Any]:
    schema = detect_joint_checkpoint_schema(path)
    if schema == MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION:
        return _modality_inventory(path)
    payload = load_joint_checkpoint(path)
    expert_hashes = payload["integrity"]["expert_sha256"]
    instances = {}
    for key, state in sorted(payload["experts"].items()):
        parameters = state["model_state_dict"]
        instances[key] = {
            "model_class": state["model_class"],
            "parameter_count": sum(value.numel() for value in parameters.values() if isinstance(value, torch.Tensor)),
            "condition_schema": state["condition_schema"],
            "physical_state_present": bool(state["physical_state"]),
            "calibration_present": bool(state["calibration"]),
            "logical_sha256": expert_hashes[key],
        }
    return {
        "checkpoint": str(path),
        "sha256": _sha256_file(path),
        "schema_version": schema,
        "metadata": payload["metadata"],
        "instances": instances,
    }


def _modality_inventory(path: Path) -> dict[str, Any]:
    payload = load_modality_joint_checkpoint(path)
    expert_hashes = payload["integrity"]["expert_sha256"]
    modalities = {}
    for modality, state in sorted(payload["experts"].items()):
        parameters = state["model_state_dict"]
        channels = payload["channel_states"][modality]
        modalities[modality] = {
            "model_class": state["model_class"],
            "parameter_count": sum(
                value.numel() for value in parameters.values() if isinstance(value, torch.Tensor)
            ),
            "condition_schema": state["condition_schema"],
            "channels": sorted(channels),
            "logical_sha256": expert_hashes[modality],
        }
    return {
        "checkpoint": str(path),
        "sha256": _sha256_file(path),
        "schema_version": MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION,
        "metadata": payload["metadata"],
        "modalities": modalities,
    }


def _assemble(paths: Sequence[Path], output: Path) -> dict[str, Any]:
    modalities = {
        CheckpointMetadata.from_dict(load_checkpoint(path)["metadata"]).expert_type
        for path in paths
    }
    if None in modalities:
        raise ValueError("every instance checkpoint must declare an expert_type")
    ordered_modalities = tuple(modality for modality in PSFModality if modality in modalities)
    payload = assemble_joint_checkpoint(
        paths,
        metadata=JointCheckpointMetadata(
            checkpoint_role="release",
            supported_modalities=ordered_modalities,
        ),
        provenance={"instance_checkpoints": [str(path) for path in paths]},
    )
    save_joint_checkpoint_payload(output, payload)
    inventory = _inventory(output)
    return {
        "checkpoint": str(output),
        "sha256": inventory["sha256"],
        "instances": sorted(inventory["instances"]),
        "supported_modalities": inventory["metadata"]["supported_modalities"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "assemble":
        result = _assemble(args.instance_checkpoints, args.output)
    else:
        inventory = _inventory(args.checkpoint)
        result = inventory if args.command == "inspect" else {
            "checkpoint": inventory["checkpoint"],
            "sha256": inventory["sha256"],
            "schema_version": inventory["schema_version"],
            "valid": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    detect_joint_checkpoint_schema,
