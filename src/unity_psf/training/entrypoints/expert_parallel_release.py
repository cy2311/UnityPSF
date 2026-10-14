"""Rank-zero joint release checkpoint and visible report publication."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, Mapping, Sequence

import torch

from unity_psf.contracts import JointExpertKey, PSFModality, ResolvedTrainingContract
from unity_psf.contracts.training import RESOLVED_TRAINING_CONTRACT_SCHEMA
from unity_psf.contracts.canonical import CANONICAL_TRAINING_ENTRYPOINT, load_canonical_release_checkpoint
from unity_psf.models import UnityPSF
from unity_psf.reporting import generate_visible_validation_report
from unity_psf.runtime import write_run_manifest, write_stage_status
from unity_psf.training import atomic_write_json
from unity_psf.training.routed import RoutedTrainingBatch
from unity_psf.training.modality_checkpoint import assemble_modality_joint_checkpoint_from_shards

from .canonical_contract import sha256
from .canonical_runtime import visual_record

VISIBLE_HELDOUT_METRICS = (
    "precision",
    "recall",
    "Jaccard",
    "RMSE_XY_nm",
    "RMSE_Z_nm",
    "photon_relative_error",
)


def publish_joint_release(
    args,
    *,
    layout,
    assignments: Sequence[PSFModality],
    rank_statuses: Sequence[Mapping[str, Any]],
    world_size: int,
    attempt_id: str,
    contract_hash: str,
    resolved_contract: ResolvedTrainingContract,
) -> dict[str, Any]:
    checkpoint_path = layout.checkpoints_dir / "unitypsf_joint.ckpt"
    assemble_modality_joint_checkpoint_from_shards(
        checkpoint_path,
        shard_paths=[status["modality_checkpoint"] for status in rank_statuses],
        required_modalities=assignments,
        provenance={
            "training_entrypoint": CANONICAL_TRAINING_ENTRYPOINT,
            "joint_config": str(args.config.resolve()),
            "execution": "expert_parallel",
            "world_size": world_size,
            "rank_assignment": "one_modality_per_rank",
            "attempt_id": attempt_id,
            "resolved_training_contract_hash": contract_hash,
            "resolved_training_contract_schema": RESOLVED_TRAINING_CONTRACT_SCHEMA,
            "resolved_training_contract": resolved_contract.to_dict(),
        },
        expected_provenance={
            "resolved_training_contract_hash": contract_hash,
            "resolved_training_contract_schema": RESOLVED_TRAINING_CONTRACT_SCHEMA,
        },
    )
    payload = load_canonical_release_checkpoint(checkpoint_path)
    loaded = UnityPSF.from_checkpoint(checkpoint_path, device="cpu")
    records = []
    for status in rank_statuses:
        modality = str(status["owned_modality"])
        validation = torch.load(
            status["validation_batches"], map_location="cpu", weights_only=False
        )
        if (
            validation.get("modality") != modality
            or validation.get("attempt_id") != attempt_id
            or validation.get("resolved_training_contract_hash") != contract_hash
        ):
            raise ValueError("validation batch identity does not match its rank status")
        for channel_id in status["channels"]:
            values = validation["channels"][channel_id]
            batch = RoutedTrainingBatch(
                images=values["images"],
                conditions=values["conditions"],
                target=None,
            )
            localization = loaded.localize(
                batch.images,
                modality=modality,
                channel_id=channel_id,
                conditions=batch.conditions,
            )
            channel_metrics = status["metrics"]["channels"][channel_id]
            record = visual_record(
                JointExpertKey(modality, channel_id).storage_key,
                batch,
                localization,
                losses=channel_metrics["losses"],
                steps=channel_metrics["steps"],
                checkpoint_hash=payload["integrity"]["expert_sha256"][modality],
            )
            heldout = validation.get("heldout")
            if isinstance(heldout, Mapping):
                artifact = heldout["artifacts"][channel_id]
                heldout_metrics = heldout["metrics"]["channels"][channel_id]
                images = artifact["input_images"]
                z_values = artifact["z_values"]
                z_errors = artifact["z_errors"]
                has_z = modality != "emitter_2d" and int(z_values.numel()) > 0
                record = replace(
                    record,
                    input_image=images.mean(dim=0).numpy(),
                    patches=tuple(frame.numpy() for frame in images),
                    route_count=int(heldout_metrics["route_count"]),
                    sample_count=int(heldout_metrics["sample_count"]),
                    prediction_xy=artifact["prediction_xy"].numpy(),
                    target_xy=artifact["target_xy"].numpy(),
                    reconstruction=artifact["reconstruction"].numpy(),
                    z_values=z_values.numpy() if has_z else None,
                    z_errors=z_errors.numpy() if has_z else None,
                    status="heldout-evaluated",
                    heldout_metrics={
                        key: heldout_metrics[key]
                        for key in VISIBLE_HELDOUT_METRICS
                    },
                )
            records.append(record)

    activation_counts = loaded.activation_audit()
    expected_activations = {
        str(status["owned_modality"]): len(status["channels"])
        for status in rank_statuses
    }
    if activation_counts != dict(sorted(expected_activations.items())):
        raise RuntimeError("joint checkpoint route activation audit does not match the channel inventory")
    cuda_confirmed = all(bool(status["cuda_confirmed"]) for status in rank_statuses)
    modality_metrics = {
        str(status["owned_modality"]): {
            key: status["metrics"]["heldout_history"][-1]["modality"][key]
            for key in VISIBLE_HELDOUT_METRICS
        }
        for status in rank_statuses
        if status["metrics"]["heldout_history"]
    }
    report = generate_visible_validation_report(
        layout.run_dir,
        records,
        run_id=args.run_id,
        provenance={
            "checkpoint_sha256": sha256(checkpoint_path),
            "execution": "expert_parallel",
            "cuda_confirmed": cuda_confirmed,
        },
        modality_metrics=modality_metrics,
    )
    summary = {
        "schema_version": "unitypsf.modality_expert_parallel_summary.v2",
        "status": "complete",
        "execution": "expert_parallel",
        "world_size": world_size,
        "attempt_id": attempt_id,
        "resolved_training_contract_hash": contract_hash,
        "resolved_training_contract_schema": RESOLVED_TRAINING_CONTRACT_SCHEMA,
        "cuda_confirmed": cuda_confirmed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "report": str(report.report_path),
        "modalities": {
            str(status["owned_modality"]): status["metrics"] for status in rank_statuses
        },
        "rank_assignments": {
            str(status["rank"]): str(status["owned_modality"])
            for status in rank_statuses
        },
        "smoke_activation_counts": activation_counts,
    }
    summary_path = atomic_write_json(
        layout.metrics_dir / "joint_training_summary.json", summary
    )
    write_run_manifest(
        layout,
        {
            "config": str(args.config.resolve()),
            "execution": "expert_parallel",
            "training_unit": "modality",
            "world_size": world_size,
            "attempt_id": attempt_id,
            "resolved_training_contract_hash": contract_hash,
            "resolved_training_contract_schema": RESOLVED_TRAINING_CONTRACT_SCHEMA,
            "rank_assignments": summary["rank_assignments"],
        },
    )
    write_stage_status(
        layout,
        "expert_parallel_training",
        "completed",
        {
            "checkpoint": str(checkpoint_path),
            "summary": str(summary_path),
            "report": str(report.report_path),
        },
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary

