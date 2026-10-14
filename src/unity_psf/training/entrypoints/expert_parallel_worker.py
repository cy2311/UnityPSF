"""Train and publish the single modality owned by a torchrun rank."""

from __future__ import annotations

import gc
import json
from typing import Any, Mapping, Sequence

import torch

from unity_psf.contracts import PSFModality, ResolvedTrainingContract
from unity_psf.contracts.training import RESOLVED_TRAINING_CONTRACT_SCHEMA
from unity_psf.training import atomic_write_json
from unity_psf.training.modality_checkpoint import restore_modality_training_shard, save_modality_training_shard
from unity_psf.training.modality_epoch import train_modality_epoch
from unity_psf.training.modality_evaluation import evaluate_modality_heldout
from unity_psf.training.modality_types import ModalityTrainingBatch
from unity_psf.training.modality_progress import (
    accumulate_epoch,
    accumulate_heldout,
    channel_progress,
    empty_metrics,
    heldout_eval_enabled,
    metrics_from_progress,
)

from .canonical_runtime import build_modality_runtime
from .expert_parallel_artifacts import (
    atomic_torch_save,
    rank_status_path,
    shard_path as modality_shard_path,
    validation_batch_path,
)

def train_owned_modality(
    args,
    *,
    layout,
    modality: PSFModality,
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    epochs: int,
    rank: int,
    local_rank: int,
    device: str,
    attempt_id: str,
    contract_hash: str,
    resolved_contract: ResolvedTrainingContract,
    epoch_end_hook=None,
) -> dict[str, Any]:
    runtime, _, formal_runtime_contracts = build_modality_runtime(
        modality,
        entries,
        joint_path=args.config,
        resolved_contract=resolved_contract,
        run_layout=layout,
        device=device,
    )
    shard_path = modality_shard_path(layout, modality)
    metrics = empty_metrics(runtime)
    start_epoch = 1
    restored_status = None
    if args.resume and shard_path.is_file():
        restored = restore_modality_training_shard(
            shard_path,
            runtime=runtime,
            expected_provenance={
                "resolved_training_contract_hash": contract_hash,
                "resolved_training_contract_schema": RESOLVED_TRAINING_CONTRACT_SCHEMA,
            },
        )
        if restored.epoch > epochs:
            raise ValueError("resume shard is ahead of the configured epoch count")
        if restored.status == "complete" and restored.epoch != epochs:
            raise ValueError("completed resume shard does not match the configured epoch count")
        metrics = metrics_from_progress(
            runtime,
            epoch=restored.epoch,
            optimizer_steps=restored.optimizer_steps,
            progress=restored.channel_progress,
        )
        start_epoch = restored.epoch + 1
        restored_status = restored.status
    elif shard_path.exists():
        raise FileExistsError(f"modality shard already exists; use --resume: {shard_path}")

    print(
        json.dumps(
            {
                "event": "modality_expert_parallel_start",
                "rank": rank,
                "local_rank": local_rank,
                "owned_modality": modality.value,
                "channels": list(runtime.channels),
                "device": device,
                "start_epoch": start_epoch,
                "stop_epoch": epochs,
                "formal_runtime_contracts": formal_runtime_contracts,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    final_heldout = None
    heldout_enabled = heldout_eval_enabled(runtime)
    for epoch in range(start_epoch, epochs + 1):
        result = train_modality_epoch(runtime=runtime, epoch=epoch)
        accumulate_epoch(metrics, result)
        if heldout_enabled:
            final_heldout = evaluate_modality_heldout(runtime)
            accumulate_heldout(metrics, final_heldout, epoch=epoch)
        physical_update = None if epoch_end_hook is None else epoch_end_hook(
            runtime=runtime,
            epoch=epoch,
            layout=layout,
        )
        save_modality_training_shard(
            shard_path,
            runtime=runtime,
            epoch=epoch,
            optimizer_steps=metrics["optimizer_steps"],
            channel_progress=channel_progress(metrics),
            status="complete" if epoch == epochs else "running",
            provenance={
                "joint_config": str(args.config.resolve()),
                "execution": "expert_parallel",
                "rank": rank,
                "attempt_id": attempt_id,
                "resolved_training_contract_hash": contract_hash,
                "resolved_training_contract_schema": RESOLVED_TRAINING_CONTRACT_SCHEMA,
            },
        )
        heldout_epoch = metrics["heldout_history"][-1] if heldout_enabled else None
        print(
            json.dumps(
                {
                    "event": "modality_epoch_complete",
                    "rank": rank,
                    "owned_modality": modality.value,
                    "epoch": epoch,
                    "attempted_optimizer_steps": result.attempted_optimizer_steps,
                    "optimizer_steps": result.optimizer_steps,
                    "skipped_optimizer_steps": result.skipped_optimizer_steps,
                    "channels": {
                        channel_id: {
                            "mean_loss": sum(result.losses_by_channel[channel_id])
                            / len(result.losses_by_channel[channel_id]),
                            "optimizer_steps": result.optimizer_steps_by_channel[channel_id],
                            "skipped_optimizer_steps": (
                                result.skipped_optimizer_steps_by_channel[channel_id]
                            ),
                            "optimizer_skip_reasons": result.optimizer_skip_reasons_by_channel[channel_id],
                            "heldout": (
                                None
                                if heldout_epoch is None
                                else {
                                    key: heldout_epoch["channels"][channel_id].get(key)
                                    for key in (
                                        "predicted_emitters",
                                        "true_positive",
                                        "false_positive",
                                        "false_negative",
                                        "precision",
                                        "recall",
                                        "Jaccard",
                                        "RMSE_XY_nm",
                                        "RMSE_Z_nm",
                                        "eval_loss",
                                    )
                                }
                            ),
                        }
                        for channel_id in runtime.channels
                    },
                    "physical_update": physical_update,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if restored_status != "complete" and metrics["epochs_completed"] != epochs:
        raise RuntimeError("modality training did not reach the configured epoch count")
    if formal_runtime_contracts is not None and metrics["optimizer_steps"] <= 0:
        raise RuntimeError(
            f"formal {modality.value} training completed no optimizer updates; "
            f"AMP skipped all {metrics['attempted_optimizer_steps']} attempted steps"
        )
    if heldout_enabled and final_heldout is None:
        final_heldout = evaluate_modality_heldout(runtime)

    validation_batches: dict[str, dict[str, torch.Tensor | None]] = {}
    for channel_id, stream in runtime.channels.items():
        batch = next(iter(stream.batches(epochs + 1)))
        if not isinstance(batch, ModalityTrainingBatch):
            raise TypeError("validation provider must return ModalityTrainingBatch")
        validation_batches[channel_id] = {
            "images": batch.images.detach().cpu(),
            "conditions": None if batch.conditions is None else batch.conditions.detach().cpu(),
        }
    validation_path = atomic_torch_save(
        {
            "modality": modality.value,
            "attempt_id": attempt_id,
            "resolved_training_contract_hash": contract_hash,
            "resolved_training_contract_schema": RESOLVED_TRAINING_CONTRACT_SCHEMA,
            "channels": validation_batches,
            "heldout": None
            if final_heldout is None
            else {
                "metrics": metrics["heldout_history"][-1],
                "artifacts": dict(final_heldout.artifacts),
            },
        },
        validation_batch_path(layout, modality),
    )
    status = {
        "rank": rank,
        "local_rank": local_rank,
        "status": "complete",
        "attempt_id": attempt_id,
        "resolved_training_contract_hash": contract_hash,
        "resolved_training_contract_schema": RESOLVED_TRAINING_CONTRACT_SCHEMA,
        "owned_modality": modality.value,
        "channels": list(runtime.channels),
        "device": device,
        "cuda_confirmed": torch.cuda.is_available() and device.startswith("cuda:"),
        "epochs": epochs,
        "parameter_count": sum(parameter.numel() for parameter in runtime.model.parameters()),
        "formal_runtime_contracts": formal_runtime_contracts,
        "metrics": metrics,
        "modality_checkpoint": str(shard_path),
        "validation_batches": str(validation_path),
    }
    atomic_write_json(rank_status_path(layout, rank), status)
    print(json.dumps({"event": "modality_expert_complete", **status}, sort_keys=True), flush=True)
    runtime.model.to("cpu")
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return status
