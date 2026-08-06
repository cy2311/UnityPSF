"""Train one multichannel expert per torchrun rank without collectives."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

from unity_psf.contracts import JointExpertKey, PSFModality, load_modality_joint_checkpoint
from unity_psf.models import UnityPSF
from unity_psf.reporting import generate_visible_validation_report
from unity_psf.runtime import ensure_run_layout, write_run_manifest, write_stage_status
from unity_psf.training import RoutedTrainingBatch, atomic_write_json
from unity_psf.training.modality_joint import (
    ModalityTrainingBatch,
    ModalityTrainingRuntime,
    assemble_modality_joint_checkpoint_from_shards,
    evaluate_modality_heldout,
    restore_modality_training_shard,
    save_modality_training_shard,
    train_modality_epoch,
)

from .train_joint import _instance_specs, _load_joint_config, _sha256, _visual_record
from .train_modality_joint import _build_modality_runtime, _config_path, _modality_groups


_VISIBLE_HELDOUT_METRICS = (
    "precision",
    "recall",
    "Jaccard",
    "RMSE_XY_nm",
    "RMSE_Z_nm",
    "photon_relative_error",
)


def _parser() -> argparse.ArgumentParser:
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


def _distributed_identity() -> tuple[int, int, int]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ValueError(f"Expert Parallel must be launched by torchrun; missing {missing}")
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])


def _attempt_id() -> str:
    value = os.environ.get("UNITYPSF_ATTEMPT_ID") or os.environ.get("TORCHELASTIC_RUN_ID")
    if value is None or not value.strip():
        raise ValueError("torchrun attempt identity is missing")
    return value.strip()


def _training_signature(
    joint_path: Path,
    specs: Mapping[str, Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    files = [("joint", joint_path.resolve())]
    files.extend(
        (key, _config_path(joint_path, specs[key]["config"]))
        for key in sorted(specs)
    )
    for label, path in files:
        encoded_label = label.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded_label).to_bytes(8, "big"))
        digest.update(encoded_label)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _rank_status_path(layout, rank: int) -> Path:
    return layout.metadata_dir / f"rank_{rank}.json"


def _validation_batch_path(layout, modality: PSFModality | str) -> Path:
    return layout.artifacts_dir / f"{PSFModality.parse(modality).value}_validation_batches.pt"


def _shard_path(layout, modality: PSFModality | str) -> Path:
    return layout.checkpoints_dir / f"{PSFModality.parse(modality).value}.resume.ckpt"


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _empty_metrics(runtime: ModalityTrainingRuntime) -> dict[str, Any]:
    return {
        "optimizer_steps": 0,
        "attempted_optimizer_steps": 0,
        "skipped_optimizer_steps": 0,
        "epochs_completed": 0,
        "heldout_history": [],
        "channels": {
            channel_id: {
                "steps": 0,
                "optimizer_steps": 0,
                "skipped_optimizer_steps": 0,
                "samples": 0,
                "losses": [],
                "heldout_history": [],
            }
            for channel_id in runtime.channels
        },
    }


def _metrics_from_progress(
    runtime: ModalityTrainingRuntime,
    *,
    epoch: int,
    optimizer_steps: int,
    progress: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(progress) != set(runtime.channels):
        raise ValueError("resume channel progress does not match the modality runtime")
    modality_histories = [
        list(progress[channel_id].get("modality_heldout_history", ()))
        for channel_id in runtime.channels
    ]
    modality_history = modality_histories[0]
    if any(history != modality_history for history in modality_histories[1:]):
        raise ValueError("resume channel copies of modality held-out history differ")
    return {
        "optimizer_steps": int(optimizer_steps),
        "attempted_optimizer_steps": sum(
            int(progress[channel_id].get("steps", 0)) for channel_id in runtime.channels
        ),
        "skipped_optimizer_steps": sum(
            int(progress[channel_id].get("skipped_optimizer_steps", 0))
            for channel_id in runtime.channels
        ),
        "epochs_completed": int(epoch),
        "heldout_history": [dict(item) for item in modality_history],
        "channels": {
            channel_id: {
                "steps": int(progress[channel_id].get("steps", 0)),
                "optimizer_steps": int(
                    progress[channel_id].get(
                        "optimizer_steps", progress[channel_id].get("steps", 0)
                    )
                ),
                "skipped_optimizer_steps": int(
                    progress[channel_id].get("skipped_optimizer_steps", 0)
                ),
                "samples": int(progress[channel_id].get("samples", 0)),
                "losses": [float(item) for item in progress[channel_id].get("losses", ())],
                "heldout_history": [
                    dict(item)
                    for item in progress[channel_id].get("heldout_history", ())
                ],
            }
            for channel_id in runtime.channels
        },
    }


def _channel_progress(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        channel_id: {
            "steps": int(values["steps"]),
            "optimizer_steps": int(values["optimizer_steps"]),
            "skipped_optimizer_steps": int(values["skipped_optimizer_steps"]),
            "samples": int(values["samples"]),
            "losses": list(values["losses"]),
            "heldout_history": list(values["heldout_history"]),
            "modality_heldout_history": list(metrics["heldout_history"]),
        }
        for channel_id, values in metrics["channels"].items()
    }


def _accumulate_epoch(metrics: dict[str, Any], result) -> None:
    metrics["optimizer_steps"] += result.optimizer_steps
    metrics["attempted_optimizer_steps"] += result.attempted_optimizer_steps
    metrics["skipped_optimizer_steps"] += result.skipped_optimizer_steps
    metrics["epochs_completed"] = result.epoch
    for channel_id, channel_metrics in metrics["channels"].items():
        channel_metrics["steps"] += result.step_counts[channel_id]
        channel_metrics["optimizer_steps"] += result.optimizer_steps_by_channel[channel_id]
        channel_metrics["skipped_optimizer_steps"] += (
            result.skipped_optimizer_steps_by_channel[channel_id]
        )
        channel_metrics["samples"] += result.sample_counts[channel_id]
        channel_metrics["losses"].extend(result.losses_by_channel[channel_id])


def _accumulate_heldout(metrics: dict[str, Any], result, *, epoch: int) -> None:
    channel_results = {}
    for channel_id, values in result.channels.items():
        channel_values = {
            **dict(values),
            "optimizer_steps": int(metrics["channels"][channel_id]["optimizer_steps"]),
            "skipped_optimizer_steps": int(
                metrics["channels"][channel_id]["skipped_optimizer_steps"]
            ),
        }
        metrics["channels"][channel_id]["heldout_history"].append(
            {"epoch": int(epoch), **channel_values}
        )
        channel_results[channel_id] = channel_values
    metrics["heldout_history"].append(
        {
            "epoch": int(epoch),
            "modality": {
                **dict(result.modality),
                "optimizer_steps": int(metrics["optimizer_steps"]),
            },
            "channels": channel_results,
        }
    )


def _heldout_eval_enabled(runtime: ModalityTrainingRuntime) -> bool:
    enabled = [
        stream.heldout_eval is not None for stream in runtime.channels.values()
    ]
    if any(enabled) and not all(enabled):
        raise ValueError("held-out eval must be enabled for all channels or none")
    return all(enabled)


def _read_completed_rank_statuses(
    layout,
    *,
    assignments: Sequence[PSFModality],
    attempt_id: str,
    training_signature: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    if timeout_seconds <= 0:
        raise ValueError("coordination timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    while True:
        statuses: list[dict[str, Any] | None] = [None] * len(assignments)
        pending: list[str] = []
        for rank, modality in enumerate(assignments):
            path = _rank_status_path(layout, rank)
            if not path.is_file():
                pending.append(f"{modality.value}:missing")
                continue
            status = json.loads(path.read_text(encoding="utf-8"))
            if (
                status.get("attempt_id") != attempt_id
                or status.get("training_signature") != training_signature
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
            if not Path(status["modality_checkpoint"]).is_file():
                pending.append(f"{modality.value}:missing-checkpoint")
                continue
            if not Path(status["validation_batches"]).is_file():
                pending.append(f"{modality.value}:missing-validation")
                continue
            statuses[rank] = status
        if all(status is not None for status in statuses):
            return [status for status in statuses if status is not None]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for completed modality ranks: {pending}")
        time.sleep(0.2)


def _train_owned_modality(
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
    training_signature: str,
) -> dict[str, Any]:
    runtime, _, formal_runtime_contracts = _build_modality_runtime(
        modality,
        entries,
        joint_path=args.config,
        run_layout=layout,
        device=device,
    )
    shard_path = _shard_path(layout, modality)
    metrics = _empty_metrics(runtime)
    start_epoch = 1
    restored_status = None
    if args.resume and shard_path.is_file():
        restored = restore_modality_training_shard(
            shard_path,
            runtime=runtime,
            expected_provenance={"training_signature": training_signature},
        )
        if restored.epoch > epochs:
            raise ValueError("resume shard is ahead of the configured epoch count")
        if restored.status == "complete" and restored.epoch != epochs:
            raise ValueError("completed resume shard does not match the configured epoch count")
        metrics = _metrics_from_progress(
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
    heldout_enabled = _heldout_eval_enabled(runtime)
    for epoch in range(start_epoch, epochs + 1):
        result = train_modality_epoch(runtime=runtime, epoch=epoch)
        _accumulate_epoch(metrics, result)
        if heldout_enabled:
            final_heldout = evaluate_modality_heldout(runtime)
            _accumulate_heldout(metrics, final_heldout, epoch=epoch)
        save_modality_training_shard(
            shard_path,
            runtime=runtime,
            epoch=epoch,
            optimizer_steps=metrics["optimizer_steps"],
            channel_progress=_channel_progress(metrics),
            status="complete" if epoch == epochs else "running",
            provenance={
                "joint_config": str(args.config.resolve()),
                "execution": "expert_parallel",
                "rank": rank,
                "attempt_id": attempt_id,
                "training_signature": training_signature,
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
    validation_path = _atomic_torch_save(
        {
            "modality": modality.value,
            "attempt_id": attempt_id,
            "training_signature": training_signature,
            "channels": validation_batches,
            "heldout": None
            if final_heldout is None
            else {
                "metrics": metrics["heldout_history"][-1],
                "artifacts": dict(final_heldout.artifacts),
            },
        },
        _validation_batch_path(layout, modality),
    )
    status = {
        "rank": rank,
        "local_rank": local_rank,
        "status": "complete",
        "attempt_id": attempt_id,
        "training_signature": training_signature,
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
    atomic_write_json(_rank_status_path(layout, rank), status)
    print(json.dumps({"event": "modality_expert_complete", **status}, sort_keys=True), flush=True)
    runtime.model.to("cpu")
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return status


def _publish_joint_release(
    args,
    *,
    layout,
    assignments: Sequence[PSFModality],
    rank_statuses: Sequence[Mapping[str, Any]],
    world_size: int,
    attempt_id: str,
    training_signature: str,
) -> dict[str, Any]:
    checkpoint_path = layout.checkpoints_dir / "unitypsf_joint.ckpt"
    assemble_modality_joint_checkpoint_from_shards(
        checkpoint_path,
        shard_paths=[status["modality_checkpoint"] for status in rank_statuses],
        required_modalities=assignments,
        provenance={
            "joint_config": str(args.config.resolve()),
            "execution": "expert_parallel",
            "world_size": world_size,
            "rank_assignment": "one_modality_per_rank",
            "attempt_id": attempt_id,
            "training_signature": training_signature,
        },
    )
    payload = load_modality_joint_checkpoint(checkpoint_path)
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
            or validation.get("training_signature") != training_signature
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
            record = _visual_record(
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
                        for key in _VISIBLE_HELDOUT_METRICS
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
            for key in _VISIBLE_HELDOUT_METRICS
        }
        for status in rank_statuses
        if status["metrics"]["heldout_history"]
    }
    report = generate_visible_validation_report(
        layout.run_dir,
        records,
        run_id=args.run_id,
        provenance={
            "checkpoint_sha256": _sha256(checkpoint_path),
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
        "training_signature": training_signature,
        "cuda_confirmed": cuda_confirmed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
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
            "training_signature": training_signature,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rank, world_size, local_rank = _distributed_identity()
    joint_config = _load_joint_config(args.config, expected_execution="expert_parallel")
    if joint_config.get("rank_assignment") != "one_modality_per_rank":
        raise ValueError("modality Expert Parallel requires rank_assignment: one_modality_per_rank")
    specs = _instance_specs(joint_config)
    current_attempt = _attempt_id()
    training_signature = _training_signature(args.config, specs)
    grouped = _modality_groups(specs)
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
        _train_owned_modality(
            args,
            layout=layout,
            modality=modality,
            entries=grouped[modality],
            epochs=epochs,
            rank=rank,
            local_rank=local_rank,
            device=device,
            attempt_id=current_attempt,
            training_signature=training_signature,
        )
    except Exception as exc:
        atomic_write_json(
            _rank_status_path(layout, rank),
            {
                "rank": rank,
                "local_rank": local_rank,
                "status": "failed",
                "attempt_id": current_attempt,
                "training_signature": training_signature,
                "owned_modality": modality.value,
                "channels": [JointExpertKey.parse(key).channel_id for key, _ in grouped[modality]],
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    if rank != 0:
        return 0
    statuses = _read_completed_rank_statuses(
        layout,
        assignments=assignments,
        attempt_id=current_attempt,
        training_signature=training_signature,
        timeout_seconds=args.coordination_timeout_seconds,
    )
    _publish_joint_release(
        args,
        layout=layout,
        assignments=assignments,
        rank_statuses=statuses,
        world_size=world_size,
        attempt_id=current_attempt,
        training_signature=training_signature,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
