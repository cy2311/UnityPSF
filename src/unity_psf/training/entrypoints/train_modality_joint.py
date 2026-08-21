"""Train shared modality experts over independent measurement channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from unity_psf.models import UnityPSF
from unity_psf.runtime import ensure_run_layout, write_run_manifest, write_stage_status
from unity_psf.training import (
    ModalityTrainingRuntime,
    commit_modality_joint_checkpoint,
    train_modality_epoch,
)
from unity_psf.training.modality_runtime import (
    build_modality_runtime,
    modality_groups,
)
from unity_psf.training.joint_config import instance_specs, load_joint_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one shared expert per PSF modality over all configured channels."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", help="Override every modality device, for example cuda:0.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    joint_config = load_joint_config(args.config)
    specs = instance_specs(joint_config)
    grouped = modality_groups(specs)
    layout = ensure_run_layout(args.run_root, args.run_id, stage_names=("modality_joint_training",))
    runtimes: dict[str, ModalityTrainingRuntime] = {}
    runtime_contracts: dict[str, dict[str, Mapping[str, Any]]] = {}
    for modality, entries in grouped.items():
        runtime, contracts, _ = build_modality_runtime(
            modality,
            entries,
            joint_path=args.config,
            run_layout=layout,
            device=args.device,
        )
        runtimes[modality.value] = runtime
        runtime_contracts[modality.value] = contracts

    epochs = int(joint_config.get("epochs", 1))
    metrics = {
        modality: {
            "optimizer_steps": 0,
            "attempted_optimizer_steps": 0,
            "skipped_optimizer_steps": 0,
            "schedule": [],
            "channels": {
                channel_id: {
                    "steps": 0,
                    "optimizer_steps": 0,
                    "skipped_optimizer_steps": 0,
                    "samples": 0,
                    "losses": [],
                }
                for channel_id in runtime.channels
            },
        }
        for modality, runtime in runtimes.items()
    }
    for epoch in range(1, epochs + 1):
        for modality, runtime in runtimes.items():
            result = train_modality_epoch(runtime=runtime, epoch=epoch)
            metrics[modality]["optimizer_steps"] += result.optimizer_steps
            metrics[modality]["attempted_optimizer_steps"] += result.attempted_optimizer_steps
            metrics[modality]["skipped_optimizer_steps"] += result.skipped_optimizer_steps
            metrics[modality]["schedule"].extend(result.schedule)
            for channel_id in runtime.channels:
                channel_metrics = metrics[modality]["channels"][channel_id]
                channel_metrics["steps"] += result.step_counts[channel_id]
                channel_metrics["optimizer_steps"] += result.optimizer_steps_by_channel[channel_id]
                channel_metrics["skipped_optimizer_steps"] += (
                    result.skipped_optimizer_steps_by_channel[channel_id]
                )
                channel_metrics["samples"] += result.sample_counts[channel_id]
                channel_metrics["losses"].extend(result.losses_by_channel[channel_id])

    checkpoint_path = layout.checkpoints_dir / "unitypsf_joint.ckpt"
    commit_modality_joint_checkpoint(
        checkpoint_path,
        runtimes=runtimes,
        completed_modalities=tuple(runtimes),
        role="release",
        provenance={"joint_config": str(args.config.resolve()), "execution": "modality_joint"},
    )
    loaded = UnityPSF.from_checkpoint(checkpoint_path, device="cpu")
    for modality, runtime in runtimes.items():
        for channel_id, stream in runtime.channels.items():
            batch = next(iter(stream.batches(epochs + 1)))
            loaded.localize(
                batch.images.detach().cpu(),
                modality=modality,
                channel_id=channel_id,
                conditions=None if batch.conditions is None else batch.conditions.detach().cpu(),
            )
    summary = {
        "schema_version": "unitypsf.modality_joint_training_summary.v2",
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "modalities": metrics,
        "smoke_activation_counts": loaded.activation_audit(),
    }
    summary_path = layout.metrics_dir / "joint_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(
        layout,
        {
            "config": str(args.config.resolve()),
            "training_unit": "modality",
            "runtime_contracts": runtime_contracts,
        },
    )
    write_stage_status(
        layout,
        "modality_joint_training",
        "completed",
        {"checkpoint": str(checkpoint_path), "summary": str(summary_path)},
    )
    print(json.dumps({"status": "complete", "checkpoint": str(checkpoint_path), "summary": str(summary_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
