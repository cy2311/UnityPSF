"""Train one dual-modality, multichannel UnityPSF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from unity_psf.config import load_config, resolve_config_reference
from unity_psf.contracts import JointExpertKey, load_joint_checkpoint, sha256_file
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.localization.training_adapter import LocalizationTrainBatch
from unity_psf.models import UnityPSF
from unity_psf.reporting import generate_visible_validation_report
from unity_psf.runtime import ensure_run_layout, write_run_manifest, write_stage_status
from unity_psf.training import (
    ExpertTrainingUnit,
    MultimodalTrainingPlan,
    RoutedTrainingBatch,
    build_trainer_runtime,
    commit_joint_training_checkpoint,
    train_round_robin_epoch,
)
from unity_psf.training.joint_config import bind_instance, instance_specs, load_joint_config
from unity_psf.training.runtime import prepare_instance_runtime
from unity_psf.training.validation import build_instance_visual_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train configured PSF modality/channel instances as one UnityPSF.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", help="Override every instance device, for example cuda:0.")
    return parser


def _routed_batches(runtime, epoch: int):
    for training_batch in runtime.batch_provider(epoch):
        yield _routed_batch(training_batch)


def _routed_batch(training_batch) -> RoutedTrainingBatch:
    local_batch = training_batch.inputs
    if not isinstance(local_batch, LocalizationTrainBatch):
        raise TypeError("joint localization training requires LocalizationTrainBatch")
    model_input = local_batch.model_input
    if isinstance(model_input, tuple):
        images, conditions = model_input
    else:
        images, conditions = model_input, None
    return RoutedTrainingBatch(images=images, conditions=conditions, target=training_batch)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    joint_config = load_joint_config(args.config)
    specs = instance_specs(joint_config)
    layout = ensure_run_layout(args.run_root, args.run_id, stage_names=("joint_training",))
    runtimes = {}
    runtime_configs = {}
    for key in specs:
        spec = specs[key]
        config_path = resolve_config_reference(spec["config"], source_path=args.config)
        instance_config = bind_instance(load_config(config_path), key, device=args.device)
        model_seed = int(spec.get("model_seed", spec.get("seed", 0)))
        data_seed = int(spec.get("data_seed", spec.get("seed", 0)))
        torch.manual_seed(model_seed)
        runtime_config = build_localization_runtime_config(
            instance_config,
            config_base_dir=config_path.parent,
            seed=data_seed,
        )
        instance_layout = ensure_run_layout(layout.run_dir / "instances", key.replace(":", "_"))
        runtime = build_trainer_runtime(
            runtime_config,
            layout=instance_layout,
            model_registry=build_localization_model_registry(),
        )
        runtime, _ = prepare_instance_runtime(
            runtime,
            runtime_config,
            config_base_dir=config_path.parent,
        )
        runtimes[key] = runtime
        runtime_configs[key] = runtime_config

    devices = {str(next(runtime.model.parameters()).device) for runtime in runtimes.values()}
    if len(devices) != 1:
        raise ValueError("round-robin execution requires every expert on the same device")
    model = UnityPSF({key: runtime.model for key, runtime in runtimes.items()}, target_device=devices.pop())
    plan = MultimodalTrainingPlan.from_step_budgets(
        {key: int(specs[key].get("step_budget", 1)) for key in specs}
    )
    if any(plan.step_budgets[key] <= 0 for key in plan.instance_keys):
        raise ValueError("dual-modality release requires a positive step budget for every expert instance")
    losses = {key: [] for key in plan.instance_keys}
    steps = {key: 0 for key in plan.instance_keys}
    schedule: list[str] = []
    for epoch in range(1, int(joint_config.get("epochs", 1)) + 1):
        units = {}
        for key in plan.instance_keys:
            runtime = runtimes[key]
            from_output = getattr(runtime.loss_fn, "from_output", None)
            if not callable(from_output):
                raise TypeError(f"loss for {key!r} does not support routed output")
            units[key] = ExpertTrainingUnit(
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                batches=_routed_batches(runtime, epoch),
                loss_fn=lambda output, batch, fn=from_output: fn(output, batch.target),
            )
        result = train_round_robin_epoch(model=model, plan=plan, units=units, epoch=epoch)
        schedule.extend(result.schedule)
        for key in plan.instance_keys:
            steps[key] += result.step_counts[key]
            losses[key].extend(result.losses_by_instance[key])

    checkpoint_path = layout.checkpoints_dir / "unitypsf_joint.ckpt"
    commit_joint_training_checkpoint(
        checkpoint_path,
        model=model,
        plan=plan,
        completed_instances=plan.instance_keys,
        optimizers={key: runtimes[key].optimizer for key in plan.instance_keys},
        schedulers={key: runtimes[key].scheduler for key in plan.instance_keys},
        role="release",
        provenance={"joint_config": str(args.config.resolve()), "execution": "round_robin"},
    )

    loaded = UnityPSF.from_checkpoint(checkpoint_path, device="cpu")
    payload = load_joint_checkpoint(checkpoint_path)
    visual_records = []
    for key in plan.instance_keys:
        batch = next(iter(_routed_batches(runtimes[key], int(joint_config.get("epochs", 1)) + 1)))
        parsed = JointExpertKey.parse(key)
        localization = loaded.localize(
            batch.images,
            modality=parsed.modality,
            channel_id=parsed.channel_id,
            conditions=batch.conditions,
        )
        visual_records.append(
            build_instance_visual_record(
                key,
                batch,
                localization,
                losses=losses[key],
                steps=steps[key],
                checkpoint_hash=payload["integrity"]["expert_sha256"][key],
            )
        )
    report = generate_visible_validation_report(
        layout.run_dir,
        visual_records,
        run_id=args.run_id,
        provenance={"checkpoint_sha256": sha256_file(checkpoint_path), "evidence_level": "synthetic-smoke"},
    )
    summary = {
        "schema_version": "unitypsf.joint_training_summary.v1",
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "instances": {
            key: {"steps": steps[key], "losses": losses[key], "parameter_count": model.describe()["parameter_counts"][key]}
            for key in plan.instance_keys
        },
        "schedule": schedule,
        "smoke_activation_counts": loaded.activation_audit(),
        "report": str(report.report_path),
    }
    summary_path = layout.metrics_dir / "joint_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(
        layout,
        {
            "config": str(args.config.resolve()),
            "instances": {
                key: {
                    "expert_instance": runtime_configs[key]["expert_instance"],
                    "resolved_contract": runtime_configs[key]["resolved_contract"],
                }
                for key in plan.instance_keys
            },
        },
    )
    write_stage_status(layout, "joint_training", "completed", {"checkpoint": str(checkpoint_path), "summary": str(summary_path)})
    print(
        json.dumps(
            {
                "status": "complete",
                "checkpoint": str(checkpoint_path),
                "summary": str(summary_path),
                "report": str(report.report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
