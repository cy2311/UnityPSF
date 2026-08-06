"""Train one dual-modality, multichannel UnityPSF model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from unity_psf.config import load_config, resolve_config_reference
from unity_psf.contracts import ChannelLayout, JointExpertKey, load_joint_checkpoint
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.localization.training_adapter import LocalizationTrainBatch
from unity_psf.models import UnityPSF
from unity_psf.reporting import InstanceVisualRecord, generate_visible_validation_report
from unity_psf.runtime import ensure_run_layout, write_run_manifest, write_stage_status
from unity_psf.training import (
    ExpertTrainingUnit,
    MultimodalTrainingPlan,
    RoutedTrainingBatch,
    build_trainer_runtime,
    commit_joint_training_checkpoint,
    train_round_robin_epoch,
)
from unity_psf.training.run_localization import _prepare_instance_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train configured PSF modality/channel instances as one UnityPSF.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", help="Override every instance device, for example cuda:0.")
    return parser


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_joint_config(path: Path, *, expected_execution: str | None = "round_robin") -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _mapping(value, "joint config")
    if config.get("schema_version") != "unitypsf.joint_training.v1":
        raise ValueError("unsupported joint training schema")
    execution = str(config.get("execution", "round_robin"))
    if expected_execution is not None and execution != expected_execution:
        raise ValueError(f"this entrypoint requires execution: {expected_execution}")
    return config


def _instance_specs(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = config.get("instances")
    if not isinstance(raw, list) or not raw:
        raise ValueError("joint config instances must be a non-empty list")
    specs: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        spec = _mapping(item, "instance")
        if "key" not in spec or "config" not in spec:
            raise ValueError("every joint instance requires key and config")
        key = JointExpertKey.parse(str(spec["key"])).storage_key
        if key in specs:
            raise ValueError(f"duplicate joint instance key {key!r}")
        if JointExpertKey.parse(key).modality.value not in {"emitter_2d", "astigmatism"}:
            raise ValueError(f"unsupported trainable modality in joint config: {key!r}")
        specs[key] = spec
    if {JointExpertKey.parse(key).modality.value for key in specs} != {"emitter_2d", "astigmatism"}:
        raise ValueError("joint config must include emitter_2d and astigmatism instances")
    return specs


def _bind_instance(config: Mapping[str, Any], key: str, *, device: str | None) -> dict[str, Any]:
    bound = deepcopy(dict(config))
    train = dict(_mapping(bound.get("train"), "train"))
    parsed = JointExpertKey.parse(key)
    layout = ChannelLayout.from_value(train.get("channel_layout", {"channels": ["main"]}))
    matching = [channel for channel in layout.channels if channel.channel_id == parsed.channel_id]
    if matching:
        channel = matching[0]
    elif len(layout.channels) == 1:
        template = layout.channels[0]
        channel = type(template)(
            channel_id=parsed.channel_id,
            crop=template.crop,
            anchor_profile=template.anchor_profile,
            calibration_ref=template.calibration_ref,
        )
    else:
        raise ValueError(f"config has no measurement channel {parsed.channel_id!r}")
    channel_value = {
        "id": channel.channel_id,
        "crop": channel.crop,
        "anchor_profile": channel.anchor_profile,
        "calibration_ref": channel.calibration_ref,
    }
    train["channel_layout"] = {
        "channels": [{name: value for name, value in channel_value.items() if value is not None}],
        **({"frame_size": list(layout.frame_size)} if layout.frame_size is not None else {}),
    }
    train["expert"] = {
        **dict(_mapping(train.get("expert", {}), "train.expert")),
        "expert_type": parsed.modality.value,
        "instance_id": parsed.channel_id,
        "channel_id": parsed.channel_id,
    }
    if device is not None:
        train["device"] = device
    bound["train"] = train
    return bound


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visual_record(
    key: str,
    batch: RoutedTrainingBatch,
    localization,
    *,
    losses: Sequence[float],
    steps: int,
    checkpoint_hash: str,
) -> InstanceVisualRecord:
    image_stack = batch.images[0].detach().cpu()
    input_image = image_stack.mean(dim=0).numpy()
    probability = localization.decoded.p[0].detach().cpu()
    photon_map = localization.decoded.pxyz_mu[0, 0].detach().cpu().clamp_min(0)
    selected = torch.topk(probability.reshape(-1), k=min(8, probability.numel())).indices
    width = probability.shape[1]
    prediction_xy = torch.stack((selected % width, selected // width), dim=1).numpy().astype(np.float32)
    reconstruction = (probability * photon_map).numpy()
    return InstanceVisualRecord(
        instance_key=key,
        input_image=input_image,
        patches=tuple(frame.numpy() for frame in image_stack),
        loss_history=tuple(losses),
        route_count=1,
        step_count=steps,
        sample_count=steps * int(batch.images.shape[0]),
        prediction_xy=prediction_xy,
        reconstruction=reconstruction,
        status="trained-smoke",
        checkpoint_hash=checkpoint_hash,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    joint_config = _load_joint_config(args.config)
    specs = _instance_specs(joint_config)
    layout = ensure_run_layout(args.run_root, args.run_id, stage_names=("joint_training",))
    runtimes = {}
    runtime_configs = {}
    for key in specs:
        spec = specs[key]
        config_path = resolve_config_reference(spec["config"], source_path=args.config)
        instance_config = _bind_instance(load_config(config_path), key, device=args.device)
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
        runtime, _ = _prepare_instance_runtime(runtime, runtime_config, config_base_dir=config_path.parent)
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
            _visual_record(
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
        provenance={"checkpoint_sha256": _sha256(checkpoint_path), "evidence_level": "synthetic-smoke"},
    )
    summary = {
        "schema_version": "unitypsf.joint_training_summary.v1",
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
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
