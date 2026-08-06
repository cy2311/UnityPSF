from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from unity_psf.config import load_config
from unity_psf.contracts import ChannelLayout, logical_sha256, sha256_file
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.runtime import ensure_run_layout, write_run_manifest, write_stage_status
from unity_psf.training import build_trainer_runtime, initialize_model_from_checkpoint, train_epochs
from unity_psf.training.localizer_eval import build_localizer_eval_provider, localizer_eval_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one UnityPSF localization expert training instance.")
    parser.add_argument("--config", required=True, help="Resolved YAML config.")
    parser.add_argument("--run-root", required=True, help="Directory containing the run directory.")
    parser.add_argument("--run-name", required=True, help="Relative run directory name.")
    parser.add_argument("--seed", type=int, default=0, help="Training batch seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config_base_dir = Path(args.config).resolve().parent
    train_cfg = config.get("train", {})
    localizer_eval = localizer_eval_route(train_cfg, config_base_dir=config_base_dir)
    layout = ensure_run_layout(Path(args.run_root), args.run_name, stage_names=("localization_training",))
    try:
        runtime_config = build_localization_runtime_config(
            config,
            config_base_dir=config_base_dir,
            seed=int(args.seed),
        )
        runtime = build_trainer_runtime(
            runtime_config,
            layout=layout,
            model_registry=build_localization_model_registry(),
        )
        runtime, initialization = _prepare_instance_runtime(
            runtime,
            runtime_config,
            config_base_dir=config_base_dir,
        )
        write_run_manifest(
            layout,
            {
                "stage": "localization_training",
                "config_path": str(args.config),
                "seed": int(args.seed),
                "localizer_eval": localizer_eval,
                "model": runtime_config["model"],
                "expert_instance": runtime_config.get("expert_instance"),
                "initialization": initialization,
            },
        )
        eval_provider = build_localizer_eval_provider(train_cfg, config_base_dir=config_base_dir)
        results = train_epochs(
            model=runtime.model,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            batch_provider=runtime.batch_provider,
            layout=runtime.layout,
            config=runtime.config,
            loss_fn=runtime.loss_fn,
            eval_provider=eval_provider,
        )
        best_checkpoint_path = runtime.layout.checkpoints_dir / "checkpoint_best.pt"
        write_stage_status(
            layout,
            "localization_training",
            "completed",
            {
                "epochs": [result.epoch for result in results],
                "global_step": results[-1].global_step if results else 0,
                "training_budget_mode": "max_batches" if runtime.config.max_batches is not None else "epochs",
                "max_batches": runtime.config.max_batches,
                "localizer_eval": {
                    **localizer_eval,
                    "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
                },
            },
        )
    except Exception as exc:
        write_stage_status(layout, "localization_training", "failed", {"error": str(exc)})
        raise
    return 0


def _prepare_instance_runtime(runtime, runtime_config: Mapping[str, Any], *, config_base_dir: Path):
    instance = runtime_config.get("expert_instance")
    if not isinstance(instance, Mapping):
        return runtime, None
    checkpoint_metadata = getattr(runtime.model, "checkpoint_metadata", None)
    if not callable(checkpoint_metadata):
        raise TypeError("an expert instance model must expose checkpoint_metadata()")

    prototype_ref = instance.get("prototype_ref")
    initialization: dict[str, Any]
    if prototype_ref is None:
        parent_hash = logical_sha256(runtime.model.state_dict())
        initialization = {"source": "random_initialization", "sha256": parent_hash}
    else:
        prototype_path = Path(str(prototype_ref))
        if not prototype_path.is_absolute():
            prototype_path = (config_base_dir / prototype_path).resolve()
        initialize_model_from_checkpoint(
            prototype_path,
            model=runtime.model,
            expected_expert_type=str(instance["expert_type"]),
        )
        parent_hash = sha256_file(prototype_path)
        initialization = {
            "source": "prototype_checkpoint",
            "path": str(prototype_path),
            "sha256": parent_hash,
        }

    channel_layout = ChannelLayout.from_value(runtime_config["channel_layout"])
    channel_id = str(instance["channel_id"])
    channel_spec = next(channel for channel in channel_layout.channels if channel.channel_id == channel_id)
    metadata = checkpoint_metadata(
        checkpoint_role="instance",
        instance_id=str(instance["instance_id"]),
        channel_spec=channel_spec,
        parent_checkpoint_hash=parent_hash,
    )
    return replace(runtime, config=replace(runtime.config, checkpoint_metadata=metadata)), initialization


if __name__ == "__main__":
    raise SystemExit(main())
