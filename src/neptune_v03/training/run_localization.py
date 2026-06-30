from __future__ import annotations

import argparse
from pathlib import Path

from neptune_v03.config import load_config
from neptune_v03.localization import build_localization_model_registry, build_localization_runtime_config
from neptune_v03.runtime import ensure_run_layout, write_run_manifest, write_stage_status
from neptune_v03.training import build_trainer_runtime, train_epochs
from neptune_v03.training.localizer_eval import build_localizer_eval_provider, localizer_eval_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neptune v0.3 localization training.")
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
    write_run_manifest(
        layout,
        {
            "stage": "localization_training",
            "config_path": str(args.config),
            "seed": int(args.seed),
            "localizer_eval": localizer_eval,
        },
    )
    try:
        runtime_config = build_localization_runtime_config(
            config,
            config_base_dir=config_base_dir,
            model_name="simple_localizer",
            model_params={"in_channels": int(config.get("train", {}).get("online_generation", {}).get("channels", 3))},
            optimizer_params={"lr": 0.01},
            seed=int(args.seed),
        )
        runtime = build_trainer_runtime(
            runtime_config,
            layout=layout,
            model_registry=build_localization_model_registry(),
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


if __name__ == "__main__":
    raise SystemExit(main())
