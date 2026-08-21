from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from unity_psf.config import load_config
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.runtime import ensure_run_layout, write_run_manifest, write_stage_status
from unity_psf.training import (
    EpochTrainingConfig,
    TrainingResumeState,
    build_trainer_runtime,
    load_training_checkpoint,
    train_epochs,
)
from unity_psf.training.channel_context import ChannelTrainingContext, sha256_file
from unity_psf.training.high_fidelity.condition_runtime import (
    condition_store_batch_provider_overrides,
    condition_store_from_runtime_config,
)
from unity_psf.training.high_fidelity.peak_bootstrap import run_peak_zmap_bootstrap_if_enabled
from unity_psf.training.high_fidelity.physical_state import (
    _physical_checkpoint_extra_fn,
    _write_initial_physical_state,
)
from unity_psf.training.high_fidelity.gamma_runtime import (
    build_roi_bank_gamma_hook,
    count_gamma_updates,
    gamma_hook_bindings,
    roi_bank_gamma_route,
)
from unity_psf.training.localizer_eval import build_localizer_eval_provider, localizer_eval_route, make_legacy_localization_eval_loss
from unity_psf.training.loop import TrainingRunEpochResult, TrainingStepResult


STAGE_NAME = "high_fidelity_localization"




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neptune v0.4 high-fidelity localization training.")
    parser.add_argument("--config", required=True, help="Resolved YAML config.")
    parser.add_argument("--run-root", required=True, help="Directory containing the run directory.")
    parser.add_argument("--run-name", required=True, help="Relative run directory name.")
    parser.add_argument("--seed", type=int, default=0, help="Training batch seed.")
    parser.add_argument("--resume-checkpoint", help="Checkpoint whose completed epoch should be resumed.")
    return parser.parse_args()


def resume_epoch_training_config(
    config: EpochTrainingConfig,
    state: TrainingResumeState,
) -> EpochTrainingConfig:
    next_epoch = int(state.epoch) + 1
    if next_epoch > int(config.stop_epoch):
        raise ValueError(
            f"resume checkpoint epoch {state.epoch} has already reached stop epoch {config.stop_epoch}"
        )
    return replace(
        config,
        start_epoch=next_epoch,
        global_step_start=int(state.global_step),
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config_base_dir = Path(args.config).resolve().parent
    layout = ensure_run_layout(Path(args.run_root), args.run_name, stage_names=("peak", STAGE_NAME))
    try:
        config = run_peak_zmap_bootstrap_if_enabled(config, config_base_dir=config_base_dir, layout=layout)
        feedback_path = _feedback_map_path(config)
        train_cfg = _mapping(config.get("train"), "train")
        runtime_config = build_localization_runtime_config(
            config,
            config_base_dir=config_base_dir,
            seed=int(args.seed),
        )
        model_name = str(runtime_config["model"]["name"])
        physical_context = (
            ChannelTrainingContext.from_runtime_config(
                runtime_config,
                layout=layout,
                metadata=config.get("metadata") if isinstance(config.get("metadata"), Mapping) else None,
            )
            if isinstance(runtime_config.get("expert_instance"), Mapping)
            else None
        )
        condition_store = (
            physical_context.condition_store
            if physical_context is not None
            else condition_store_from_runtime_config(runtime_config)
        )
        gamma_route = roi_bank_gamma_route(train_cfg, config=config, config_base_dir=config_base_dir)
        localizer_eval = localizer_eval_route(train_cfg, config_base_dir=config_base_dir)
        runtime = build_trainer_runtime(
            runtime_config,
            layout=layout,
            model_registry=build_localization_model_registry(),
            batch_provider_overrides=condition_store_batch_provider_overrides(condition_store),
        )
        resume_state = None
        if args.resume_checkpoint is not None:
            resume_path = Path(args.resume_checkpoint).resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(resume_path)
            resume_state = load_training_checkpoint(
                resume_path,
                model=runtime.model,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                map_location=next(runtime.model.parameters()).device,
                **_resume_identity_kwargs(runtime_config, config_base_dir=config_base_dir),
            )
            runtime = replace(
                runtime,
                config=resume_epoch_training_config(runtime.config, resume_state),
            )
        if physical_context is not None:
            if resume_state is None:
                physical_context.write_physical_state(source="initial")
            else:
                physical_context.restore_physical_state(
                    resume_state.physical_state,
                    resume_state.physical_coeff_maps,
                    initial_state_hash=resume_state.initial_physical_state_hash,
                )
        else:
            _write_initial_physical_state(layout, runtime_config)
        write_run_manifest(
            layout,
            {
                "stage": STAGE_NAME,
                "config_path": str(args.config),
                "seed": int(args.seed),
                "model_name": model_name,
                "batch_provider": runtime_config["batch_provider"]["name"],
                "feedback_map_path": feedback_path,
                "resolved_contract": runtime_config.get("resolved_contract", {}),
                "roi_bank_gamma": gamma_route,
                "localizer_eval": localizer_eval,
                **(physical_context.manifest_fields() if physical_context is not None else {}),
                **({"physical_context": physical_context.manifest_fields()} if physical_context is not None else {}),
                "resume": None
                if resume_state is None
                else {
                    "checkpoint_path": str(resume_state.path),
                    "checkpoint_epoch": int(resume_state.epoch),
                    "checkpoint_global_step": int(resume_state.global_step),
                    "checkpoint_format": resume_state.checkpoint_format,
                    "next_epoch": int(runtime.config.start_epoch),
                },
            },
        )
        gamma_hook = build_roi_bank_gamma_hook(
            train_cfg,
            config=config,
            config_base_dir=config_base_dir,
            model=runtime.model,
            layout=layout,
            condition_store=condition_store,
            physical_context=physical_context,
        )
        gamma_epoch_hook, gamma_batch_hook = gamma_hook_bindings(gamma_hook, train_cfg.get("roi_bank_gamma"))
        if resume_state is not None and gamma_epoch_hook is not None:
            gamma_epoch_hook(
                TrainingRunEpochResult(
                    epoch=int(resume_state.epoch),
                    step_count=int(resume_state.step_count),
                    mean_loss=0.0,
                    metrics_path=layout.metrics_dir / "training_metrics.jsonl",
                    checkpoint_path=resume_state.path,
                    global_step=int(resume_state.global_step),
                )
            )
        eval_provider = build_localizer_eval_provider(
            train_cfg,
            config_base_dir=config_base_dir,
            root_config=config,
            condition_store=condition_store,
        )
        eval_loss_fn = make_legacy_localization_eval_loss(runtime.loss_fn, train_cfg, root_config=config) if eval_provider is not None else None
        results = train_epochs(
            model=runtime.model,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            batch_provider=runtime.batch_provider,
            layout=runtime.layout,
            config=runtime.config,
            on_epoch_end=gamma_epoch_hook,
            on_batch_end=gamma_batch_hook,
            loss_fn=runtime.loss_fn,
            eval_provider=eval_provider,
            eval_loss_fn=eval_loss_fn,
            checkpoint_extra_fn=_physical_checkpoint_extra_fn(layout, physical_context=physical_context),
        )
        gamma_updates = count_gamma_updates(layout)
        best_checkpoint_path = layout.checkpoints_dir / "checkpoint_best.pt"
        write_stage_status(
            layout,
            STAGE_NAME,
            "completed",
            {
                "epochs": [result.epoch for result in results],
                "global_step": results[-1].global_step if results else 0,
                "training_budget_mode": "max_batches" if runtime.config.max_batches is not None else "epochs",
                "max_batches": runtime.config.max_batches,
                "model_name": model_name,
                "resume": None
                if resume_state is None
                else {
                    "checkpoint_path": str(resume_state.path),
                    "checkpoint_epoch": int(resume_state.epoch),
                    "checkpoint_global_step": int(resume_state.global_step),
                    "checkpoint_format": resume_state.checkpoint_format,
                },
                "resolved_contract": runtime_config.get("resolved_contract", {}),
                "roi_bank_gamma": {**gamma_route, "updates": gamma_updates},
                "localizer_eval": {
                    **localizer_eval,
                    "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
                },
            },
        )
    except Exception as exc:
        write_stage_status(layout, STAGE_NAME, "failed", {"error": str(exc)})
        raise
    return 0


def _feedback_map_path(config: Mapping[str, Any]) -> str | None:
    train_cfg = _mapping(config.get("train"), "train")
    feedback_cfg = train_cfg.get("feedback")
    if isinstance(feedback_cfg, Mapping) and "map_path" in feedback_cfg:
        return str(feedback_cfg["map_path"])
    return None




def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _resume_identity_kwargs(
    runtime_config: Mapping[str, Any],
    *,
    config_base_dir: Path | None = None,
) -> dict[str, object]:
    instance = runtime_config.get("expert_instance")
    if not isinstance(instance, Mapping):
        return {}
    output: dict[str, object] = {}
    for source_key, target_key in (
        ("expert_type", "expected_expert_type"),
        ("instance_id", "expected_instance_id"),
        ("channel_id", "expected_channel_id"),
        ("parent_checkpoint_hash", "expected_parent_checkpoint_hash"),
    ):
        value = instance.get(source_key)
        if value is not None:
            output[target_key] = value
    prototype_ref = instance.get("prototype_ref")
    if prototype_ref is not None and "expected_parent_checkpoint_hash" not in output:
        reference = Path(str(prototype_ref))
        if config_base_dir is not None and not reference.is_absolute():
            reference = config_base_dir / reference
        if reference.is_file():
            output["expected_parent_checkpoint_hash"] = sha256_file(reference)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
