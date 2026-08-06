"""Train shared modality experts over independent measurement channels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from unity_psf.config import load_config, resolve_config_reference
from unity_psf.contracts import ChannelLayout, JointExpertKey, PSFModality
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.localization.online import OnlineBatchProviderConfig, build_online_batch_provider
from unity_psf.localization.training_adapter import LocalizationTrainBatch
from unity_psf.models import UnityPSF
from unity_psf.optics.empirical_psf import load_empirical_focal_psf
from unity_psf.runtime import ensure_run_layout, write_run_manifest, write_stage_status
from unity_psf.training import (
    ChannelTrainingContext,
    ModalityChannelStream,
    ModalityTrainingBatch,
    ModalityTrainingRuntime,
    build_runtime_batch_provider,
    build_runtime_loss,
    build_trainer_runtime,
    commit_modality_joint_checkpoint,
    train_modality_epoch,
)
from unity_psf.training.localizer_eval import (
    build_localizer_eval_provider,
    evaluate_localizer_heldout,
    localizer_eval_route,
    make_legacy_localization_eval_loss,
)
from unity_psf.training.run_localization import _prepare_instance_runtime
from unity_psf.training.channel_context import sha256_file

from .train_joint import _bind_instance, _instance_specs, _load_joint_config


FORMAL_DENSITY_UM2 = 0.5
FORMAL_BACKGROUND_PHOTONS = 110.0
FORMAL_PHOTON_MEAN = 20000.0
FORMAL_PHOTON_SIGMA = 1000.0
FORMAL_PHOTON_RANGE = (0.0, 31000.0)
FORMAL_RECENTER_MODE = "fd_deeploc_exact_recenter"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one shared expert per PSF modality over all configured channels."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", help="Override every modality device, for example cuda:0.")
    return parser


def _config_path(joint_path: Path, value: object) -> Path:
    return resolve_config_reference(str(value), source_path=joint_path)


def _modality_groups(
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[PSFModality, list[tuple[str, Mapping[str, Any]]]]:
    grouped: dict[PSFModality, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for key, spec in specs.items():
        grouped[JointExpertKey.parse(key).modality].append((key, spec))
    return dict(grouped)


def _shared_runtime_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    training_runtime = config.get("resolved_contract", {}).get("training_runtime", {})
    return {
        "device": config.get("device"),
        "model": config.get("model"),
        "optimizer": config.get("optimizer"),
        "training_runtime": training_runtime,
    }


def _modality_batches(provider, channel_id: str):
    def batches(epoch: int):
        for training_batch in provider(epoch):
            local_batch = training_batch.inputs
            if not isinstance(local_batch, LocalizationTrainBatch):
                raise TypeError("modality joint training requires LocalizationTrainBatch")
            model_input = local_batch.model_input
            if isinstance(model_input, tuple):
                images, conditions = model_input
            else:
                images, conditions = model_input, None
            yield ModalityTrainingBatch(
                images=images,
                conditions=conditions,
                target=training_batch,
                channel_id=channel_id,
            )

    return batches


def _channel_metadata(
    runtime_config: Mapping[str, Any],
    *,
    physical_context: ChannelTrainingContext,
    config_path: Path,
    key: str,
    data_seed: int,
    experiment_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parsed = JointExpertKey.parse(key)
    layout = ChannelLayout.from_value(runtime_config["channel_layout"])
    channel = layout[parsed.channel_id]
    physical_state = json.loads(
        physical_context.physical_state_path.read_text(encoding="utf-8")
    )
    calibration = {"calibration_ref": channel.calibration_ref}
    provenance = {
        "training_instance": key,
        "config": str(config_path),
        "data_seed": int(data_seed),
    }
    provider = runtime_config.get("batch_provider", {})
    provider_params = provider.get("params", {}) if isinstance(provider, Mapping) else {}
    sampling = _sampling_recenter_metadata(provider_params)
    provenance.update(sampling)
    calibration.update(
        {
            name: sampling[name]
            for name in (
                "camera_baseline_adu",
                "camera_qe",
                "camera_e_per_adu",
                "train_background_adu",
                "recenter_mode",
            )
        }
    )
    if isinstance(provider_params, Mapping) and provider_params.get("psf_type") == "empirical_focal":
        empirical = load_empirical_focal_psf(
            str(provider_params["empirical_psf_path"]),
            channel_id=str(provider_params["empirical_psf_channel"]),
            focus_index=int(provider_params["empirical_psf_focus_index"]),
        )
        calibration.update(empirical.metadata())
    if isinstance(provider_params, Mapping) and provider_params.get("psf_type") == "vector":
        entries = provider_params.get("dual_domain_coeff_maps", ())
        if isinstance(entries, (list, tuple)) and len(entries) == 1 and isinstance(entries[0], Mapping):
            path_value = entries[0].get("coeff_maps_npz")
            if path_value is not None:
                path = Path(str(path_value)).resolve()
                calibration.update(
                    {
                        "psf_type": "vector",
                        "coefficient_map": str(path),
                        "coefficient_map_sha256": sha256_file(path),
                    }
                )
    if experiment_metadata is not None:
        for name in ("real_sample_root", "real_sample_acquisition", "real_sample_frames"):
            if name in experiment_metadata:
                provenance[name] = experiment_metadata[name]
    return physical_state, calibration, provenance


def _sampling_recenter_metadata(provider_params: Mapping[str, Any]) -> dict[str, Any]:
    density = provider_params.get("emitter_density_um2")
    background = tuple(float(value) for value in provider_params.get("background_range", ()))
    photon_range = tuple(float(value) for value in provider_params.get("photon_range", ()))
    photon_mean = provider_params.get("photon_mean")
    photon_sigma = provider_params.get("photon_sigma")
    camera_baseline = float(provider_params.get("camera_baseline", 0.0))
    camera_qe = float(provider_params.get("camera_qe", 0.0))
    camera_e_per_adu = float(provider_params.get("camera_e_per_adu", 0.0))
    width = int(provider_params.get("width", 0))
    height = int(provider_params.get("height", 0))
    pixel_size_nm_x = float(provider_params.get("pixel_size_nm_x", 0.0))
    pixel_size_nm_y = float(provider_params.get("pixel_size_nm_y", 0.0))
    density_value = None if density is None else float(density)
    target_active = None
    if density_value is not None:
        target_active = (
            density_value
            * width
            * pixel_size_nm_x
            / 1000.0
            * height
            * pixel_size_nm_y
            / 1000.0
        )
    train_background_photons = background[0] if len(background) == 2 and background[0] == background[1] else None
    train_background_adu = None
    if train_background_photons is not None and camera_e_per_adu > 0.0:
        train_background_adu = (
            camera_baseline + train_background_photons * camera_qe / camera_e_per_adu
        )
    return {
        "emitter_density_um2": density_value,
        "target_active_emitters_per_frame": target_active,
        "background_range_photons": list(background),
        "photon_mean": None if photon_mean is None else float(photon_mean),
        "photon_sigma": None if photon_sigma is None else float(photon_sigma),
        "photon_range": list(photon_range),
        "camera_baseline_adu": camera_baseline,
        "camera_qe": camera_qe,
        "camera_e_per_adu": camera_e_per_adu,
        "train_background_photons": train_background_photons,
        "train_background_adu": train_background_adu,
        "recenter_mode": FORMAL_RECENTER_MODE,
    }


def _condition_store_provider_overrides(context: ChannelTrainingContext):
    def online_train_batch(params: dict[str, object]):
        return build_online_batch_provider(
            OnlineBatchProviderConfig(**params),
            condition_store=context.condition_store,
        )

    return {"online_train_batch": online_train_batch}


def _physical_state_snapshot(context: ChannelTrainingContext) -> dict[str, Any]:
    return json.loads(context.physical_state_path.read_text(encoding="utf-8"))


def _audit_formal_runtime_contracts(
    modality: PSFModality | str,
    runtime_configs: Mapping[str, Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> dict[str, dict[str, Any]]:
    parsed_modality = PSFModality.parse(modality)
    required_channels = (
        {"left"}
        if parsed_modality is PSFModality.EMITTER_2D
        else {"left", "right"}
    )
    if require_complete and set(runtime_configs) != required_channels:
        channel_label = "left" if required_channels == {"left"} else "left and right"
        raise ValueError(
            f"formal {parsed_modality.value} training requires {channel_label} channels"
        )

    physical_hashes: dict[str, str] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for channel_id, runtime_config in runtime_configs.items():
        optimizer = runtime_config.get("optimizer", {})
        if str(optimizer.get("name", "")).lower() != "adamw":
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} training requires AdamW")
        optimizer_params = optimizer.get("params", {})
        if not isinstance(optimizer_params, Mapping) or float(optimizer_params.get("weight_decay", -1.0)) != 0.1:
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} AdamW requires weight_decay=0.1")
        if float(optimizer_params.get("lr", -1.0)) != 0.0006:
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} AdamW requires lr=0.0006")

        loss = runtime_config.get("loss", {})
        loss_params = loss.get("params", {}) if isinstance(loss, Mapping) else {}
        if not isinstance(loss, Mapping) or loss.get("name") != "active_smlm_gmm_loss":
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} training requires GMM loss")
        if not isinstance(loss_params, Mapping) or loss_params.get("target_order") != "legacy_iwae":
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} GMM target order must be legacy_iwae")
        if loss_params.get("gmm_backend") != "mixture_same_family":
            raise ValueError(
                f"formal {parsed_modality.value}:{channel_id} GMM backend must be mixture_same_family"
            )
        if (
            int(loss_params.get("gmm_target_chunk", -1)) != 4
            or int(loss_params.get("gmm_component_chunk", -1)) != 64
        ):
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} GMM chunks must be 4/64")

        training = runtime_config.get("resolved_contract", {}).get("training_runtime", {})
        amp = training.get("amp", {}) if isinstance(training, Mapping) else {}
        grad_clip = training.get("grad_clip", {}) if isinstance(training, Mapping) else {}
        scheduler = training.get("scheduler", {}) if isinstance(training, Mapping) else {}
        if not isinstance(amp, Mapping) or amp.get("active") is not True or amp.get("dtype") != "float16":
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} training requires float16 AMP")
        if not isinstance(grad_clip, Mapping) or float(grad_clip.get("configured_norm", -1.0)) != 0.03:
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} training requires grad_clip_norm=0.03")
        scheduler_params = scheduler.get("params", {}) if isinstance(scheduler, Mapping) else {}
        if (
            not isinstance(scheduler, Mapping)
            or scheduler.get("name") != "StepLR"
            or scheduler.get("step_unit") != "epoch"
            or not isinstance(scheduler_params, Mapping)
            or int(scheduler_params.get("step_size", -1)) != 10
            or float(scheduler_params.get("gamma", -1.0)) != 0.9
        ):
            raise ValueError(
                f"formal {parsed_modality.value}:{channel_id} requires StepLR(step_size=10, gamma=0.9) per epoch"
            )

        evidence[channel_id] = {
            "optimizer": "AdamW",
            "learning_rate": 0.0006,
            "weight_decay": 0.1,
            "scheduler": "StepLR",
            "scheduler_step_size": 10,
            "scheduler_gamma": 0.9,
            "scheduler_step_unit": "epoch",
            "loss": "active_smlm_gmm_loss",
            "target_order": "legacy_iwae",
            "gmm_backend": "mixture_same_family",
            "gmm_target_chunk": 4,
            "gmm_component_chunk": 64,
            "amp_dtype": "float16",
            "grad_clip_norm": 0.03,
        }

        provider = runtime_config.get("batch_provider", {})
        provider_params = provider.get("params", {}) if isinstance(provider, Mapping) else {}
        if not isinstance(provider_params, Mapping):
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} provider params are invalid")
        sampling = _sampling_recenter_metadata(provider_params)
        if sampling["emitter_density_um2"] != FORMAL_DENSITY_UM2:
            raise ValueError(
                f"formal {parsed_modality.value}:{channel_id} requires emitter density=0.5 emitters/um2"
            )
        if tuple(sampling["background_range_photons"]) != (
            FORMAL_BACKGROUND_PHOTONS,
            FORMAL_BACKGROUND_PHOTONS,
        ):
            raise ValueError(
                f"formal {parsed_modality.value}:{channel_id} requires fixed background=110 photons/pixel"
            )
        if (
            sampling["photon_mean"] != FORMAL_PHOTON_MEAN
            or sampling["photon_sigma"] != FORMAL_PHOTON_SIGMA
            or tuple(sampling["photon_range"]) != FORMAL_PHOTON_RANGE
        ):
            raise ValueError(
                f"formal {parsed_modality.value}:{channel_id} requires photon mean/sigma=20000/1000 "
                "and range=0..31000"
            )
        evidence[channel_id].update(sampling)

        if parsed_modality in {PSFModality.EMITTER_2D, PSFModality.ASTIGMATISM}:
            if provider_params.get("psf_type") != "vector" or provider_params.get("simulation_backend") != "lut":
                raise ValueError(
                    f"formal {parsed_modality.value}:{channel_id} requires the shared vector LUT infrastructure"
                )
            entries = provider_params.get("dual_domain_coeff_maps", ()) if isinstance(provider_params, Mapping) else ()
            if not isinstance(entries, (list, tuple)) or len(entries) != 1:
                raise ValueError(f"formal {parsed_modality.value}:{channel_id} requires one coefficient map")
            entry = entries[0]
            path_value = entry.get("coeff_maps_npz") if isinstance(entry, Mapping) else None
            if path_value is None:
                raise ValueError(f"formal {parsed_modality.value}:{channel_id} coefficient map path is missing")
            path = Path(str(path_value)).resolve()
            if not path.is_file():
                raise ValueError(
                    f"formal {parsed_modality.value}:{channel_id} coefficient map does not exist: {path}"
                )
            physical_hashes[channel_id] = sha256_file(path)
            evidence[channel_id].update(
                {
                    "psf_type": "vector",
                    "simulation_backend": "lut",
                    "z_range_um": list(provider_params.get("z_range", ())),
                    "coefficient_map": str(path),
                    "coefficient_map_sha256": physical_hashes[channel_id],
                }
            )
        if parsed_modality is PSFModality.EMITTER_2D:
            if tuple(float(value) for value in provider_params.get("z_range", ())) != (-0.1, 0.1):
                raise ValueError(f"formal emitter_2d:{channel_id} requires nuisance z range -0.1..0.1 um")
            model_params = runtime_config.get("model", {}).get("params", {})
            disabled = model_params.get("disabled_attr", ()) if isinstance(model_params, Mapping) else ()
            if 3 not in tuple(int(value) for value in disabled) or int(loss_params.get("disable_attr", -1)) != 3:
                raise ValueError(f"formal emitter_2d:{channel_id} must disable z head and z loss")

    if require_complete and len(required_channels) > 1 and parsed_modality in {
        PSFModality.EMITTER_2D,
        PSFModality.ASTIGMATISM,
    }:
        if len(set(physical_hashes.values())) != len(physical_hashes):
            raise ValueError(
                f"formal {parsed_modality.value} left/right coefficient maps must be physically distinct"
            )
    return evidence


def _build_modality_runtime(
    modality: PSFModality,
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    joint_path: Path,
    run_layout,
    device: str | None,
) -> tuple[
    ModalityTrainingRuntime,
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, Any]] | None,
]:
    if not entries:
        raise ValueError(f"modality {modality.value!r} has no configured channels")
    channels: dict[str, ModalityChannelStream] = {}
    budgets: dict[str, int] = {}
    runtime_configs: dict[str, Mapping[str, Any]] = {}
    shared_runtime = None
    shared_contract = None
    shared_model_seed = None
    formal_channels: set[str] = set()
    formal_runtime_evidence = None
    model_registry = build_localization_model_registry()
    for key, spec in entries:
        parsed = JointExpertKey.parse(key)
        config_path = _config_path(joint_path, spec["config"])
        bound_config = _bind_instance(load_config(config_path), key, device=device)
        data_seed = int(spec.get("data_seed", spec.get("seed", 0)))
        model_seed = int(spec.get("model_seed", spec.get("seed", 0)))
        runtime_config = build_localization_runtime_config(
            bound_config,
            config_base_dir=config_path.parent,
            seed=data_seed,
        )
        metadata = bound_config.get("metadata")
        formal_contract = (
            isinstance(metadata, Mapping)
            and metadata.get("formal_training_contract") == "unitypsf.formal.v1"
        )
        if formal_contract:
            formal_channels.add(parsed.channel_id)
            _audit_formal_runtime_contracts(
                modality,
                {**runtime_configs, parsed.channel_id: runtime_config},
                require_complete=False,
            )
        contract = _shared_runtime_contract(runtime_config)
        channel_layout = ensure_run_layout(
            run_layout.run_dir / "channels",
            f"{modality.value}_{parsed.channel_id}",
        )
        physical_context = ChannelTrainingContext.from_runtime_config(
            runtime_config,
            layout=channel_layout,
            metadata=(
                bound_config.get("metadata")
                if isinstance(bound_config.get("metadata"), Mapping)
                else None
            ),
        )
        physical_context.write_physical_state(source="initial")
        provider_overrides = _condition_store_provider_overrides(physical_context)
        if shared_runtime is None:
            torch.manual_seed(model_seed)
            shared_runtime = build_trainer_runtime(
                runtime_config,
                layout=channel_layout,
                model_registry=model_registry,
                batch_provider_overrides=provider_overrides,
            )
            shared_runtime, _ = _prepare_instance_runtime(
                shared_runtime,
                runtime_config,
                config_base_dir=config_path.parent,
            )
            shared_contract = contract
            shared_model_seed = model_seed
            provider = shared_runtime.batch_provider
            loss_fn = shared_runtime.loss_fn
        else:
            if model_seed != shared_model_seed or contract != shared_contract:
                raise ValueError(
                    f"all {modality.value} channels must share model seed, model, optimizer, and scheduler"
                )
            provider = build_runtime_batch_provider(
                runtime_config,
                batch_provider_overrides=provider_overrides,
            )
            loss_fn = build_runtime_loss(runtime_config)
        from_output = getattr(loss_fn, "from_output", None)
        if not callable(from_output):
            raise TypeError(f"loss for {key!r} does not support routed output")
        train_cfg = bound_config.get("train", {})
        if not isinstance(train_cfg, Mapping):
            raise ValueError(f"train config for {key!r} must be a mapping")
        eval_route = localizer_eval_route(train_cfg, config_base_dir=config_path.parent)
        if (
            eval_route.get("source") == "online_generation"
            and int(eval_route["seed"]) == data_seed
        ):
            raise ValueError(f"held-out eval seed overlaps training seed for {key!r}")
        eval_provider = build_localizer_eval_provider(
            train_cfg,
            config_base_dir=config_path.parent,
            root_config=bound_config,
            condition_store=physical_context.condition_store,
        )
        eval_loss_fn = (
            None
            if eval_provider is None
            else make_legacy_localization_eval_loss(
                loss_fn,
                train_cfg,
                root_config=bound_config,
            )
        )
        physical_state, calibration, provenance = _channel_metadata(
            runtime_config,
            physical_context=physical_context,
            config_path=config_path,
            key=key,
            data_seed=data_seed,
            experiment_metadata=(metadata if isinstance(metadata, Mapping) else None),
        )
        channels[parsed.channel_id] = ModalityChannelStream(
            channel_id=parsed.channel_id,
            batches=_modality_batches(provider, parsed.channel_id),
            loss_fn=lambda output, batch, fn=from_output: fn(output, batch.target),
            physical_state=physical_state,
            calibration=calibration,
            provenance=provenance,
            snapshot_physical_state=(
                lambda context=physical_context: _physical_state_snapshot(context)
            ),
            restore_physical_state=(
                lambda state, context=physical_context: context.restore_physical_state(state)
            ),
            heldout_eval=(
                None
                if eval_provider is None or eval_loss_fn is None
                else lambda model, provider=eval_provider, fn=eval_loss_fn: evaluate_localizer_heldout(
                    model,
                    provider=provider,
                    eval_loss_fn=fn,
                )
            ),
        )
        budgets[parsed.channel_id] = int(spec.get("step_budget", 1))
        runtime_configs[parsed.channel_id] = runtime_config
    assert shared_runtime is not None
    if formal_channels:
        if formal_channels != set(runtime_configs):
            raise ValueError(f"all {modality.value} channels must use the same formal training contract")
        formal_runtime_evidence = _audit_formal_runtime_contracts(modality, runtime_configs)
        if not isinstance(shared_runtime.optimizer, torch.optim.AdamW):
            raise ValueError(
                f"formal {modality.value} runtime built {type(shared_runtime.optimizer).__name__}, not AdamW"
            )
        if shared_runtime.scheduler is None or type(shared_runtime.scheduler).__name__ != "StepLR":
            scheduler_name = (
                None if shared_runtime.scheduler is None else type(shared_runtime.scheduler).__name__
            )
            raise ValueError(
                f"formal {modality.value} runtime built scheduler {scheduler_name}, not StepLR"
            )
        for item in formal_runtime_evidence.values():
            item["runtime_optimizer_class"] = type(shared_runtime.optimizer).__name__
            item["runtime_scheduler_class"] = type(shared_runtime.scheduler).__name__
    return (
        ModalityTrainingRuntime(
            modality=modality,
            model=shared_runtime.model,
            optimizer=shared_runtime.optimizer,
            scheduler=shared_runtime.scheduler,
            channels=channels,
            step_budgets=budgets,
            scheduler_step_unit=shared_runtime.config.scheduler_step_unit,
            grad_clip_norm=shared_runtime.config.grad_clip_norm,
            amp_enabled=shared_runtime.config.amp_enabled,
            amp_dtype=shared_runtime.config.amp_dtype,
        ),
        runtime_configs,
        formal_runtime_evidence,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    joint_config = _load_joint_config(args.config)
    specs = _instance_specs(joint_config)
    grouped = _modality_groups(specs)
    layout = ensure_run_layout(args.run_root, args.run_id, stage_names=("modality_joint_training",))
    runtimes: dict[str, ModalityTrainingRuntime] = {}
    runtime_contracts: dict[str, dict[str, Mapping[str, Any]]] = {}
    for modality, entries in grouped.items():
        runtime, contracts, _ = _build_modality_runtime(
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
