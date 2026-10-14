"""Runtime construction for canonical UnityPSF training.

Contract validation and physical provenance live in sibling modules. This
module owns only the runtime assembly path used by the canonical executor.
Public helpers are imported from their implementation owners. Legacy private
names remain aliases for the existing entrypoint and focused tests.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from unity_psf.config import load_config
from unity_psf.contracts import JointExpertKey, PSFModality, ResolvedTrainingContract
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.reporting import InstanceVisualRecord
from unity_psf.runtime import ensure_run_layout
from unity_psf.training import (
    ChannelTrainingContext,
    ModalityChannelStream,
    ModalityTrainingRuntime,
    RoutedTrainingBatch,
    build_runtime_batch_provider,
    build_runtime_loss,
    build_trainer_runtime,
)
from unity_psf.training.localizer_eval import (
    build_localizer_eval_provider,
    evaluate_localizer_heldout,
    localizer_eval_route,
    make_legacy_localization_eval_loss,
)
from unity_psf.training.run_localization import prepare_instance_runtime
from unity_psf.training.modality_runtime import (
    condition_store_provider_overrides,
    modality_batches,
    physical_state_snapshot,
    shared_runtime_contract,
)

from .canonical_contract import (
    bind_instance,
    config_path,
    instance_specs,
    load_joint_config,
    load_resolved_training_contract,
    mapping,
    modality_groups,
    sha256,
)
from .canonical_physics import (
    FORMAL_BACKGROUND_PHOTONS,
    FORMAL_DENSITY_UM2,
    FORMAL_PHOTON_MEAN,
    FORMAL_PHOTON_RANGE,
    FORMAL_PHOTON_SIGMA,
    FORMAL_RECENTER_MODE,
    audit_formal_runtime_contracts,
    channel_metadata,
    physical_provenance,
    sampling_recenter_metadata,
)


__all__ = [
    "audit_formal_runtime_contracts",
    "bind_instance",
    "build_modality_runtime",
    "channel_metadata",
    "config_path",
    "instance_specs",
    "load_joint_config",
    "load_resolved_training_contract",
    "mapping",
    "modality_groups",
    "physical_provenance",
    "sampling_recenter_metadata",
    "sha256",
    "visual_record",
    "FORMAL_BACKGROUND_PHOTONS",
    "FORMAL_DENSITY_UM2",
    "FORMAL_PHOTON_MEAN",
    "FORMAL_PHOTON_RANGE",
    "FORMAL_PHOTON_SIGMA",
    "FORMAL_RECENTER_MODE",
    "_mapping",
    "_load_joint_config",
    "_load_resolved_training_contract",
    "_instance_specs",
    "_bind_instance",
    "_sha256",
    "_config_path",
    "_modality_groups",
    "_channel_metadata",
    "_physical_provenance",
    "_sampling_recenter_metadata",
    "_audit_formal_runtime_contracts",
    "_visual_record",
    "_build_modality_runtime",
]


def visual_record(
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
    prediction_xy = torch.stack((selected % width, selected // width), dim=1).numpy().astype("float32")
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


def build_modality_runtime(
    modality: PSFModality,
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    joint_path: Path,
    resolved_contract: ResolvedTrainingContract,
    run_layout,
    device: str | None,
) -> tuple[
    ModalityTrainingRuntime,
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, Any]] | None,
]:
    if not entries:
        raise ValueError(f"modality {modality.value!r} has no configured channels")
    initialization_seed = int(entries[0][1]['model_seed'])
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    channels: dict[str, ModalityChannelStream] = {}
    budgets: dict[str, int] = {}
    runtime_configs: dict[str, Mapping[str, Any]] = {}
    shared_runtime = None
    shared_contract = None
    shared_model_seed = None
    formal_channels: set[str] = set()
    formal_runtime_evidence = None
    model_registry = build_localization_model_registry()
    contract_instances = {item.key: item for item in resolved_contract.instances}
    for key, spec in entries:
        parsed = JointExpertKey.parse(key)
        instance_config_path = config_path(joint_path, spec["config"])
        contract_instance = contract_instances.get(key)
        if contract_instance is None or instance_config_path.resolve() != Path(contract_instance.config).resolve():
            raise ValueError(f"runtime instance {key!r} is not owned by the resolved training contract")
        if (
            int(spec.get("model_seed", -1)) != contract_instance.model_seed
            or int(spec.get("data_seed", -1)) != contract_instance.data_seed
            or int(spec.get("step_budget", -1)) != contract_instance.step_budget
        ):
            raise ValueError(f"runtime instance {key!r} seed or budget differs from the resolved contract")

        bound_config = bind_instance(load_config(instance_config_path), key, device=device)
        data_seed = int(spec.get("data_seed", spec.get("seed", 0)))
        model_seed = int(spec.get("model_seed", spec.get("seed", 0)))
        runtime_config = build_localization_runtime_config(
            bound_config,
            config_base_dir=instance_config_path.parent,
            seed=data_seed,
        )
        metadata = bound_config.get("metadata")
        if isinstance(metadata, Mapping) and "physical_initialization" in metadata:
            runtime_config["physical_initialization"] = metadata["physical_initialization"]
        formal_contract = (
            isinstance(metadata, Mapping)
            and metadata.get("formal_training_contract") == "unitypsf.formal.v1"
        )
        if formal_contract:
            formal_channels.add(parsed.channel_id)
            audit_formal_runtime_contracts(
                modality,
                {**runtime_configs, parsed.channel_id: runtime_config},
                require_complete=False,
            )

        contract = shared_runtime_contract(runtime_config)
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
        provider_overrides = condition_store_provider_overrides(physical_context)
        if shared_runtime is None:
            torch.manual_seed(model_seed)
            shared_runtime = build_trainer_runtime(
                runtime_config,
                layout=channel_layout,
                model_registry=model_registry,
                batch_provider_overrides=provider_overrides,
            )
            shared_runtime, _ = prepare_instance_runtime(
                shared_runtime,
                runtime_config,
                config_base_dir=instance_config_path.parent,
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
        eval_route = localizer_eval_route(train_cfg, config_base_dir=instance_config_path.parent)
        if (
            eval_route.get("source") == "online_generation"
            and int(eval_route["seed"]) == data_seed
        ):
            raise ValueError(f"held-out eval seed overlaps training seed for {key!r}")
        eval_provider = build_localizer_eval_provider(
            train_cfg,
            config_base_dir=instance_config_path.parent,
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
        physical_state, calibration, provenance = channel_metadata(
            runtime_config,
            physical_context=physical_context,
            config_path=instance_config_path,
            key=key,
            data_seed=data_seed,
            experiment_metadata=(metadata if isinstance(metadata, Mapping) else None),
        )
        channels[parsed.channel_id] = ModalityChannelStream(
            channel_id=parsed.channel_id,
            batches=modality_batches(provider, parsed.channel_id),
            loss_fn=lambda output, batch, fn=from_output: fn(output, batch.target),
            physical_state=physical_state,
            calibration=calibration,
            provenance=provenance,
            snapshot_physical_state=lambda context=physical_context: physical_state_snapshot(context),
            restore_physical_state=lambda state, context=physical_context: context.restore_physical_state(state),
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
        formal_runtime_evidence = audit_formal_runtime_contracts(modality, runtime_configs)
        if not isinstance(shared_runtime.optimizer, torch.optim.AdamW):
            raise ValueError(
                f"formal {modality.value} runtime built {type(shared_runtime.optimizer).__name__}, not AdamW"
            )
        if modality is not PSFModality.DOUBLE_HELIX and (
            shared_runtime.scheduler is None
            or type(shared_runtime.scheduler).__name__ != "StepLR"
        ):
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


_mapping = mapping
_load_joint_config = load_joint_config
_load_resolved_training_contract = load_resolved_training_contract
_instance_specs = instance_specs
_bind_instance = bind_instance
_sha256 = sha256
_config_path = config_path
_modality_groups = modality_groups
_channel_metadata = channel_metadata
_physical_provenance = physical_provenance
_sampling_recenter_metadata = sampling_recenter_metadata
_audit_formal_runtime_contracts = audit_formal_runtime_contracts
_visual_record = visual_record
_build_modality_runtime = build_modality_runtime
