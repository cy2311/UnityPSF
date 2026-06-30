from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from neptune_v03.runtime.layout import RunLayout
from neptune_v03.training.loop import BatchProvider, EpochTrainingConfig, LossFn


ModelFactory = Callable[[dict[str, object]], torch.nn.Module]
BatchProviderFactory = Callable[[dict[str, object]], BatchProvider]
LossFactory = Callable[[dict[str, object]], LossFn]


@dataclass(frozen=True)
class TrainerRuntime:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    batch_provider: BatchProvider
    config: EpochTrainingConfig
    layout: RunLayout
    loss_fn: LossFn | None = None


def build_trainer_runtime(
    config: Mapping[str, object],
    *,
    layout: RunLayout,
    model_registry: Mapping[str, ModelFactory],
    batch_provider_registry: Mapping[str, BatchProviderFactory] | None = None,
    batch_provider_overrides: Mapping[str, BatchProviderFactory] | None = None,
    loss_registry: Mapping[str, LossFactory] | None = None,
) -> TrainerRuntime:
    batch_provider_registry = {
        **_builtin_batch_provider_registry(),
        **dict(batch_provider_registry or {}),
        **dict(batch_provider_overrides or {}),
    }
    loss_registry = {
        **_builtin_loss_registry(),
        **dict(loss_registry or {}),
    }

    model_spec = _component_spec(config, "model")
    model = _build_registered(model_spec, registry=model_registry, label="model")
    device = _runtime_device(config)
    model.to(device=device)

    optimizer_spec = _component_spec(config, "optimizer")
    optimizer = _build_optimizer(optimizer_spec, model.parameters())
    scheduler = _build_scheduler(config, optimizer)

    batch_spec = _component_spec(config, "batch_provider")
    batch_provider = _build_registered(batch_spec, registry=batch_provider_registry, label="batch_provider")

    loss_fn = None
    if "loss" in config:
        loss_spec = _component_spec(config, "loss")
        loss_fn = _build_registered(loss_spec, registry=loss_registry, label="loss")

    epochs = _mapping(config.get("epochs"), "epochs")
    return TrainerRuntime(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_provider=batch_provider,
        config=EpochTrainingConfig(
            start_epoch=int(epochs["start"]),
            stop_epoch=int(epochs["stop"]),
            max_batches=None if config.get("max_batches") is None else int(config["max_batches"]),
            grad_clip_norm=_grad_clip_norm(config),
            amp_enabled=_amp_enabled(config),
            amp_dtype=_amp_dtype(config),
            scheduler_step_unit=_scheduler_step_unit(config),
        ),
        layout=layout,
        loss_fn=loss_fn,
    )


def _runtime_device(config: Mapping[str, object]) -> torch.device:
    value = config.get("device", "cpu")
    device = torch.device(str(value))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is not available: {device}")
    return device


def _component_spec(config: Mapping[str, object], key: str) -> dict[str, object]:
    return dict(_mapping(config.get(key), key))


def _build_registered(
    spec: Mapping[str, object],
    *,
    registry: Mapping[str, Callable[[dict[str, object]], object]],
    label: str,
):
    name = str(spec["name"])
    if name not in registry:
        raise ValueError(f"unknown {label}: {name}")
    return registry[name](dict(_mapping(spec.get("params", {}), f"{label}.params")))


def _build_optimizer(spec: Mapping[str, object], parameters) -> torch.optim.Optimizer:
    name = str(spec["name"])
    params = dict(_mapping(spec.get("params", {}), "optimizer.params"))
    key = name.lower()
    if key == "sgd":
        return torch.optim.SGD(parameters, **params)
    if key == "adamw":
        return torch.optim.AdamW(parameters, **params)
    raise ValueError(f"unknown optimizer: {name}")


def _build_scheduler(config: Mapping[str, object], optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_spec = _active_scheduler_spec(config)
    if scheduler_spec is None:
        return None
    name = str(scheduler_spec["name"])
    params = dict(_mapping(scheduler_spec.get("params", {}), "scheduler.params"))
    step_unit = str(scheduler_spec.get("step_unit"))
    if step_unit not in {"optimizer_step", "epoch"}:
        raise ValueError(f"active scheduler step_unit must be optimizer_step or epoch, got {step_unit!r}")
    if name.lower() == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    raise ValueError(f"unknown scheduler: {name}")


def _scheduler_step_unit(config: Mapping[str, object]) -> str:
    scheduler_spec = _active_scheduler_spec(config)
    if scheduler_spec is None:
        return "optimizer_step"
    step_unit = str(scheduler_spec.get("step_unit"))
    if step_unit not in {"optimizer_step", "epoch"}:
        raise ValueError(f"active scheduler step_unit must be optimizer_step or epoch, got {step_unit!r}")
    return step_unit


def _active_scheduler_spec(config: Mapping[str, object]) -> Mapping[str, object] | None:
    resolved_contract = config.get("resolved_contract")
    if not isinstance(resolved_contract, Mapping):
        return None
    training_runtime = resolved_contract.get("training_runtime")
    if not isinstance(training_runtime, Mapping):
        return None
    scheduler = training_runtime.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("active") is not True:
        return None
    return scheduler


def _amp_spec(config: Mapping[str, object]) -> Mapping[str, object] | None:
    resolved_contract = config.get("resolved_contract")
    if not isinstance(resolved_contract, Mapping):
        return None
    training_runtime = resolved_contract.get("training_runtime")
    if not isinstance(training_runtime, Mapping):
        return None
    amp = training_runtime.get("amp")
    return amp if isinstance(amp, Mapping) else None


def _amp_enabled(config: Mapping[str, object]) -> bool:
    amp = _amp_spec(config)
    return bool(amp is not None and amp.get("active") is True)


def _amp_dtype(config: Mapping[str, object]) -> torch.dtype:
    amp = _amp_spec(config)
    dtype = None if amp is None else amp.get("dtype")
    if dtype in (None, "", "float16", "fp16"):
        return torch.float16
    if dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"unknown AMP dtype: {dtype}")


def _grad_clip_norm(config: Mapping[str, object]) -> float | None:
    resolved_contract = config.get("resolved_contract")
    if not isinstance(resolved_contract, Mapping):
        return None
    training_runtime = resolved_contract.get("training_runtime")
    if not isinstance(training_runtime, Mapping):
        return None
    grad_clip = training_runtime.get("grad_clip")
    if not isinstance(grad_clip, Mapping) or grad_clip.get("active") is not True:
        return None
    configured_norm = grad_clip.get("configured_norm")
    return None if configured_norm is None else float(configured_norm)


def _builtin_batch_provider_registry() -> dict[str, BatchProviderFactory]:
    def deterministic_synthetic_online(params: dict[str, object]) -> BatchProvider:
        from neptune_v03.localization.synthetic import (
            SyntheticOnlineBatchConfig,
            build_synthetic_online_batch_provider,
        )

        return build_synthetic_online_batch_provider(SyntheticOnlineBatchConfig(**params))

    def online_train_batch(params: dict[str, object]) -> BatchProvider:
        from neptune_v03.localization.online import OnlineBatchProviderConfig, build_online_batch_provider

        return build_online_batch_provider(OnlineBatchProviderConfig(**params))

    def microtube_tiff_train_batch(params: dict[str, object]) -> BatchProvider:
        from neptune_v03.localization.microtube_tiff import (
            MicrotubeTiffBatchProviderConfig,
            build_microtube_tiff_batch_provider,
        )

        return build_microtube_tiff_batch_provider(MicrotubeTiffBatchProviderConfig(**params))

    return {
        "deterministic_synthetic_online": deterministic_synthetic_online,
        "microtube_tiff_train_batch": microtube_tiff_train_batch,
        "online_train_batch": online_train_batch,
    }


def _builtin_loss_registry() -> dict[str, LossFactory]:
    def active_smlm_loss(params: dict[str, object]) -> LossFn:
        from neptune_v03.localization.losses import ActiveSMLMLoss
        from neptune_v03.localization.training_adapter import make_localization_loss

        return make_localization_loss(ActiveSMLMLoss(**params))

    def active_smlm_gmm_loss(params: dict[str, object]) -> LossFn:
        from neptune_v03.localization.losses import ActiveSMLMGMMLoss
        from neptune_v03.localization.training_adapter import make_localization_loss

        return make_localization_loss(ActiveSMLMGMMLoss(**params))

    def localization_mse(params: dict[str, object]) -> LossFn:
        from neptune_v03.localization.smlm_output import decode_smlm_output
        from neptune_v03.localization.training_adapter import make_localization_loss

        class _MSECriterion:
            def forward(self, y_out, detect_tar, pxyz_tar, mask_tar, bkg_tar):
                if hasattr(y_out, "detection_logits"):
                    detection = y_out.detection_logits
                elif y_out.dim() == 4 and y_out.shape[1] == 10:
                    detection = decode_smlm_output(y_out).detection_prob
                elif y_out.dim() == 4 and y_out.shape[1] == 1:
                    detection = y_out.squeeze(1)
                elif y_out.dim() == 4:
                    raise ValueError("localization_mse expects single-channel or 10-channel SMLM tensor output")
                else:
                    detection = y_out.squeeze(1)
                return torch.nn.functional.mse_loss(detection, detect_tar, reduction="none").flatten(start_dim=1)

        return make_localization_loss(_MSECriterion())

    return {
        "active_smlm_gmm_loss": active_smlm_gmm_loss,
        "active_smlm_loss": active_smlm_loss,
        "localization_mse": localization_mse,
    }


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value
