"""Shared construction helpers used by modality training entrypoints."""

from __future__ import annotations

import json
from typing import Any, Mapping

from unity_psf.localization.data.online_types import OnlineBatchProviderConfig
from unity_psf.localization.data.online_provider import build_online_batch_provider
from unity_psf.localization.training_adapter import LocalizationTrainBatch
from unity_psf.training.modality_types import ModalityTrainingBatch


def shared_runtime_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    training_runtime = config.get("resolved_contract", {}).get("training_runtime", {})
    return {
        "device": config.get("device"),
        "model": config.get("model"),
        "optimizer": config.get("optimizer"),
        "training_runtime": training_runtime,
    }


def modality_batches(provider, channel_id: str):
    def batches(epoch: int):
        for training_batch in provider(epoch):
            local_batch = training_batch.inputs
            if hasattr(training_batch, "targets") and not isinstance(local_batch, LocalizationTrainBatch):
                yield ModalityTrainingBatch(images=local_batch, conditions=None, target=training_batch, channel_id=channel_id)
                continue
            if not isinstance(local_batch, LocalizationTrainBatch):
                raise TypeError("modality joint training requires LocalizationTrainBatch")
            model_input = local_batch.model_input
            images, conditions = model_input if isinstance(model_input, tuple) else (model_input, None)
            yield ModalityTrainingBatch(images=images, conditions=conditions, target=training_batch, channel_id=channel_id)

    refresh = getattr(provider, "refresh_physical_model", None)
    if callable(refresh):
        batches.refresh_physical_model = refresh
    return batches


def condition_store_provider_overrides(context):
    def online_train_batch(params: dict[str, object]):
        return build_online_batch_provider(OnlineBatchProviderConfig(**params), condition_store=context.condition_store)

    return {"online_train_batch": online_train_batch}


def physical_state_snapshot(context) -> dict[str, Any]:
    return json.loads(context.physical_state_path.read_text(encoding="utf-8"))
