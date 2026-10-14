"""Held-out evaluation and sufficient-statistic aggregation for modalities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from unity_psf.contracts import PSFModality
from unity_psf.training.evaluation_mode import preserve_module_training_modes
from unity_psf.training.modality_types import ModalityHeldoutEvalResult, ModalityTrainingRuntime


def evaluate_modality_heldout(
    runtime: ModalityTrainingRuntime,
) -> ModalityHeldoutEvalResult:
    """Evaluate every channel and recompute modality metrics from sufficient statistics."""

    missing = [
        channel_id
        for channel_id, stream in runtime.channels.items()
        if stream.heldout_eval is None
    ]
    if missing:
        raise ValueError(f"held-out eval is not configured for channels: {missing}")
    channel_metrics: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Mapping[str, Any]] = {}
    with preserve_module_training_modes(runtime.model):
        for channel_id, stream in runtime.channels.items():
            assert stream.heldout_eval is not None
            result = stream.heldout_eval(runtime.model)
            if not isinstance(result, Mapping):
                raise TypeError("held-out eval callback must return a mapping")
            metrics = result.get("metrics")
            artifact = result.get("artifacts", {})
            if not isinstance(metrics, Mapping) or not isinstance(artifact, Mapping):
                raise TypeError("held-out eval result requires metrics and artifacts mappings")
            channel_metrics[channel_id] = normalize_heldout_metrics(
                metrics,
                modality=runtime.modality,
            )
            artifacts[channel_id] = artifact
    return ModalityHeldoutEvalResult(
        modality=aggregate_heldout_metrics(
            channel_metrics.values(),
            modality=runtime.modality,
        ),
        channels=channel_metrics,
        artifacts=artifacts,
    )


def normalize_heldout_metrics(
    metrics: Mapping[str, Any],
    *,
    modality: PSFModality,
) -> dict[str, Any]:
    required = (
        "eval_loss",
        "true_positive",
        "false_positive",
        "false_negative",
        "lateral_sq_error_nm2_sum",
        "axial_sq_error_nm2_sum",
        "photon_relative_error_sum",
        "matched_photons",
        "sample_count",
        "route_count",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"held-out eval metrics are missing {missing}")
    result = dict(metrics)
    true_positive = int(result["true_positive"])
    false_positive = int(result["false_positive"])
    false_negative = int(result["false_negative"])
    matched_photons = int(result["matched_photons"])
    result.update(
        {
            "eval_loss": float(result["eval_loss"]),
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
            "Jaccard": true_positive
            / max(true_positive + false_positive + false_negative, 1),
            "RMSE_XY_nm": None
            if true_positive == 0
            else (
                float(result["lateral_sq_error_nm2_sum"])
                / true_positive
            )
            ** 0.5,
            "RMSE_Z_nm": None
            if modality is PSFModality.EMITTER_2D or true_positive == 0
            else (
                float(result["axial_sq_error_nm2_sum"])
                / true_positive
            )
            ** 0.5,
            "photon_relative_error": None
            if matched_photons == 0
            else float(result["photon_relative_error_sum"]) / matched_photons,
            "sample_count": int(result["sample_count"]),
            "route_count": int(result["route_count"]),
        }
    )
    return result


def aggregate_heldout_metrics(
    channels: Iterable[Mapping[str, Any]],
    *,
    modality: PSFModality,
) -> dict[str, Any]:
    values = list(channels)
    sample_count = sum(int(item["sample_count"]) for item in values)
    totals = {
        key: sum(float(item[key]) for item in values)
        for key in (
            "true_positive",
            "false_positive",
            "false_negative",
            "lateral_sq_error_nm2_sum",
            "axial_sq_error_nm2_sum",
            "photon_relative_error_sum",
            "matched_photons",
            "route_count",
        )
    }
    totals.update(
        {
            "eval_loss": sum(
                float(item["eval_loss"]) * int(item["sample_count"])
                for item in values
            )
            / max(sample_count, 1),
            "sample_count": sample_count,
        }
    )
    return normalize_heldout_metrics(totals, modality=modality)
