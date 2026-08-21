"""Shared progress and metric state for sequential and expert-parallel modality training."""

from __future__ import annotations

from typing import Any, Mapping


def empty_metrics(runtime) -> dict[str, Any]:
    return {
        "optimizer_steps": 0,
        "attempted_optimizer_steps": 0,
        "skipped_optimizer_steps": 0,
        "epochs_completed": 0,
        "heldout_history": [],
        "channels": {
            channel_id: {
                "steps": 0,
                "optimizer_steps": 0,
                "skipped_optimizer_steps": 0,
                "samples": 0,
                "losses": [],
                "heldout_history": [],
            }
            for channel_id in runtime.channels
        },
    }


def metrics_from_progress(runtime, *, epoch: int, optimizer_steps: int, progress: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(progress) != set(runtime.channels):
        raise ValueError("resume channel progress does not match the modality runtime")
    histories = [list(progress[channel_id].get("modality_heldout_history", ())) for channel_id in runtime.channels]
    modality_history = histories[0]
    if any(history != modality_history for history in histories[1:]):
        raise ValueError("resume channel copies of modality held-out history differ")
    return {
        "optimizer_steps": int(optimizer_steps),
        "attempted_optimizer_steps": sum(int(progress[channel_id].get("steps", 0)) for channel_id in runtime.channels),
        "skipped_optimizer_steps": sum(int(progress[channel_id].get("skipped_optimizer_steps", 0)) for channel_id in runtime.channels),
        "epochs_completed": int(epoch),
        "heldout_history": [dict(item) for item in modality_history],
        "channels": {
            channel_id: {
                "steps": int(progress[channel_id].get("steps", 0)),
                "optimizer_steps": int(progress[channel_id].get("optimizer_steps", progress[channel_id].get("steps", 0))),
                "skipped_optimizer_steps": int(progress[channel_id].get("skipped_optimizer_steps", 0)),
                "samples": int(progress[channel_id].get("samples", 0)),
                "losses": [float(item) for item in progress[channel_id].get("losses", ())],
                "heldout_history": [dict(item) for item in progress[channel_id].get("heldout_history", ())],
            }
            for channel_id in runtime.channels
        },
    }


def channel_progress(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        channel_id: {
            "steps": int(values["steps"]),
            "optimizer_steps": int(values["optimizer_steps"]),
            "skipped_optimizer_steps": int(values["skipped_optimizer_steps"]),
            "samples": int(values["samples"]),
            "losses": list(values["losses"]),
            "heldout_history": list(values["heldout_history"]),
            "modality_heldout_history": list(metrics["heldout_history"]),
        }
        for channel_id, values in metrics["channels"].items()
    }


def accumulate_epoch(metrics: dict[str, Any], result) -> None:
    metrics["optimizer_steps"] += result.optimizer_steps
    metrics["attempted_optimizer_steps"] += result.attempted_optimizer_steps
    metrics["skipped_optimizer_steps"] += result.skipped_optimizer_steps
    metrics["epochs_completed"] = result.epoch
    for channel_id, values in metrics["channels"].items():
        values["steps"] += result.step_counts[channel_id]
        values["optimizer_steps"] += result.optimizer_steps_by_channel[channel_id]
        values["skipped_optimizer_steps"] += result.skipped_optimizer_steps_by_channel[channel_id]
        values["samples"] += result.sample_counts[channel_id]
        values["losses"].extend(result.losses_by_channel[channel_id])


def accumulate_heldout(metrics: dict[str, Any], result, *, epoch: int) -> None:
    channel_results = {}
    for channel_id, values in result.channels.items():
        channel_values = {
            **dict(values),
            "optimizer_steps": int(metrics["channels"][channel_id]["optimizer_steps"]),
            "skipped_optimizer_steps": int(metrics["channels"][channel_id]["skipped_optimizer_steps"]),
        }
        metrics["channels"][channel_id]["heldout_history"].append({"epoch": int(epoch), **channel_values})
        channel_results[channel_id] = channel_values
    metrics["heldout_history"].append({"epoch": int(epoch), "modality": {**dict(result.modality), "optimizer_steps": int(metrics["optimizer_steps"])}, "channels": channel_results})


def heldout_eval_enabled(runtime) -> bool:
    enabled = [stream.heldout_eval is not None for stream in runtime.channels.values()]
    if any(enabled) and not all(enabled):
        raise ValueError("held-out eval must be enabled for all channels or none")
    return all(enabled)
