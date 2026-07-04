from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import torch

from neptune_v03.runtime import profiling
from neptune_v03.runtime.layout import RunLayout
from neptune_v03.training.loop import TrainingRunEpochResult, TrainingStepResult


GammaObjective = Callable[[torch.Tensor], torch.Tensor]
GammaTriggerResult = TrainingRunEpochResult | TrainingStepResult
GammaMetrics = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, GammaTriggerResult, dict[str, object]], dict[str, object]]
GammaFeedback = Callable[[torch.Tensor, GammaTriggerResult, dict[str, object]], dict[str, object]]
GammaPrepare = Callable[[GammaTriggerResult], dict[str, object] | None]


@dataclass(frozen=True)
class GammaUpdateConfig:
    start_epoch: int
    stop_epoch: int
    update_interval_epochs: int
    lr: float
    steps: int
    metrics_name: str = "gamma_update_metrics.jsonl"
    start_batch: int | None = None
    stop_batch: int | None = None
    update_interval_batches: int | None = None
    optimizer: str = "adam"
    heldout_accept_policy: str = "monitor"
    checkpoint_policy: str = "final_step"


@dataclass(frozen=True)
class GammaUpdateState:
    gamma: torch.nn.Parameter


def build_gamma_update_hook(
    *,
    state: GammaUpdateState,
    localizer: torch.nn.Module,
    layout: RunLayout,
    config: GammaUpdateConfig,
    objective_fn: GammaObjective,
    prepare_fn: GammaPrepare | None = None,
    metrics_fn: GammaMetrics | None = None,
    feedback_fn: GammaFeedback | None = None,
):
    updated_triggers: set[tuple[str, int]] = set()

    def hook(result: GammaTriggerResult) -> None:
        if not _should_update(result, config):
            return
        trigger_unit = "batch" if config.start_batch is not None else "epoch"
        trigger_value = int(result.global_step) if trigger_unit == "batch" else int(result.epoch)
        trigger_key = (trigger_unit, trigger_value)
        if trigger_key in updated_triggers:
            return
        updated_triggers.add(trigger_key)
        _trace_phase(layout, "start", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)
        cuda_device = state.gamma.device if state.gamma.device.type == "cuda" else None
        cuda_memory_before_allocated = cuda_memory_before_reserved = None
        if cuda_device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
            cuda_memory_before_allocated = int(torch.cuda.memory_allocated(cuda_device))
            cuda_memory_before_reserved = int(torch.cuda.memory_reserved(cuda_device))

        with profiling.time_block("gamma_update_total"):
            with profiling.time_block("gamma_prepare"):
                _trace_phase(layout, "prepare_start", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)
                prepared_metrics = prepare_fn(result) if prepare_fn is not None else None
                localizer_before = [param.detach().clone() for param in localizer.parameters()]
                before = state.gamma.detach().clone()
                optimizer = _make_optimizer(state.gamma, config)
                _trace_phase(layout, "prepare_done", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)
            checkpoint_policy = _normalize_checkpoint_policy(config.checkpoint_policy)
            use_observed_best = checkpoint_policy == "best_observed_loss"
            last_loss = state.gamma.detach().new_tensor(float("nan"))
            best_gamma = before.detach().clone() if use_observed_best else None
            best_optimizer_state = _clone_optimizer_state(optimizer.state_dict()) if use_observed_best else None
            best_loss_value: float | None = None
            best_step: int | None = None
            final_loss_before_best_restore: float | None = None
            for step_index in range(int(config.steps)):
                _trace_phase(layout, "step_start", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value, step=step_index + 1)
                with profiling.time_block("gamma_step_zero_grad"):
                    optimizer.zero_grad(set_to_none=True)
                with profiling.time_block("gamma_step_forward"):
                    loss = objective_fn(state.gamma)
                last_loss = loss.detach()
                if not bool(loss.requires_grad):
                    raise RuntimeError(
                        "gamma update objective returned a loss without a gradient path; "
                        "check posterior sampling selected gamma-sensitive emitters"
                    )
                with profiling.time_block("gamma_step_backward"):
                    loss.backward()
                with profiling.time_block("gamma_step_optimizer"):
                    optimizer.step()
                if use_observed_best:
                    with torch.no_grad():
                        with profiling.time_block("gamma_step_observed_eval"):
                            observed_loss = objective_fn(state.gamma).detach()
                        with profiling.time_block("gamma_step_observed_scalar"):
                            observed_loss_value = float(observed_loss.cpu().item())
                    final_loss_before_best_restore = observed_loss_value
                    if best_loss_value is None or observed_loss_value < best_loss_value:
                        best_loss_value = observed_loss_value
                        best_step = int(step_index + 1)
                        with profiling.time_block("gamma_best_state_snapshot"):
                            best_gamma = state.gamma.detach().clone()
                            best_optimizer_state = _clone_optimizer_state(optimizer.state_dict())
                _trace_phase(layout, "step_done", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value, step=step_index + 1)
            if use_observed_best:
                with profiling.time_block("gamma_restore_best_state"):
                    _trace_phase(layout, "restore_best_start", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)
                    if best_gamma is None or best_optimizer_state is None:
                        raise RuntimeError("best observed gamma state was not recorded")
                    with torch.no_grad():
                        state.gamma.copy_(best_gamma.to(device=state.gamma.device, dtype=state.gamma.dtype))
                    optimizer.load_state_dict(best_optimizer_state)
                    _trace_phase(layout, "restore_best_done", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)
            else:
                with profiling.time_block("gamma_final_loss_scalar"):
                    final_loss_value = float(last_loss.detach().cpu().item())
                best_loss_value = final_loss_value
                best_step = int(config.steps)
                final_loss_before_best_restore = final_loss_value
            with profiling.time_block("gamma_localizer_integrity_check"):
                for before_param, after_param in zip(localizer_before, localizer.parameters()):
                    if not torch.equal(before_param, after_param.detach()):
                        raise RuntimeError("gamma update hook must not modify localizer parameters")

            delta = state.gamma.detach() - before
            metrics = {
                "epoch": int(result.epoch),
                "global_step": int(result.global_step),
                "gamma_delta_norm": float(torch.linalg.norm(delta).item()),
                "gamma_before_norm": float(torch.linalg.norm(before).item()),
                "gamma_after_norm": float(torch.linalg.norm(state.gamma.detach()).item()),
                "steps": int(config.steps),
                "lr": float(config.lr),
                "optimizer": str(config.optimizer),
                "trigger_unit": trigger_unit,
                "best_step": best_step,
                "selected_step": best_step,
                "best_loss": best_loss_value,
                "final_loss_before_best_restore": final_loss_before_best_restore,
                "selected_checkpoint_policy": checkpoint_policy,
            }
            if prepared_metrics:
                metrics.update(prepared_metrics)
            if metrics_fn is not None:
                with profiling.time_block("gamma_metrics"):
                    _trace_phase(layout, "metrics_start", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)
                    metrics.update(metrics_fn(state.gamma.detach(), last_loss, before, result, dict(metrics)))
                    _trace_phase(layout, "metrics_done", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)
            with profiling.time_block("gamma_accept_policy"):
                metrics["heldout_accept_policy"] = str(config.heldout_accept_policy)
                accepted, reject_reason = _accept_update(metrics, config)
            metrics["gamma_update_accepted"] = bool(accepted)
            if not accepted:
                with profiling.time_block("gamma_reject_restore"):
                    with torch.no_grad():
                        state.gamma.copy_(before.to(device=state.gamma.device, dtype=state.gamma.dtype))
                metrics["gamma_update_reject_reason"] = reject_reason
                metrics["feedback_skipped"] = True
                metrics["gamma_after_norm"] = float(torch.linalg.norm(state.gamma.detach()).item())
                metrics["gamma_delta_norm"] = 0.0
            elif feedback_fn is not None:
                with profiling.time_block("gamma_feedback"):
                    _trace_phase(layout, "feedback_start", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)
                    feedback_metrics = dict(metrics)
                    feedback_metrics["_gamma_before"] = before.detach().clone()
                    metrics.update(feedback_fn(state.gamma.detach(), result, feedback_metrics))
                    _trace_phase(layout, "feedback_done", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)
            elif feedback_fn is None:
                metrics["feedback_skipped"] = True
        if cuda_device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(cuda_device)
            peak_allocated = int(torch.cuda.max_memory_allocated(cuda_device))
            peak_reserved = int(torch.cuda.max_memory_reserved(cuda_device))
            after_allocated = int(torch.cuda.memory_allocated(cuda_device))
            after_reserved = int(torch.cuda.memory_reserved(cuda_device))
            metrics.update(
                {
                    "profile_gamma_cuda_memory_before_allocated_mb": _bytes_to_mib(cuda_memory_before_allocated),
                    "profile_gamma_cuda_memory_before_reserved_mb": _bytes_to_mib(cuda_memory_before_reserved),
                    "profile_gamma_cuda_memory_peak_allocated_mb": _bytes_to_mib(peak_allocated),
                    "profile_gamma_cuda_memory_peak_reserved_mb": _bytes_to_mib(peak_reserved),
                    "profile_gamma_cuda_memory_after_allocated_mb": _bytes_to_mib(after_allocated),
                    "profile_gamma_cuda_memory_after_reserved_mb": _bytes_to_mib(after_reserved),
                }
            )
        metrics.update(profiling.drain())
        _rewrite_summary_if_present(layout, metrics)
        _append_jsonl(layout.metrics_dir / config.metrics_name, _gamma_core_metrics_row(metrics))
        _trace_phase(layout, "metrics_written", result=result, trigger_unit=trigger_unit, trigger_value=trigger_value)

    return hook


def _trace_phase(layout: RunLayout, phase: str, *, result: GammaTriggerResult, trigger_unit: str, trigger_value: int, step: int | None = None) -> None:
    if str(os.environ.get("NEPTUNE_V03_GAMMA_PHASE_TRACE", "0")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    payload = {
        "time": time.time(),
        "phase": str(phase),
        "epoch": int(result.epoch),
        "global_step": int(result.global_step),
        "trigger_unit": str(trigger_unit),
        "trigger_value": int(trigger_value),
    }
    if step is not None:
        payload["step"] = int(step)
    _append_jsonl(layout.logs_dir / "gamma_phase_trace.jsonl", payload)


def _make_optimizer(gamma: torch.nn.Parameter, config: GammaUpdateConfig) -> torch.optim.Optimizer:
    name = str(config.optimizer).strip().lower()
    if name == "adam":
        return torch.optim.Adam([gamma], lr=float(config.lr))
    if name == "sgd":
        return torch.optim.SGD([gamma], lr=float(config.lr))
    raise ValueError(f"Unsupported gamma update optimizer: {config.optimizer!r}")


def _clone_optimizer_state(state_dict: dict[str, object]) -> dict[str, object]:
    return {
        key: _clone_optimizer_value(value)
        for key, value in state_dict.items()
    }


def _clone_optimizer_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_optimizer_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_value(item) for item in value)
    return value


def _accept_update(metrics: dict[str, object], config: GammaUpdateConfig) -> tuple[bool, str | None]:
    policy = str(config.heldout_accept_policy).strip().lower()
    if policy == "monitor":
        return True, None
    raise ValueError(f"Unsupported heldout_accept_policy: {config.heldout_accept_policy!r}")


def _normalize_checkpoint_policy(value: str) -> str:
    policy = str(value).strip().lower()
    aliases = {
        "final": "final_step",
        "final_step": "final_step",
        "last": "final_step",
        "last_step": "final_step",
        "best": "best_observed_loss",
        "best_observed": "best_observed_loss",
        "best_observed_loss": "best_observed_loss",
    }
    try:
        return aliases[policy]
    except KeyError as exc:
        raise ValueError("gamma checkpoint_policy must be 'final_step' or 'best_observed_loss'") from exc


def _should_update(result: GammaTriggerResult, config: GammaUpdateConfig) -> bool:
    if int(config.update_interval_epochs) <= 0:
        raise ValueError("update_interval_epochs must be positive")
    if int(config.steps) <= 0:
        raise ValueError("steps must be positive")
    if config.start_batch is not None:
        if config.update_interval_batches is None or int(config.update_interval_batches) <= 0:
            raise ValueError("update_interval_batches must be positive")
        batch = int(result.global_step)
        stop_batch = config.stop_batch
        if batch < int(config.start_batch) or (stop_batch is not None and batch > int(stop_batch)):
            return False
        return (batch - int(config.start_batch)) % int(config.update_interval_batches) == 0
    if isinstance(result, TrainingStepResult):
        return False
    epoch = int(result.epoch)
    if epoch < int(config.start_epoch) or epoch > int(config.stop_epoch):
        return False
    return (epoch - int(config.start_epoch)) % int(config.update_interval_epochs) == 0


def _append_jsonl(path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _bytes_to_mib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _gamma_core_metrics_row(metrics: dict[str, object]) -> dict[str, object]:
    core_keys = (
        "epoch",
        "global_step",
        "steps",
        "lr",
        "optimizer",
        "trigger_unit",
        "heldout_accept_policy",
        "gamma_update_accepted",
        "gamma_update_reject_reason",
        "gamma_delta_norm",
        "gamma_before_norm",
        "gamma_after_norm",
        "best_loss",
        "best_step",
        "selected_step",
        "selected_checkpoint_policy",
        "final_loss_before_best_restore",
        "selected_poisson_nll",
        "selected_sample_count",
        "selected_sampled_emitter_count",
        "heldout_available",
        "heldout_monitor_mode",
        "heldout_initial_loss",
        "heldout_final_loss",
        "heldout_loss_delta",
        "heldout_loss_delta_percent",
        "heldout_roi_count",
        "condition_store_version",
        "physical_state_path",
        "checkpoint_path",
        "summary_path",
        "report_path",
        "diagnostic_png_path",
        "feedback_version",
        "feedback_skipped",
        "feedback_deferred",
        "feedback_deferred_commit",
        "feedback_deferred_commit_skipped",
        "feedback_deferred_pending_domain_count",
        "feedback_deferred_committed_domain_count",
        "feedback_deferred_committed_domains",
        "prepared_epoch",
    )
    row = {key: metrics[key] for key in core_keys if key in metrics}
    row.update({key: value for key, value in metrics.items() if str(key).startswith("profile_")})
    return row


def _rewrite_summary_if_present(layout: RunLayout, payload: dict[str, object]) -> None:
    summary = payload.get("summary_path")
    if not isinstance(summary, str) or not summary:
        return
    path = layout.run_dir / summary
    if not path.is_file():
        return
    clean = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    path.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
