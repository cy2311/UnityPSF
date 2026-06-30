from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import torch

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
    def hook(result: GammaTriggerResult) -> None:
        if not _should_update(result, config):
            return

        prepared_metrics = prepare_fn(result) if prepare_fn is not None else None
        localizer_before = [param.detach().clone() for param in localizer.parameters()]
        before = state.gamma.detach().clone()
        optimizer = _make_optimizer(state.gamma, config)
        last_loss = state.gamma.detach().new_tensor(float("nan"))
        best_gamma = before.detach().clone()
        best_optimizer_state = _clone_optimizer_state(optimizer.state_dict())
        best_loss_value: float | None = None
        best_step: int | None = None
        final_loss_before_best_restore: float | None = None
        for step_index in range(int(config.steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = objective_fn(state.gamma)
            last_loss = loss.detach()
            if not bool(loss.requires_grad):
                raise RuntimeError(
                    "gamma update objective returned a loss without a gradient path; "
                    "check posterior sampling selected gamma-sensitive emitters"
                )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                observed_loss = objective_fn(state.gamma).detach()
            observed_loss_value = float(observed_loss.cpu().item())
            final_loss_before_best_restore = observed_loss_value
            if best_loss_value is None or observed_loss_value < best_loss_value:
                best_loss_value = observed_loss_value
                best_step = int(step_index + 1)
                best_gamma = state.gamma.detach().clone()
                best_optimizer_state = _clone_optimizer_state(optimizer.state_dict())
        with torch.no_grad():
            state.gamma.copy_(best_gamma.to(device=state.gamma.device, dtype=state.gamma.dtype))
        optimizer.load_state_dict(best_optimizer_state)
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
            "best_step": best_step,
            "selected_step": best_step,
            "best_loss": best_loss_value,
            "final_loss_before_best_restore": final_loss_before_best_restore,
            "selected_checkpoint_policy": "best_observed_loss",
        }
        if prepared_metrics:
            metrics.update(prepared_metrics)
        if metrics_fn is not None:
            metrics.update(metrics_fn(state.gamma.detach(), last_loss, before, result, dict(metrics)))
        metrics["heldout_accept_policy"] = str(config.heldout_accept_policy)
        accepted, reject_reason = _accept_update(metrics, config)
        metrics["gamma_update_accepted"] = bool(accepted)
        if not accepted:
            with torch.no_grad():
                state.gamma.copy_(before.to(device=state.gamma.device, dtype=state.gamma.dtype))
            metrics["gamma_update_reject_reason"] = reject_reason
            metrics["feedback_skipped"] = True
            metrics["gamma_after_norm"] = float(torch.linalg.norm(state.gamma.detach()).item())
            metrics["gamma_delta_norm"] = 0.0
        elif feedback_fn is not None:
            feedback_metrics = dict(metrics)
            feedback_metrics["_gamma_before"] = before.detach().clone()
            metrics.update(feedback_fn(state.gamma.detach(), result, feedback_metrics))
        elif feedback_fn is None:
            metrics["feedback_skipped"] = True
        _rewrite_summary_if_present(layout, metrics)
        _append_jsonl(layout.metrics_dir / config.metrics_name, metrics)

    return hook


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


def _should_update(result: TrainingRunEpochResult, config: GammaUpdateConfig) -> bool:
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
    epoch = int(result.epoch)
    if epoch < int(config.start_epoch) or epoch > int(config.stop_epoch):
        return False
    return (epoch - int(config.start_epoch)) % int(config.update_interval_epochs) == 0


def _append_jsonl(path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _rewrite_summary_if_present(layout: RunLayout, payload: dict[str, object]) -> None:
    summary = payload.get("summary_path")
    if not isinstance(summary, str) or not summary:
        return
    path = layout.run_dir / summary
    if not path.is_file():
        return
    clean = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    path.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
