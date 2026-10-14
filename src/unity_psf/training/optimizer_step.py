from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch

from unity_psf.runtime import profiling


@dataclass(frozen=True)
class OptimizerStepResult:
    loss: float
    optimizer_step_performed: bool
    skip_reason: str | None = None


def run_optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    compute_loss: Callable[[], torch.Tensor],
    scaler: torch.amp.GradScaler | None,
    autocast_enabled: bool,
    amp_dtype: torch.dtype,
    grad_clip_norm: float | None,
    loss_context: str,
    profile: bool = False,
) -> OptimizerStepResult:
    scaler_enabled = scaler is not None and scaler.is_enabled()
    with profiling.time_block("optimizer_zero_grad") if profile else nullcontext():
        optimizer.zero_grad(set_to_none=True)
    with profiling.time_block("model_forward_and_loss") if profile else nullcontext():
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast_enabled):
            loss = compute_loss()
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError(f"loss must be one scalar tensor for {loss_context}")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError(f"non-finite loss for {loss_context}")
    with profiling.time_block("backward") if profile else nullcontext():
        if scaler_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    performed = True
    skip_reason = None
    with profiling.time_block("optimizer_step") if profile else nullcontext():
        if scaler_enabled:
            scaler.unscale_(optimizer)
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if scaler_enabled:
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            performed = scaler.get_scale() >= scale_before_step
            if not performed:
                skip_reason = "amp_scale_reduced"
        else:
            optimizer.step()
    return OptimizerStepResult(float(loss.detach().cpu().item()), performed, skip_reason)
