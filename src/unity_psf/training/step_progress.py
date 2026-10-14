from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class OptimizerStepProgress:
    attempted_optimizer_steps: int = 0
    optimizer_steps: int = 0

    @property
    def skipped_optimizer_steps(self) -> int:
        return self.attempted_optimizer_steps - self.optimizer_steps

    def record(self, *, performed: bool) -> None:
        self.attempted_optimizer_steps += 1
        self.optimizer_steps += int(performed)


def validate_scheduler_step_unit(step_unit: str) -> None:
    if step_unit not in {"optimizer_step", "epoch"}:
        raise ValueError("scheduler_step_unit must be optimizer_step or epoch")


def advance_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    step_unit: str,
    event: Literal["optimizer_step", "epoch"],
    optimizer_steps: int,
) -> None:
    validate_scheduler_step_unit(step_unit)
    if scheduler is not None and step_unit == event and optimizer_steps > 0:
        scheduler.step()
