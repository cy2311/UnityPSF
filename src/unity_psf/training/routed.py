"""Small batch value object used by the canonical routed trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RoutedTrainingBatch:
    images: torch.Tensor
    conditions: torch.Tensor | None
    target: Any


__all__ = ["RoutedTrainingBatch"]
