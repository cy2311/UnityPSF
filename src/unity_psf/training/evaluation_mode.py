"""Utilities for exception-safe evaluation mode transitions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def preserve_module_training_modes(module: torch.nn.Module) -> Iterator[None]:
    """Temporarily switch a module tree to eval mode and restore every mode."""

    modes = tuple((submodule, submodule.training) for submodule in module.modules())
    try:
        module.eval()
        yield
    finally:
        for submodule, was_training in modes:
            submodule.training = was_training
