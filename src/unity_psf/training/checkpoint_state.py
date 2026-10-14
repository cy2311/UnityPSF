from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    bit_generator, state, position, has_gauss, cached_gaussian = np.random.get_state()
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(bit_generator),
            "state": np.asarray(state, dtype=np.uint32).tolist(),
            "pos": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(value: object) -> None:
    if not isinstance(value, Mapping):
        return
    python_state = value.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = value.get("numpy")
    if isinstance(numpy_state, Mapping):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    elif isinstance(numpy_state, (tuple, list)) and len(numpy_state) == 5:
        np.random.set_state(
            (
                str(numpy_state[0]), np.asarray(numpy_state[1], dtype=np.uint32),
                int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4]),
            )
        )
    torch_cpu_state = value.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(torch.as_tensor(torch_cpu_state, dtype=torch.uint8, device="cpu"))
    torch_cuda_state = value.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(torch_cuda_state, (tuple, list)):
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in torch_cuda_state]
        )


def build_training_state_snapshot(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    amp_scaler: torch.amp.GradScaler | None = None,
    progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(progress or {}))
    reserved = {"model_state_dict", "optimizer", "scheduler", "amp_scaler"}
    collisions = sorted(reserved.intersection(payload))
    if collisions:
        raise ValueError(f"reserved checkpoint field(s) in progress: {collisions}")
    if "rng_state" not in payload:
        payload["rng_state"] = capture_rng_state()
    payload.update(
        {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "amp_scaler": (
                amp_scaler.state_dict() if amp_scaler is not None and amp_scaler.is_enabled() else None
            ),
        }
    )
    return payload
