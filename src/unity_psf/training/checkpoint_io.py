from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from unity_psf.contracts.checkpoint import CheckpointMetadata, build_checkpoint
from unity_psf.training.checkpoint_state import build_training_state_snapshot, capture_rng_state
from unity_psf.training.loop_types import CheckpointExtraFn


def checkpoint_extra(checkpoint_extra_fn: CheckpointExtraFn | None) -> dict[str, object]:
    if checkpoint_extra_fn is None:
        return {}
    payload = checkpoint_extra_fn()
    if payload is None:
        return {}
    return dict(payload)


def checkpoint_extra_with_rng(checkpoint_extra_fn: CheckpointExtraFn | None) -> dict[str, object]:
    rng_state = capture_rng_state()
    extra = checkpoint_extra(checkpoint_extra_fn)
    extra.setdefault("rng_state", rng_state)
    return extra


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    amp_scaler: torch.amp.GradScaler | None = None,
    epoch: int,
    step_count: int,
    global_step: int,
    extra: dict[str, object] | None = None,
    checkpoint_metadata: CheckpointMetadata | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    training_extra: dict[str, object] = {
        "epoch": epoch,
        "step_count": step_count,
        "global_step": global_step,
    }
    if extra:
        training_extra.update(extra)
    snapshot = build_training_state_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=amp_scaler,
        progress=training_extra,
    )
    model_state = snapshot.pop("model_state_dict")
    optimizer_state = snapshot.pop("optimizer")
    scheduler_state = snapshot.pop("scheduler")
    scaler_state = snapshot.pop("amp_scaler")
    if scaler_state is not None:
        snapshot["scaler_state_dict"] = scaler_state
    if checkpoint_metadata is None:
        payload: dict[str, object] = {
            **snapshot,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
        }
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler_state
    else:
        payload = build_checkpoint(
            model_state_dict=model_state,
            metadata=checkpoint_metadata,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            extra=snapshot,
        )
    atomic_torch_save(payload, path)


def atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    """Write a checkpoint beside its destination, then replace it atomically."""

    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temporary_path = Path(temporary_name)
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def existing_best_eval_loss(path: Path) -> float | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    eval_loss = checkpoint.get("eval_loss")
    return None if eval_loss is None else float(eval_loss)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
