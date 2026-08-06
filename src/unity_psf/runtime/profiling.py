from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time

import torch

from .environment import get_env


_TIMINGS: defaultdict[str, float] = defaultdict(float)


def enabled() -> bool:
    return str(get_env("UNITY_V04_PROFILE_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}


def _sync_cuda() -> None:
    if not enabled():
        return
    if str(get_env("UNITY_V04_PROFILE_SYNC_CUDA", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def time_block(name: str):
    if not enabled():
        yield
        return
    key = str(name)
    _sync_cuda()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync_cuda()
        _TIMINGS[key] += time.perf_counter() - start


def drain(prefix: str = "profile_") -> dict[str, float]:
    if not enabled():
        return {}
    values = {f"{prefix}{key}_s": float(value) for key, value in _TIMINGS.items() if float(value) != 0.0}
    _TIMINGS.clear()
    return values
