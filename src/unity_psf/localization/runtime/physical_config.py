"""Physical artifact resolution for localization runtime configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .paths import resolve_path


def dual_domain_coeff_maps(value: Any, *, base_dir: Path | None = None) -> tuple[dict[str, str], ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("dual_domain_coeff_maps must be a list")
    maps: list[dict[str, str]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("dual_domain_coeff_maps entries must be mappings")
        name = str(item.get("name", f"domain{idx}"))
        path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
        if path is None:
            raise ValueError(
                "dual_domain_coeff_maps entries must include coeff_maps_npz, "
                "alternating_coeff_maps_npz, or path"
            )
        maps.append({"name": name, "coeff_maps_npz": resolve_path(str(path), base_dir=base_dir)})
    return tuple(maps)


def select_single_channel_coeff_maps(
    entries: tuple[dict[str, str], ...],
    train_cfg: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Bind a single-channel runtime to its own physical coefficient map."""

    if not entries:
        return ()
    expert_cfg = train_cfg.get("expert")
    channel_id = "main"
    if isinstance(expert_cfg, Mapping) and expert_cfg.get("channel_id") is not None:
        channel_id = str(expert_cfg["channel_id"])
    matching = tuple(item for item in entries if str(item.get("name", "")) == channel_id)
    if len(entries) > 1:
        if len(matching) != 1:
            names = sorted(str(item.get("name", "")) for item in entries)
            raise ValueError(
                "single-channel runtime has no unique coefficient map for "
                f"channel={channel_id!r}; available domains={names!r}"
            )
        selected = matching[0]
    else:
        selected = entries[0]
    return ({**selected, "name": channel_id},)


def pupil_carrier_complex(value: Any, *, base_dir: Path | None) -> torch.Tensor | None:
    if value in (None, ""):
        return None
    path = Path(resolve_path(str(value), base_dir=base_dir))
    with np.load(path, allow_pickle=False) as payload:
        key = "carrier_complex" if "carrier_complex" in payload else "complex_pupil"
        carrier = np.asarray(payload[key], dtype=np.complex64)
    if carrier.ndim != 2 or carrier.shape[0] != carrier.shape[1]:
        raise ValueError("pupil carrier must be a square complex pupil array")
    return torch.from_numpy(carrier)


__all__ = [
    "dual_domain_coeff_maps",
    "pupil_carrier_complex",
    "select_single_channel_coeff_maps",
]
