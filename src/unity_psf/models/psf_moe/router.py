"""Explicit modality router for the UnityPSF expert bank."""

from __future__ import annotations

from typing import Sequence

from torch import nn

from unity_psf.contracts.modality import PSFModality
from unity_psf.contracts.joint_checkpoint import JointExpertKey

class InstanceRouter(nn.Module):
    """Legacy v1 router retained only for read-only checkpoint compatibility."""

    def __init__(self, routes: Sequence[JointExpertKey | str]) -> None:
        super().__init__()
        parsed = tuple(JointExpertKey.parse(item) for item in routes)
        if not parsed or len(parsed) != len(set(parsed)):
            raise ValueError("instance routes must be non-empty and unique")
        self.routes = parsed
        self._route_keys = frozenset(item.storage_key for item in parsed)

    def resolve(self, modality: PSFModality | str, channel_id: str) -> JointExpertKey:
        selected = JointExpertKey(modality, channel_id)
        if selected.storage_key not in self._route_keys:
            available = ", ".join(sorted(self._route_keys))
            raise ValueError(
                f"unsupported UnityPSF route {selected.storage_key!r}; available routes: {available}"
            )
        return selected


class ModalityRouter(nn.Module):
    """Resolve exactly one complete expert using only the PSF modality."""

    def __init__(self, modalities: Sequence[PSFModality | str]) -> None:
        super().__init__()
        parsed = tuple(PSFModality.parse(item) for item in modalities)
        if not parsed or len(parsed) != len(set(parsed)):
            raise ValueError("modality routes must be non-empty and unique")
        self.modalities = parsed
        self._modalities = frozenset(parsed)

    def resolve(self, modality: PSFModality | str) -> PSFModality:
        selected = PSFModality.parse(modality)
        if selected not in self._modalities:
            available = ", ".join(sorted(item.value for item in self._modalities))
            raise ValueError(
                f"unsupported UnityPSF modality {selected.value!r}; available modalities: {available}"
            )
        return selected


__all__ = [
    "InstanceRouter",
    "ModalityRouter",
]
