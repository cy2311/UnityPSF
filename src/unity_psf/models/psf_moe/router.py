"""Explicit modality router for the UnityPSF expert bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from unity_psf.contracts.modality import ModalityBatch, PSFModality
from unity_psf.contracts.joint_checkpoint import JointExpertKey


DEFAULT_MODALITY_ORDER = tuple(PSFModality)


@dataclass(frozen=True)
class RoutingResult:
    modalities: ModalityBatch
    weights: torch.Tensor
    order: tuple[PSFModality, ...]

    def selected_indices(self) -> torch.Tensor:
        return torch.argmax(self.weights, dim=1)


class PSFRouter(nn.Module):
    """Routes each sample to one of the three named PSF experts.

    The initial contract is label-routed: a modality label is required because
    a one-channel image is ambiguous between a 2D emitter and a double helix.
    A learned image router can be added later without changing expert inputs.
    """

    def __init__(self, *, order: Sequence[PSFModality] = DEFAULT_MODALITY_ORDER, mode: str = "hard") -> None:
        super().__init__()
        parsed = tuple(PSFModality.parse(item) for item in order)
        if len(parsed) != len(set(parsed)) or not parsed:
            raise ValueError("router order must contain unique non-empty modalities")
        if mode not in {"hard", "soft"}:
            raise ValueError("router mode must be 'hard' or 'soft'")
        self.order = parsed
        self.mode = mode
        self.soft_gate_logits = nn.Parameter(torch.eye(len(parsed), dtype=torch.float32) * 4.0)

    def forward(
        self,
        modality: PSFModality | str | Sequence[PSFModality | str],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RoutingResult:
        labels = ModalityBatch.from_value(modality, batch_size=batch_size)
        indices = labels.indices(self.order).to(device=device)
        hard_weights = torch.nn.functional.one_hot(indices, num_classes=len(self.order)).to(dtype=dtype)
        if self.mode == "hard":
            weights = hard_weights
        else:
            logits = self.soft_gate_logits.index_select(0, indices).to(dtype=dtype)
            weights = torch.softmax(logits, dim=1)
        return RoutingResult(modalities=labels, weights=weights, order=self.order)


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
    "DEFAULT_MODALITY_ORDER",
    "InstanceRouter",
    "ModalityRouter",
    "PSFRouter",
    "RoutingResult",
]
