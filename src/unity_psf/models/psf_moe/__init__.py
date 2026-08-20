"""Three-expert PSF mixture-of-experts model for UnityPSF."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from unity_psf.contracts.checkpoint import CheckpointMetadata
from unity_psf.contracts.modality import PSFExpertOutput, PSFModality

from .base import SharedPSFStem
from .experts.astigmatism import (
    DEFAULT_ASTIGMATISM_CONDITION_FIELDS,
    AstigmatismExpert,
    LegacyAstigmatismExpert,
)
from .experts.double_helix import DoubleHelixExpert
from .experts.emitter_2d import Emitter2DExpert, LegacyEmitter2DExpert
from .instances import (
    AstigmatismExpertInstance,
    build_instance_optimizer,
    create_expert_instance_from_prototype,
    parameter_state_hash,
)
from .router import DEFAULT_MODALITY_ORDER, PSFRouter, RoutingResult


class PSFMoE(nn.Module):
    """Shared encoder with independent 2D, astigmatic, and DH experts."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        feature_channels: int = 32,
        router_mode: str = "hard",
        modality_order: Sequence[PSFModality] = DEFAULT_MODALITY_ORDER,
    ) -> None:
        super().__init__()
        self.stem = SharedPSFStem(in_channels=in_channels, feature_channels=feature_channels)
        self.router = PSFRouter(order=modality_order, mode=router_mode)
        self.experts = nn.ModuleDict(
            {
                PSFModality.EMITTER_2D.value: LegacyEmitter2DExpert(feature_channels),
                # The original shared-stem scaffold still consumes feature tensors.
                # The complete AstigmatismExpert is intentionally a separate image
                # entry point until the modality router is migrated in a later task.
                PSFModality.ASTIGMATISM.value: LegacyAstigmatismExpert(feature_channels),
                PSFModality.DOUBLE_HELIX.value: DoubleHelixExpert(feature_channels),
            }
        )
        missing = set(item.value for item in self.router.order).difference(self.experts)
        if missing:
            raise ValueError(f"router modalities have no expert implementation: {sorted(missing)}")

    def forward(
        self,
        images: torch.Tensor,
        modality: PSFModality | str | Sequence[PSFModality | str],
    ) -> PSFExpertOutput:
        features = self.stem(images)
        routing = self.router(
            modality,
            batch_size=images.shape[0],
            device=features.device,
            dtype=features.dtype,
        )
        return self._dispatch(features, routing)

    def _dispatch(self, features: torch.Tensor, routing: RoutingResult) -> PSFExpertOutput:
        batch_size = features.shape[0]
        aggregate: dict[str, torch.Tensor | None] = {
            "detection_logits": None,
            "xy_offset": None,
            "z": None,
            "photons": None,
        }
        auxiliary: dict[str, torch.Tensor] = {}
        for expert_index, label in enumerate(routing.order):
            sample_indices = torch.nonzero(routing.weights[:, expert_index] > 0, as_tuple=False).flatten()
            if sample_indices.numel() == 0:
                continue
            expert = self.experts[label.value]
            output = expert(features.index_select(0, sample_indices))
            sample_weights = routing.weights.index_select(0, sample_indices)[:, expert_index]
            for name in aggregate:
                value = getattr(output, name)
                weights = sample_weights.to(dtype=value.dtype)
                weighted = value * weights.view(-1, *([1] * (value.ndim - 1)))
                full_shape = (batch_size, *value.shape[1:])
                full = value.new_zeros(full_shape).index_copy(0, sample_indices, weighted)
                aggregate[name] = full if aggregate[name] is None else aggregate[name] + full
            for name, value in output.auxiliary.items():
                weights = sample_weights.to(dtype=value.dtype)
                weighted = value * weights.view(-1, *([1] * (value.ndim - 1)))
                full_shape = (batch_size, *value.shape[1:])
                full = value.new_zeros(full_shape).index_copy(0, sample_indices, weighted)
                auxiliary[name] = full if name not in auxiliary else auxiliary[name] + full
        if any(value is None for value in aggregate.values()):
            raise RuntimeError("router produced no expert output")
        return PSFExpertOutput(
            detection_logits=aggregate["detection_logits"],  # type: ignore[arg-type]
            xy_offset=aggregate["xy_offset"],  # type: ignore[arg-type]
            z=aggregate["z"],  # type: ignore[arg-type]
            photons=aggregate["photons"],  # type: ignore[arg-type]
            auxiliary=auxiliary,
        ).validate(batch_size=batch_size)

    def checkpoint_metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            experts=tuple(item.value for item in self.router.order),
            shared_feature_channels=self.stem.feature_channels,
            router_mode=self.router.mode,
        )


__all__ = [
    "AstigmatismExpert",
    "AstigmatismExpertInstance",
    "build_instance_optimizer",
    "create_expert_instance_from_prototype",
    "DEFAULT_ASTIGMATISM_CONDITION_FIELDS",
    "LegacyAstigmatismExpert",
    "LegacyEmitter2DExpert",
    "PSFMoE",
    "PSFRouter",
    "RoutingResult",
    "parameter_state_hash",
]
