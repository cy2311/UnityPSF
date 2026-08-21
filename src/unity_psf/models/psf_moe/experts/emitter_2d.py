"""Complete FiLM-conditioned expert for conventional 2D emitter PSFs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from unity_psf.contracts.checkpoint import CheckpointMetadata
from unity_psf.contracts.modality import InputFrameSpec, MeasurementChannelSpec, PSFModality
from unity_psf.localization.losses import ActiveSMLMGMMLoss
from unity_psf.models.psf_moe.experts.astigmatism import AstigmatismExpert


DEFAULT_EMITTER_2D_CONDITION_FIELDS = ("field_x", "field_y")
_Z_ATTRIBUTE_INDEX = 3


def _disabled_2d_attributes(value: int | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return (_Z_ATTRIBUTE_INDEX,)
    selected = (int(value),) if isinstance(value, int) else tuple(int(item) for item in value)
    if _Z_ATTRIBUTE_INDEX not in selected:
        raise ValueError("Emitter2DExpert must disable the z attribute")
    return selected


class Emitter2DExpert(AstigmatismExpert):
    """Independent image-to-localization network with a disabled z head."""

    modality = PSFModality.EMITTER_2D

    def __init__(
        self,
        *,
        condition_dim: int | None = None,
        condition_fields: Sequence[str] = DEFAULT_EMITTER_2D_CONDITION_FIELDS,
        disabled_attr: int | Sequence[int] | None = None,
        **model_config: Any,
    ) -> None:
        super().__init__(
            condition_dim=condition_dim,
            condition_fields=condition_fields,
            disabled_attr=_disabled_2d_attributes(disabled_attr),
            **model_config,
        )
        self.condition_schema = {
            "name": "emitter_2d_film_v1",
            "fields": list(self.condition_fields),
            "dimension": self.condition_dim,
        }
        self.model_config.update(
            {
                "condition_dim": self.condition_dim,
                "condition_fields": list(self.condition_fields),
                "disabled_attr": list(_disabled_2d_attributes(disabled_attr)),
            }
        )

    def forward(
        self,
        images: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if conditions is None and isinstance(images, tuple) and len(images) == 2:
            images, conditions = images
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must have shape (N,C,H,W)")
        if conditions is None:
            conditions = images.new_zeros((images.shape[0], self.condition_dim))
        return super().forward(images, conditions)

    def checkpoint_metadata(
        self,
        *,
        checkpoint_role: str = "prototype",
        instance_id: str | None = None,
        channel_spec: MeasurementChannelSpec | None = None,
        parent_checkpoint_hash: str | None = None,
        code_version: str = "0.4.0",
    ) -> CheckpointMetadata:
        return CheckpointMetadata(
            model_name="emitter_2d_expert",
            checkpoint_role=checkpoint_role,
            expert_type=self.modality,
            model_config=dict(self.model_config),
            input_frame_spec=InputFrameSpec(input_frame_channels=self.nch_in),
            instance_id=instance_id,
            channel_spec=channel_spec,
            condition_schema=dict(self.condition_schema),
            parent_checkpoint_hash=parent_checkpoint_hash,
            code_version=code_version,
            experts=(self.modality.value,),
            shared_feature_channels=int(self.network.feature_channels),
            router_mode="hard",
        )


class Emitter2DGMMLoss(ActiveSMLMGMMLoss):
    """Active SMLM GMM loss with the target z attribute permanently masked."""

    def __init__(self, **kwargs: Any) -> None:
        requested = kwargs.pop("disable_attr", _Z_ATTRIBUTE_INDEX)
        if _Z_ATTRIBUTE_INDEX not in _disabled_2d_attributes(requested):
            raise ValueError("Emitter2DGMMLoss must disable the z target")
        super().__init__(disable_attr=_Z_ATTRIBUTE_INDEX, **kwargs)


__all__ = [
    "DEFAULT_EMITTER_2D_CONDITION_FIELDS",
    "Emitter2DExpert",
    "Emitter2DGMMLoss",
]
