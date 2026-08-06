"""Complete image-to-localization expert for astigmatic PSFs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from unity_psf.contracts.checkpoint import CheckpointMetadata
from unity_psf.contracts.modality import InputFrameSpec, MeasurementChannelSpec, PSFModality
from unity_psf.localization.film import FiLMConditionedDoubleUNet
from unity_psf.localization.smlm_output import SMLMOutputChannels
from unity_psf.localization.smlm_unet import DoubleUNet
from unity_psf.models.psf_moe.base import AdaptedPSFExpert


DEFAULT_ASTIGMATISM_CONDITION_FIELDS = (
    "zernike_0",
    "zernike_1",
    "field_x",
    "field_y",
)
_MISSING_CONDITIONS = object()


def _normalize_condition_fields(fields: Sequence[str]) -> tuple[str, ...]:
    if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
        raise TypeError("condition_fields must be a sequence of strings")
    if any(not isinstance(field, str) for field in fields):
        raise TypeError("condition_fields must contain only strings")
    normalized = tuple(field.strip() for field in fields)
    if not normalized or any(not field for field in normalized):
        raise ValueError("condition_fields must contain non-empty names")
    if len(set(normalized)) != len(normalized):
        raise ValueError("condition_fields must be unique")
    return normalized


def _serializable_activation(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty activation name string")
    return value


class AstigmatismExpert(nn.Module):
    """A standalone FiLM-conditioned DoubleUNet for astigmatic localization.

    The expert owns its complete image encoder, union backbone, FiLM module, and
    SMLM heads.  It deliberately accepts the raw preprocessed image rather than
    a feature tensor produced by ``SharedPSFStem``.
    """

    modality = PSFModality.ASTIGMATISM
    output_channels = SMLMOutputChannels.count

    def __init__(
        self,
        *,
        nch_in: int = 3,
        depth_shared: int = 1,
        depth_union: int = 1,
        nfeatures_init: int = 32,
        nfeatures_inter: int | None = None,
        norm_start_level: int = 0,
        norm_groups: int = 0,
        activation: str = "ELU",
        dropout_start_level: int | None = None,
        p_dropout: float = 0.1,
        pool_mode: str = "StrideConv",
        upsample_mode: str = "bilinear",
        inter_activation: str = "ELU",
        norm_head_groups: int = 0,
        final_activation: str = "ELU",
        disabled_attr: int | tuple[int, ...] | list[int] | None = None,
        kaiming_normal: bool = True,
        depthwise: bool = True,
        z_mu_activation: str | None = None,
        condition_dim: int | None = None,
        condition_fields: Sequence[str] = DEFAULT_ASTIGMATISM_CONDITION_FIELDS,
        film_hidden_dim: int = 32,
        base_model: DoubleUNet | None = None,
    ) -> None:
        super().__init__()
        fields = _normalize_condition_fields(condition_fields)
        activation_name = _serializable_activation(activation, field_name="activation")
        inter_activation_name = _serializable_activation(inter_activation, field_name="inter_activation")
        final_activation_name = _serializable_activation(final_activation, field_name="final_activation")
        resolved_condition_dim = len(fields) if condition_dim is None else int(condition_dim)
        if resolved_condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if resolved_condition_dim != len(fields):
            raise ValueError("condition_dim must equal the number of condition_fields")

        if base_model is None:
            base_model = DoubleUNet(
                nch_in=int(nch_in),
                depth_shared=int(depth_shared),
                depth_union=int(depth_union),
                nfeatures_init=int(nfeatures_init),
                nfeatures_inter=None if nfeatures_inter is None else int(nfeatures_inter),
                norm_start_level=int(norm_start_level),
                norm_groups=int(norm_groups),
                activation=activation_name,
                dropout_start_level=dropout_start_level,
                p_dropout=float(p_dropout),
                pool_mode=str(pool_mode),
                upsample_mode=str(upsample_mode),
                inter_activation=inter_activation_name,
                norm_head_groups=int(norm_head_groups),
                final_activation=final_activation_name,
                disabled_attr=disabled_attr,
                kaiming_normal=bool(kaiming_normal),
                depthwise=bool(depthwise),
                z_mu_activation=z_mu_activation,
            )
        elif not isinstance(base_model, DoubleUNet):
            raise TypeError(f"base_model must be DoubleUNet, got {type(base_model)!r}")

        activation_name = _serializable_activation(base_model.activation, field_name="base_model.activation")
        inter_activation_name = _serializable_activation(
            base_model.inter_activation,
            field_name="base_model.inter_activation",
        )
        final_activation_name = _serializable_activation(
            base_model.final_activation,
            field_name="base_model.final_activation",
        )

        self.network = FiLMConditionedDoubleUNet.from_base(
            base_model,
            condition_dim=resolved_condition_dim,
            hidden_dim=int(film_hidden_dim),
            copy_base=True,
        )
        self.nch_in = int(self.network.nch_in)
        self.condition_dim = resolved_condition_dim
        self.condition_fields = fields
        self.condition_schema = {
            "name": "astigmatism_film_v1",
            "fields": list(fields),
            "dimension": self.condition_dim,
        }
        self.model_config = {
            "nch_in": self.nch_in,
            "depth_shared": int(base_model.depth_shared),
            "depth_union": int(base_model.depth_union),
            "nfeatures_init": int(base_model.nfeatures_init),
            "nfeatures_inter": (
                None if base_model.nfeatures_inter is None else int(base_model.nfeatures_inter)
            ),
            "norm_start_level": int(base_model.norm_start_level),
            "norm_groups": int(base_model.norm_groups),
            "activation": activation_name,
            "dropout_start_level": base_model.dropout_start_level,
            "p_dropout": float(base_model.p_dropout),
            "pool_mode": str(base_model.pool_mode),
            "upsample_mode": str(base_model.upsample_mode),
            "inter_activation": inter_activation_name,
            "norm_head_groups": int(base_model.norm_head_groups),
            "final_activation": final_activation_name,
            "disabled_attr": base_model.disabled_attr_ix,
            "kaiming_normal": bool(base_model.kaiming_normal),
            "depthwise": bool(base_model.depthwise),
            "z_mu_activation": "tanh" if 4 in base_model.ch_ix_tanh else "sigmoid",
            "condition_dim": self.condition_dim,
            "condition_fields": list(self.condition_fields),
            "film_hidden_dim": int(film_hidden_dim),
        }

    @property
    def backbone(self) -> FiLMConditionedDoubleUNet:
        """Return the complete private localization network."""

        return self.network

    @property
    def model(self) -> FiLMConditionedDoubleUNet:
        """Compatibility alias for callers that refer to the localizer as model."""

        return self.network

    def forward(self, images: torch.Tensor | tuple[torch.Tensor, torch.Tensor], conditions: torch.Tensor | object = _MISSING_CONDITIONS) -> torch.Tensor:
        if conditions is _MISSING_CONDITIONS:
            if isinstance(images, tuple) and len(images) == 2:
                images, conditions = images
            else:
                raise TypeError("AstigmatismExpert.forward requires images and conditions")
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(f"images must have shape (N,C,H,W), got {getattr(images, 'shape', None)}")
        if images.shape[1] != self.nch_in:
            raise ValueError(f"AstigmatismExpert expects nch_in={self.nch_in}, got {images.shape[1]}")
        if not isinstance(conditions, torch.Tensor) or conditions.ndim != 2:
            raise ValueError(f"conditions must have shape (N,{self.condition_dim}), got {getattr(conditions, 'shape', None)}")
        if conditions.shape[0] != images.shape[0]:
            raise ValueError("condition batch size must match image batch size")
        if conditions.shape[1] != self.condition_dim:
            raise ValueError(f"AstigmatismExpert expects condition_dim={self.condition_dim}, got {conditions.shape[1]}")
        output = self.network((images, conditions))
        if output.shape[1] != self.output_channels:
            raise RuntimeError(f"AstigmatismExpert produced {output.shape[1]} channels, expected {self.output_channels}")
        return output

    def checkpoint_metadata(
        self,
        *,
        checkpoint_role: str = "prototype",
        instance_id: str | None = None,
        channel_spec: MeasurementChannelSpec | None = None,
        parent_checkpoint_hash: str | None = None,
        code_version: str = "0.4.0",
    ) -> CheckpointMetadata:
        """Build metadata that records the expert and ordered FiLM conditions."""

        return CheckpointMetadata(
            model_name="astigmatism_expert",
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


class LegacyAstigmatismExpert(AdaptedPSFExpert):
    """Feature-tensor adapter retained only for the early ``PSFMoE`` scaffold."""

    modality = PSFModality.ASTIGMATISM

    def __init__(self, feature_channels: int = 32) -> None:
        super().__init__(feature_channels, auxiliary_channels={"astigmatism_width": 2})


__all__ = [
    "AstigmatismExpert",
    "DEFAULT_ASTIGMATISM_CONDITION_FIELDS",
    "LegacyAstigmatismExpert",
]
