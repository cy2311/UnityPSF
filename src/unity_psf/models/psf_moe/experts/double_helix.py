"""Complete image-to-localization expert for double-helix PSFs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from unity_psf.contracts.checkpoint import CheckpointMetadata
from unity_psf.contracts.modality import InputFrameSpec, MeasurementChannelSpec, PSFModality
from unity_psf.data.normalization import LunarInputNormalization
from unity_psf.localization.film import FiLMConditionedDoubleUNet
from unity_psf.localization.smlm_output import SMLMOutputChannels
from unity_psf.localization.smlm_unet import DoubleUNet


_MISSING_CONDITIONS = object()


class DoubleHelixExpert(nn.Module):
    """Standalone DoubleUNet using the 10-channel UnityPSF GMM contract."""

    modality = PSFModality.DOUBLE_HELIX
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
        input_normalization: Mapping[str, object] | None = None,
        condition_dim: int | None = None,
        condition_fields: Sequence[str] = (),
        film_hidden_dim: int = 32,
        ppg_map_path: str | None = None,
        domain_count: int = 1,
    ) -> None:
        super().__init__()
        if int(domain_count) <= 0:
            raise ValueError("domain_count must be positive")
        base_network = DoubleUNet(
            nch_in=int(nch_in),
            depth_shared=int(depth_shared),
            depth_union=int(depth_union),
            nfeatures_init=int(nfeatures_init),
            nfeatures_inter=None if nfeatures_inter is None else int(nfeatures_inter),
            norm_start_level=int(norm_start_level),
            norm_groups=int(norm_groups),
            activation=str(activation),
            dropout_start_level=dropout_start_level,
            p_dropout=float(p_dropout),
            pool_mode=str(pool_mode),
            upsample_mode=str(upsample_mode),
            inter_activation=str(inter_activation),
            norm_head_groups=int(norm_head_groups),
            final_activation=str(final_activation),
            disabled_attr=disabled_attr,
            kaiming_normal=bool(kaiming_normal),
            depthwise=bool(depthwise),
            z_mu_activation=z_mu_activation,
        )
        resolved_condition_dim = 0 if condition_dim is None else int(condition_dim)
        fields = tuple(str(field).strip() for field in condition_fields)
        if resolved_condition_dim < 0:
            raise ValueError("condition_dim must be non-negative")
        if resolved_condition_dim != len(fields):
            raise ValueError("condition_dim must equal the number of condition_fields")
        self.network = (
            FiLMConditionedDoubleUNet.from_base(
                base_network,
                condition_dim=resolved_condition_dim,
                hidden_dim=int(film_hidden_dim),
                copy_base=True,
            )
            if resolved_condition_dim
            else base_network
        )
        self.nch_in = int(nch_in)
        self.condition_dim = resolved_condition_dim
        self.condition_fields = fields
        self.condition_schema = (
            {"name": "none", "fields": [], "dimension": 0}
            if not resolved_condition_dim
            else {"name": "ppg_film_v1", "fields": list(fields), "dimension": resolved_condition_dim}
        )
        self.input_normalization = (
            None
            if input_normalization is None
            else LunarInputNormalization.from_config(input_normalization)
        )
        self.model_config = {
            "nch_in": self.nch_in,
            "depth_shared": int(depth_shared),
            "depth_union": int(depth_union),
            "nfeatures_init": int(nfeatures_init),
            "nfeatures_inter": None if nfeatures_inter is None else int(nfeatures_inter),
            "norm_start_level": int(norm_start_level),
            "norm_groups": int(norm_groups),
            "activation": str(activation),
            "dropout_start_level": dropout_start_level,
            "p_dropout": float(p_dropout),
            "pool_mode": str(pool_mode),
            "upsample_mode": str(upsample_mode),
            "inter_activation": str(inter_activation),
            "norm_head_groups": int(norm_head_groups),
            "final_activation": str(final_activation),
            "disabled_attr": disabled_attr,
            "kaiming_normal": bool(kaiming_normal),
            "depthwise": bool(depthwise),
            "z_mu_activation": "tanh" if z_mu_activation is None else str(z_mu_activation),
            "input_normalization": (
                None if self.input_normalization is None else self.input_normalization.to_config()
            ),
            "condition_dim": self.condition_dim,
            "condition_fields": list(self.condition_fields),
            "film_hidden_dim": int(film_hidden_dim),
            "ppg_map_path": ppg_map_path,
        }

    @property
    def backbone(self) -> nn.Module:
        return self.network

    def forward(
        self,
        images: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        conditions: torch.Tensor | object = _MISSING_CONDITIONS,
    ) -> torch.Tensor:
        if self.condition_dim:
            if conditions is _MISSING_CONDITIONS:
                if isinstance(images, tuple) and len(images) == 2:
                    images, conditions = images
                else:
                    raise TypeError("conditioned DoubleHelixExpert.forward requires images and conditions")
        elif conditions is not _MISSING_CONDITIONS:
            raise TypeError("unconditioned DoubleHelixExpert does not accept conditions")
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(f"images must have shape (N,C,H,W), got {getattr(images, 'shape', None)}")
        if images.shape[1] != self.nch_in:
            raise ValueError(f"DoubleHelixExpert expects nch_in={self.nch_in}, got {images.shape[1]}")
        if self.input_normalization is not None:
            images = self.input_normalization.normalize(images)
        output = self.network(images if conditions is _MISSING_CONDITIONS else (images, conditions))
        if output.shape[1] != self.output_channels:
            raise RuntimeError(
                f"DoubleHelixExpert produced {output.shape[1]} channels, expected {self.output_channels}"
            )
        return output

    def checkpoint_metadata(
        self,
        *,
        checkpoint_role: str = "prototype",
        instance_id: str | None = None,
        channel_spec: MeasurementChannelSpec | None = None,
        parent_checkpoint_hash: str | None = None,
        code_version: str = "0.3.0",
    ) -> CheckpointMetadata:
        return CheckpointMetadata(
            model_name="double_helix_expert",
            checkpoint_role=checkpoint_role,
            expert_type=self.modality,
            model_config=dict(self.model_config),
            input_frame_spec=InputFrameSpec(input_frame_channels=self.nch_in),
            instance_id=instance_id,
            channel_spec=channel_spec,
            condition_schema=self.condition_schema,
            parent_checkpoint_hash=parent_checkpoint_hash,
            code_version=code_version,
            experts=(self.modality.value,),
            shared_feature_channels=int(
                self.network.feature_channels
                if self.condition_dim
                else self.network.nfeatures_init
            ),
            router_mode="hard",
        )


__all__ = ["DoubleHelixExpert"]
