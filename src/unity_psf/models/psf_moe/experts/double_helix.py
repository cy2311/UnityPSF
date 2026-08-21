"""Double-helix expert with lobe geometry auxiliary heads."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from unity_psf.contracts.checkpoint import CheckpointMetadata
from unity_psf.contracts.modality import InputFrameSpec, MeasurementChannelSpec, PSFExpertOutput, PSFModality
from unity_psf.localization.smlm_unet import DoubleUNet


class DoubleHelixDirectXYZLoss:
    """Dense supervised loss for the DH direct-XYZ output contract."""

    required_targets = ("detection", "xy_offset", "z", "photons", "lobe_angle", "lobe_separation")

    def __init__(self, *, auxiliary_weight: float = 0.25) -> None:
        self.auxiliary_weight = float(auxiliary_weight)

    def __call__(self, output, targets: dict[str, Any]) -> Any:
        missing = [name for name in self.required_targets if name not in targets]
        if missing:
            raise ValueError(f"DH direct-XYZ targets missing keys: {', '.join(missing)}")
        detection = torch.nn.functional.binary_cross_entropy_with_logits(
            output.detection_logits, targets["detection"].to(dtype=output.detection_logits.dtype)
        )
        direct = (
            torch.nn.functional.mse_loss(output.xy_offset, targets["xy_offset"].to(dtype=output.xy_offset.dtype))
            + torch.nn.functional.mse_loss(output.z, targets["z"].to(dtype=output.z.dtype))
            + torch.nn.functional.mse_loss(output.photons, targets["photons"].to(dtype=output.photons.dtype))
        )
        auxiliary = sum(
            torch.nn.functional.mse_loss(output.auxiliary[name], targets[name].to(dtype=output.auxiliary[name].dtype))
            for name in ("lobe_angle", "lobe_separation")
            if name in output.auxiliary
        )
        return detection + direct + self.auxiliary_weight * auxiliary


class DoubleHelixImageExpert(nn.Module):
    """Standalone DH image expert with its own complete image backbone.

    This is the formal DH expert and consumes raw temporal frames directly.
    """

    modality = PSFModality.DOUBLE_HELIX

    def __init__(
        self,
        *,
        nch_in: int = 3,
        in_channels: int | None = None,
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
        feature_channels: int | None = None,
        condition_dim: int = 0,
        domain_count: int = 1,
    ) -> None:
        super().__init__()
        if in_channels is not None:
            nch_in = int(in_channels)
        if feature_channels is not None:
            nfeatures_init = int(feature_channels)
            nfeatures_inter = int(feature_channels)
        self.backbone = DoubleUNet(
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
        channels = int(nfeatures_init if nfeatures_inter is None else nfeatures_inter)
        self.detection = nn.Conv2d(channels, 1, kernel_size=1)
        self.xy_offset = nn.Conv2d(channels, 2, kernel_size=1)
        self.z = nn.Conv2d(channels, 1, kernel_size=1)
        self.photons = nn.Conv2d(channels, 1, kernel_size=1)
        self.auxiliary = nn.ModuleDict(
            {
                "lobe_angle": nn.Conv2d(channels, 1, kernel_size=1),
                "lobe_separation": nn.Conv2d(channels, 1, kernel_size=1),
            }
        )
        self.nch_in = int(nch_in)
        self.feature_channels = channels
        self.condition_dim = int(condition_dim)
        self.domain_count = int(domain_count)
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
            "z_mu_activation": z_mu_activation,
        }

    def forward(self, images: torch.Tensor):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(f"images must have shape (N,C,H,W), got {getattr(images, 'shape', None)}")
        if images.shape[1] != self.nch_in:
            raise ValueError(f"DoubleHelixImageExpert expects nch_in={self.nch_in}, got {images.shape[1]}")
        features = self.backbone._forward_core(images)
        return PSFExpertOutput(
            detection_logits=self.detection(features).squeeze(1),
            xy_offset=torch.tanh(self.xy_offset(features)) * 0.5,
            z=self.z(features).squeeze(1),
            photons=torch.nn.functional.softplus(self.photons(features).squeeze(1)),
            auxiliary={name: head(features) for name, head in self.auxiliary.items()},
        ).validate(batch_size=images.shape[0])

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
            model_name="double_helix_image_expert",
            checkpoint_role=checkpoint_role,
            expert_type=self.modality,
            model_config=dict(self.model_config),
            input_frame_spec=InputFrameSpec(input_frame_channels=self.nch_in),
            instance_id=instance_id,
            channel_spec=channel_spec,
            parent_checkpoint_hash=parent_checkpoint_hash,
            code_version=code_version,
            experts=(self.modality.value,),
            shared_feature_channels=self.feature_channels,
            router_mode="hard",
        )


__all__ = ["DoubleHelixDirectXYZLoss", "DoubleHelixImageExpert"]
