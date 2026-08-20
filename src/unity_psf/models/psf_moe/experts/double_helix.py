"""Double-helix expert with lobe geometry auxiliary heads."""

from __future__ import annotations

from typing import Any

import torch

from unity_psf.contracts.checkpoint import CheckpointMetadata
from unity_psf.contracts.modality import InputFrameSpec, MeasurementChannelSpec
from unity_psf.contracts.modality import PSFModality
from unity_psf.models.psf_moe.base import AdaptedPSFExpert


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


class DoubleHelixExpert(AdaptedPSFExpert):
    modality = PSFModality.DOUBLE_HELIX

    def __init__(self, feature_channels: int = 32) -> None:
        super().__init__(
            feature_channels,
            auxiliary_channels={"lobe_angle": 1, "lobe_separation": 1},
        )

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
            model_name="double_helix_expert",
            checkpoint_role=checkpoint_role,
            expert_type=self.modality,
            model_config={"feature_channels": int(self.adapter[0].in_channels), "auxiliary_channels": ["lobe_angle", "lobe_separation"]},
            input_frame_spec=InputFrameSpec(input_frame_channels=3),
            instance_id=instance_id,
            channel_spec=channel_spec,
            parent_checkpoint_hash=parent_checkpoint_hash,
            code_version=code_version,
            experts=(self.modality.value,),
            shared_feature_channels=int(self.adapter[0].in_channels),
            router_mode="hard",
        )


__all__ = ["DoubleHelixDirectXYZLoss", "DoubleHelixExpert"]
