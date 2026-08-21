"""Shared validation-record construction for training entrypoints."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from unity_psf.reporting import InstanceVisualRecord
from unity_psf.training.multimodal_joint import RoutedTrainingBatch


def build_instance_visual_record(
    key: str,
    batch: RoutedTrainingBatch,
    localization,
    *,
    losses: Sequence[float],
    steps: int,
    checkpoint_hash: str,
) -> InstanceVisualRecord:
    image_stack = batch.images[0].detach().cpu()
    input_image = image_stack.mean(dim=0).numpy()
    probability = localization.decoded.p[0].detach().cpu()
    photon_map = localization.decoded.pxyz_mu[0, 0].detach().cpu().clamp_min(0)
    selected = torch.topk(probability.reshape(-1), k=min(8, probability.numel())).indices
    width = probability.shape[1]
    prediction_xy = torch.stack((selected % width, selected // width), dim=1).numpy().astype(np.float32)
    reconstruction = (probability * photon_map).numpy()
    return InstanceVisualRecord(
        instance_key=key,
        input_image=input_image,
        patches=tuple(frame.numpy() for frame in image_stack),
        loss_history=tuple(losses),
        route_count=1,
        step_count=steps,
        sample_count=steps * int(batch.images.shape[0]),
        prediction_xy=prediction_xy,
        reconstruction=reconstruction,
        status="trained-smoke",
        checkpoint_hash=checkpoint_hash,
    )


__all__ = ["build_instance_visual_record"]
