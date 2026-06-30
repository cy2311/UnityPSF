from __future__ import annotations

from dataclasses import dataclass

import torch

from neptune_v03.localization.model import LocalizationModelOutput
from neptune_v03.localization.legacy_decode import decode_legacy_smlm_emitters, spatial_integration_probability
from neptune_v03.localization.smlm_output import SMLMOutputChannels, decode_smlm_output
from neptune_v03.localization.training_adapter import LocalizationTrainBatch, localization_batch_to_device


@dataclass(frozen=True)
class DetectionPosteriorSamples:
    xyzph: torch.Tensor
    mask: torch.Tensor
    logits: torch.Tensor
    metadata: dict[str, object]


def sample_detection_posterior(
    *,
    model: torch.nn.Module,
    batch: LocalizationTrainBatch,
    threshold: float,
    max_emitters: int,
    seed: int = 0,
    photon_scale: float | None = None,
    z_scale: float | None = None,
    candidate_threshold: float = 0.3,
    split_threshold: float = 0.6,
) -> DetectionPosteriorSamples:
    if int(max_emitters) <= 0:
        raise ValueError("max_emitters must be positive")

    was_training = model.training
    model.eval()
    batch = localization_batch_to_device(batch, _model_device(model))
    with torch.no_grad():
        output = model(batch.model_input)
    if was_training:
        model.train()

    if isinstance(output, LocalizationModelOutput):
        logits = output.detection_logits.detach()
        xy_offset = output.xy_offset.detach()
        z_map = output.z.detach()
        photon_map = output.photons.detach()
    elif isinstance(output, torch.Tensor) and output.ndim == 4 and int(output.shape[1]) == SMLMOutputChannels.count:
        decoded = decode_smlm_output(output.detach())
        logits = decoded.p.detach()
        spatial_p = spatial_integration_probability(
            logits,
            raw_th=float(candidate_threshold),
            split_th=float(split_threshold),
        )
        emitters = decode_legacy_smlm_emitters(
            output.detach(),
            raw_th=float(candidate_threshold),
            split_th=float(split_threshold),
            accept_th=float(threshold),
            photon_scale=photon_scale,
            z_scale=z_scale,
            max_emitters=int(max_emitters),
        )
        xyzph = torch.zeros((int(logits.shape[0]), int(max_emitters), 4), dtype=torch.float32)
        mask = torch.zeros((int(logits.shape[0]), int(max_emitters)), dtype=torch.bool)
        counts = [0 for _ in range(int(logits.shape[0]))]
        for row_idx in range(int(emitters.xyz_px_nm.shape[0])):
            batch_idx = int(emitters.batch_index[row_idx].item())
            slot = counts[batch_idx]
            if slot >= int(max_emitters):
                continue
            xyzph[batch_idx, slot, :3] = emitters.xyz_px_nm[row_idx]
            xyzph[batch_idx, slot, 3] = emitters.photons[row_idx]
            mask[batch_idx, slot] = True
            counts[batch_idx] += 1
        return DetectionPosteriorSamples(
            xyzph=xyzph.detach().cpu(),
            mask=mask.detach().cpu(),
            logits=spatial_p.detach().cpu(),
            metadata={
                "seed": int(seed),
                "source": "legacy_spatial_integration_posterior",
                "photon_scale": None if photon_scale is None else float(photon_scale),
                "z_scale": None if z_scale is None else float(z_scale),
                "candidate_threshold": float(candidate_threshold),
                "split_threshold": float(split_threshold),
            },
        )
    else:
        logits = output.squeeze(1).detach()
        xy_offset = torch.zeros((logits.shape[0], 2, logits.shape[1], logits.shape[2]), dtype=logits.dtype)
        z_map = torch.zeros_like(logits)
        photon_map = torch.sigmoid(logits)

    batch_size, height, width = int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2])
    scores = logits.flatten(start_dim=1)
    top_values, top_indices = torch.topk(scores, k=min(int(max_emitters), scores.shape[1]), dim=1)
    xyzph = torch.zeros((batch_size, int(max_emitters), 4), dtype=torch.float32, device=logits.device)
    mask = torch.zeros((batch_size, int(max_emitters)), dtype=torch.bool, device=logits.device)
    for batch_idx in range(batch_size):
        count = int(top_indices.shape[1])
        cols = (top_indices[batch_idx] % width).to(dtype=torch.float32)
        rows = torch.div(top_indices[batch_idx], width, rounding_mode="floor").to(dtype=torch.float32)
        xyzph[batch_idx, :count, 0] = cols + xy_offset[batch_idx, 0].flatten()[top_indices[batch_idx]].to(dtype=torch.float32)
        xyzph[batch_idx, :count, 1] = rows + xy_offset[batch_idx, 1].flatten()[top_indices[batch_idx]].to(dtype=torch.float32)
        xyzph[batch_idx, :count, 2] = _physical_z(
            z_map[batch_idx].flatten()[top_indices[batch_idx]].to(dtype=torch.float32),
            z_scale=z_scale,
        )
        xyzph[batch_idx, :count, 3] = _physical_photons(
            photon_map[batch_idx].flatten()[top_indices[batch_idx]].to(dtype=torch.float32),
            photon_scale=photon_scale,
        )
        mask[batch_idx, :count] = top_values[batch_idx] >= float(threshold)
    return DetectionPosteriorSamples(
        xyzph=xyzph.detach().cpu(),
        mask=mask.detach().cpu(),
        logits=logits.detach().cpu(),
        metadata={
            "seed": int(seed),
            "source": "detection_posterior",
            "photon_scale": None if photon_scale is None else float(photon_scale),
            "z_scale": None if z_scale is None else float(z_scale),
        },
    )


def _physical_photons(value: torch.Tensor, *, photon_scale: float | None) -> torch.Tensor:
    if photon_scale is None:
        return value
    return value * float(photon_scale)


def _physical_z(value: torch.Tensor, *, z_scale: float | None) -> torch.Tensor:
    if z_scale is None:
        return value
    scale = abs(float(z_scale))
    scale_nm = scale * 1000.0 if scale <= 10.0 else scale
    return value * scale_nm


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
