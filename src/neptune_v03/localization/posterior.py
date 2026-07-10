from __future__ import annotations

from dataclasses import dataclass

import torch

from neptune_v03.localization.model import LocalizationModelOutput
from neptune_v03.localization.smlm_output import SMLMOutputChannels, decode_smlm_output
from neptune_v03.localization.training_adapter import LocalizationTrainBatch, localization_batch_to_device


@dataclass(frozen=True)
class DetectionPosteriorSamples:
    xyzph: torch.Tensor
    mask: torch.Tensor
    logits: torch.Tensor
    metadata: dict[str, object]


def select_posterior_emitters(
    y_out: torch.Tensor,
    *,
    threshold: float,
    max_emitters: int,
    photon_scale: float | None,
    z_scale: float | None,
    candidate_threshold: float,
    split_threshold: float,
) -> DetectionPosteriorSamples:
    decoded = decode_smlm_output(y_out.detach())
    logits = decoded.p.detach()
    spatial_p = _posterior_spatial_integration_probability(
        logits,
        candidate_threshold=float(candidate_threshold),
        split_threshold=float(split_threshold),
    )
    active = spatial_p >= float(threshold)
    active = _topk_mask(spatial_p, active, max_emitters=int(max_emitters))
    xyzph = torch.zeros((int(logits.shape[0]), int(max_emitters), 4), dtype=torch.float32, device=y_out.device)
    mask = torch.zeros((int(logits.shape[0]), int(max_emitters)), dtype=torch.bool, device=y_out.device)
    for batch_idx in range(int(logits.shape[0])):
        row_ix, col_ix = active[batch_idx].nonzero(as_tuple=True)
        for slot, (row, col) in enumerate(zip(row_ix.tolist(), col_ix.tolist())):
            xyzph[batch_idx, slot, 0] = float(col) + 0.5 + y_out[batch_idx, SMLMOutputChannels.x_mu, row, col]
            xyzph[batch_idx, slot, 1] = float(row) + 0.5 + y_out[batch_idx, SMLMOutputChannels.y_mu, row, col]
            xyzph[batch_idx, slot, 2] = _physical_z(
                y_out[batch_idx, SMLMOutputChannels.z_mu, row, col].to(dtype=torch.float32),
                z_scale=z_scale,
            )
            xyzph[batch_idx, slot, 3] = _physical_photons(
                y_out[batch_idx, SMLMOutputChannels.photons_mu, row, col].to(dtype=torch.float32),
                photon_scale=photon_scale,
            )
            mask[batch_idx, slot] = True
    return DetectionPosteriorSamples(
        xyzph=xyzph.detach().cpu(),
        mask=mask.detach().cpu(),
        logits=spatial_p.detach().cpu(),
        metadata={
            "source": "legacy_spatial_integration_posterior",
            "photon_scale": None if photon_scale is None else float(photon_scale),
            "z_scale": None if z_scale is None else float(z_scale),
            "candidate_threshold": float(candidate_threshold),
            "split_threshold": float(split_threshold),
        },
    )


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
        selected = select_posterior_emitters(
            output.detach(),
            threshold=float(threshold),
            max_emitters=int(max_emitters),
            photon_scale=photon_scale,
            z_scale=z_scale,
            candidate_threshold=float(candidate_threshold),
            split_threshold=float(split_threshold),
        )
        return DetectionPosteriorSamples(
            xyzph=selected.xyzph,
            mask=selected.mask,
            logits=selected.logits,
            metadata={**selected.metadata, "seed": int(seed)},
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


def _posterior_spatial_integration_probability(
    p: torch.Tensor,
    *,
    candidate_threshold: float,
    split_threshold: float,
) -> torch.Tensor:
    filt = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=p.dtype,
        device=p.device,
    ).view(1, 1, 3, 3)
    conv = torch.nn.functional.conv2d(p.unsqueeze(1), filt, padding=1)
    p_clip = torch.where(p > float(candidate_threshold), p, torch.zeros_like(p))
    pool = torch.nn.functional.max_pool2d(p_clip.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    max_mask1 = torch.eq(p.unsqueeze(1), pool)
    p_ps1 = max_mask1.to(dtype=p.dtype) * conv
    p_copy = p.unsqueeze(1) * (1.0 - max_mask1.to(dtype=p.dtype))
    max_mask2 = torch.where(p_copy > float(split_threshold), torch.ones_like(p_copy), torch.zeros_like(p_copy))
    p_ps2 = max_mask2 * conv
    return torch.clamp(p_ps1 + p_ps2, 0.0, 1.0).squeeze(1)


def _topk_mask(p: torch.Tensor, active: torch.Tensor, *, max_emitters: int) -> torch.Tensor:
    limited = torch.zeros_like(active)
    for batch_idx in range(int(p.shape[0])):
        values = torch.where(active[batch_idx], p[batch_idx], torch.full_like(p[batch_idx], -torch.inf))
        keep = min(int(max_emitters), int(active[batch_idx].sum().item()))
        if keep <= 0:
            continue
        indices = torch.topk(values.reshape(-1), k=keep).indices
        limited[batch_idx].reshape(-1)[indices] = True
    return limited


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
