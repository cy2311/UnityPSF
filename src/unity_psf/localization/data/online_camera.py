"""Camera conversion used by online localization providers."""

from __future__ import annotations

import torch


def poisson_camera_readout(config, frame_photons: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Convert expected photons into deterministic simulated raw ADU frames."""
    generator = torch.Generator(device=frame_photons.device).manual_seed(int(seed))
    electrons = frame_photons * float(config.camera_qe) + float(config.camera_spurious_charge)
    raw_adu = (
        torch.poisson(electrons, generator=generator) / max(float(config.camera_e_per_adu), 1e-12)
        + float(config.camera_baseline)
    )
    if float(config.camera_read_sigma) > 0:
        raw_adu = raw_adu + torch.randn(
            raw_adu.shape,
            dtype=raw_adu.dtype,
            device=raw_adu.device,
            generator=generator,
        ) * float(config.camera_read_sigma)
    return raw_adu
