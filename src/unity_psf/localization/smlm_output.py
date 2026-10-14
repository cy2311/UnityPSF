from __future__ import annotations

from dataclasses import dataclass

import torch


class SMLMOutputChannels:
    p: int = 0
    photons_mu: int = 1
    x_mu: int = 2
    y_mu: int = 3
    z_mu: int = 4
    pxyz_mu: slice = slice(1, 5)
    photons_sigma: int = 5
    x_sigma: int = 6
    y_sigma: int = 7
    z_sigma: int = 8
    pxyz_sigma: slice = slice(5, 9)
    pxyz_sig: slice = pxyz_sigma
    bg: int = 9
    count: int = 10


@dataclass(frozen=True)
class SMLMOutput:
    raw: torch.Tensor
    p: torch.Tensor
    pxyz_mu: torch.Tensor
    pxyz_sigma: torch.Tensor
    bg: torch.Tensor

    @property
    def detection(self) -> torch.Tensor:
        return self.p

    @property
    def detection_prob(self) -> torch.Tensor:
        return self.p

    @property
    def z_mu(self) -> torch.Tensor:
        return self.raw[:, SMLMOutputChannels.z_mu]

    @property
    def z_sigma(self) -> torch.Tensor:
        return self.raw[:, SMLMOutputChannels.z_sigma]


def decode_smlm_output(output: torch.Tensor) -> SMLMOutput:
    if output.ndim != 4:
        raise ValueError(f"Expected SMLM output as (B,10,H,W), got {tuple(output.shape)}")
    if int(output.shape[1]) != SMLMOutputChannels.count:
        raise ValueError(f"Expected 10-channel SMLM tensor, got {int(output.shape[1])} channels")
    return SMLMOutput(
        raw=output,
        p=output[:, SMLMOutputChannels.p],
        pxyz_mu=output[:, SMLMOutputChannels.pxyz_mu],
        pxyz_sigma=output[:, SMLMOutputChannels.pxyz_sigma],
        bg=output[:, SMLMOutputChannels.bg],
    )


__all__ = ["SMLMOutput", "SMLMOutputChannels", "decode_smlm_output"]
