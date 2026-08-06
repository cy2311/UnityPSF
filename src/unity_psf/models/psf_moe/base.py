"""Shared feature and output heads for the UnityPSF PSF experts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import torch
from torch import nn

from unity_psf.contracts.modality import PSFExpertOutput, PSFModality


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if int(channels) % groups == 0:
            return groups
    return 1


class PSFExpert(nn.Module, ABC):
    """Base contract implemented by every modality-specific expert."""

    modality: PSFModality

    @abstractmethod
    def forward(self, features: torch.Tensor) -> PSFExpertOutput:
        raise NotImplementedError


class SharedPSFStem(nn.Module):
    """Small shared image encoder with explicit channel padding semantics."""

    def __init__(self, *, in_channels: int = 3, feature_channels: int = 32) -> None:
        super().__init__()
        if int(in_channels) <= 0 or int(feature_channels) <= 0:
            raise ValueError("in_channels and feature_channels must be positive")
        self.in_channels = int(in_channels)
        self.feature_channels = int(feature_channels)
        hidden = max(16, self.feature_channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, self.feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.feature_channels), self.feature_channels),
            nn.SiLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(f"images must have shape (N,C,H,W), got {type(images)!r} {getattr(images, 'shape', None)}")
        if images.shape[1] > self.in_channels:
            raise ValueError(f"images provide {images.shape[1]} channels, but stem accepts at most {self.in_channels}")
        if images.shape[1] < self.in_channels:
            padding = images.new_zeros(images.shape[0], self.in_channels - images.shape[1], *images.shape[2:])
            images = torch.cat((images, padding), dim=1)
        return self.net(images)


class PSFExpertHead(nn.Module):
    """Common localization heads plus optional modality-specific dense heads."""

    def __init__(self, feature_channels: int, *, auxiliary_channels: Mapping[str, int] = ()) -> None:
        super().__init__()
        channels = int(feature_channels)
        self.detection = nn.Conv2d(channels, 1, kernel_size=1)
        self.xy_offset = nn.Conv2d(channels, 2, kernel_size=1)
        self.z = nn.Conv2d(channels, 1, kernel_size=1)
        self.photons = nn.Conv2d(channels, 1, kernel_size=1)
        self.auxiliary = nn.ModuleDict(
            {name: nn.Conv2d(channels, int(width), kernel_size=1) for name, width in dict(auxiliary_channels).items()}
        )

    def forward(self, features: torch.Tensor) -> PSFExpertOutput:
        auxiliary = {name: head(features) for name, head in self.auxiliary.items()}
        return PSFExpertOutput(
            detection_logits=self.detection(features).squeeze(1),
            xy_offset=torch.tanh(self.xy_offset(features)) * 0.5,
            z=self.z(features).squeeze(1),
            photons=torch.nn.functional.softplus(self.photons(features).squeeze(1)),
            auxiliary=auxiliary,
        ).validate(batch_size=features.shape[0])


class AdaptedPSFExpert(PSFExpert):
    """Convenience base for independent modality adapters and heads."""

    def __init__(self, feature_channels: int, *, auxiliary_channels: Mapping[str, int] = ()) -> None:
        super().__init__()
        channels = int(feature_channels)
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        self.head = PSFExpertHead(channels, auxiliary_channels=auxiliary_channels)

    def forward(self, features: torch.Tensor) -> PSFExpertOutput:
        return self.head(self.adapter(features))


__all__ = ["AdaptedPSFExpert", "PSFExpert", "PSFExpertHead", "SharedPSFStem"]
