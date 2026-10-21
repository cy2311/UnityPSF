from __future__ import annotations

import torch
import torch.nn as nn


class LocalSetPredictor(nn.Module):
    def __init__(self, *, in_channels: int, hidden_channels: int, num_slots: int, patch_size: int):
        super().__init__()
        self.num_slots = int(num_slots)
        self.patch_size = int(patch_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(hidden_channels * 2, self.num_slots * 9)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x).flatten(1)
        raw = self.head(feat).view(x.shape[0], self.num_slots, 9)

        prob = torch.sigmoid(raw[..., 0])
        mu = torch.zeros((x.shape[0], self.num_slots, 4), dtype=raw.dtype, device=raw.device)
        sig = torch.zeros_like(mu)

        mu[..., 0] = torch.sigmoid(raw[..., 1])
        mu[..., 1] = torch.sigmoid(raw[..., 2]) * float(self.patch_size)
        mu[..., 2] = torch.sigmoid(raw[..., 3]) * float(self.patch_size)
        mu[..., 3] = torch.tanh(raw[..., 4])

        sig[..., 0] = torch.nn.functional.softplus(raw[..., 5]) + 1e-3
        sig[..., 1] = torch.nn.functional.softplus(raw[..., 6]) + 1e-3
        sig[..., 2] = torch.nn.functional.softplus(raw[..., 7]) + 1e-3
        sig[..., 3] = torch.nn.functional.softplus(raw[..., 8]) + 1e-3

        return {"prob": prob, "mu": mu, "sig": sig}

