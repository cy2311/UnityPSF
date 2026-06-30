from __future__ import annotations

import copy

import torch
import torch.nn as nn

from neptune_v03.localization.smlm_unet import DoubleUNet


class FiLMModulator(nn.Module):
    def __init__(self, *, condition_dim: int, feature_channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if int(condition_dim) <= 0:
            raise ValueError("condition_dim must be positive")
        if int(feature_channels) <= 0:
            raise ValueError("feature_channels must be positive")
        self.condition_dim = int(condition_dim)
        self.feature_channels = int(feature_channels)
        hidden = max(int(hidden_dim), self.condition_dim, 1)
        self.net = nn.Sequential(
            nn.Linear(self.condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.feature_channels),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.dim() != 2:
            raise ValueError(f"Expected condition as (N,C), got {tuple(condition.shape)}")
        if condition.shape[0] != features.shape[0]:
            raise ValueError("FiLM condition batch size must match feature batch size")
        if condition.shape[1] != self.condition_dim:
            raise ValueError(f"Expected condition_dim={self.condition_dim}, got {condition.shape[1]}")
        gamma_beta = self.net(condition.to(device=features.device, dtype=features.dtype))
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(features.shape[0], self.feature_channels, 1, 1)
        beta = beta.view(features.shape[0], self.feature_channels, 1, 1)
        return features * (1.0 + gamma) + beta


def split_conditioned_input(
    x,
    *,
    image_channels: int,
    condition_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(x):
        if x.dim() != 4 or x.shape[1] != int(image_channels):
            raise ValueError(f"Expected image input (N,{int(image_channels)},H,W), got {tuple(x.shape)}")
        condition = torch.zeros((x.shape[0], int(condition_dim)), device=device, dtype=dtype)
        return x, condition
    if isinstance(x, (tuple, list)) and len(x) == 2:
        image, condition = x
        if not torch.is_tensor(image) or not torch.is_tensor(condition):
            raise TypeError("FiLM input must be (image_tensor, condition_tensor)")
        if image.dim() != 4 or image.shape[1] != int(image_channels):
            raise ValueError(f"Expected image input (N,{int(image_channels)},H,W), got {tuple(image.shape)}")
        if condition.shape[0] != image.shape[0]:
            raise ValueError("FiLM condition batch size must match image batch size")
        if condition.shape[1] != int(condition_dim):
            raise ValueError(f"Expected condition_dim={int(condition_dim)}, got {condition.shape[1]}")
        return image, condition
    raise TypeError(f"Unsupported FiLM input type: {type(x)!r}")


class FiLMConditionedDoubleUNet(nn.Module):
    def __init__(self, *, base_model: DoubleUNet, condition_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if not isinstance(base_model, DoubleUNet):
            raise TypeError(f"base_model must be DoubleUNet, got {type(base_model)!r}")
        self.base_model = base_model
        self.nch_in = int(base_model.nch_in)
        self.condition_dim = int(condition_dim)
        self.feature_channels = int(base_model.nfeatures_inter or base_model.nfeatures_init)
        self.film_modulator = FiLMModulator(
            condition_dim=self.condition_dim,
            feature_channels=self.feature_channels,
            hidden_dim=int(hidden_dim),
        )
        self.film_condition_mode = "feature_modulation_after_union"

    @classmethod
    def from_base(cls, base_model: DoubleUNet, *, condition_dim: int, hidden_dim: int = 32, copy_base: bool = True):
        return cls(
            base_model=copy.deepcopy(base_model) if bool(copy_base) else base_model,
            condition_dim=int(condition_dim),
            hidden_dim=int(hidden_dim),
        )

    def forward(self, x) -> torch.Tensor:
        parameter = next(self.parameters())
        image, condition = split_conditioned_input(
            x,
            image_channels=self.nch_in,
            condition_dim=self.condition_dim,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        image = image.to(device=parameter.device, dtype=parameter.dtype)
        condition = condition.to(device=parameter.device, dtype=parameter.dtype)
        features = self.base_model._forward_core(image)
        features = self.film_modulator(features, condition)
        output = torch.cat([head(features) for head in self.base_model.mt_heads], dim=1)
        return self.base_model._apply_nonlinearity(output)
