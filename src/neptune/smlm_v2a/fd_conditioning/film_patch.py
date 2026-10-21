from __future__ import annotations

from collections.abc import Sequence
from types import MethodType

import torch
import torch.nn as nn

from neptune.smlm_v2a.models.smlm_model import DoubleUNet, SingleUNet


def _conv_out_channels(module: nn.Module) -> int:
    if isinstance(module, nn.Sequential):
        for child in reversed(module):
            if isinstance(child, nn.Conv2d):
                return int(child.out_channels)
    elif isinstance(module, nn.Conv2d):
        return int(module.out_channels)
    raise TypeError(f"Unsupported convolution block type: {type(module)!r}")


class ZeroInitFiLM2d(nn.Module):
    def __init__(self, *, condition_dim: int, channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.channels = int(channels)
        hidden = max(int(hidden_dim), 16)
        self.pre = nn.Sequential(
            nn.LayerNorm(self.condition_dim),
            nn.Linear(self.condition_dim, hidden),
            nn.SiLU(),
        )
        self.to_gamma_beta = nn.Linear(hidden, 2 * self.channels)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected feature tensor with shape (N,C,H,W), got {tuple(x.shape)}")
        if condition.ndim != 2:
            raise ValueError(f"Expected condition tensor with shape (N,D), got {tuple(condition.shape)}")
        if condition.size(0) != x.size(0):
            raise ValueError(f"Batch mismatch between features {tuple(x.shape)} and condition {tuple(condition.shape)}")
        condition = condition.to(device=x.device, dtype=x.dtype)
        gamma, beta = self.to_gamma_beta(self.pre(condition)).chunk(2, dim=1)
        return x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)


def _patch_unet_with_film(unet: nn.Module, *, condition_dim: int, hidden_dim: int) -> nn.Module:
    unet.fd_encoder_film = nn.ModuleList(
        [
            ZeroInitFiLM2d(condition_dim=condition_dim, channels=_conv_out_channels(block.conv2), hidden_dim=hidden_dim)
            for block in unet.encoder
        ]
    )
    unet.fd_base_film = ZeroInitFiLM2d(
        condition_dim=condition_dim,
        channels=_conv_out_channels(unet.base.conv2),
        hidden_dim=hidden_dim,
    )
    unet.fd_decoder_film = nn.ModuleList(
        [
            ZeroInitFiLM2d(condition_dim=condition_dim, channels=_conv_out_channels(block.conv2), hidden_dim=hidden_dim)
            for block in unet.decoder
        ]
    )
    unet.fd_condition_dim = int(condition_dim)
    unet.fd_condition_hidden_dim = int(hidden_dim)

    def _forward_conditioned(self, input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = input
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            x = self.fd_encoder_film[level](x, condition)
            encoder_out.append(x)
            x = self.pooler[level](x)

        x = self.base(x)
        x = self.fd_base_film(x, condition)

        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsampler[level](x)
            x = self.decoder[level](self._crop_and_concat(x, encoder_out[level]))
            x = self.fd_decoder_film[level](x, condition)

        if self.out_conv is not None:
            x = self.out_conv(x)
            if self.out_activation is not None:
                x = self.out_activation(x)
        return x

    unet.forward_conditioned = MethodType(_forward_conditioned, unet)
    return unet


def _unpack_conditioned_input(
    x: torch.Tensor | Sequence[torch.Tensor],
    *,
    image_channels: int,
    condition_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(x, Sequence) and not isinstance(x, torch.Tensor):
        if len(x) != 2:
            raise ValueError("Conditioned model expects input as (frames, condition_vector).")
        image, condition = x
    else:
        image, condition = x, None
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("Frames input must be a tensor with shape (N,C,H,W).")
    if image.size(1) != int(image_channels):
        raise ValueError(f"Expected {image_channels} image channels, got {image.size(1)}.")
    if condition is None:
        condition = torch.zeros((image.size(0), int(condition_dim)), dtype=image.dtype, device=image.device)
    elif torch.is_tensor(condition):
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2:
            raise ValueError(f"Condition vector must be shaped as (N,D), got {tuple(condition.shape)}")
        if condition.size(0) != image.size(0):
            raise ValueError(f"Batch mismatch between frames {tuple(image.shape)} and condition {tuple(condition.shape)}")
        if condition.size(1) != int(condition_dim):
            raise ValueError(f"Expected condition dim {condition_dim}, got {condition.size(1)}.")
        condition = condition.to(device=image.device, dtype=image.dtype)
    else:
        raise TypeError("Condition input must be a tensor.")
    return image, condition


def _patch_single_unet(model: SingleUNet, *, condition_dim: int, hidden_dim: int) -> SingleUNet:
    model.image_input_channels = int(model.nch_in)
    model.fd_condition_dim = int(condition_dim)
    model.fd_condition_hidden_dim = int(hidden_dim)
    model.fd_condition_strategy = "film_tile_vector_v1"
    _patch_unet_with_film(model.unet, condition_dim=condition_dim, hidden_dim=hidden_dim)

    def _forward_conditioned(self: SingleUNet, x: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        image, condition = _unpack_conditioned_input(
            x,
            image_channels=self.image_input_channels,
            condition_dim=self.fd_condition_dim,
        )
        features = self.unet.forward_conditioned(image, condition)
        return self._apply_nonlinearity(torch.cat([mt_head(features) for mt_head in self.mt_heads], dim=1))

    model.forward = MethodType(_forward_conditioned, model)
    return model


def _patch_double_unet(model: DoubleUNet, *, condition_dim: int, hidden_dim: int) -> DoubleUNet:
    model.image_input_channels = int(model.nch_in)
    model.fd_condition_dim = int(condition_dim)
    model.fd_condition_hidden_dim = int(hidden_dim)
    model.fd_condition_strategy = "film_tile_vector_v1"
    _patch_unet_with_film(model.unet_shared, condition_dim=condition_dim, hidden_dim=hidden_dim)
    _patch_unet_with_film(model.unet_union, condition_dim=condition_dim, hidden_dim=hidden_dim)

    def _forward_core_conditioned(self: DoubleUNet, x: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        image, condition = _unpack_conditioned_input(
            x,
            image_channels=self.image_input_channels,
            condition_dim=self.fd_condition_dim,
        )
        if self.nch_in > 1:
            outputs = [self.unet_shared.forward_conditioned(image[:, [idx]], condition) for idx in range(self.nch_in)]
            shared = torch.cat(outputs, dim=1)
        elif self.nch_in == 1:
            shared = self.unet_shared.forward_conditioned(image, condition)
        else:
            raise ValueError(f"nch_in={self.nch_in} is invalid, should be positive integers")
        return self.unet_union.forward_conditioned(shared, condition)

    def _forward_conditioned(self: DoubleUNet, x: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        features = self._forward_core(x)
        return self._apply_nonlinearity(torch.cat([mt_head(features) for mt_head in self.mt_heads], dim=1))

    model._forward_core = MethodType(_forward_core_conditioned, model)
    model.forward = MethodType(_forward_conditioned, model)
    return model


def patch_model_for_fd_film_conditioning(
    model: nn.Module,
    *,
    condition_dim: int,
    hidden_dim: int = 64,
) -> nn.Module:
    if int(condition_dim) <= 0:
        raise ValueError("condition_dim must be positive.")
    if isinstance(model, DoubleUNet):
        return _patch_double_unet(model, condition_dim=int(condition_dim), hidden_dim=int(hidden_dim))
    if isinstance(model, SingleUNet):
        return _patch_single_unet(model, condition_dim=int(condition_dim), hidden_dim=int(hidden_dim))
    raise TypeError(f"Unsupported model type for fd film conditioning: {type(model)!r}")
