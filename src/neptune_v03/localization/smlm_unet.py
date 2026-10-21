from __future__ import annotations

import numbers
import warnings
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import TracerWarning

from neptune_v03.localization.smlm_output import SMLMOutputChannels


def get_activation(activation: str | nn.Module | None, kwargs: dict | None = None):
    kwargs = {} if kwargs is None else dict(kwargs)
    if activation is None:
        return None
    if isinstance(activation, str):
        allowed = {
            "ReLU",
            "LeakyReLU",
            "ELU",
            "PReLU",
            "SELU",
            "GELU",
            "SiLU",
            "Mish",
            "Tanh",
            "Sigmoid",
            "Identity",
        }
        if activation not in allowed:
            raise ValueError(f"Unknown activation: {activation}")
        if activation == "LeakyReLU" and "negative_slope" not in kwargs:
            kwargs["negative_slope"] = 0.01
        if activation in {"ReLU", "LeakyReLU", "ELU", "SELU", "SiLU", "Mish"}:
            return getattr(nn, activation)(inplace=True, **kwargs)
        return getattr(nn, activation)(**kwargs)
    if isinstance(activation, nn.Module):
        return activation
    raise TypeError("activation must be str, nn.Module, or None")


class ReparamLargeKernelDepthwise(nn.Module):
    """RepLKNet-style train-time small/large depthwise branches.

    Both branches include BatchNorm and are fused into one depthwise large
    kernel for deployment. The layer is intentionally depthwise-only so the
    surrounding pointwise projection remains unchanged.
    """

    def __init__(self, *, channels: int, large_kernel_size: int, small_kernel_size: int = 5) -> None:
        super().__init__()
        large = int(large_kernel_size)
        small = int(small_kernel_size)
        if large < 1 or large % 2 == 0 or small < 1 or small % 2 == 0 or small > large:
            raise ValueError("large/small kernel sizes must be positive odd values with small <= large")
        self.channels = int(channels)
        self.large_kernel_size = large
        self.small_kernel_size = small
        self.large_conv = nn.Conv2d(self.channels, self.channels, large, padding=large // 2, groups=self.channels, bias=False)
        self.large_bn = nn.BatchNorm2d(self.channels)
        self.small_conv: nn.Conv2d | None = nn.Conv2d(
            self.channels, self.channels, small, padding=small // 2, groups=self.channels, bias=False
        )
        self.small_bn: nn.BatchNorm2d | None = nn.BatchNorm2d(self.channels)
        self.deploy_conv: nn.Conv2d | None = None

    @staticmethod
    def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = conv.weight * scale.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * scale
        return weight, bias

    def switch_to_deploy(self) -> None:
        if self.deploy_conv is not None:
            return
        assert self.small_conv is not None and self.small_bn is not None
        large_weight, large_bias = self._fuse(self.large_conv, self.large_bn)
        small_weight, small_bias = self._fuse(self.small_conv, self.small_bn)
        pad = (self.large_kernel_size - self.small_kernel_size) // 2
        fused_weight = large_weight.clone()
        fused_weight[:, :, pad : pad + self.small_kernel_size, pad : pad + self.small_kernel_size] += small_weight
        fused_bias = large_bias + small_bias
        deploy = nn.Conv2d(
            self.channels,
            self.channels,
            self.large_kernel_size,
            padding=self.large_kernel_size // 2,
            groups=self.channels,
            bias=True,
        ).to(device=fused_weight.device, dtype=fused_weight.dtype)
        deploy.weight.data.copy_(fused_weight)
        deploy.bias.data.copy_(fused_bias)
        self.deploy_conv = deploy
        self.large_conv = None  # type: ignore[assignment]
        self.large_bn = None  # type: ignore[assignment]
        self.small_conv = None
        self.small_bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy_conv is not None:
            return self.deploy_conv(x)
        assert self.small_conv is not None and self.small_bn is not None
        return self.large_bn(self.large_conv(x)) + self.small_bn(self.small_conv(x))


class UniRepLKDepthwise(nn.Module):
    """UniRepLK-style parallel dilated depthwise branches with K=11 fusion."""

    _default_branch_specs = ((5, 1), (5, 2), (3, 3), (3, 4), (3, 5))
    _liteloc_branch_specs = ((3, 1), (3, 2), (3, 4), (3, 8), (3, 16))

    def __init__(self, *, channels: int, kernel_size: int = 11, branch_specs: tuple[tuple[int, int], ...] | None = None) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = int(channels)
        self.kernel_size = kernel_size
        self.branch_specs = tuple(branch_specs or (self._liteloc_branch_specs if kernel_size >= 33 else self._default_branch_specs))
        if not self.branch_specs or any(
            kernel < 1 or kernel % 2 == 0 or dilation < 1 or (kernel - 1) * dilation + 1 > kernel_size
            for kernel, dilation in self.branch_specs
        ):
            raise ValueError("all branch supports must be odd and fit inside kernel_size")
        self.large_conv = nn.Conv2d(self.channels, self.channels, kernel_size, padding=kernel_size // 2, groups=self.channels, bias=False)
        self.large_bn = nn.BatchNorm2d(self.channels)
        self.branches = nn.ModuleList(
            nn.Conv2d(self.channels, self.channels, kernel, padding=(kernel // 2) * dilation,
                      dilation=dilation, groups=self.channels, bias=False)
            for kernel, dilation in self.branch_specs
        )
        self.branch_norms = nn.ModuleList(nn.BatchNorm2d(self.channels) for _ in self.branch_specs)
        self.deploy_conv: nn.Conv2d | None = None

    @staticmethod
    def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return conv.weight * scale.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * scale

    def _embed(self, weight: torch.Tensor, branch_kernel: int, dilation: int) -> torch.Tensor:
        dense = weight.new_zeros((self.channels, 1, self.kernel_size, self.kernel_size))
        support = (branch_kernel - 1) * dilation + 1
        start = (self.kernel_size - support) // 2
        for row in range(branch_kernel):
            for col in range(branch_kernel):
                dense[:, :, start + row * dilation, start + col * dilation] = weight[:, :, row, col]
        return dense

    def switch_to_deploy(self) -> None:
        if self.deploy_conv is not None:
            return
        fused_weight, fused_bias = self._fuse(self.large_conv, self.large_bn)
        for conv, norm, (kernel, dilation) in zip(self.branches, self.branch_norms, self.branch_specs):
            weight, bias = self._fuse(conv, norm)
            fused_weight = fused_weight + self._embed(weight, kernel, dilation)
            fused_bias = fused_bias + bias
        deploy = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.kernel_size // 2,
                           groups=self.channels, bias=True).to(device=fused_weight.device, dtype=fused_weight.dtype)
        deploy.weight.data.copy_(fused_weight)
        deploy.bias.data.copy_(fused_bias)
        self.deploy_conv = deploy
        self.large_conv = None  # type: ignore[assignment]
        self.large_bn = None  # type: ignore[assignment]
        self.branches = nn.ModuleList()
        self.branch_norms = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy_conv is not None:
            return self.deploy_conv(x)
        output = self.large_bn(self.large_conv(x))
        for conv, norm in zip(self.branches, self.branch_norms):
            output = output + norm(conv(x))
        return output


class WaveletDepthwiseConv(nn.Module):
    """Multi-level Haar wavelet convolution with dense local subband filters."""

    def __init__(self, *, channels: int, levels: int, kernel_size: int = 5) -> None:
        super().__init__()
        if int(levels) < 1:
            raise ValueError("levels must be positive")
        if int(kernel_size) < 1 or int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = int(channels)
        self.levels = int(levels)
        self.kernel_size = int(kernel_size)
        haar = torch.tensor(
            [[[0.5, 0.5], [0.5, 0.5]], [[-0.5, -0.5], [0.5, 0.5]],
             [[-0.5, 0.5], [-0.5, 0.5]], [[0.5, -0.5], [-0.5, 0.5]]],
            dtype=torch.float32,
        )
        self.register_buffer("haar_filters", haar.unsqueeze(1), persistent=False)
        self.subband_convs = nn.ModuleList(
            nn.Conv2d(
                4 * self.channels,
                4 * self.channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                groups=4 * self.channels,
                bias=False,
            )
            for _ in range(self.levels)
        )
        self.subband_norms = nn.ModuleList(nn.BatchNorm2d(4 * self.channels) for _ in range(self.levels))

    def _filters(self, x: torch.Tensor) -> torch.Tensor:
        return self.haar_filters.to(dtype=x.dtype).repeat(self.channels, 1, 1, 1)

    def _dwt(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self._filters(x), stride=2, groups=self.channels)

    def _idwt(self, coeffs: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(coeffs, self._filters(coeffs), stride=2, groups=self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        factor = 2**self.levels
        if height % factor or width % factor:
            raise ValueError(f"WTConv input must be divisible by {factor}, got {(height, width)}")
        encoded: list[torch.Tensor] = []
        low = x
        for conv, norm in zip(self.subband_convs, self.subband_norms):
            coeffs = norm(conv(self._dwt(low))).reshape(low.shape[0], self.channels, 4, low.shape[-2] // 2, low.shape[-1] // 2)
            encoded.append(coeffs)
            low = coeffs[:, :, 0]
        for coeffs in reversed(encoded):
            low = self._idwt(torch.cat((low.unsqueeze(2), coeffs[:, :, 1:]), dim=2).flatten(1, 2))
        return low


class WTConvLocalResidualContext(nn.Module):
    """Stride-2 context block that fuses dense 5x5 local and wavelet paths."""

    def __init__(self, *, channels: int, levels: int = 2, local_kernel_size: int = 5) -> None:
        super().__init__()
        if int(local_kernel_size) < 1 or int(local_kernel_size) % 2 == 0:
            raise ValueError("local_kernel_size must be a positive odd integer")
        self.levels = int(levels)
        self.local_kernel_size = int(local_kernel_size)
        self.local = nn.Sequential(
            nn.Conv2d(int(channels), int(channels), self.local_kernel_size, padding=self.local_kernel_size // 2, groups=int(channels), bias=False),
            nn.BatchNorm2d(int(channels)),
        )
        self.wavelet = WaveletDepthwiseConv(channels=int(channels), levels=self.levels, kernel_size=self.local_kernel_size)
        self.wavelet_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.local(x) + self.wavelet_scale * self.wavelet(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        *,
        level: int,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        norm_start_level: int | None = 0,
        norm_groups: int = 0,
        activation: str | nn.Module = "ReLU",
        activation_kwargs: dict | None = None,
        dropout_start_level: int | None = 3,
        p_dropout: float = 0.1,
        depthwise: bool = True,
        kernel_size: int = 3,
        small_kernel_size: int = 5,
        unireplk: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = self.create_conv(int(in_channels), int(out_channels), int(padding), bool(depthwise), int(kernel_size), int(small_kernel_size), bool(unireplk))
        self.conv2 = self.create_conv(int(out_channels), int(out_channels), int(padding), bool(depthwise), int(kernel_size), int(small_kernel_size), bool(unireplk))

        if norm_start_level is not None and int(level) >= int(norm_start_level) and int(norm_groups) >= 0:
            if int(norm_groups) > 0:
                if int(out_channels) % int(norm_groups) != 0:
                    raise ValueError("norm_groups must divide out_channels")
                self.norm1 = nn.GroupNorm(int(norm_groups), int(out_channels))
                self.norm2 = nn.GroupNorm(int(norm_groups), int(out_channels))
            else:
                self.norm1 = nn.BatchNorm2d(int(out_channels))
                self.norm2 = nn.BatchNorm2d(int(out_channels))
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.activation1 = get_activation(activation, activation_kwargs)
        self.activation2 = get_activation(activation, activation_kwargs)
        if dropout_start_level is not None and int(level) >= int(dropout_start_level) and float(p_dropout) > 0:
            self.dropout = nn.Dropout2d(float(p_dropout))
        else:
            self.dropout = nn.Identity()

    @staticmethod
    def create_conv(
        in_channels: int,
        out_channels: int,
        padding: int,
        depthwise: bool = True,
        kernel_size: int = 3,
        small_kernel_size: int = 5,
        unireplk: bool = False,
    ) -> nn.Module:
        if depthwise:
            if unireplk:
                spatial = UniRepLKDepthwise(channels=int(in_channels), kernel_size=int(kernel_size))
                return nn.Sequential(spatial, nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1))
            spatial = (
                ReparamLargeKernelDepthwise(channels=int(in_channels), large_kernel_size=int(kernel_size), small_kernel_size=int(small_kernel_size))
                if int(kernel_size) > 3
                else nn.Conv2d(
                    int(in_channels), int(in_channels), kernel_size=int(kernel_size),
                    padding=int(padding), groups=int(in_channels)
                )
            )
            return nn.Sequential(
                spatial,
                nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1),
            )
        return nn.Conv2d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), padding=int(padding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        if self.activation1 is not None:
            x = self.activation1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.activation2 is not None:
            x = self.activation2(x)
        return self.dropout(x)


class Pooler(nn.Module):
    pool_types = {"MaxPool", "StrideConv"}

    def __init__(self, *, n_channels: int, pool_type: str = "StrideConv") -> None:
        super().__init__()
        if pool_type not in self.pool_types:
            raise ValueError(f"{pool_type} is not supported pooling method")
        if pool_type == "StrideConv":
            self.pooler = nn.Conv2d(int(n_channels), int(n_channels), kernel_size=2, stride=2)
        else:
            self.pooler = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooler(x)


class Upsample(nn.Module):
    def __init__(
        self,
        *,
        scale_factor: int,
        nch_in: int,
        nch_out: int,
        mode: str = "bilinear",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        align = align_corners if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
        self.scale_factor = int(scale_factor)
        self.mode = str(mode)
        self.align_corners = align
        self.conv = None if int(nch_in) == int(nch_out) else nn.Conv2d(int(nch_in), int(nch_out), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x if self.conv is None else self.conv(x)


class UNet2d(nn.Module):
    def __init__(
        self,
        *,
        nch_in: int,
        nch_out: int | None,
        depth: int,
        nfeatures_init: int = 64,
        pad_convs: bool = True,
        norm_start_level: int = 0,
        norm_groups: int = 0,
        activation: str | nn.Module = "ReLU",
        dropout_start_level: int | None = None,
        p_dropout: float = 0.1,
        pool_mode: str = "StrideConv",
        upsample_mode: str = "bilinear",
        out_activation: str | nn.Module = "ReLU",
        depthwise: bool = True,
        base_kernel_size: int = 3,
        bottleneck_small_kernel_size: int = 5,
        wtconv_context_levels: int = 0,
        wtconv_context_kernel_size: int = 5,
        unireplk_kernel_size: int = 0,
    ) -> None:
        super().__init__()
        self.nch_in = int(nch_in)
        self.nch_out = None if nch_out is None else int(nch_out)
        self.depth = int(depth)
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        self.padding = 1 if pad_convs else 0
        self.norm_start_level = int(norm_start_level)
        self.norm_groups = int(norm_groups)
        self.activation = activation
        self.dropout_start_level = dropout_start_level
        self.p_dropout = float(p_dropout)
        self.pool_mode = str(pool_mode)
        self.upsample_mode = str(upsample_mode)
        self.depthwise = bool(depthwise)
        self.base_kernel_size = int(base_kernel_size)
        self.bottleneck_small_kernel_size = int(bottleneck_small_kernel_size)
        if self.base_kernel_size < 1 or self.base_kernel_size % 2 == 0:
            raise ValueError("base_kernel_size must be a positive odd integer")
        self.wtconv_context_levels = int(wtconv_context_levels)
        self.wtconv_context_kernel_size = int(wtconv_context_kernel_size)
        self.unireplk_kernel_size = int(unireplk_kernel_size)
        if self.wtconv_context_levels < 0:
            raise ValueError("wtconv_context_levels must be non-negative")

        self.encoder = nn.ModuleList()
        self.pooler = nn.ModuleList()
        for level in range(self.depth):
            in_channels = self.nch_in if level == 0 else int(nfeatures_init) * 2 ** (level - 1)
            out_channels = int(nfeatures_init) * 2**level
            self.encoder.append(self._conv_block(level, in_channels, out_channels))
            self.pooler.append(Pooler(n_channels=out_channels, pool_type=self.pool_mode))

        stride2_channels = int(nfeatures_init) * 2
        self.stride2_context = (
            WTConvLocalResidualContext(
                channels=stride2_channels,
                levels=self.wtconv_context_levels,
                local_kernel_size=self.wtconv_context_kernel_size,
            )
            if self.depth >= 2 and self.wtconv_context_levels > 0
            else None
        )

        base_in = int(nfeatures_init) * 2 ** (self.depth - 1)
        base_out = int(nfeatures_init) * 2**self.depth
        self.base = self._conv_block(
            self.depth,
            base_in,
            base_out,
            kernel_size=(self.unireplk_kernel_size if self.unireplk_kernel_size > 0 else self.base_kernel_size),
            small_kernel_size=self.bottleneck_small_kernel_size,
            unireplk=self.unireplk_kernel_size > 0,
        )

        self.upsampler = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for level in range(self.depth, 0, -1):
            in_channels = int(nfeatures_init) * 2**level
            out_channels = int(nfeatures_init) * 2 ** (level - 1)
            self.upsampler.append(
                Upsample(scale_factor=2, nch_in=in_channels, nch_out=out_channels, mode=self.upsample_mode)
            )
            self.decoder.append(self._conv_block(level, in_channels, out_channels))

        self.out_conv = None if self.nch_out is None else nn.Conv2d(int(nfeatures_init), self.nch_out, kernel_size=1)
        self.out_activation = get_activation(out_activation)

    def _conv_block(
        self,
        level: int,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        small_kernel_size: int = 5,
        unireplk: bool = False,
    ) -> ConvBlock:
        return ConvBlock(
            level=int(level),
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            padding=(int(kernel_size) // 2 if self.padding else 0),
            norm_start_level=self.norm_start_level,
            norm_groups=self.norm_groups,
            activation=self.activation,
            dropout_start_level=self.dropout_start_level,
            p_dropout=self.p_dropout,
            depthwise=self.depthwise,
            kernel_size=int(kernel_size),
            small_kernel_size=int(small_kernel_size),
            unireplk=bool(unireplk),
        )

    @staticmethod
    def _crop_tensor(input_: torch.Tensor, shape_to_crop: torch.Size) -> torch.Tensor:
        input_shape = input_.shape
        shape_diff = tuple((ish - csh) // 2 for ish, csh in zip(input_shape, shape_to_crop))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=TracerWarning)
            if all(sd == 0 for sd in shape_diff):
                return input_
        crop = tuple(slice(sd, sh - sd) for sd, sh in zip(shape_diff, input_shape))
        return input_[crop]

    def _crop_and_concat(self, from_decoder: torch.Tensor, from_encoder: torch.Tensor) -> torch.Tensor:
        return torch.cat((self._crop_tensor(from_encoder, from_decoder.shape), from_decoder), dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            if level == 1 and self.stride2_context is not None:
                x = self.stride2_context(x)
            encoder_out.append(x)
            x = self.pooler[level](x)
        x = self.base(x)
        for level, from_encoder in enumerate(reversed(encoder_out)):
            x = self.upsampler[level](x)
            x = self.decoder[level](self._crop_and_concat(x, from_encoder))
        if self.out_conv is not None:
            x = self.out_conv(x)
            if self.out_activation is not None:
                x = self.out_activation(x)
        return x


class MultiHeads(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        last_kernelsz: int,
        pad_convs: bool = True,
        norm_groups: int = 0,
        activation: str | nn.Module = "ReLU",
    ) -> None:
        super().__init__()
        padding = 1 if pad_convs else 0
        if int(norm_groups) > 0 and int(in_channels) % int(norm_groups) != 0:
            raise ValueError("in_channels must be divisible by norm_groups")
        self.in_conv = nn.Conv2d(int(in_channels), int(in_channels), kernel_size=3, padding=padding)
        if int(norm_groups) > 0:
            self.norm = nn.GroupNorm(int(norm_groups), int(in_channels))
        elif int(norm_groups) == 0:
            self.norm = nn.BatchNorm2d(int(in_channels))
        else:
            self.norm = nn.Identity()
        self.activation = get_activation(activation)
        self.out_conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=int(last_kernelsz), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.out_conv(x)


class SMLMUNet(nn.Module):
    out_channels_heads = (1, 4, 4, 1)
    sigma_eps_default = 0.001

    def __init__(
        self,
        *,
        nch_in: int,
        nfeatures_init: int,
        nfeatures_inter: int | None = None,
        norm_start_level: int = 0,
        norm_groups: int = 0,
        activation: str | nn.Module = "ELU",
        dropout_start_level: int | None = None,
        p_dropout: float = -0.1,
        pool_mode: str = "StrideConv",
        upsample_mode: str = "bilinear",
        inter_activation: str | nn.Module = "ELU",
        norm_head_groups: int = 0,
        final_activation: str | nn.Module = "ELU",
        disabled_attr: int | tuple[int, ...] | list[int] | None = None,
        kaiming_normal: bool = True,
        z_mu_activation: str | None = None,
    ) -> None:
        super().__init__()
        self.register_parameter("sigma_eps", nn.Parameter(torch.tensor([self.sigma_eps_default]), requires_grad=False))
        self.nch_in = int(nch_in)
        self.nfeatures_init = int(nfeatures_init)
        self.nfeatures_inter = None if nfeatures_inter is None else int(nfeatures_inter)
        self.norm_start_level = int(norm_start_level)
        self.norm_groups = int(norm_groups)
        self.activation = activation
        self.dropout_start_level = dropout_start_level
        self.p_dropout = float(p_dropout)
        self.pool_mode = str(pool_mode)
        self.upsample_mode = str(upsample_mode)
        self.inter_activation = inter_activation
        self.norm_head_groups = int(norm_head_groups)
        self.final_activation = final_activation
        self.disabled_attr_ix = [disabled_attr] if isinstance(disabled_attr, numbers.Integral) else disabled_attr
        self.kaiming_normal = bool(kaiming_normal)

        self.ch_ix_sigmoid = [0, 1, 4, 5, 6, 7, 8, 9]
        self.ch_ix_tanh = [2, 3]
        self._configure_z_mu_activation(z_mu_activation)

        head_in_channels = self.nfeatures_init if self.nfeatures_inter is None else self.nfeatures_inter
        self.mt_heads = nn.ModuleList(
            [
                MultiHeads(
                    in_channels=head_in_channels,
                    out_channels=out_channels,
                    last_kernelsz=1,
                    pad_convs=True,
                    norm_groups=self.norm_head_groups,
                    activation=self.final_activation,
                )
                for out_channels in self.out_channels_heads
            ]
        )

    def _configure_z_mu_activation(self, z_mu_activation: str | None) -> None:
        z_mu_act = ""
        if z_mu_activation is not None:
            z_mu_act = str(z_mu_activation).strip().lower()
        if not z_mu_act:
            z_mu_act = "tanh"
        if z_mu_act == "tanh":
            if SMLMOutputChannels.z_mu in self.ch_ix_sigmoid:
                self.ch_ix_sigmoid.remove(SMLMOutputChannels.z_mu)
            if SMLMOutputChannels.z_mu not in self.ch_ix_tanh:
                self.ch_ix_tanh.append(SMLMOutputChannels.z_mu)
        elif z_mu_act == "sigmoid":
            if SMLMOutputChannels.z_mu in self.ch_ix_tanh:
                self.ch_ix_tanh.remove(SMLMOutputChannels.z_mu)
            if SMLMOutputChannels.z_mu not in self.ch_ix_sigmoid:
                self.ch_ix_sigmoid.append(SMLMOutputChannels.z_mu)
        else:
            raise ValueError("z_mu_activation must be 'tanh' or 'sigmoid'")

    @staticmethod
    def weight_init(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def _apply_nonlinearity(self, o: torch.Tensor) -> torch.Tensor:
        o = o.clone()
        o[:, [SMLMOutputChannels.p]] = torch.clamp(o[:, [SMLMOutputChannels.p]], min=-8.0, max=8.0)
        o[:, self.ch_ix_sigmoid] = torch.sigmoid(o[:, self.ch_ix_sigmoid])
        o[:, self.ch_ix_tanh] = torch.tanh(o[:, self.ch_ix_tanh])
        o[:, SMLMOutputChannels.pxyz_sig] = 3.0 * o[:, SMLMOutputChannels.pxyz_sig] + self.sigma_eps
        if self.disabled_attr_ix is not None:
            for ix in self.disabled_attr_ix:
                o[:, 1 + int(ix)] = 0.0
                o[:, 5 + int(ix)] = 0.1
        return o


class DoubleUNet(SMLMUNet):
    def __init__(
        self,
        *,
        nch_in: int,
        depth_shared: int,
        depth_union: int,
        nfeatures_init: int,
        nfeatures_inter: int | None = None,
        norm_start_level: int = 0,
        norm_groups: int = 0,
        activation: Union[str, nn.Module] = "ELU",
        dropout_start_level: int | None = None,
        p_dropout: float = 0.1,
        pool_mode: str = "StrideConv",
        upsample_mode: str = "bilinear",
        inter_activation: Union[str, nn.Module] = "ELU",
        norm_head_groups: int = 0,
        final_activation: Union[str, nn.Module] = "ELU",
        disabled_attr: int | tuple[int, ...] | list[int] | None = None,
        kaiming_normal: bool = True,
        depthwise: bool = True,
        bottleneck_kernel_size: int = 3,
        bottleneck_small_kernel_size: int = 5,
        wtconv_context_levels: int = 0,
        wtconv_context_kernel_size: int = 5,
        unireplk_kernel_size: int = 0,
        z_mu_activation: str | None = None,
    ) -> None:
        super().__init__(
            nch_in=nch_in,
            nfeatures_init=nfeatures_init,
            nfeatures_inter=nfeatures_inter,
            norm_start_level=norm_start_level,
            norm_groups=norm_groups,
            activation=activation,
            dropout_start_level=dropout_start_level,
            p_dropout=p_dropout,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            inter_activation=inter_activation,
            norm_head_groups=norm_head_groups,
            final_activation=final_activation,
            disabled_attr=disabled_attr,
            kaiming_normal=kaiming_normal,
            z_mu_activation=z_mu_activation,
        )
        self.depth_shared = int(depth_shared)
        self.depth_union = int(depth_union)
        self.depthwise = bool(depthwise)
        self.bottleneck_kernel_size = int(bottleneck_kernel_size)
        self.bottleneck_small_kernel_size = int(bottleneck_small_kernel_size)
        self.wtconv_context_levels = int(wtconv_context_levels)
        self.wtconv_context_kernel_size = int(wtconv_context_kernel_size)
        self.unireplk_kernel_size = int(unireplk_kernel_size)
        shared_out = self.nfeatures_init if self.nfeatures_inter is None else self.nfeatures_inter

        self.unet_shared = UNet2d(
            nch_in=1,
            nch_out=self.nfeatures_inter,
            depth=self.depth_shared,
            nfeatures_init=self.nfeatures_init,
            pad_convs=True,
            norm_start_level=self.norm_start_level,
            norm_groups=self.norm_groups,
            activation=self.activation,
            dropout_start_level=self.dropout_start_level,
            p_dropout=self.p_dropout,
            pool_mode=self.pool_mode,
            upsample_mode=self.upsample_mode,
            out_activation=self.inter_activation,
            depthwise=self.depthwise,
            base_kernel_size=self.bottleneck_kernel_size,
            bottleneck_small_kernel_size=self.bottleneck_small_kernel_size,
            unireplk_kernel_size=self.unireplk_kernel_size,
            wtconv_context_levels=self.wtconv_context_levels,
            wtconv_context_kernel_size=self.wtconv_context_kernel_size,
        )
        self.unet_union = UNet2d(
            nch_in=self.nch_in * shared_out,
            nch_out=self.nfeatures_inter,
            depth=self.depth_union,
            nfeatures_init=self.nfeatures_init,
            pad_convs=True,
            norm_start_level=self.norm_start_level,
            norm_groups=self.norm_groups,
            activation=self.activation,
            dropout_start_level=self.dropout_start_level,
            p_dropout=self.p_dropout,
            pool_mode=self.pool_mode,
            upsample_mode=self.upsample_mode,
            out_activation=self.inter_activation,
            depthwise=self.depthwise,
            base_kernel_size=self.bottleneck_kernel_size,
            bottleneck_small_kernel_size=self.bottleneck_small_kernel_size,
            unireplk_kernel_size=self.unireplk_kernel_size,
            wtconv_context_levels=self.wtconv_context_levels,
            wtconv_context_kernel_size=self.wtconv_context_kernel_size,
        )

        if self.kaiming_normal:
            self.apply(self.weight_init)
            nn.init.kaiming_normal_(self.mt_heads[0].in_conv.weight, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_normal_(self.mt_heads[0].out_conv.weight, mode="fan_in", nonlinearity="linear")
            nn.init.constant_(self.mt_heads[0].out_conv.bias, -6.0)

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.nch_in > 1:
            outputs = [self.unet_shared(x[:, [ix]]) for ix in range(self.nch_in)]
            features = torch.cat(outputs, dim=1)
        elif self.nch_in == 1:
            features = self.unet_shared(x)
        else:
            raise ValueError(f"nch_in={self.nch_in} is invalid")
        return self.unet_union(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_core(x)
        output = torch.cat([mt_head(features) for mt_head in self.mt_heads], dim=1)
        return self._apply_nonlinearity(output)
