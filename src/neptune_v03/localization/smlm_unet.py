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
    ) -> None:
        super().__init__()
        self.conv1 = self.create_conv(int(in_channels), int(out_channels), int(padding), bool(depthwise))
        self.conv2 = self.create_conv(int(out_channels), int(out_channels), int(padding), bool(depthwise))

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
    def create_conv(in_channels: int, out_channels: int, padding: int, depthwise: bool = True) -> nn.Module:
        if depthwise:
            return nn.Sequential(
                nn.Conv2d(int(in_channels), int(in_channels), kernel_size=3, padding=int(padding), groups=int(in_channels)),
                nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1),
            )
        return nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=int(padding))

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

        self.encoder = nn.ModuleList()
        self.pooler = nn.ModuleList()
        for level in range(self.depth):
            in_channels = self.nch_in if level == 0 else int(nfeatures_init) * 2 ** (level - 1)
            out_channels = int(nfeatures_init) * 2**level
            self.encoder.append(self._conv_block(level, in_channels, out_channels))
            self.pooler.append(Pooler(n_channels=out_channels, pool_type=self.pool_mode))

        base_in = int(nfeatures_init) * 2 ** (self.depth - 1)
        base_out = int(nfeatures_init) * 2**self.depth
        self.base = self._conv_block(self.depth, base_in, base_out)

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

    def _conv_block(self, level: int, in_channels: int, out_channels: int) -> ConvBlock:
        return ConvBlock(
            level=int(level),
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            padding=self.padding,
            norm_start_level=self.norm_start_level,
            norm_groups=self.norm_groups,
            activation=self.activation,
            dropout_start_level=self.dropout_start_level,
            p_dropout=self.p_dropout,
            depthwise=self.depthwise,
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
