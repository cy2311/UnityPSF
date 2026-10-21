from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn

from neptune.smlm_v2a.models.smlm_model import DoubleUNet, SingleUNet


def _expanded_conv2d(conv: nn.Conv2d, *, in_channels: int, out_channels: int | None = None, groups: int | None = None) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=int(in_channels),
        out_channels=conv.out_channels if out_channels is None else int(out_channels),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups if groups is None else int(groups),
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )


def _expand_depthwise_pointwise_stem(stem: nn.Sequential, extra_in_channels: int) -> nn.Sequential:
    if int(extra_in_channels) <= 0:
        return stem
    if len(stem) != 2 or not isinstance(stem[0], nn.Conv2d) or not isinstance(stem[1], nn.Conv2d):
        raise TypeError("Expected a 2-layer depthwise/pointwise Conv2d stem.")
    depthwise_old = stem[0]
    pointwise_old = stem[1]
    new_in_channels = int(depthwise_old.in_channels) + int(extra_in_channels)
    depthwise_new = _expanded_conv2d(
        depthwise_old,
        in_channels=new_in_channels,
        out_channels=new_in_channels,
        groups=new_in_channels,
    )
    pointwise_new = _expanded_conv2d(pointwise_old, in_channels=new_in_channels)
    with torch.no_grad():
        depthwise_new.weight.zero_()
        depthwise_new.weight[: depthwise_old.out_channels].copy_(depthwise_old.weight)
        if depthwise_old.bias is not None:
            depthwise_new.bias.zero_()
            depthwise_new.bias[: depthwise_old.out_channels].copy_(depthwise_old.bias)
        pointwise_new.weight.zero_()
        pointwise_new.weight[:, : pointwise_old.in_channels].copy_(pointwise_old.weight)
        mean_projection = pointwise_old.weight.mean(dim=1, keepdim=True)
        pointwise_new.weight[:, pointwise_old.in_channels :].copy_(
            mean_projection.expand(-1, int(extra_in_channels), -1, -1)
        )
        if pointwise_old.bias is not None:
            pointwise_new.bias.copy_(pointwise_old.bias)
    return nn.Sequential(depthwise_new, pointwise_new)


def _expand_plain_conv_input(conv: nn.Conv2d, extra_in_channels: int) -> nn.Conv2d:
    if int(extra_in_channels) <= 0:
        return conv
    new_conv = _expanded_conv2d(conv, in_channels=int(conv.in_channels) + int(extra_in_channels))
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, : conv.in_channels].copy_(conv.weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _patch_unet_stem_input(unet: nn.Module, extra_in_channels: int) -> None:
    stem = unet.encoder[0].conv1
    if isinstance(stem, nn.Sequential):
        unet.encoder[0].conv1 = _expand_depthwise_pointwise_stem(stem, int(extra_in_channels))
    elif isinstance(stem, nn.Conv2d):
        unet.encoder[0].conv1 = _expand_plain_conv_input(stem, int(extra_in_channels))
    else:
        raise TypeError(f"Unsupported UNet stem type: {type(stem)!r}")
    unet.nch_in = int(unet.nch_in) + int(extra_in_channels)


def _patch_single_unet(model: SingleUNet, condition_channels: int) -> SingleUNet:
    image_channels = int(model.nch_in)
    _patch_unet_stem_input(model.unet, int(condition_channels))
    model.image_input_channels = image_channels
    model.zmap_condition_channels = int(condition_channels)
    model.expected_input_channels = image_channels + int(condition_channels)

    def _forward_with_optional_zmap(self: SingleUNet, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input with shape (N,C,H,W), got {tuple(x.shape)}")
        if x.size(1) == self.image_input_channels:
            cond = torch.zeros((x.size(0), self.zmap_condition_channels, x.size(2), x.size(3)), dtype=x.dtype, device=x.device)
            x_in = torch.cat([x, cond], dim=1)
        elif x.size(1) == self.expected_input_channels:
            x_in = x
        else:
            raise ValueError(f"Expected {self.image_input_channels} or {self.expected_input_channels} channels, got {x.size(1)}.")
        features = self.unet(x_in)
        return self._apply_nonlinearity(torch.cat([mt_head(features) for mt_head in self.mt_heads], dim=1))

    model.forward = MethodType(_forward_with_optional_zmap, model)
    return model


def _patch_double_unet(model: DoubleUNet, condition_channels: int) -> DoubleUNet:
    image_channels = int(model.nch_in)
    _patch_unet_stem_input(model.unet_shared, int(condition_channels))
    model.image_input_channels = image_channels
    model.zmap_condition_channels = int(condition_channels)
    model.expected_input_channels = image_channels + int(condition_channels)

    def _forward_core_with_optional_zmap(self: DoubleUNet, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input with shape (N,C,H,W), got {tuple(x.shape)}")
        if x.size(1) == self.image_input_channels:
            cond = torch.zeros((x.size(0), self.zmap_condition_channels, x.size(2), x.size(3)), dtype=x.dtype, device=x.device)
            image = x
        elif x.size(1) == self.expected_input_channels:
            image = x[:, : self.image_input_channels]
            cond = x[:, self.image_input_channels :]
        else:
            raise ValueError(f"Expected {self.image_input_channels} or {self.expected_input_channels} channels, got {x.size(1)}.")
        outputs = [self.unet_shared(torch.cat([image[:, [idx]], cond], dim=1)) for idx in range(self.image_input_channels)]
        return self.unet_union(torch.cat(outputs, dim=1))

    def _forward_with_optional_zmap(self: DoubleUNet, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_core(x)
        return self._apply_nonlinearity(torch.cat([mt_head(features) for mt_head in self.mt_heads], dim=1))

    model._forward_core = MethodType(_forward_core_with_optional_zmap, model)
    model.forward = MethodType(_forward_with_optional_zmap, model)
    return model


def patch_model_for_zmap_conditioning(model: nn.Module, *, condition_channels: int) -> nn.Module:
    if int(condition_channels) <= 0:
        raise ValueError("condition_channels must be positive.")
    if isinstance(model, DoubleUNet):
        return _patch_double_unet(model, int(condition_channels))
    if isinstance(model, SingleUNet):
        return _patch_single_unet(model, int(condition_channels))
    raise TypeError(f"Unsupported model type for zmap conditioning: {type(model)!r}")
