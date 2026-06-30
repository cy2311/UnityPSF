from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalizationModelOutput:
    detection_logits: torch.Tensor
    xy_offset: torch.Tensor
    z: torch.Tensor
    photons: torch.Tensor

    @property
    def probability(self) -> torch.Tensor:
        return torch.sigmoid(self.detection_logits)


class SimpleLocalizationModel(torch.nn.Module):
    def __init__(self, *, in_channels: int = 3, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int(hidden_channels), 1, kernel_size=1),
        )

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.net(model_input)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
            torch.nn.GroupNorm(_group_count(channels), int(channels)),
            torch.nn.SiLU(),
            torch.nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
            torch.nn.GroupNorm(_group_count(channels), int(channels)),
        )
        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class EncoderStage(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, downsample: bool) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, stride=stride, padding=1),
            torch.nn.GroupNorm(_group_count(out_channels), int(out_channels)),
            torch.nn.SiLU(),
        )
        self.residual = ResidualBlock(int(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.projection(x))


class DecoderStage(torch.nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(int(in_channels) + int(skip_channels), int(out_channels), kernel_size=3, padding=1),
            torch.nn.GroupNorm(_group_count(out_channels), int(out_channels)),
            torch.nn.SiLU(),
        )
        self.residual = ResidualBlock(int(out_channels))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.residual(self.fuse(torch.cat([up, skip], dim=1)))


class ProductionLocalizationModel(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 32,
        depth: int = 3,
        refinement_blocks: int = 2,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        base = int(hidden_channels if hidden_channels is not None else base_channels)
        levels = max(2, int(depth))
        channels = [base * (2**level) for level in range(levels)]
        self.encoder_stages = torch.nn.ModuleList(
            EncoderStage(
                int(in_channels) if level == 0 else channels[level - 1],
                channels[level],
                downsample=level > 0,
            )
            for level in range(levels)
        )
        self.decoder_stages = torch.nn.ModuleList(
            DecoderStage(channels[level], channels[level - 1], channels[level - 1])
            for level in range(levels - 1, 0, -1)
        )
        self.refinement = torch.nn.ModuleList(ResidualBlock(channels[0]) for _ in range(max(0, int(refinement_blocks))))
        self.detection_head = torch.nn.Conv2d(channels[0], 1, kernel_size=1)
        self.xy_head = torch.nn.Conv2d(channels[0], 2, kernel_size=1)
        self.z_head = torch.nn.Conv2d(channels[0], 1, kernel_size=1)
        self.photon_head = torch.nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, model_input: torch.Tensor) -> LocalizationModelOutput:
        skips = []
        features = model_input
        for stage in self.encoder_stages:
            features = stage(features)
            skips.append(features)
        for decoder, skip in zip(self.decoder_stages, reversed(skips[:-1])):
            features = decoder(features, skip)
        for block in self.refinement:
            features = block(features)
        return LocalizationModelOutput(
            detection_logits=self.detection_head(features).squeeze(1),
            xy_offset=torch.tanh(self.xy_head(features)) * 0.5,
            z=self.z_head(features).squeeze(1),
            photons=torch.nn.functional.softplus(self.photon_head(features).squeeze(1)),
        )


def build_localization_model_registry():
    def simple_localizer(params: dict[str, object]) -> torch.nn.Module:
        return SimpleLocalizationModel(
            in_channels=int(params.get("in_channels", 3)),
            hidden_channels=int(params.get("hidden_channels", 16)),
        )

    def production_localizer(params: dict[str, object]) -> torch.nn.Module:
        return ProductionLocalizationModel(
            in_channels=int(params.get("in_channels", 3)),
            base_channels=int(params.get("base_channels", params.get("hidden_channels", 32))),
            depth=int(params.get("depth", 3)),
            refinement_blocks=int(params.get("refinement_blocks", 2)),
        )

    def active_smlm_double_unet(params: dict[str, object]) -> torch.nn.Module:
        from neptune_v03.localization.smlm_unet import DoubleUNet

        nfeatures_inter = params.get("nfeatures_inter", params.get("base_channels", 32))
        return DoubleUNet(
            nch_in=int(params.get("nch_in", params.get("in_channels", 3))),
            depth_shared=int(params.get("depth_shared", 1)),
            depth_union=int(params.get("depth_union", 1)),
            nfeatures_init=int(params.get("nfeatures_init", params.get("base_channels", 32))),
            nfeatures_inter=None if nfeatures_inter is None else int(nfeatures_inter),
            norm_start_level=int(params.get("norm_start_level", 0)),
            norm_groups=int(params.get("norm_groups", 0)),
            activation=_optional_activation(params.get("activation", "ELU")),
            dropout_start_level=(
                None if params.get("dropout_start_level") is None else int(params["dropout_start_level"])
            ),
            p_dropout=float(params.get("p_dropout", 0.1)),
            pool_mode=str(params.get("pool_mode", "StrideConv")),
            upsample_mode=str(params.get("upsample_mode", "bilinear")),
            inter_activation=_optional_activation(params.get("inter_activation", "ELU")),
            norm_head_groups=int(params.get("norm_head_groups", 0)),
            final_activation=_optional_activation(params.get("final_activation", "ELU")),
            disabled_attr=params.get("disabled_attr"),
            kaiming_normal=bool(params.get("kaiming_normal", True)),
            depthwise=bool(params.get("depthwise", True)),
            z_mu_activation=(None if params.get("z_mu_activation") is None else str(params["z_mu_activation"])),
        )

    def active_smlm_soft_moe_double_unet(params: dict[str, object]) -> torch.nn.Module:
        from neptune_v03.localization.soft_moe import SoftMoEFiLMExperts

        base = active_smlm_double_unet(params)
        return SoftMoEFiLMExperts(
            base_model=base,
            condition_dim=int(params["condition_dim"]),
            domain_count=int(params.get("domain_count", 2)),
            hidden_dim=int(params.get("film_hidden_dim", params.get("hidden_dim", 32))),
        )

    return {
        "active_smlm_double_unet": active_smlm_double_unet,
        "active_smlm_soft_moe_double_unet": active_smlm_soft_moe_double_unet,
        "simple_localizer": simple_localizer,
        "production_localizer": production_localizer,
    }


def _optional_activation(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text.strip().lower() in {"", "none", "null"} else text


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if int(channels) % groups == 0:
            return groups
    return 1
