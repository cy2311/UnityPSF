from __future__ import annotations

from dataclasses import dataclass

import torch

from neptune_v03.localization.training_adapter import LocalizationTrainBatch, to_training_batch
from neptune_v03.training.loop import TrainingBatch


@dataclass(frozen=True)
class SyntheticOnlineBatchConfig:
    batch_size: int
    channels: int = 3
    height: int = 128
    width: int = 128
    emitters_per_sample: int = 8
    seed: int = 0
    background: float = 0.0
    signal: float = 1.0


def build_synthetic_online_batch_provider(config: SyntheticOnlineBatchConfig):
    def provider(epoch: int) -> list[TrainingBatch]:
        seed = int(config.seed) + int(epoch)
        loc_batch = build_synthetic_localization_batch(config, epoch=int(epoch), seed=seed)
        return [to_training_batch(loc_batch)]

    return provider


def build_synthetic_localization_batch(
    config: SyntheticOnlineBatchConfig,
    *,
    epoch: int,
    seed: int,
    step: int | None = None,
    source: str = "deterministic_synthetic_online",
) -> LocalizationTrainBatch:
    generator = torch.Generator().manual_seed(int(seed))
    return _build_batch(
        config,
        epoch=int(epoch),
        seed=int(seed),
        step=step,
        source=source,
        generator=generator,
    )


def _build_batch(
    config: SyntheticOnlineBatchConfig,
    *,
    epoch: int,
    seed: int,
    step: int | None,
    source: str,
    generator: torch.Generator,
) -> LocalizationTrainBatch:
    batch_size = int(config.batch_size)
    channels = int(config.channels)
    height = int(config.height)
    width = int(config.width)
    emitters = int(config.emitters_per_sample)
    if min(batch_size, channels, height, width, emitters) <= 0:
        raise ValueError("synthetic batch dimensions and emitters_per_sample must be positive")

    model_input = torch.full(
        (batch_size, channels, height, width),
        float(config.background),
        dtype=torch.float32,
    )
    detect = torch.zeros((batch_size, height, width), dtype=torch.float32)
    bkg = torch.full((batch_size, height, width), float(config.background), dtype=torch.float32)
    pxyz = torch.zeros((batch_size, emitters, 4), dtype=torch.float32)
    mask = torch.ones((batch_size, emitters), dtype=torch.bool)

    ys = torch.randint(0, height, (batch_size, emitters), generator=generator)
    xs = torch.randint(0, width, (batch_size, emitters), generator=generator)
    for batch_idx in range(batch_size):
        for emitter_idx in range(emitters):
            y = int(ys[batch_idx, emitter_idx])
            x = int(xs[batch_idx, emitter_idx])
            detect[batch_idx, y, x] = 1.0
            model_input[batch_idx, :, y, x] = float(config.signal)
            pxyz[batch_idx, emitter_idx] = torch.tensor(
                [float(x), float(y), 0.0, float(config.signal)],
                dtype=torch.float32,
            )

    metadata = {
        "epoch": epoch,
        "seed": seed,
        "source": source,
    }
    if step is not None:
        metadata["step"] = int(step)

    return LocalizationTrainBatch(
        model_input=model_input,
        detect_tar=detect,
        bkg_tar=bkg,
        pxyz_tar=pxyz,
        mask_tar=mask,
        metadata=metadata,
    )
