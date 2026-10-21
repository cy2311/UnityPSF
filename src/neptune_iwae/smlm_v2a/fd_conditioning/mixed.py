from __future__ import annotations

import random
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


def append_conditioning_channels(frames: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 3 or conditioning.ndim != 3:
        raise ValueError("frames and conditioning must both be 3D tensors shaped as (C,H,W).")
    if frames.shape[-2:] != conditioning.shape[-2:]:
        raise ValueError("frames and conditioning spatial shapes must match.")
    return torch.cat((frames, conditioning), dim=0)


class ChannelBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        channel_to_indices: dict[str, list[int]],
        batch_size: int,
        batches_per_epoch: int,
        seed: int = 0,
    ) -> None:
        self.channel_to_indices = {
            str(channel): [int(idx) for idx in indices]
            for channel, indices in channel_to_indices.items()
            if indices
        }
        if not self.channel_to_indices:
            raise ValueError("channel_to_indices must contain at least one non-empty channel.")
        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        channels = list(self.channel_to_indices.keys())
        for _ in range(self.batches_per_epoch):
            channel = rng.choice(channels)
            population = self.channel_to_indices[channel]
            if len(population) >= self.batch_size:
                yield rng.sample(population, self.batch_size)
            else:
                yield [rng.choice(population) for _ in range(self.batch_size)]
