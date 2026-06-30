from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch

from neptune_v03.data import CameraCalibration, TrainNormalization, adu_to_photons, normalize_train_input, read_tiff_stack
from neptune_v03.localization.training_adapter import LocalizationTrainBatch, to_training_batch
from neptune_v03.training.loop import TrainingBatch


@dataclass(frozen=True)
class MicrotubeTiffBatchProviderConfig:
    tiff_path: str | Path
    batch_size: int
    channels: int = 3
    height: int | None = None
    width: int | None = None
    steps_per_epoch: int = 1
    frame_start: int = 0
    frame_stop: int | None = None
    crop_top: int = 0
    crop_left: int = 0
    seed: int = 0
    calibration: Mapping[str, float] = field(default_factory=dict)
    normalization: Mapping[str, float] = field(default_factory=dict)


def build_microtube_tiff_batch_provider(config: MicrotubeTiffBatchProviderConfig):
    frames_adu = read_tiff_stack(config.tiff_path)
    frame_stop = frames_adu.shape[0] if config.frame_stop is None else int(config.frame_stop)
    frames_adu = frames_adu[int(config.frame_start) : frame_stop]
    if frames_adu.shape[0] < int(config.channels):
        raise ValueError("microtube TIFF provider requires at least channels frames")

    calibration = CameraCalibration(
        baseline_adu=float(config.calibration.get("baseline_adu", config.calibration.get("baseline", 0.0))),
        e_per_adu=float(config.calibration.get("e_per_adu", 1.0)),
        qe=float(config.calibration.get("qe", 1.0)),
        em_gain=float(config.calibration.get("em_gain", 1.0)),
        spurious_charge=float(config.calibration.get("spurious_charge", 0.0)),
    )
    normalization = TrainNormalization(
        input_offset=float(config.normalization.get("input_offset", 0.0)),
        input_scale=float(config.normalization.get("input_scale", 1.0)),
        photon_scale=float(config.normalization.get("photon_scale", 1.0)),
    )
    frames = normalize_train_input(adu_to_photons(frames_adu, calibration), normalization)
    crop_height = frames.shape[-2] - int(config.crop_top) if config.height is None else int(config.height)
    crop_width = frames.shape[-1] - int(config.crop_left) if config.width is None else int(config.width)
    _validate_config(config, frame_count=frames.shape[0], crop_height=crop_height, crop_width=crop_width)

    def provider(epoch: int) -> list[TrainingBatch]:
        batches = []
        max_start = frames.shape[0] - int(config.channels)
        cursor = (int(config.seed) + int(epoch) - 1) % (max_start + 1)
        for step_idx in range(int(config.steps_per_epoch)):
            loc_batches = []
            frame_windows = []
            for batch_idx in range(int(config.batch_size)):
                start = (cursor + step_idx * int(config.batch_size) + batch_idx) % (max_start + 1)
                stop = start + int(config.channels)
                window = frames[
                    start:stop,
                    int(config.crop_top) : int(config.crop_top) + crop_height,
                    int(config.crop_left) : int(config.crop_left) + crop_width,
                ]
                loc_batches.append(torch.as_tensor(window, dtype=torch.float32))
                frame_windows.append((int(config.frame_start) + start, int(config.frame_start) + stop))
            model_input = torch.stack(loc_batches, dim=0)
            batches.append(to_training_batch(_empty_target_batch(model_input, frame_windows=frame_windows, epoch=epoch, step=step_idx + 1)))
        return batches

    return provider


def _empty_target_batch(
    model_input: torch.Tensor,
    *,
    frame_windows: list[tuple[int, int]],
    epoch: int,
    step: int,
) -> LocalizationTrainBatch:
    batch_size, _, height, width = model_input.shape
    return LocalizationTrainBatch(
        model_input=model_input,
        detect_tar=torch.zeros((batch_size, height, width), dtype=torch.float32),
        bkg_tar=torch.zeros((batch_size, height, width), dtype=torch.float32),
        pxyz_tar=torch.zeros((batch_size, 0, 4), dtype=torch.float32),
        mask_tar=torch.zeros((batch_size, 0), dtype=torch.bool),
        metadata={
            "source": "microtube_tiff",
            "epoch": int(epoch),
            "step": int(step),
            "frame_windows": frame_windows,
        },
    )


def _validate_config(
    config: MicrotubeTiffBatchProviderConfig,
    *,
    frame_count: int,
    crop_height: int,
    crop_width: int,
) -> None:
    if min(int(config.batch_size), int(config.channels), int(config.steps_per_epoch), int(crop_height), int(crop_width)) <= 0:
        raise ValueError("microtube TIFF provider dimensions must be positive")
    if int(config.crop_top) < 0 or int(config.crop_left) < 0:
        raise ValueError("microtube TIFF crop offsets must be non-negative")
    if frame_count < int(config.channels):
        raise ValueError("microtube TIFF frame range is shorter than channels")
