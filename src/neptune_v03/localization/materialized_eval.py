from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from neptune_v03.localization.training_adapter import LocalizationTrainBatch, to_training_batch
from neptune_v03.training.loop import TrainingBatch


@dataclass(frozen=True)
class MaterializedDatasetEvalConfig:
    source_path: str | Path
    dataset_id: str
    sample_id: str
    batch_size: int
    batch_count: int
    frame_range: tuple[int, int] | list[int]
    crop: Mapping[str, Any]
    heldout_split: Mapping[str, Any]


def build_materialized_dataset_eval_provider(config: MaterializedDatasetEvalConfig):
    if int(config.batch_size) <= 0 or int(config.batch_count) <= 0:
        raise ValueError("materialized eval batch_size and batch_count must be positive")
    heldout_split = dict(config.heldout_split)
    if heldout_split.get("allow_train_eval_overlap", False) is not False:
        raise ValueError("materialized eval requires no train/eval overlap by default")
    source_path = Path(config.source_path)
    arrays = _load_npz_arrays(source_path)
    sample_count = int(arrays["model_input"].shape[0])
    if sample_count <= 0:
        raise ValueError("materialized eval source must contain at least one sample")

    fixed_batches = []
    for step_idx in range(int(config.batch_count)):
        indices = [int((step_idx * int(config.batch_size) + batch_idx) % sample_count) for batch_idx in range(int(config.batch_size))]
        loc_batch = LocalizationTrainBatch(
            model_input=torch.as_tensor(arrays["model_input"][indices], dtype=torch.float32),
            detect_tar=torch.as_tensor(arrays["detect_tar"][indices], dtype=torch.float32),
            bkg_tar=torch.as_tensor(arrays["bkg_tar"][indices], dtype=torch.float32),
            pxyz_tar=torch.as_tensor(arrays["pxyz_tar"][indices], dtype=torch.float32),
            mask_tar=torch.as_tensor(arrays["mask_tar"][indices], dtype=torch.bool),
            metadata={
                "source": "materialized_dataset",
                "dataset_id": str(config.dataset_id),
                "sample_id": str(config.sample_id),
                "source_path": str(source_path.resolve()),
                "frame_range": [int(item) for item in config.frame_range],
                "crop": dict(config.crop),
                "heldout_split": heldout_split,
                "step": step_idx + 1,
                "sample_indices": indices,
            },
        )
        fixed_batches.append(to_training_batch(loc_batch))

    return lambda: list(fixed_batches)


def _load_npz_arrays(source_path: Path) -> dict[str, np.ndarray]:
    if source_path.suffix != ".npz":
        raise ValueError("materialized eval source_path must point to a .npz fixture for Slice 6.16")
    with np.load(source_path) as data:
        arrays = {key: data[key] for key in ("model_input", "detect_tar", "bkg_tar", "pxyz_tar", "mask_tar")}
    _validate_arrays(arrays)
    return arrays


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    model_input = arrays["model_input"]
    detect_tar = arrays["detect_tar"]
    bkg_tar = arrays["bkg_tar"]
    pxyz_tar = arrays["pxyz_tar"]
    mask_tar = arrays["mask_tar"]
    if model_input.ndim != 4:
        raise ValueError("materialized eval model_input must have shape [N, C, H, W]")
    sample_count, _, height, width = model_input.shape
    if detect_tar.shape != (sample_count, height, width):
        raise ValueError("materialized eval detect_tar must have shape [N, H, W]")
    if bkg_tar.shape != (sample_count, height, width):
        raise ValueError("materialized eval bkg_tar must have shape [N, H, W]")
    if pxyz_tar.ndim != 3 or pxyz_tar.shape[0] != sample_count or pxyz_tar.shape[2] != 4:
        raise ValueError("materialized eval pxyz_tar must have shape [N, M, 4]")
    if mask_tar.shape != pxyz_tar.shape[:2]:
        raise ValueError("materialized eval mask_tar must have shape [N, M]")
