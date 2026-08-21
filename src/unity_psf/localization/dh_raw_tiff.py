from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import tifffile
import torch

from unity_psf.models.psf_moe.experts.double_helix import (
    DoubleHelixDirectXYZLoss,
    DoubleHelixImageExpert,
)
from unity_psf.localization.training_adapter import LocalizationTrainBatch


@dataclass(frozen=True)
class DHRawTiffBatch:
    inputs: torch.Tensor
    targets: Mapping[str, torch.Tensor]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class DHRawTiffBatchProviderConfig:
    raw_tiff_path: str | Path
    target_npz_path: str | Path
    frames_npz_path: str | Path | None = None
    physical_state_path: str | Path | None = None
    simulation_metadata_path: str | Path | None = None
    batch_size: int = 8
    steps_per_epoch: int = 12
    channels: int = 3
    frame_start: int = 0
    frame_stop: int | None = 75000
    train_background_adu: float = 495.58422534346505
    signal_gain: float = 0.03
    gain_scope: str = "global_run"
    seed: int = 0


def _build_targets(npz: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, torch.Tensor]:
    emitters = np.asarray(npz["emitter_xy_z_um_photons"], dtype=np.float32)
    emitter_mask = np.asarray(npz["emitter_mask"], dtype=bool)
    lobes = np.asarray(npz["dh_lobe_targets"], dtype=np.float32)
    lobe_mask = np.asarray(npz["dh_lobe_mask"], dtype=bool)
    _, _, height, width = np.asarray(npz["frames_adu"]).shape
    n = len(indices)
    raw = {key: np.zeros((n, height, width), np.float32) for key in ("detection", "z", "photons")}
    raw["xy_offset"] = np.zeros((n, 2, height, width), np.float32)
    raw["lobe_angle"] = np.zeros((n, 1, height, width), np.float32)
    raw["lobe_separation"] = np.zeros((n, 1, height, width), np.float32)
    for local, source in enumerate(indices):
        occupied: dict[tuple[int, int], float] = {}
        for emitter_index in np.flatnonzero(emitter_mask[source]):
            x, y, z, photons = map(float, emitters[source, emitter_index])
            xi = int(np.clip(round(x), 0, width - 1)); yi = int(np.clip(round(y), 0, height - 1))
            if (yi, xi) in occupied and photons <= occupied[(yi, xi)]:
                continue
            occupied[(yi, xi)] = photons
            raw["detection"][local, yi, xi] = 1.0
            raw["xy_offset"][local, :, yi, xi] = (x - xi, y - yi)
            raw["z"][local, yi, xi] = z / 1.5
            raw["photons"][local, yi, xi] = photons / 31000.0
            pair = 2 * int(emitter_index)
            if pair + 1 < lobe_mask.shape[1] and lobe_mask[source, pair] and lobe_mask[source, pair + 1]:
                first, second = lobes[source, pair], lobes[source, pair + 1]
                raw["lobe_angle"][local, 0, yi, xi] = first[0]
                raw["lobe_separation"][local, 0, yi, xi] = np.hypot(first[1] - second[1], first[2] - second[2]) / 31.0
    return {key: torch.from_numpy(value) for key, value in raw.items()}


def build_dh_raw_tiff_batch_provider(config: DHRawTiffBatchProviderConfig):
    if not Path(config.raw_tiff_path).is_file() or not Path(config.target_npz_path).is_file():
        raise FileNotFoundError("DH raw TIFF provider requires raw_tiff_path and target_npz_path")
    target_data = np.load(config.target_npz_path, allow_pickle=False)
    if config.frames_npz_path is not None:
        frames_data = np.load(config.frames_npz_path, allow_pickle=False)
        target_data = frames_data
    target_count = int(target_data["frames_adu"].shape[0])
    physical_state = None
    if config.physical_state_path is not None:
        state_path = Path(config.physical_state_path)
        if not state_path.is_file():
            raise FileNotFoundError(f"DH physical update state is missing: {state_path}")
        physical_state = __import__("json").loads(state_path.read_text(encoding="utf-8"))
        if physical_state.get("source") != "gamma_feedback":
            raise ValueError("DH provider requires physical state source='gamma_feedback'")
    if config.frames_npz_path is not None and config.simulation_metadata_path is not None:
        metadata_path = Path(config.simulation_metadata_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"DH simulation metadata is missing: {metadata_path}")
        simulation_metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
        if simulation_metadata.get("source") != "dh_vector_lut_simulation_from_physical_update":
            raise ValueError("DH simulation metadata must declare physical-update vector/LUT source")
        if simulation_metadata.get("simulation_backend") != "lut" or simulation_metadata.get("psf_type") != "vector":
            raise ValueError("DH simulation metadata must declare vector/LUT rendering")
        if physical_state is not None:
            entries = physical_state.get("coeff_maps", [])
            state_map = entries[0].get("coeff_maps_npz") if len(entries) == 1 and isinstance(entries[0], dict) else None
            if state_map is None or Path(str(state_map)).resolve() != Path(str(simulation_metadata.get("coefficient_map"))).resolve():
                raise ValueError("DH simulation coefficient map does not match physical-update state")
    if config.frames_npz_path is not None:
        frames_source = np.load(config.frames_npz_path, allow_pickle=False)
        tiff = np.asarray(frames_source["frames_adu"], dtype=np.float32)
    else:
        tiff = tifffile.memmap(config.raw_tiff_path, mode="r")
    simulated_samples = tiff.ndim == 4
    stop = tiff.shape[0] if config.frame_stop is None else min(int(config.frame_stop), tiff.shape[0])
    if not simulated_samples and stop - int(config.frame_start) < int(config.channels):
        raise ValueError("DH raw TIFF has fewer frames than one temporal window")
    if int(config.channels) != 3:
        raise ValueError("DH direct-XYZ provider requires exactly 3 temporal frames")
    if simulated_samples and tiff.shape[1] != int(config.channels):
        raise ValueError("DH vector/LUT simulation must provide exactly three input frames per sample")
    required_samples = int(config.batch_size) * int(config.steps_per_epoch)
    if simulated_samples and target_count < required_samples:
        raise ValueError(
            "materialized DH batch has fewer samples than the configured epoch requires; "
            f"need {required_samples}, found {target_count}"
        )

    def provider(epoch: int) -> Iterable[DHRawTiffBatch]:
        rng = np.random.default_rng(int(config.seed) + int(epoch))
        max_start = stop - int(config.channels) - int(config.frame_start)
        epoch_indices = (
            rng.permutation(target_count)[:required_samples]
            if simulated_samples
            else None
        )
        for step in range(int(config.steps_per_epoch)):
            if simulated_samples:
                assert epoch_indices is not None
                indices = epoch_indices[
                    step * int(config.batch_size):(step + 1) * int(config.batch_size)
                ]
                windows = np.asarray(tiff[indices], dtype=np.float32)
                frame_windows = [(int(index), int(index) + 1) for index in indices]
            else:
                starts = (rng.integers(0, max_start + 1, size=int(config.batch_size)) + step * int(config.batch_size)) % (max_start + 1)
                windows = []
                for start in starts:
                    raw = np.asarray(tiff[int(config.frame_start) + int(start): int(config.frame_start) + int(start) + 3], dtype=np.float32)
                    centered = (raw - float(config.train_background_adu)) * float(config.signal_gain)
                    windows.append(centered)
                windows = np.stack(windows, axis=0)
                indices = np.arange(step * int(config.batch_size), (step + 1) * int(config.batch_size)) % target_count
                frame_windows = [(int(config.frame_start) + int(start), int(config.frame_start) + int(start) + 3) for start in starts]
            yield DHRawTiffBatch(
                inputs=torch.from_numpy(windows),
                targets=_build_targets(target_data, indices),
                metadata={"source": "dh_vector_lut_simulation_from_physical_update" if simulated_samples else "dh_raw_tiff_75000", "epoch": int(epoch), "step": step + 1, "frame_windows": frame_windows, "recenter_mode": "fd_deeploc_exact_recenter", "train_background_adu": float(config.train_background_adu), "signal_gain": float(config.signal_gain), "gain_scope": "global_run"},
            )

    return provider


def _direct_targets_from_online(batch: LocalizationTrainBatch, *, z_scale: float, photon_scale: float) -> dict[str, torch.Tensor]:
    images = batch.model_input[0] if isinstance(batch.model_input, tuple) else batch.model_input
    count, _, height, width = images.shape
    device = batch.pxyz_tar.device
    targets = {
        "detection": torch.zeros((count, height, width), device=device),
        "xy_offset": torch.zeros((count, 2, height, width), device=device),
        "z": torch.zeros((count, height, width), device=device),
        "photons": torch.zeros((count, height, width), device=device),
        "lobe_angle": torch.zeros((count, 1, height, width), device=device),
        "lobe_separation": torch.zeros((count, 1, height, width), device=device),
    }
    for sample in range(count):
        occupied: dict[tuple[int, int], float] = {}
        for emitter in torch.nonzero(batch.mask_tar[sample], as_tuple=False).flatten().tolist():
            x, y, z, photons = batch.pxyz_tar[sample, emitter].tolist()
            xi = int(max(0, min(width - 1, round(x))))
            yi = int(max(0, min(height - 1, round(y))))
            if photons <= occupied.get((yi, xi), float("-inf")):
                continue
            occupied[(yi, xi)] = photons
            targets["detection"][sample, yi, xi] = 1.0
            targets["xy_offset"][sample, :, yi, xi] = torch.tensor((x - xi, y - yi), device=device)
            targets["z"][sample, yi, xi] = z / z_scale
            targets["photons"][sample, yi, xi] = photons / photon_scale
    return targets


def build_dh_online_direct_xyz_batch_provider(params: dict[str, object]):
    from unity_psf.localization.online import OnlineBatchProviderConfig, build_online_batch_provider

    config = OnlineBatchProviderConfig(**params)
    online_provider = build_online_batch_provider(config)

    def provider(epoch: int) -> Iterable[DHRawTiffBatch]:
        for item in online_provider(epoch):
            batch = item.inputs
            if not isinstance(batch, LocalizationTrainBatch):
                raise TypeError("DH online provider requires LocalizationTrainBatch inputs")
            images = batch.model_input[0] if isinstance(batch.model_input, tuple) else batch.model_input
            yield DHRawTiffBatch(
                inputs=images,
                targets=_direct_targets_from_online(
                    batch,
                    z_scale=float(config.z_scale or max(abs(value) for value in config.z_range)),
                    photon_scale=float(config.photon_scale or max(config.photon_range)),
                ),
                metadata={**batch.metadata, "source": "dh_online_vector_lut_direct_xyz"},
            )

    return provider


class DHDirectXYZLossAdapter:
    def __init__(self, params: dict[str, object] | None = None):
        self.loss = DoubleHelixDirectXYZLoss(auxiliary_weight=float((params or {}).get("auxiliary_weight", 0.25)))

    def from_output(self, output, batch: DHRawTiffBatch):
        device = output.detection_logits.device
        targets = {key: value.to(device=device) for key, value in batch.targets.items()}
        return self.loss(output, targets)
