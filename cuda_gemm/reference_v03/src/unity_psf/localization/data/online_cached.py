"""Cached epoch scheduling and window selection."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import hashlib
import math
import numpy as np
import torch
from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.localization.training_adapter import LocalizationTrainBatch
from unity_psf.runtime.profiling import time_block
from .online_cached_sequences import CachedSequenceStore
from .online_types import OnlineBatchProviderConfig
from .online_physical import VectorRendererCache
from .online_validation import online_item_seed
from .online_validation import resolve_cached_window_cache_frames
from .online_validation import resolve_cached_window_cache_storage
from .online_validation import resolve_field_origin_sampling_mode
from .online_cached_window import slice_cached_window_batch


class CachedWindowEpochBatches(CachedSequenceStore):
    def __init__(
        self,
        config: OnlineBatchProviderConfig,
        *,
        epoch: int,
        cache_frames: int | None = None,
        condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
        condition_store_version: int | None = None,
        cache_generation: int = 0,
        condition_state_getter: Callable[
            [], tuple[int | None, tuple[tuple[str | None, FullResZernikeConditioning], ...], int]
        ]
        | None = None,
        renderer_cache: VectorRendererCache | None = None,
    ) -> None:
        self.config = config
        self.epoch = int(epoch)
        self.cache_epoch = int(epoch)
        self.condition_providers = condition_providers
        self.condition_store_version = condition_store_version
        self.cache_generation = int(cache_generation)
        self.condition_state_getter = condition_state_getter
        self.renderer_cache = renderer_cache
        self.field_origin_sampling_mode = resolve_field_origin_sampling_mode(config)
        self.window_size = int(config.channels)
        if self.window_size != 3:
            raise ValueError("cached_window currently expects channels/window_size=3")
        self.batch_size = int(config.batch_size)
        self.center_samples = int(
            resolve_cached_window_cache_frames(config)
            if cache_frames is None
            else int(cache_frames)
        )
        if self.center_samples % self.batch_size != 0:
            raise ValueError("cached window center sample count must be divisible by batch_size")
        if self.field_origin_sampling_mode == "per_step_random":
            self.sequence_count = max(1, int(config.steps_per_epoch))
        else:
            self.sequence_count = max(1, int(config.sequence_count))
        self.centers_per_sequence = int(math.ceil(float(self.center_samples) / float(self.sequence_count)))
        self.frames_per_sequence = self.centers_per_sequence + self.window_size - 1
        self.base_seed = online_item_seed(
            data_seed=int(config.seed),
            epoch=self.cache_epoch,
            item_index=0,
            items_per_epoch=self.sequence_count,
        )
        self.global_step_base = (self.cache_epoch - 1) * int(config.steps_per_epoch)
        self._sequence_cache: OrderedDict[int, dict[str, object]] = OrderedDict()
        self.cached_window_order = (
            resolve_cached_window_order(config)
        )
        if self.field_origin_sampling_mode == "per_step_random" and self.cached_window_order not in {"sequence_block", "temporal"}:
            raise ValueError("field_origin_sampling_mode='per_step_random' requires cached_window_order='sequence_block', 'temporal', or 'auto' with renderer output")
        self.max_cached_sequences = resolve_cached_window_max_cached_sequences(config)
        self.sequence_cache_evictions = 0
        self._reported_metric_sequence_indices: set[int] = set()
        self.center_records: list[tuple[int, int, int]] = []
        for sequence_idx in range(self.sequence_count):
            for center_offset in range(self.centers_per_sequence):
                if len(self.center_records) >= self.center_samples:
                    break
                center_idx = center_offset + self.window_size // 2
                self.center_records.append((sequence_idx, center_idx, len(self.center_records)))
        rng = np.random.default_rng(self.base_seed)
        self.order = self._build_order(rng)
        self.cache_batch_count = self.center_samples // self.batch_size
        if int(config.steps_per_epoch) > self.cache_batch_count:
            raise ValueError(
                "cached_window steps_per_epoch cannot exceed the number of full cache batches "
                f"({int(config.steps_per_epoch)} > {self.cache_batch_count})"
            )
        self.data_pool_fingerprint = ""

    def materialize_all(self) -> None:
        with time_block("cached_window_pool_materialize"):
            for sequence_idx in range(self.sequence_count):
                self.sequence(sequence_idx)
        self.data_pool_fingerprint = self._build_data_pool_fingerprint()

    def _build_data_pool_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.epoch).encode("utf-8"))
        digest.update(str(self.cache_generation).encode("utf-8"))
        digest.update(str(self.condition_store_version).encode("utf-8"))
        for sequence_idx in range(min(self.sequence_count, 2)):
            payload = self.sequence(sequence_idx)
            for key in ("frames", "pxyz", "mask", "bkg"):
                value = payload.get(key)
                if isinstance(value, torch.Tensor):
                    digest.update(key.encode("utf-8"))
                    digest.update(value.detach().to(device="cpu").contiguous().numpy().tobytes())
        return digest.hexdigest()

    def __len__(self) -> int:
        return int(self.config.steps_per_epoch)

    def __iter__(self):
        for batch_idx in range(len(self)):
            yield self[batch_idx]

    def __getitem__(self, batch_idx: int) -> LocalizationTrainBatch:
        self.refresh_condition_state()
        batch_idx = int(batch_idx)
        selected = [
            self.center_records[int(idx)]
            for idx in self.order[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        ]
        sequence_indices = sorted({int(item[0]) for item in selected})
        sequences = [self.sequence(index) for index in sequence_indices]
        sequence_by_index = {int(index): sequence for index, sequence in zip(sequence_indices, sequences, strict=True)}
        metric_sequence_indices = set(sequence_indices) - self._reported_metric_sequence_indices
        ordered_sequences = [
            sequence_with_batch_metric_visibility(
                sequence_by_index[int(sequence_idx)],
                include_metrics=int(sequence_idx) in metric_sequence_indices,
            )
            for sequence_idx, _, _ in selected
        ]
        remapped_selected = [(sample_idx, center_idx, global_center_idx) for sample_idx, (_, center_idx, global_center_idx) in enumerate(selected)]
        loc_batch = slice_cached_window_batch(
            self.config,
            sequences=ordered_sequences,
            selected=remapped_selected,
            epoch=self.epoch,
            batch_idx=batch_idx,
            center_samples=self.center_samples,
            sequence_count=self.sequence_count,
            frames_per_sequence=self.frames_per_sequence,
            condition_providers=self.condition_providers,
            condition_store_version=self.condition_store_version,
        )
        loc_batch.metadata.update(
            {
                "cached_window_order": self.cached_window_order,
                "cached_window_max_cached_sequences": int(self.max_cached_sequences),
                "cached_window_cached_sequence_count": int(len(self._sequence_cache)),
                "cached_window_sequence_cache_evictions": int(self.sequence_cache_evictions),
                "field_origin_sampling_mode": self.field_origin_sampling_mode,
                "cached_window_cache_epoch": int(self.cache_epoch),
                "cached_window_cache_frames": int(self.center_samples),
                "cached_window_prefetch_batches": int(self.config.cached_window_prefetch_batches),
                "condition_store_version": self.condition_store_version,
                "physical_model_version": self.condition_store_version,
                "cache_generation": int(self.cache_generation),
                "data_refresh_epoch": int(self.epoch),
                "data_refresh_seed": int(self.base_seed),
                "data_pool_fingerprint": self.data_pool_fingerprint,
            }
        )
        self._reported_metric_sequence_indices.update(sequence_indices)
        return loc_batch

    def _build_order(self, rng: np.random.Generator) -> np.ndarray:
        if self.cached_window_order == "temporal":
            return np.arange(len(self.center_records), dtype=np.int64)
        if self.cached_window_order == "shuffle":
            order = np.arange(len(self.center_records), dtype=np.int64)
            rng.shuffle(order)
            return order
        if self.cached_window_order != "sequence_block":
            raise ValueError(f"unsupported cached_window_order: {self.cached_window_order}")
        blocks: list[int] = []
        sequence_order = np.arange(self.sequence_count, dtype=np.int64)
        rng.shuffle(sequence_order)
        records_by_sequence: list[list[int]] = [[] for _ in range(self.sequence_count)]
        for record_index, (sequence_idx, _center_idx, _global_center_idx) in enumerate(self.center_records):
            records_by_sequence[int(sequence_idx)].append(int(record_index))
        for sequence_idx in sequence_order.tolist():
            sequence_records = np.asarray(records_by_sequence[int(sequence_idx)], dtype=np.int64)
            rng.shuffle(sequence_records)
            blocks.extend(int(item) for item in sequence_records.tolist())
        return np.asarray(blocks, dtype=np.int64)






def build_cached_window_epoch_batches(
    config: OnlineBatchProviderConfig,
    *,
    epoch: int,
    cache_frames: int | None = None,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
    cache_generation: int = 0,
    condition_state_getter: Callable[
        [], tuple[int | None, tuple[tuple[str | None, FullResZernikeConditioning], ...], int]
    ]
    | None = None,
    renderer_cache: VectorRendererCache | None = None,
) -> CachedWindowEpochBatches:
    return CachedWindowEpochBatches(
        config,
        epoch=int(epoch),
        cache_frames=None if cache_frames is None else int(cache_frames),
        condition_providers=condition_providers,
        condition_store_version=condition_store_version,
        cache_generation=cache_generation,
        condition_state_getter=condition_state_getter,
        renderer_cache=renderer_cache,
    )


def resolve_cached_window_order(config: OnlineBatchProviderConfig) -> str:
    raw = str(config.cached_window_order or "auto").strip().lower()
    if raw == "auto":
        return "shuffle"
    if raw in {"shuffle", "sequence_block", "temporal"}:
        return raw
    raise ValueError("cached_window_order must be 'auto', 'shuffle', 'sequence_block', or 'temporal'")


def resolve_cached_window_max_cached_sequences(config: OnlineBatchProviderConfig) -> int:
    if resolve_cached_window_cache_storage(config) == "cpu":
        return 0
    value = int(config.cached_window_max_gpu_sequences)
    if value > 0:
        return value
    if str(config.simulation_output_device).strip().lower() == "renderer":
        return 4
    return 0


def sequence_with_batch_metric_visibility(sequence: dict[str, object], *, include_metrics: bool) -> dict[str, object]:
    if include_metrics:
        return sequence
    metadata = dict(sequence.get("metadata", {}))
    for key in (
        "renderer_cache_hits",
        "renderer_cache_misses",
        "render_many_calls",
        "render_frames_calls",
    ):
        metadata[key] = 0
    return {**sequence, "metadata": metadata}
