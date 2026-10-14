"""Sequence generation, cache invalidation and LRU eviction."""

from __future__ import annotations

from unity_psf.runtime.profiling import time_block
from .online_geometry import cached_sequence_condition_origin
from .online_validation import cached_window_precompute_enabled
from .online_cached_payload import materialize_cached_sequence_payload_cpu
from .online_cached_payload import precompute_cached_sequence_payload
from .online_geometry import random_sequence_condition_origin
from .online_validation import resolve_cached_window_cache_storage
from .online_conditioning import domain_index as resolve_domain_index
from .online_conditioning import select_condition_provider
from .online_native import simulate_native_sequence
from .online_physical import simulator_config
from .online_geometry import sliding_window_sequence_condition_origin


class CachedSequenceStore:
    def refresh_condition_state(self) -> None:
        if self.condition_state_getter is None:
            return
        version, providers, cache_generation = self.condition_state_getter()
        version_key = None if version is None else int(version)
        cache_generation_i = int(cache_generation)
        if (
            version_key == self.condition_store_version
            and cache_generation_i == self.cache_generation
            and providers is self.condition_providers
        ):
            return
        self.condition_store_version = version_key
        self.condition_providers = providers
        self.cache_generation = cache_generation_i
        self._sequence_cache.clear()
        self._reported_metric_sequence_indices.clear()

    def sequence(self, sequence_idx: int) -> dict[str, object]:
        sequence_idx = int(sequence_idx)
        if sequence_idx in self._sequence_cache:
            payload = self._sequence_cache.pop(sequence_idx)
            self._sequence_cache[sequence_idx] = payload
            return payload
        config = self.config
        self.evict_sequence_cache_to_fit(new_items=1)
        domain_index = resolve_domain_index(config, global_step=self.global_step_base + sequence_idx)
        condition_provider, condition_source, _ = select_condition_provider(
            config,
            domain_index=domain_index,
            condition_providers=self.condition_providers,
        )
        field_origin_index = None
        field_origin_bank_size = None
        if self.field_origin_sampling_mode == "per_step_random":
            field_origin = random_sequence_condition_origin(
                config,
                condition_provider=condition_provider,
                epoch=self.cache_epoch,
                sequence_idx=sequence_idx,
                domain_index=domain_index,
            )
        elif self.field_origin_sampling_mode == "sliding_window":
            field_origin, field_origin_index, field_origin_bank_size = sliding_window_sequence_condition_origin(
                config,
                condition_provider=condition_provider,
                epoch=self.cache_epoch,
                sequence_idx=sequence_idx,
                sequence_count=self.sequence_count,
            )
        else:
            field_origin = cached_sequence_condition_origin(
                config,
                condition_provider=condition_provider,
                epoch=self.cache_epoch,
                sequence_idx=sequence_idx,
                sequence_count=self.sequence_count,
            )
        sim_config = simulator_config(
            config,
            seed=self.base_seed + sequence_idx,
            field_origin=field_origin,
            frames_per_sample=1,
            condition_provider=condition_provider,
        )
        sequence = simulate_native_sequence(
            config,
            sim_config,
            epoch=self.cache_epoch,
            sequence_idx=sequence_idx,
            frame_count=self.frames_per_sequence,
            physical_model_version=self.condition_store_version,
            cache_generation=self.cache_generation,
            renderer_cache=self.renderer_cache,
        )
        payload = {
            **sequence,
            "sequence_idx": sequence_idx,
            "domain_index": domain_index,
            "domain_name": (
                None
                if not self.condition_providers
                else self.condition_providers[int(domain_index) % len(self.condition_providers)][0]
            ),
            "field_origin": field_origin,
            "field_origin_index": field_origin_index,
            "field_origin_bank_size": field_origin_bank_size,
            "global_step": self.global_step_base + sequence_idx,
        }
        if cached_window_precompute_enabled():
            precompute_cached_sequence_payload(
                config,
                payload,
                condition_provider=condition_provider,
                condition_source=condition_source,
            )
        if resolve_cached_window_cache_storage(config) == "cpu":
            with time_block("cached_sequence_cpu_materialize"):
                materialize_cached_sequence_payload_cpu(payload)
        self._sequence_cache[sequence_idx] = payload
        self.evict_sequence_cache_over_limit()
        return payload

    def evict_sequence_cache_to_fit(self, *, new_items: int) -> None:
        if self.max_cached_sequences <= 0:
            return
        while len(self._sequence_cache) + int(new_items) > int(self.max_cached_sequences):
            self._sequence_cache.popitem(last=False)
            self.sequence_cache_evictions += 1

    def evict_sequence_cache_over_limit(self) -> None:
        if self.max_cached_sequences <= 0:
            return
        while len(self._sequence_cache) > int(self.max_cached_sequences):
            self._sequence_cache.popitem(last=False)
            self.sequence_cache_evictions += 1

    _refresh_condition_state = refresh_condition_state
    _sequence = sequence
    _evict_sequence_cache_to_fit = evict_sequence_cache_to_fit
    _evict_sequence_cache_over_limit = evict_sequence_cache_over_limit
