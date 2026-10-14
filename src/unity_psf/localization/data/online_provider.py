"""Online strategy orchestration and training batch adaptation."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from unity_psf.localization.conditioning import ConditioningProviderStore
from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.localization.training_adapter import to_training_batch
from unity_psf.optics import resolve_circular_psf_anchor_profile
from unity_psf.training.loop_types import TrainingBatch
from .online_cached import CachedWindowEpochBatches
from .online_types import OnlineBatchProviderConfig
from .online_physical import PhysicalVersionedRendererLifecycle
from .online_physical import VectorRendererCache
from .online_cached import build_cached_window_epoch_batches
from .online_native import build_native_online_batch
from .online_conditioning import conditioning_profile_source
from .online_conditioning import default_condition_provider
from .online_conditioning import load_condition_providers
from .online_validation import online_item_seed
from .online_validation import resolve_cached_window_cache_epochs
from .online_validation import resolve_cached_window_cache_frames
from .online_validation import validate_online_simulation_modes


def build_online_batch_provider(config: OnlineBatchProviderConfig, *, condition_store: ConditioningProviderStore | None = None):
    if config.simulation_backend not in {"native", "cuda_gemm", "ppg_direct"}:
        raise ValueError("simulation_backend must be 'native' or 'cuda_gemm' or 'ppg_direct'")
    profile_key = str(config.conditioning_profile).strip().lower().replace("-", "_")
    if (
        (profile_key.startswith("circular_2d") or profile_key.startswith("circular_psf"))
        and config.circular_psf_radius_nm is None
    ):
        config = replace(
            config,
            circular_psf_radius_nm=resolve_circular_psf_anchor_profile(profile_key).radius_nm,
        )
    validate_online_simulation_modes(config)
    if config.fixed_astig_anchor_nm is not None and float(config.fixed_astig_anchor_nm) != 0.0:
        if config.dual_domain_coeff_maps:
            raise ValueError("fixed_astig_anchor_nm cannot be combined with absolute coefficient maps")
        if conditioning_profile_source(config) == "astigmatism_anchor_profile":
            raise ValueError("fixed_astig_anchor_nm cannot be combined with astigmatism conditioning_profile")
    if config.circular_psf_radius_nm is not None:
        radius_nm = float(config.circular_psf_radius_nm)
        if not np.isfinite(radius_nm) or radius_nm <= 0.0:
            raise ValueError("circular_psf_radius_nm must be finite and positive")
        if config.dual_domain_coeff_maps:
            raise ValueError("circular_psf_radius_nm cannot be combined with absolute coefficient maps")
    if config.dual_domain_coeff_maps and int(config.domain_count) != len(config.dual_domain_coeff_maps):
        raise ValueError("domain_count must match dual_domain_coeff_maps length")
    if condition_store is None and config.dual_domain_coeff_maps:
        condition_store = ConditioningProviderStore.from_coeff_maps(config.dual_domain_coeff_maps)
    condition_providers = None if condition_store is not None else load_condition_providers(config)
    default_condition_providers = ((None, default_condition_provider(config)),)
    renderer_cache = VectorRendererCache()
    cached_window_pool: dict[tuple[int | None, int, int], CachedWindowEpochBatches] = {}
    cached_window_pool_signature: tuple[int | None, int] | None = None
    lifecycle = PhysicalVersionedRendererLifecycle(renderer_cache)
    if condition_store is not None:
        initial_version, initial_providers = condition_store.snapshot()
        if config.fixed_astig_anchor_nm and initial_providers:
            raise ValueError("fixed_astig_anchor_nm cannot be combined with absolute coefficient maps")
        lifecycle.activate(
            version=initial_version,
            condition_providers=initial_providers or default_condition_providers,
        )
        condition_store.add_update_listener(
            lambda version, providers: lifecycle.activate(
                version=version,
                condition_providers=providers or default_condition_providers,
            )
        )
    else:
        lifecycle.activate(version=None, condition_providers=condition_providers or default_condition_providers)

    def condition_state() -> tuple[int | None, tuple[tuple[str | None, FullResZernikeConditioning], ...], int]:
        condition_store_version = None
        active_condition_providers = condition_providers
        if condition_store is not None:
            condition_store_version, active_condition_providers = condition_store.snapshot()
            if config.fixed_astig_anchor_nm and active_condition_providers:
                raise ValueError("fixed_astig_anchor_nm cannot be combined with absolute coefficient maps")
        if active_condition_providers is None:
            active_condition_providers = default_condition_providers
        lifecycle.activate(version=condition_store_version, condition_providers=active_condition_providers)
        return condition_store_version, active_condition_providers, lifecycle.cache_generation

    def provider(epoch: int) -> list[TrainingBatch]:
        nonlocal cached_window_pool_signature
        if int(config.steps_per_epoch) <= 0:
            raise ValueError("steps_per_epoch must be positive")
        condition_store_version, active_condition_providers, cache_generation = condition_state()

        if str(config.batch_strategy) == "cached_window":
            fixed_cache = int(config.cached_window_cache_frames) > 0
            cache_epochs = resolve_cached_window_cache_epochs(config)
            signature = (
                None if condition_store_version is None else int(condition_store_version),
                int(cache_generation),
            )
            if signature != cached_window_pool_signature:
                cached_window_pool.clear()
                cached_window_pool_signature = signature
            epoch_i = int(epoch)
            cache_slot = 0 if fixed_cache else (epoch_i - 1) % cache_epochs
            pool_key = (signature[0], signature[1], cache_slot)
            epoch_batches = cached_window_pool.get(pool_key)
            if epoch_batches is None:
                epoch_batches = build_cached_window_epoch_batches(
                    config,
                    epoch=epoch_i,
                    cache_epoch=1 if fixed_cache else epoch_i,
                    cache_slot=cache_slot,
                    cache_frames=resolve_cached_window_cache_frames(config),
                    condition_providers=active_condition_providers,
                    condition_store_version=condition_store_version,
                    cache_generation=cache_generation,
                    condition_state_getter=condition_state,
                    renderer_cache=renderer_cache,
                )
                cached_window_pool[pool_key] = epoch_batches
                if fixed_cache:
                    epoch_batches.materialize_all()
                    if epoch_i != 1:
                        epoch_batches.set_epoch(epoch_i)
            else:
                epoch_batches.set_epoch(epoch_i)
            return TrainingBatchSequence(
                epoch_batches,
                prefetch_batches=int(config.cached_window_prefetch_batches),
            )

        batches: list[TrainingBatch] = []
        for step_idx in range(int(config.steps_per_epoch)):
            seed = online_item_seed(
                data_seed=int(config.seed),
                epoch=int(epoch),
                item_index=step_idx,
                items_per_epoch=int(config.steps_per_epoch),
            )
            loc_batch = build_native_online_batch(
                config,
                epoch=int(epoch),
                seed=seed,
                step=step_idx + 1,
                global_step=(int(epoch) - 1) * int(config.steps_per_epoch) + step_idx,
                condition_providers=active_condition_providers,
                condition_store_version=condition_store_version,
                cache_generation=cache_generation,
                renderer_cache=renderer_cache,
            )
            batches.append(to_training_batch(loc_batch))
        return batches

    def refresh_physical_model() -> None:
        nonlocal default_condition_providers, cached_window_pool_signature
        cached_window_pool.clear()
        cached_window_pool_signature = None
        if str(config.psf_type).strip().lower() == "ppg_direct":
            default_condition_providers = ((None, default_condition_provider(config)),)
        lifecycle.invalidate()

    provider.refresh_physical_model = refresh_physical_model
    return provider


class TrainingBatchSequence:
    def __init__(self, loc_batches, *, prefetch_batches: int = 0) -> None:
        self._loc_batches = loc_batches
        self.prefetch_batches = max(0, int(prefetch_batches))

    def __len__(self) -> int:
        return len(self._loc_batches)

    def __iter__(self):
        for loc_batch in self._loc_batches:
            yield to_training_batch(loc_batch)

    def __getitem__(self, index: int) -> TrainingBatch:
        return to_training_batch(self._loc_batches[int(index)])
