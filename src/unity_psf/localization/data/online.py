"""Compatibility facade; canonical implementations never import this module."""

from unity_psf.field_origin import build_sliding_window_origin_bank as build_sliding_window_origin_bank
from unity_psf.localization.conditioning import _load_conditioning_providers as _load_conditioning_providers
from unity_psf.localization.simulator import _build_emitter_renderer as _build_emitter_renderer
from unity_psf.localization.simulator import _max_emitters_per_sample as _max_emitters_per_sample
from unity_psf.localization.simulator import _sample_counts as _sample_counts
from unity_psf.localization.simulator import _sample_photons as _sample_photons
from unity_psf.localization.simulator import _sample_range as _sample_range
from .online_types import OnlineBatchProviderConfig as OnlineBatchProviderConfig
from .online_provider import build_online_batch_provider as build_online_batch_provider
from .online_physical import VectorRendererCache as _VectorRendererCache
from .online_physical import renderer_cache_key as _renderer_cache_key
from .online_physical import PhysicalVersionedRendererLifecycle as _PhysicalVersionedRendererLifecycle
from .online_native import build_native_online_batch as _build_native_online_batch
from .online_conditioning import attach_film_conditioning as _attach_film_conditioning
from .online_provider import TrainingBatchSequence as _TrainingBatchSequence
from .online_cached import CachedWindowEpochBatches as _CachedWindowEpochBatches
from .online_cached import resolve_cached_window_order as _resolve_cached_window_order
from .online_cached import resolve_cached_window_max_cached_sequences as _resolve_cached_window_max_cached_sequences
from .online_cached_payload import materialize_cached_sequence_payload_cpu as _materialize_cached_sequence_payload_cpu
from .online_cached import sequence_with_batch_metric_visibility as _sequence_with_batch_metric_visibility
from .online_cached import build_cached_window_epoch_batches as _build_cached_window_epoch_batches
from .online_cached_payload import precompute_cached_sequence_payload as _precompute_cached_sequence_payload
from .online_native import simulate_native_sequence as _simulate_native_sequence
from .online_native import simulate_native_sequence_batched as _simulate_native_sequence_batched
from .online_cached_window import slice_cached_window_batch as _slice_cached_window_batch
from .online_conditioning import attach_cached_window_film_conditioning as _attach_cached_window_film_conditioning
from .online_conditioning import attach_cached_window_precomputed_film_conditioning as _attach_cached_window_precomputed_film_conditioning
from .online_physical import simulator_config as _simulator_config
from .online_native import build_sequence_window_batch as _build_sequence_window_batch
from .online_camera import poisson_camera_readout as _poisson_camera_readout
from .online_geometry import cached_sequence_condition_origin as _cached_sequence_condition_origin
from .online_geometry import sliding_window_sequence_condition_origin as _sliding_window_sequence_condition_origin
from .online_geometry import random_sequence_condition_origin as _random_sequence_condition_origin
from .online_conditioning import build_condition_features as _build_condition_features
from .online_conditioning import select_condition_provider as _condition_provider
from .online_conditioning import conditioning_profile_source as _conditioning_profile_source
from .online_conditioning import default_condition_provider as _default_condition_provider
from .online_conditioning import load_condition_providers as _load_condition_providers
from .online_geometry import condition_roi_extent as _condition_roi_extent
from .online_geometry import grid_axis_origin as _grid_axis_origin
from .online_geometry import condition_origin as _condition_origin
from .online_conditioning import condition_feature_dim as _condition_feature_dim
from .online_conditioning import condition_feature_order_for_config as _condition_feature_order_for_config
from .online_conditioning import condition_vector_for_config as _condition_vector_for_config
from .online_conditioning import domain_index as _domain_index
from .online_conditioning import match_feature_dim as _match_feature_dim
from .online_targets import detect_from_v03_targets as _detect_from_v03_targets
from .online_targets import finalize_pxyz_targets as _finalize_pxyz_targets
from .online_targets import pad_pxyz_targets as _pad_pxyz_targets
from .online_targets import pxyz_target_order_tuple as _pxyz_target_order_tuple
from .online_targets import target_contract_for_config as _target_contract_for_config
from .online_validation import cached_window_precompute_enabled as _cached_window_precompute_enabled
from .online_validation import online_item_seed as _online_item_seed
from .online_validation import resolve_cached_window_cache_epochs as _resolve_cached_window_cache_epochs
from .online_validation import resolve_cached_window_cache_frames as _resolve_cached_window_cache_frames
from .online_validation import resolve_cached_window_cache_storage as _resolve_cached_window_cache_storage
from .online_validation import resolve_field_origin_sampling_mode as _resolve_field_origin_sampling_mode
from .online_validation import validate_online_simulation_modes as _validate_online_simulation_modes

__all__ = [
    '_load_conditioning_providers',
    '_build_emitter_renderer',
    '_max_emitters_per_sample',
    '_sample_counts',
    '_sample_photons',
    '_sample_range',
    'build_sliding_window_origin_bank',
    'OnlineBatchProviderConfig',
    'build_online_batch_provider',
    '_VectorRendererCache',
    '_renderer_cache_key',
    '_PhysicalVersionedRendererLifecycle',
    '_build_native_online_batch',
    '_attach_film_conditioning',
    '_TrainingBatchSequence',
    '_CachedWindowEpochBatches',
    '_resolve_cached_window_order',
    '_resolve_cached_window_max_cached_sequences',
    '_materialize_cached_sequence_payload_cpu',
    '_sequence_with_batch_metric_visibility',
    '_build_cached_window_epoch_batches',
    '_precompute_cached_sequence_payload',
    '_simulate_native_sequence',
    '_simulate_native_sequence_batched',
    '_slice_cached_window_batch',
    '_attach_cached_window_film_conditioning',
    '_attach_cached_window_precomputed_film_conditioning',
    '_simulator_config',
    '_build_sequence_window_batch',
    '_poisson_camera_readout',
    '_cached_sequence_condition_origin',
    '_sliding_window_sequence_condition_origin',
    '_random_sequence_condition_origin',
    '_build_condition_features',
    '_condition_provider',
    '_conditioning_profile_source',
    '_default_condition_provider',
    '_load_condition_providers',
    '_condition_roi_extent',
    '_grid_axis_origin',
    '_condition_origin',
    '_condition_feature_dim',
    '_condition_feature_order_for_config',
    '_condition_vector_for_config',
    '_domain_index',
    '_match_feature_dim',
    '_detect_from_v03_targets',
    '_finalize_pxyz_targets',
    '_pad_pxyz_targets',
    '_pxyz_target_order_tuple',
    '_target_contract_for_config',
    '_cached_window_precompute_enabled',
    '_online_item_seed',
    '_resolve_cached_window_cache_epochs',
    '_resolve_cached_window_cache_frames',
    '_resolve_cached_window_cache_storage',
    '_resolve_field_origin_sampling_mode',
    '_validate_online_simulation_modes',
]
