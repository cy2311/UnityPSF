from __future__ import annotations

from .model import (
    LocalizationModelOutput,
    ProductionLocalizationModel,
    SimpleLocalizationModel,
    build_localization_model_registry,
)
from .conditioning import FullResZernikeConditioning, FullResZernikeStats, build_default_conditioning_maps
from .film import FiLMConditionedDoubleUNet, FiLMModulator, split_conditioned_input
from .losses import ActiveSMLMGMMTargetAdapter, ActiveSMLMGMMLoss, ActiveSMLMLoss
from .materialized_eval import MaterializedDatasetEvalConfig, build_materialized_dataset_eval_provider
from .microtube_tiff import MicrotubeTiffBatchProviderConfig, build_microtube_tiff_batch_provider
from .online import OnlineBatchProviderConfig, build_online_batch_provider
from .posterior import DetectionPosteriorSamples, sample_detection_posterior
from .roi_batches import build_roi_batch_provider
from .runtime_config import build_localization_runtime_config
from .simulator import LocalizationSimulatorConfig, simulate_localization_batch
from .smlm_output import SMLMOutput, SMLMOutputChannels, decode_smlm_output
from .smlm_targets import SMLMTargetConvention, absolute_pxyz_to_local_targets, legacy_iwae_target_process_to_v03, target_pixel_indices
from .smlm_unet import ConvBlock, DoubleUNet, MultiHeads, UNet2d
from .soft_moe import SoftMoEFiLMExperts, domain_index_from_condition
from .synthetic import (
    SyntheticOnlineBatchConfig,
    build_synthetic_localization_batch,
    build_synthetic_online_batch_provider,
)
from .training_adapter import LocalizationTrainBatch, make_localization_loss, to_training_batch

__all__ = [
    "LocalizationTrainBatch",
    "LocalizationSimulatorConfig",
    "LocalizationModelOutput",
    "MicrotubeTiffBatchProviderConfig",
    "MaterializedDatasetEvalConfig",
    "OnlineBatchProviderConfig",
    "ProductionLocalizationModel",
    "ActiveSMLMGMMTargetAdapter",
    "ActiveSMLMGMMLoss",
    "ActiveSMLMLoss",
    "ConvBlock",
    "DoubleUNet",
    "FiLMConditionedDoubleUNet",
    "FiLMModulator",
    "FullResZernikeConditioning",
    "FullResZernikeStats",
    "MultiHeads",
    "SMLMOutputChannels",
    "SMLMTargetConvention",
    "SMLMOutput",
    "SimpleLocalizationModel",
    "SoftMoEFiLMExperts",
    "SyntheticOnlineBatchConfig",
    "UNet2d",
    "DetectionPosteriorSamples",
    "build_localization_model_registry",
    "build_default_conditioning_maps",
    "build_localization_runtime_config",
    "build_materialized_dataset_eval_provider",
    "build_microtube_tiff_batch_provider",
    "build_online_batch_provider",
    "build_roi_batch_provider",
    "build_synthetic_localization_batch",
    "build_synthetic_online_batch_provider",
    "domain_index_from_condition",
    "decode_smlm_output",
    "absolute_pxyz_to_local_targets",
    "legacy_iwae_target_process_to_v03",
    "make_localization_loss",
    "sample_detection_posterior",
    "simulate_localization_batch",
    "split_conditioned_input",
    "target_pixel_indices",
    "to_training_batch",
]
