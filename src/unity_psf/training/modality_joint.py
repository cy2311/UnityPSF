"""Compatibility facade for shared-expert, multichannel modality training."""

from unity_psf.training.checkpoint_state import (
    build_training_state_snapshot as build_training_state_snapshot,
)
from unity_psf.training.modality_checkpoint import (
    assemble_modality_joint_checkpoint_from_shards,
    commit_modality_joint_checkpoint,
    restore_modality_training_shard,
    save_modality_training_shard,
)
from unity_psf.training.modality_epoch import train_modality_epoch
from unity_psf.training.modality_evaluation import (
    aggregate_heldout_metrics,
    evaluate_modality_heldout,
    normalize_heldout_metrics,
)
from unity_psf.training.modality_physical_state import (
    physical_provenance_registry,
    snapshot_channel_physical_state,
)
from unity_psf.training.modality_types import (
    HeldoutEvalFn as HeldoutEvalFn,
    ModalityBatchProvider as ModalityBatchProvider,
    ModalityChannelStream,
    ModalityEpochResult,
    ModalityHeldoutEvalResult,
    ModalityLossFn as ModalityLossFn,
    ModalityTrainingBatch,
    ModalityTrainingRuntime,
    ModalityTrainingShardState,
    PhysicalStateRestore as PhysicalStateRestore,
    PhysicalStateSnapshot as PhysicalStateSnapshot,
)


_aggregate_heldout_metrics = aggregate_heldout_metrics
_normalize_heldout_metrics = normalize_heldout_metrics
_physical_provenance_registry = physical_provenance_registry
_snapshot_channel_physical_state = snapshot_channel_physical_state


__all__ = [
    "ModalityChannelStream",
    "ModalityEpochResult",
    "ModalityHeldoutEvalResult",
    "ModalityTrainingBatch",
    "ModalityTrainingRuntime",
    "ModalityTrainingShardState",
    "assemble_modality_joint_checkpoint_from_shards",
    "commit_modality_joint_checkpoint",
    "evaluate_modality_heldout",
    "restore_modality_training_shard",
    "save_modality_training_shard",
    "train_modality_epoch",
]
