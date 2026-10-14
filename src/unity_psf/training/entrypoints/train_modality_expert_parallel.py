"""Compatibility entrypoint for single-owner modality Expert Parallel training.

CLI dispatch, rank coordination, worker execution, release publication and
filesystem artifacts each have one implementation owner in sibling modules.
Legacy helper imports remain available for existing callers and focused tests.
"""

from .canonical_runtime import (
    build_modality_runtime as _build_modality_runtime,
    instance_specs as _instance_specs,
    load_joint_config as _load_joint_config,
    load_resolved_training_contract as _load_resolved_training_contract,
    modality_groups as _modality_groups,
    sha256 as _sha256,
    visual_record as _visual_record,
)
from unity_psf.training.modality_progress import (
    accumulate_epoch as _accumulate_epoch,
    accumulate_heldout as _accumulate_heldout,
    channel_progress as _channel_progress,
    empty_metrics as _empty_metrics,
    heldout_eval_enabled as _heldout_eval_enabled,
    metrics_from_progress as _metrics_from_progress,
)

from .expert_parallel_artifacts import (
    atomic_torch_save as _atomic_torch_save,
    rank_status_path as _rank_status_path,
    shard_path as _shard_path,
    validation_batch_path as _validation_batch_path,
)
from .expert_parallel_cli import (
    attempt_id as _attempt_id,
    distributed_identity as _distributed_identity,
    main as main,
    parser as _parser,
)
from .expert_parallel_coordination import read_completed_rank_statuses as _read_completed_rank_statuses
from .expert_parallel_release import (
    VISIBLE_HELDOUT_METRICS as _VISIBLE_HELDOUT_METRICS,
    publish_joint_release as _publish_joint_release,
)
from .expert_parallel_worker import train_owned_modality as _train_owned_modality


__all__ = [
    "main",
    "_build_modality_runtime",
    "_instance_specs",
    "_load_joint_config",
    "_load_resolved_training_contract",
    "_modality_groups",
    "_sha256",
    "_visual_record",
    "_accumulate_epoch",
    "_accumulate_heldout",
    "_atomic_torch_save",
    "_attempt_id",
    "_channel_progress",
    "_distributed_identity",
    "_empty_metrics",
    "_heldout_eval_enabled",
    "_metrics_from_progress",
    "_parser",
    "_publish_joint_release",
    "_rank_status_path",
    "_read_completed_rank_statuses",
    "_shard_path",
    "_train_owned_modality",
    "_validation_batch_path",
    "_VISIBLE_HELDOUT_METRICS",
]


if __name__ == "__main__":
    raise SystemExit(main())
