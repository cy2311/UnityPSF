"""Compatibility exports for the generic training API."""

from unity_psf.training.loop_types import (
    BatchEndHook as BatchEndHook,
    BatchProvider as BatchProvider,
    CheckpointExtraFn as CheckpointExtraFn,
    EpochEndHook as EpochEndHook,
    EpochTrainingConfig as EpochTrainingConfig,
    EvalProvider as EvalProvider,
    LossFn as LossFn,
    ModelInitializationState as ModelInitializationState,
    TrainingBatch as TrainingBatch,
    TrainingConfig as TrainingConfig,
    TrainingEpochResult as TrainingEpochResult,
    TrainingResumeState as TrainingResumeState,
    TrainingRunEpochResult as TrainingRunEpochResult,
    TrainingStepResult as TrainingStepResult,
)
from unity_psf.training.epoch_loop import train_one_epoch as train_one_epoch
from unity_psf.training.training_run import train_epochs as train_epochs
from unity_psf.training.resume import (
    initialize_model_from_checkpoint as initialize_model_from_checkpoint,
    load_training_checkpoint as load_training_checkpoint,
    resume_training_checkpoint as resume_training_checkpoint,
)
from unity_psf.training.evaluation import evaluate as evaluate
from unity_psf.training.checkpoint_io import existing_best_eval_loss as _existing_best_eval_loss

__all__ = [
    "BatchEndHook", "BatchProvider", "CheckpointExtraFn", "EpochEndHook", "EpochTrainingConfig",
    "EvalProvider", "LossFn", "ModelInitializationState", "TrainingBatch", "TrainingConfig",
    "TrainingEpochResult", "TrainingResumeState", "TrainingRunEpochResult", "TrainingStepResult",
    "train_one_epoch", "train_epochs", "initialize_model_from_checkpoint", "load_training_checkpoint",
    "resume_training_checkpoint", "evaluate", "_existing_best_eval_loss",
]
