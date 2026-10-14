# Training core convergence

## CORE-01: Shared optimizer step

Both `training.loop.train_one_epoch` and
`training.modality_joint.train_modality_epoch` use
`training.optimizer_step.run_optimizer_step` for zeroing gradients, autocast,
scalar/finite loss validation, backward, AMP unscaling, gradient clipping,
optimizer updates and AMP skipped-step detection.

The result contains the detached loss and whether an optimizer update actually
occurred. Channel routing, scheduler advancement, attempted/successful step
counts, hooks and checkpoint publication remain owned by the outer loops.
The generic loop retains its profiling blocks; the modality loop does not add
profiling events to the shared timing accumulator.

Scaler creation and cross-epoch ownership remain outside the step helper.
Existing generic and modality initial-scale policies and supplied/restored
scalers are preserved; unifying initialization defaults would change training
policy and is not part of this extraction.

The generic loop now rejects non-tensor and non-scalar losses with `ValueError`
before backward, matching the modality loop's validation contract. Non-finite
losses continue to raise `FloatingPointError`.

## CORE-02: Scheduler and optimizer progress

Both loops use `OptimizerStepProgress` to accumulate attempted, successful and
skipped updates and `advance_scheduler` for the advancement policy:

- An optimizer-step scheduler advances only after a successful update.
- An epoch scheduler advances once after an epoch with at least one successful
  update. An all-skipped epoch does not advance it.
- `train_one_epoch` owns epoch-scheduler advancement before writing its latest
  checkpoint. `train_epochs` no longer advances it or rewrites that checkpoint.

Scheduler units are validated when generic and modality configurations are
constructed, before provider consumption or mutation of model, optimizer and
scaler state. The shared advancement helper also validates its policy.

The all-skipped epoch policy follows the existing modality loop. This fixes
the generic loop's previous unconditional epoch advancement. Direct callers of
`train_one_epoch` now also receive a finalized epoch scheduler state.

Generic epoch results expose `optimizer_steps`, `skipped_optimizer_steps` and
`attempted_optimizer_steps`; batch hooks expose `optimizer_step_performed`.
These are per-epoch counts, not cumulative resume identities. Existing
`step_count`, `global_step` and `max_batches` still count attempted batches,
including AMP skips. Checkpoint schemas and persisted progress fields are
unchanged; shared checkpoint snapshots are covered by CORE-03 below.

Regression tests exercise both scheduler modes across generic single-epoch,
multi-epoch and modality loops, including all/partial/no skips, channel counts,
batch and epoch hooks, single latest publication, checkpoint scheduler state,
interrupted resume equivalence and attempted-batch budgets.

## CORE-03: Shared training-state snapshots

`training.checkpoint_state.build_training_state_snapshot` owns the common
inventory: model, optimizer, scheduler, enabled AMP scaler, RNG and supplied
progress. Generic training and modality resume writers consume this builder.
RNG capture/restore also live in this module; modality training no longer
imports private RNG helpers from the generic loop.

The adapters preserve existing file formats:

- Generic legacy/v2 files retain `model_state_dict`, `optimizer_state_dict`,
  optional `scheduler_state_dict` and optional `scaler_state_dict`.
- Joint resume files put model weights in `experts`, and optimizer, scheduler,
  scaler, RNG and modality progress in `training_state`.
- Release files never construct a training snapshot or carry training state.

Two saving policies are intentionally corrected: generic legacy training files
now include RNG, and direct joint resume commits capture RNG when callers do
not supply it. Explicit historical RNG supplied in progress is preserved.
Absent or disabled scalers normalize to `None` in the shared snapshot; generic
files omit the scaler field, while joint resume files retain `amp_scaler=None`.
Existing readers continue to accept historical RNG/scaler representations.

Checkpoint extension callbacks preserve the previous ordering: RNG is captured
before the callback runs. Progress keys that collide with canonical state keys
are rejected explicitly instead of being silently overwritten.

Progress is copied independently from callers. Component state dictionaries
remain synchronous serialization inputs, not asynchronous frozen tensor
copies; the builder does not duplicate model/optimizer tensors on GPU. Channel
physical state remains owned by its existing callbacks and adapters. This work
does not make multi-artifact physical publication transactional or change
resume validation ordering; those remain separate work.

Tests cover generic legacy/v2 and joint/shard inventories, enabled/disabled
scalers, RNG round trips for Python/NumPy/Torch, historical NumPy tuples,
explicit RNG preservation, copied progress and release-state isolation.

## Remaining stages

CORE-04 resume preflight is implemented by the canonical resolved-contract,
provenance, checkpoint-role, runtime-identity, and state-compatibility checks
used before formal resume state application. It remains covered by the
selection, contract, checkpoint, and resume-equivalence tests.

## CORE-05: Shared evaluation mode restoration

`training.evaluation_mode.preserve_module_training_modes` owns temporary
evaluation transitions for generic evaluation, modality held-out evaluation,
and high-fidelity ROI inference. It snapshots every nested module's original
`training` flag, switches the complete tree to evaluation mode, and restores
all flags in a `finally` block. This preserves intentionally mixed states such
as a training-mode parent with an evaluation-mode BatchNorm child and also
restores state when evaluation raises.

Restoration assigns each module's saved flag directly, rather than calling
recursive `train()` methods. This also preserves shared submodules with mixed
parent modes. Entering evaluation still calls `model.eval()`.

Regression tests cover nested and shared mixed modes, exception-safe
restoration, generic evaluation, modality callbacks and ROI inference.

CORE-05 is complete. The remaining work is CUDA release validation and any
optional package-resource cleanup; neither is a training-state correctness
refactor.

Each stage requires separate regression verification before proceeding. This
stage does not change global-step identity, checkpoint schemas or physical
feedback behavior.

## Verification

Targeted tests cover both loop validation paths, CPU AMP updates and overflow
skips, clipping after unscaling, scheduler progress and the shared checkpoint
inventory. Existing modality tests also cover scaler reuse across epochs and
checkpoint restoration. CPU GradScaler tests do not establish CUDA kernel or
GPU autocast correctness.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q \
  tests/training/test_optimizer_step.py \
  tests/training/test_scheduler_progress.py \
  tests/training/test_checkpoint_state.py \
  tests/training/test_loop_amp_guards.py \
  tests/training/test_modality_joint_training.py \
  -o cache_dir=.local/cache/pytest
```

CORE-02 targeted verification on 2026-10-12: 64 passed, including instance
checkpoint/resume tests. CORE-03 targeted verification: 67 passed, including generic legacy/v2 and
joint/shard checkpoint inventory tests plus callback-order and reserved-field
regressions. Ruff checks the
full source/test tree; the existing mypy gate still covers only its six
configured modules. No GPU job was run.
Final CORE-03 full-suite verification: 339 passed, 1 skipped in 68.52 seconds
(exit 0).
Ruff and the configured mypy gate both exited 0. Independent review found a
late policy-validation issue; regression tests and constructor validation
resolved it, and the follow-up review found no remaining blockers.
