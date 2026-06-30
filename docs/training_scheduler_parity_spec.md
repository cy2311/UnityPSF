# Training Scheduler Parity Spec

## Scope

This is the Slice 6.4-6.8 contract for localizer training scheduler parity.
It records the old active-route scheduler contract and the v0.3 runtime
mapping for the current StepLR active route.

## Old Route

Reference files:

- `neptune_iwae/neptune_core/localization_runtime/model_setup.py`
- `neptune_iwae/neptune_core/online_train.py`
- `neptune_iwae/neptune_core/cached_window_train.py`
- `neptune_iwae/neptune_core/starter_model.py`
- `neptune_iwae/Normalization/pipeline_density1_online_lut_gridmap2_film_nat_fdstyle_norm_roi128_b24_depthwise_mixture_fdstrict_30kbatches.json`

The active cached-window route uses:

```text
smlm_overrides.lr_scheduler = StepLR
smlm_overrides.lr_step_size = 1000
smlm_overrides.lr_gamma = 0.9
smlm_overrides.lr_step_unit = optimizer_step
```

Old runtime behavior:

- `starter_model.py` maps `smlm_overrides.lr_scheduler` into
  `HyperParameter.learning_rate_scheduler`.
- `starter_model.py` maps `lr_step_size` and `lr_gamma` into
  `HyperParameter.learning_rate_scheduler_param`.
- `model_setup.py` constructs either `torch.optim.lr_scheduler.StepLR` or
  `ReduceLROnPlateau`.
- `online_train.py` and `cached_window_train.py` resolve scheduler step unit:
  - `ReduceLROnPlateau` always steps at epoch end with eval loss when
    available, otherwise train loss.
  - StepLR defaults to epoch stepping unless `smlm_overrides.lr_step_unit`
    is one of `optimizer_step`, `step`, `batch`, `iteration`, or `iter`.
  - The active route sets `optimizer_step`, so StepLR steps after every
    optimizer update.

## Current v0.3 Runtime

v0.3 records scheduler runtime state in:

```text
resolved_contract.training_runtime.scheduler
```

For the active microtube route, the contract is:

```json
{
  "name": "StepLR",
  "active": true,
  "step_unit": "optimizer_step",
  "params": {"step_size": 1000, "gamma": 0.9},
  "inactive_reason": null,
  "legacy_source": "smlm_overrides"
}
```

The generic trainer now builds `torch.optim.lr_scheduler.StepLR` for active
contracts with `step_unit=optimizer_step`. It calls `scheduler.step()` after
each `optimizer.step()` and persists `scheduler_state_dict` in epoch and best
checkpoints. `load_training_checkpoint` restores scheduler state when a
scheduler is supplied. If a caller supplies a scheduler while loading a
checkpoint that does not contain `scheduler_state_dict`, loading fails fast
before mutating model or optimizer state instead of silently continuing with a
reset scheduler.

When no scheduler is configured, the contract is:

```json
{
  "name": "none",
  "active": false,
  "step_unit": null,
  "params": {},
  "inactive_reason": "not_configured"
}
```

## Non-Goals

- Slice 6.8 does not implement epoch-step schedulers.
- Slice 6.8 does not implement ReduceLROnPlateau.
- Slice 6.8 does not change optimizer choice.
- Slice 6.8 does not implement AMP.
- Slice 6.8 does not change loss or gamma semantics.

## Future Acceptance

Remaining scheduler work should be separate and test:

- Epoch-step schedulers step only once per epoch.
- ReduceLROnPlateau uses eval loss when present and train loss otherwise.
- Any future scheduler variants compose with optimizer checkpoint/resume.
