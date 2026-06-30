# Training AMP Parity Spec

## Scope

This records the AMP policy and implementation state for localizer training
runtime parity. v0.3 now interprets old-route AMP intent and wires the CUDA
runtime path with `autocast` and `GradScaler`.

Slice 6.10 created the original AMP policy/spec; later work wired the runtime
behavior described here.

## Current Contract

v0.3 records AMP intent in:

```text
resolved_contract.training_runtime.amp
```

The current materialized active route records:

```json
{
  "configured": true,
  "dtype": "float16",
  "active": true,
  "inactive_reason": null
}
```

When AMP is not configured, v0.3 records:

```json
{
  "configured": false,
  "dtype": null,
  "active": false,
  "inactive_reason": "not_configured"
}
```

## Runtime Policy

AMP runtime execution is CUDA-only. The resolved contract records old-route
intent with `active=true` when AMP is configured; the training loop enables
actual autocast/scaler behavior only when the model is on CUDA. CPU execution
therefore remains plain FP32 even if a configured contract is parsed.

CUDA behavior:

- If `resolved_contract.training_runtime.amp.configured=true` and CUDA is
  available, use `torch.amp.autocast` around forward/loss computation.
- Use `torch.amp.GradScaler` for `dtype=float16`.
- Accept `dtype=float16` as the first supported CUDA AMP dtype.
- Treat unsupported AMP dtypes as configuration errors when runtime AMP is
  active.
- On CPU, accept the same contract but do not create a scaler or write scaler
  state.

Checkpoint behavior:

- When AMP runtime is active, checkpoint payloads must include
  `scaler_state_dict`.
- When AMP runtime is inactive, checkpoint payloads should not include a
  placeholder scaler state.

## Testing Strategy

Required CPU tests:

- CPU smoke accepts configured AMP without CUDA scaler side effects.
- No `scaler_state_dict` is written for inactive AMP runs.
- Active AMP contracts are parsed by the generic trainer and propagated into
  `EpochTrainingConfig`.

CUDA tests:

- CUDA AMP active route writes `active=true`.
- CUDA AMP active route writes `scaler_state_dict`.
- CUDA AMP active route composes with optimizer, scheduler, and global step
  semantics.

CUDA tests should be skipped when CUDA is unavailable. CPU tests must remain
the default CI/smoke path.

## Slice 6.11a AMP inactive runtime guard

Slice 6.11a implemented the temporary CPU inactive guard before CUDA runtime
wiring:

- `build_trainer_runtime` rejected `resolved_contract.training_runtime.amp`
  when `active=true`.
- The materialized microtube high-fidelity CPU smoke verified inactive AMP
  checkpoints did not contain `scaler_state_dict`.

## Slice 6.11b Generic Inactive AMP Coverage

Slice 6.11b broadened inactive AMP guard coverage before runtime wiring:

- Generic `build_trainer_runtime` accepts `configured=true, active=false` AMP
  contracts.
- Generic training checkpoints for inactive AMP runs do not contain
  `scaler_state_dict`.
- Active AMP contracts failed fast until runtime AMP was wired.

## Runtime Wiring

The CUDA AMP runtime path now tests:

- `autocast` wraps forward/loss computation only when AMP is active.
- `GradScaler` scales loss, unscales before grad clipping, steps optimizer,
  updates scaler state, and composes with StepLR ordering.
- `scaler_state_dict` is saved with checkpoints on CUDA.
- CPU configured AMP remains plain FP32 unless a later CPU AMP policy is
  explicitly accepted.
- This does not change loss or gamma semantics.
