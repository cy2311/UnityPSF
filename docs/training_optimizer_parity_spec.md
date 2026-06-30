# Training Optimizer Parity Spec

## Scope

This is the Slice 6.5-6.7 contract for localizer optimizer parity. It records
the old active-route optimizer intent, the current v0.3 contract mapping, and
the AdamW runtime support added in Slice 6.6-6.7.

## Old Route

Reference files:

- `neptune_iwae/smlm_v2a/utils/reference_files/param_reference.yaml`
- `neptune_iwae/neptune_core/localization_runtime/model_setup.py`
- `neptune_iwae/neptune_core/starter_model.py`

The old training parameter template defines:

```text
HyperParameter.optimizer = AdamW
HyperParameter.opt_param.lr = 0.0006
HyperParameter.opt_param.weight_decay = 0.1
```

Old runtime behavior:

- `model_setup.py` supports only `Adam` and `AdamW` for this route.
- Optimizer LR priority is:
  1. `train.online_generation.optimizer_lr`
  2. `train.learning_rate`
  3. `HyperParameter.opt_param.lr`
- Weight decay priority is:
  1. `train.online_generation.weight_decay`
  2. `HyperParameter.opt_param.weight_decay`
- `betas` and `eps` are passed through only when present on
  `HyperParameter.opt_param`.

## Current v0.3 Contract

v0.3 records the actual runtime optimizer in:

```text
resolved_contract.training_runtime.optimizer
```

The current v0.3 high-fidelity runtime config resolves `smlm_overrides.optimizer`
when the caller does not explicitly request an optimizer. For the materialized
microtube active route this selects:

```json
{
  "name": "adamw",
  "params": {"lr": 0.002, "weight_decay": 0.1}
}
```

The `lr=0.002` value in the high-fidelity fixture comes from
`train.online_generation.optimizer_lr`, which has higher priority than
`train.learning_rate` and `smlm_overrides.optimizer_lr`. Explicit runtime
optimizer specs still take precedence, so tests can intentionally keep an SGD
runtime while recording the legacy AdamW target separately.

v0.3 records the old target optimizer separately in:

```text
resolved_contract.training_runtime.legacy_optimizer
```

For the active microtube route, the legacy contract is:

```json
{
  "name": "AdamW",
  "params": {"lr": 0.002, "weight_decay": 0.1},
  "active": true,
  "inactive_reason": null,
  "legacy_source": "smlm_overrides"
}
```

When `train.online_generation.optimizer_lr` is absent but `train.learning_rate`
is provided, the legacy LR follows old priority and uses `train.learning_rate`.
When both are present, `train.online_generation.optimizer_lr` wins.

## Non-Goals

- Slice 6.7 does not add Adam runtime support.
- Slice 6.7 does not change scheduler behavior.
- Slice 6.7 does not change checkpoint schema.
- Slice 6.7 does not change gamma optimizer behavior.
- Slice 6.7 does not change loss or gamma semantics.

## Implemented Runtime

The generic v0.3 trainer can now instantiate `torch.optim.AdamW` when the
runtime optimizer spec uses `adamw` or `AdamW`. The materialized microtube
high-fidelity route now emits `adamw` from legacy `smlm_overrides.optimizer`
when no explicit optimizer is provided. Optimizer state is already persisted
through the existing checkpoint helper because checkpoints save
`optimizer.state_dict()`.

## Future Acceptance

The remaining optimizer-adjacent parity work should be separate and test:

- Scheduler implementation composes with the selected optimizer.
- Any future Adam support is explicit and does not disturb the active AdamW
  route.
