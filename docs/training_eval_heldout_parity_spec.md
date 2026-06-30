# Training Eval / Held-Out Parity Spec

## Scope

This document started as the Slice 6.12 localizer eval / held-out provider spec.
Slice 6.12 does not implement the provider. Slice 6.13 implements the first
provider for `source=online_generation`. Slice 6.14 hardens that online eval
contract without adding materialized eval runtime. Slice 6.15 adds the
dataset-agnostic materialized eval contract in
`docs/materialized_localizer_eval_contract.md`.

The generic trainer already supports an `EvalProvider`, writes
`eval_metrics.jsonl`, and saves `checkpoint_best.pt`. The missing piece is the
high-fidelity/localization runtime config and entrypoint contract that decides
which fixed eval batches are used for old-route parity. Slice 6.13 wires that
contract for online synthetic eval batches only.

## Boundary With ROI-Bank Gamma Held-Out

Localizer eval / held-out and ROI-bank gamma held-out are not the same
contract. They are not the same contract even when both use the words
held-out.

ROI-bank gamma held-out:

- Lives under `train.roi_bank_gamma`.
- Evaluates gamma-update objectives on selected/held-out ROI banks.
- Writes held-out gamma monitor fields in `gamma_update_metrics.jsonl`.
- Uses posterior samples and ROI-bank projection diagnostics.

Localizer eval / held-out:

- Lives under the training runtime for the localizer.
- Supplies fixed eval batches to the generic `EvalProvider`.
- Writes localizer eval loss to `eval_metrics.jsonl`.
- Updates `checkpoint_best.pt` according to localizer eval loss.
- Must not run gamma updates or ROI-bank projection objectives.

## Current Generic Trainer Contract

The generic trainer already provides the following behavior:

- `EvalProvider` is a callable returning fixed `TrainingBatch` objects.
- `train_epochs(..., eval_provider=...)` evaluates once per epoch.
- `eval_metrics.jsonl` records `epoch`, `global_step`, and `eval_loss`.
- `checkpoint_best.pt` is written when eval loss improves.
- Best checkpoints include model, optimizer, and active scheduler state.

Slice 6.12 keeps this generic behavior unchanged.

## Slice 6.13 Implementation Status

Implemented:

- `train.eval.enabled=true` with `source=online_generation`.
- Independent fixed eval seed.
- Fixed eval batch construction through the native online batch provider.
- `eval_metrics.jsonl` and `checkpoint_best.pt` through the existing generic
  trainer.
- Manifest/status recording for both `run_high_fidelity` and
  `run_localization`.
- Unsupported eval sources fail fast with `train.eval.source must be
  online_generation`.

Not implemented:

- `source=materialized_dataset`.
- Dataset-agnostic held-out frame/crop selection for materialized samples.
- Any coupling between localizer eval and ROI-bank gamma held-out metrics.
- AMP, optimizer, scheduler, loss, gamma, or target semantic changes.

## Slice 6.14 Hardening Status

Implemented:

- `train.eval.batch_count` and `train.eval.batch_size` must be positive.
- Default eval seed remains independent from train seed and defaults to
  `train.online_generation.seed + 100000`.
- Eval batch size defaults through the explicit localizer eval route.
- The online eval provider reuses the same runtime config mapping as online
  training for FiLM/SoftMoE/domain-aware fields.
- SoftMoE/domain-onehot eval batches are covered through active model/loss
  smoke tests.
- Relative `dual_domain_coeff_maps` paths are resolved against the config
  directory before eval batch construction.

Still not implemented:

- Materialized dataset eval frame/crop selection.
- A `source=materialized_dataset` provider.

## Slice 6.15 Materialized Contract Status

Implemented as spec/tests only:

- `docs/materialized_localizer_eval_contract.md` defines a dataset-agnostic
  materialized localizer eval contract.
- The future preferred source is `materialized_dataset`.
- `materialized_microtube` is limited to a compatibility fixture or legacy
  alias mapped onto the same generic contract.
- The contract records `dataset_id`, `sample_id`, `source_path`,
  `frame_range`, `crop`, and `heldout_split`.
- The contract requires no train/eval overlap by default.
- The contract keeps localizer eval separate from ROI-bank gamma held-out.
- Runtime guard coverage verifies materialized sources still fail fast until
  Slice 6.16 wires a provider.

## Slice 6.16 Materialized Provider Status

Implemented:

- `source=materialized_dataset` builds fixed supervised eval batches from a
  dataset-agnostic `.npz` fixture.
- The `.npz` fixture stores `model_input`, `detect_tar`, `bkg_tar`,
  `pxyz_tar`, and `mask_tar`.
- Relative `source_path` resolves against the config directory.
- Manifest/status record `dataset_id`, `sample_id`, `source_path`,
  `frame_range`, `crop`, and `heldout_split`.
- `materialized_microtube` maps to the generic materialized contract as a
  compatibility alias when the generic fields are provided.
- `heldout_split.allow_train_eval_overlap=true` fails fast in the first
  provider implementation.

Still not implemented:

- Direct TIFF/HDF5/manifest ingestion for materialized eval.
- Automatic train/eval overlap comparison against materialized train providers.
- Unsupervised materialized diagnostics.

## Proposed Runtime Config Contract

The next implementation slice should add an explicit localizer eval section,
for example:

```yaml
train:
  eval:
    enabled: true
    source: online_generation
    seed: 123
    batch_count: 2
    batch_size: 1
```

Required fields:

- `enabled`: enables localizer eval batches.
- `source`: first implementation should support `online_generation`.
- `seed`: fixed eval seed independent from train seed.
- `batch_count`: number of fixed eval batches.
- `batch_size`: eval batch size, defaulting to train batch size when omitted.

Materialized support should use `source=materialized_dataset`, because
held-out frame and crop selection has different semantics from online synthetic
eval. `materialized_microtube` may remain a compatibility fixture or alias, but
new behavior should map onto the dataset-agnostic contract. Materialized
localizer eval should not be implied by ROI-bank gamma
`auto_heldout_min_rois` or `heldout_roi_library_path`.

## Source Policies

### `online_generation`

The first implementation should build fixed eval batches from the same online
generation contract as training but with an independent seed and no epoch/step
mutation after construction. The same eval batches must be reused every epoch.

Acceptance criteria:

- Fixed eval batches are deterministic for the same config and seed.
- Eval loss is written to `eval_metrics.jsonl`.
- `checkpoint_best.pt` updates only when eval loss improves.
- Resume preserves `global_step` and continues appending train/eval metrics.

### `materialized_dataset`

Materialized dataset eval is future work. Slice 6.15 specifies the contract in
`docs/materialized_localizer_eval_contract.md`; Slice 6.16 should implement the
first provider.

Acceptance criteria for the future materialized route:

- Held-out frames/crops are selected deterministically.
- Training frames and eval frames do not overlap unless explicitly requested.
- Manifest/status record the eval source and held-out frame/crop selection.
- The contract remains separate from ROI-bank gamma held-out.

## Metrics And Manifest

When localizer eval is configured, high-fidelity manifest/status should record:

- eval enabled/disabled
- eval source
- eval seed
- eval batch count
- eval batch size
- best checkpoint path when available

`eval_metrics.jsonl` remains the localizer eval metrics file. ROI-bank gamma
held-out metrics remain in `gamma_update_metrics.jsonl`.

## Non-Goals

- Slice 6.12 does not implement the provider.
- Slice 6.12 does not change `train_epochs`.
- Slice 6.12 does not change ROI-bank gamma held-out behavior.
- Slice 6.12 does not add materialized microtube held-out frame selection.
- Slice 6.12 does not change optimizer, scheduler, AMP, loss, gamma, or target
  semantics.

## Slice 6.13 Acceptance

Slice 6.13 implements and tests:

- `source=online_generation` produces fixed eval batches.
- `eval_metrics.jsonl` is written by the high-fidelity/localization entrypoint.
- `checkpoint_best.pt` corresponds to the lowest localizer eval loss.
- Manifest/status record the localizer eval contract.
- ROI-bank gamma held-out remains independent from localizer eval.
