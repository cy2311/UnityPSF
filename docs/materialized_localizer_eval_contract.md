# Materialized Localizer Eval Contract

## Scope

Slice 6.15 defines a dataset-agnostic materialized localizer eval contract.
Slice 6.16 implements the first supervised `.npz` provider against this
contract.

This contract is intentionally broader than microtube. Microtube is only the
first compatibility fixture because existing v0.3 smoke data and old-route
examples already cover that dataset family. The runtime contract must also
support other materialized samples with different acquisition layouts,
normalization metadata, frame windows, and crop policies.

## Source Names

The preferred future source name is:

```yaml
train:
  eval:
    enabled: true
    source: materialized_dataset
```

`materialized_microtube` may remain as a compatibility fixture or legacy alias,
but it should map onto the same dataset-agnostic contract. New code should not
add microtube-only fields when a generic name can describe the same concept.

Current Slice 6.16 behavior:

- `source=online_generation` remains implemented.
- `source=materialized_dataset` is implemented for supervised `.npz` fixtures.
- `source=materialized_microtube` maps to `materialized_dataset` as a
  compatibility alias when the generic fields are provided.
- Unsupported sources must not silently fall back to online eval.

## Dataset Identity

Each materialized eval source must identify the sample and dataset explicitly:

```yaml
train:
  eval:
    enabled: true
    source: materialized_dataset
    dataset_id: experiment_or_collection_name
    sample_id: sample_or_field_of_view_name
    source_path: relative/or/absolute/path/to/materialized/data
```

Required identity fields:

- `dataset_id`: stable name for a collection, experiment, or acquisition
  family.
- `sample_id`: stable name for the sample, FOV, dish, tissue section, cell
  line, or synthetic materialization.
- `source_path`: path to the materialized frames or an index/manifest that
  resolves them.

Paths should resolve relative to the config file directory, matching the v0.3
runtime config policy used for other file-backed sources.

## Frame And Crop Selection

Materialized localizer eval must define deterministic held-out frame and crop
selection. A minimal generic contract is:

```yaml
train:
  eval:
    source: materialized_dataset
    frame_range: [1000, 1200]
    frame_stride: 1
    channels: 3
    crop:
      mode: fixed_window
      top: 0
      left: 0
      height: 128
      width: 128
```

Frame fields:

- `frame_range`: half-open `[start, stop]` frame window.
- `frame_stride`: optional deterministic stride inside the window.
- `channels`: number of frames per localizer sample.

Crop fields:

- `crop.mode`: explicit crop policy, initially `fixed_window` or
  `roi_records`.
- `crop.top`, `crop.left`, `crop.height`, `crop.width`: required for
  `fixed_window`.
- `crop.records_path`: optional future source for reusable crop records.

The provider must materialize fixed eval batches before training epochs begin,
then reuse the same batches each epoch.

## Heldout Split

The materialized eval route owns a localizer heldout_split, not a gamma update
split. The held-out split must be recorded in manifest/status and must be
deterministic.

```yaml
train:
  eval:
    heldout_split:
      mode: explicit_frame_range
      allow_train_eval_overlap: false
```

Required policy:

- no train/eval overlap by default.
- Any allowed overlap must be explicit with
  `heldout_split.allow_train_eval_overlap: true`.
- If the training provider uses materialized frames, the eval provider must be
  able to compare training and eval frame/crop selections before training
  starts.
- If overlap cannot be proven or rejected, the provider should fail fast rather
  than silently continue.

## Target And Normalization Contract

A materialized localizer eval batch must still produce the same v0.3 training
batch contract consumed by localizer losses:

```text
model_input
detect_tar
bkg_tar
pxyz_tar = x_px, y_px, z, photons
mask_tar
```

The materialized source must record how targets and normalization are obtained:

- `target_source`: explicit source for detections/emitters, such as
  `embedded`, `sidecar`, `simulated`, or `none_for_unsupervised_future_route`.
- `normalization`: camera or source-specific frame normalization policy.
- `units`: frame intensity units and pxyz units after adaptation.

Slice 6.16 implements only a supervised route that can produce targets
compatible with the existing localizer loss from `.npz` arrays. Unsupervised
diagnostic eval and direct TIFF/manifest ingestion can be specified later.

## Boundary With ROI-Bank Gamma Held-Out

Materialized localizer eval and ROI-bank gamma held-out are not the same
contract. They are not the same contract even when both are configured from a
materialized dataset.

Materialized localizer eval:

- Lives under `train.eval`.
- Supplies fixed localizer eval batches to `EvalProvider`.
- Writes `eval_metrics.jsonl`.
- Updates `checkpoint_best.pt`.
- Records `dataset_id`, `sample_id`, `source_path`, `frame_range`, `crop`, and
  `heldout_split`.

ROI-bank gamma held-out:

- Lives under `train.roi_bank_gamma`.
- Monitors gamma update objectives on ROI banks.
- Writes held-out monitor fields to `gamma_update_metrics.jsonl`.
- Uses posterior samples and ROI projection diagnostics.

The presence of `train.roi_bank_gamma.auto_heldout_*` or
`heldout_roi_library_path` must not enable localizer eval. The presence of
`train.eval.source=materialized_dataset` must not run gamma updates.

## Microtube Compatibility Fixture

Microtube should be the first compatibility fixture, not the whole design.

A microtube fixture can map:

- `dataset_id`: experiment or acquisition folder.
- `sample_id`: FOV or sample label.
- `source_path`: TIFF path or materialized frame manifest.
- `frame_range`: held-out TIFF frame window.
- `crop`: the same generic crop structure.
- `normalization`: existing camera calibration and ADU/photon scaling fields.

This mapping should be implemented as a translation into
`source=materialized_dataset`, not as a separate microtube-only provider
surface unless a real incompatibility appears.

## Slice 6.16 Acceptance

Slice 6.16 implements and tests:

- `source=materialized_dataset` produces fixed eval batches.
- Relative `source_path` resolves against the config directory.
- `dataset_id`, `sample_id`, `frame_range`, `crop`, and `heldout_split` are
  recorded in manifest/status.
- The provider rejects ambiguous train/eval overlap by default.
- The microtube compatibility fixture maps onto the generic contract.
- `eval_metrics.jsonl` and `checkpoint_best.pt` still come from the generic
  trainer path.

## Non-Goals

- Slice 6.15 does not implement a provider.
- Slice 6.15 does not add materialized eval runtime behavior.
- Slice 6.15 does not change ROI-bank gamma held-out behavior.
- Slice 6.15 does not change loss, target, optimizer, scheduler, AMP, or gamma
  semantics.
- Slice 6.16 does not implement direct TIFF, HDF5, or manifest ingestion.
- Slice 6.16 does not implement unsupervised materialized diagnostics.
