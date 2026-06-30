# Baseline 3052: Microtube ROI-Bank Gamma Loop

## Status

Frozen reference baseline, 2026-06-18.

This baseline records the latest completed `neptune_iwae` formal run that
`neptune_v0.3` should preserve semantically while rebuilding the implementation
cleanly. It is a reference for behavior and artifacts, not a source tree to copy
wholesale.

## Source Run

Run directory:

```text
neptune_iwae/output/microtube_real_tiff_switch_train_anchor99_roi_gamma_long_default_interval5_steps100_epoch300_3052
```

Primary source files:

```text
summary.json
metrics.jsonl
checkpoint_latest.pt
checkpoint_best.pt
nat_wake_latest.pt
```

Final feedback coefficient maps:

```text
roi_bank_gamma_interval5_lr025_steps100_epoch300/epoch_0300/feedback/left/coeff_maps.npz
roi_bank_gamma_interval5_lr025_steps100_epoch300/epoch_0300/feedback/right/coeff_maps.npz
```

## Training Contract

The run completed the active conservative route:

```text
raw TIFF
  -> peak/ZMap bootstrap
  -> localizer training with current gamma feedback
  -> scheduled fixed-ROI gamma updates
  -> detached posterior sampling from the current localizer
  -> gamma-only Poisson ROI objective
  -> feedback coefficient maps for later epochs
```

Run summary:

| Field | Value |
| --- | ---: |
| epochs | 300 |
| batch size | 24 |
| center samples per epoch | 10008 |
| sequence count | 64 |
| global steps completed | 125100 |

Gamma-update schedule:

| Field | Value |
| --- | ---: |
| enabled | true |
| start epoch | 30 |
| update interval | 5 epochs |
| stop epoch | 300 |
| objective | `importance_wake` |
| gamma steps | 100 |
| gamma learning rate | 0.025 |
| posterior samples | 25 |
| target projected emitters | 5000 |
| max sampling rounds | 20 |
| fixed ROI library | true |
| ROI source | `loc_infer_raw_tiff` |
| candidate mode | `dense_tile_temporal` |
| ROI frame range | `[0, 100]` |
| ROI grid shape | `[4, 4]` |
| ROI size | 128 px |
| over-cut crop | 8 px |
| background smoothing | 9 x 9 |
| auto held-out ROI count | 20 per domain |

The default route keeps `joint_training.real_loc_enabled=false`: localizer
parameters are not updated by the real-data gamma objective.

## Data And Domains

Raw TIFF reference:

```text
neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif
```

Domain crops:

| Domain | Crop left | Crop top | Width | Height |
| --- | ---: | ---: | ---: | ---: |
| left | 0 | 0 | 600 | 1200 |
| right | 600 | 0 | 600 | 1200 |

Initial ZMap references:

```text
neptune_iwae/test_data/microtube/zmap/left/alternating_full_roi_zernike_maps_nm.npz
neptune_iwae/test_data/microtube/zmap/right/alternating_full_roi_zernike_maps_nm.npz
```

## Frozen Diagnostics

### Initial vs Epoch-300 ZMap

Diagnostic directory:

```text
neptune_iwae/output/3052_initial_vs_epoch300_zmap
```

Physical coefficient-map changes:

| Domain | Delta abs mean | Delta abs max | Dominant delta mode |
| --- | ---: | ---: | --- |
| left | 23.3555 nm | 174.4230 nm | `[2, 2]` |
| right | 23.0473 nm | 172.6435 nm | `[2, 2]` |

The before/after comparison uses the peak-bootstrap base map plus the final
ROI-bank delta feedback map.

### Raw TIFF Patch Reconstruction

Diagnostic directory:

```text
neptune_iwae/output/3052_epoch300_raw_tiff_patch_recon_gpu
```

| Domain | Harvest accepted | Selected patches | Initial loss | Final loss | Poisson NLL mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| left | 155379 | 13 | 112.9516 | 55.9580 | 55.5442 |
| right | 168424 | 5 | 115.6050 | 57.2261 | 56.8014 |

These patch diagnostics are sanity evidence only. They are not a substitute for
the fixed held-out ROI-bank monitor.

### Epoch-300 ROI128 Reconstruction, First Frames

Diagnostic directory:

```text
neptune_iwae/output/3052_epoch300_raw_tiff_roi128_recon_gpu_first5
```

ROI-bank summary:

| Field | Value |
| --- | ---: |
| selected ROI count | 300 |
| selected emitter count | 21201 |
| candidate count | 300 |
| target emitters reached | true |

Per-domain reconstruction:

| Domain | ROI count | Rendered count | Initial NLL | Epoch-300 NLL | Initial RMS | Final RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| left | 150 | 3 | 22.9451 | 22.4735 | 34.7146 nm | 23.1678 nm |
| right | 150 | 3 | 35.0833 | 32.2437 | 34.7486 nm | 23.6114 nm |

### Epoch-300 ROI128 Reconstruction, Frames 100-110

Diagnostic directory:

```text
neptune_iwae/output/3052_epoch300_raw_tiff_roi128_recon_gpu_frames100_110_roi5
```

ROI-bank summary:

| Field | Value |
| --- | ---: |
| selected ROI count | 800 |
| selected emitter count | 52773 |
| candidate count | 800 |
| target emitters reached | true |

Per-domain reconstruction:

| Domain | ROI count | Rendered count | Initial NLL | Epoch-300 NLL | Initial RMS | Final RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| left | 400 | 5 | 22.5777 | 21.4888 | 34.7146 nm | 23.1678 nm |
| right | 400 | 5 | 41.9441 | 38.7975 | 34.7486 nm | 23.6114 nm |

## What v0.3 Must Preserve

- The active config semantics above, especially epoch 30 start, 5-epoch update
  interval, 100 gamma steps, 25 posterior samples, ROI128, and 8 px over-cut.
- Fixed ROI-library geometry/raw crops for a run, with current-localizer
  posterior resampling during gamma updates.
- Detached localizer posterior samples and detached smoothed background for the
  default gamma objective.
- Per-domain independent gamma/ZMap feedback for left and right domains.
- Compact monitor summaries and explicit artifact paths rather than hidden
  runtime state.

## What v0.3 Should Not Copy

- The large `cached_window_train.py` orchestration structure.
- Dynamic import fallbacks and script-path patching from `neptune_iwae`.
- Historical config/SLURM copies that are not part of this active route.
- Runtime output directories as source artifacts.
- Direct real-data DReG localizer updates in the default route.

## Migration Use

Use this baseline to evaluate each v0.3 migration slice:

1. Config materialization must reproduce the schedule contract.
2. `gamma_update` tests must preserve over-cut, background smoothing,
   posterior-sample semantics, and gamma-only gradients.
3. Training-loop work must call a narrow gamma-update API rather than importing
   old trainer internals.
4. Reconstruction diagnostics should report before/after values in the same
   terms as this baseline, even if implementation details differ.
