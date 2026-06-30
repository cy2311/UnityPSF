# Active Pipeline: Microtube ROI-Bank Gamma Loop

## Current Scope

The first supported workflow is the microtube raw-TIFF ROI-bank gamma loop. It
is reconstructed from the current active `neptune_iwae` configuration:

`neptune_iwae/config/microtube_real_tiff_switch_train_anchor99_roi_gamma_long.json`

In v0.3, the equivalent resolved config is:

```bash
python -m pip install -e ".[dev]"
python -m neptune_v03.config.materialize \
  --base configs/microtube_base.yaml \
  --override configs/overrides/gamma_start50_interval10.yaml \
  --output output/resolved_microtube_roi_gamma_default.yaml
```

The override filename is currently kept for compatibility with the initial
scaffold. Its contents now represent the active default: start at epoch 30,
update every 5 epochs, use 128 x 128 ROI windows, 25 posterior samples, and
100 gamma steps.

The frozen reference run for this route is documented in
[`baselines/3052_microtube_roi_gamma.md`](baselines/3052_microtube_roi_gamma.md).
Use it as the behavioral baseline for v0.3 migration, not as code to copy
wholesale from `neptune_iwae`.

The current training/localization implementation checkpoint and next-task plan
are tracked in
[`training_localization_plan.md`](training_localization_plan.md).

## Data Source

The active raw TIFF path is:

```text
neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif
```

The two domain crops are configured as:

| Domain | Crop left | Crop top | Width | Height |
| --- | ---: | ---: | ---: | ---: |
| `left` | 0 | 0 | 600 | 1200 |
| `right` | 600 | 0 | 600 | 1200 |

v0.3 keeps those paths as references while the code is being rebuilt. The
future data layer should make this an explicit dataset input rather than a
hard-coded dependency on `neptune_iwae`.

## Loop Contract

The intended training loop is:

```text
raw TIFF
  -> peak bootstrap and gamma initialization
  -> train localizer with current gamma feedback
  -> at scheduled epochs, use current localizer on the fixed ROI library
  -> sample detached emitter posterior values per ROI
  -> project sampled emitters with current gamma
  -> optimize gamma coefficients only by ROI wake objective
  -> export feedback maps for later localizer epochs
```

The localizer is not updated by the gamma objective in the conservative route.
Gamma updates use detached localizer posterior samples and detached background.
Per-step raw/NAT DReG localizer updates are disabled by default through
`joint_training.real_loc_enabled=false`; enabling them is an explicit ablation,
not part of the active pipeline.

## Active Gamma Schedule

The current formal schedule is encoded by
`configs/overrides/gamma_start50_interval10.yaml`:

| Field | Value |
| --- | ---: |
| `train.epochs` | 300 |
| `train.roi_bank_gamma.enabled` | true |
| `train.roi_bank_gamma.start_epoch` | 30 |
| `train.roi_bank_gamma.update_interval_epochs` | 5 |
| `train.roi_bank_gamma.stop_epoch` | 300 |
| `train.roi_bank_gamma.fixed_roi_library` | true |
| `train.roi_bank_gamma.auto_build_roi_bank` | true |
| `train.roi_bank_gamma.roi_bank_candidate_mode` | `dense_tile_temporal` |
| `train.roi_bank_gamma.roi_bank_frame_range` | `[0, 100]` |
| `train.roi_bank_gamma.target_projected_emitters` | 5000 |
| `train.roi_bank_gamma.gamma_steps` | 100 |
| `train.roi_bank_gamma.gamma_lr` | 0.025 |
| `train.roi_bank_gamma.num_posterior_samples` | 25 |
| `train.roi_bank_gamma.roi_size_px` | 128 |
| `train.roi_bank_gamma.auto_heldout_min_rois` | 20 |
| `train.roi_bank_gamma.auto_heldout_max_rois` | 20 |
| `train.roi_bank_gamma.roi_bank_over_cut_px` | 8 |

## ROI-Library Semantics

v0.3 should keep a fixed ROI library for a run, then re-sample posterior values
from the current localizer during gamma updates. The ROI bank is a geometry and
raw-data library, not a frozen emitter table.

The active `neptune_iwae` training route keeps this library in memory and does
not save it into checkpoints. HDF5 is still supported for offline diagnostics
and smoke benchmarks, but the formal training path should avoid repeatedly
writing long-lived H5 caches.

The HDF5 ROI bank should store:

- ROI id, domain, origin, frame index/window, and grid-cell metadata.
- Photon-count raw ROI crops, not full duplicated TIFF frames.
- Enough raw/input metadata to re-run the current localizer on each ROI.
- Compact ragged emitter records only when fixed detections are needed for
  offline diagnostics.
- A stable held-out ROI split, plus fixed held-out posterior samples for
  cross-update reconstruction monitoring.

Large numeric arrays must be chunked and compressed in HDF5. CSV, dense JSON
numeric arrays, and per-emitter text tables are out of scope.

## Objective Details To Preserve

The gamma objective should align with the current Lunar-style wake route:

- Use the fixed ROI library as the raw-data source.
- Run the current localizer on each selected ROI when sampling posterior values.
- Sample emitter existence and continuous `xyzph` values from localizer
  posterior parameters.
- Smooth detached background with a 9 x 9 average filter.
- Render/project the sampled emitters with the current gamma coefficients.
- Crop away `roi_bank_over_cut_px` pixels before computing the wake loss so
  boundary PSF truncation does not dominate gamma gradients.
- Update gamma coefficients only.

## Monitor Contract

Every gamma update should produce a compact monitor summary with:

- `best_step`
- fixed held-out `initial`, `final`, `delta`, and percentage delta
- `gamma_delta_norm`
- selected ROI `poisson_nll`, projected emitters, background mean, and photons
- links to the checkpoint/report and optional raw-vs-recon PNG diagnostics

In `neptune_iwae`, this is provided by
`scripts/diagnostics/gamma_update_monitor.py`. v0.3 now exposes the same
read-only contract as `python -m neptune_v03.diagnostics.gamma_update_monitor`.

## Implemented Now

- Package scaffold under `src/neptune_v03`.
- Config loader and deep-merge materializer.
- Base microtube config plus schedule/smoke overrides.
- ROI-bank HDF5 data model, compact ragged emitter table, save/load, and
  subset loading.
- ROI candidate geometry, emitter-centered candidate construction, overlap
  checks, and FOV-balanced selection.
- Minimal raw-TIFF fixed ROI-library construction from TIFF/array inputs,
  domain crops, camera backward conversion, injected inference results, and
  optional HDF5 export.
- Native microtube TIFF training-batch provider for high-fidelity smoke runs:
  bounded TIFF frame windows, camera normalization, train input scaling, and
  `neptune-v03-train-high-fidelity` routing through
  `microtube_tiff_train_batch`.
- Native production localizer architecture: residual encoder-decoder backbone,
  decoder skip fusion, refinement residual blocks, and detection/xy/z/photon
  output heads that train through the v0.3 trainer.
- Gamma update monitor discovery, summary coercion, latest-by-domain payloads,
  and Markdown reports.
- Slice 5.1 high-fidelity entrypoint hook wiring: when
  `train.roi_bank_gamma.enabled=true`, `neptune-v03-train-high-fidelity`
  attaches a scheduled gamma update hook, writes `gamma_update_metrics.jsonl`,
  records the gamma route in the manifest/status payload, and verifies gamma
  updates do not modify localizer parameters. This is a smoke wiring step; the
  fixed ROI-library wake objective is still future work.
- Slice 5.2 ROI projection objective smoke wiring: when
  `train.roi_bank_gamma.smoke_roi_library=true`, the high-fidelity hook builds
  a tiny native ROI bank, runs the current localizer posterior sampler, calls
  `GammaProjectionObjective`, and records `objective_source`,
  `roi_count`, `posterior_max_emitters`, and `over_cut_px` in
  `gamma_update_metrics.jsonl`. This proves the hook can use the ROI/posterior
  projection path; captured fixed ROI libraries and monitor summaries remain
  future work.
- Slice 5.3 configured ROI-library objective wiring: when
  `train.roi_bank_gamma.roi_library_path` points to an HDF5 ROI bank,
  `neptune-v03-train-high-fidelity` loads that bank, samples posterior values
  from the current localizer on its ROI batches, calls `GammaProjectionObjective`,
  and records `objective_source=roi_projection_hdf5` plus the ROI library path
  in `gamma_update_metrics.jsonl` and the run manifest. Full held-out monitor
  parity remains future work.
- Slice 5.4 gamma monitor payload parity: ROI projection gamma updates now add
  the old compact monitor field names to `gamma_update_metrics.jsonl`,
  including `best_step`, selected ROI NLL/emitter/photon/background summaries,
  held-out monitor fields, and checkpoint/report link fields. Held-out values
  are explicitly marked `not_configured` until the held-out ROI split is wired.
- Slice 5.5 configured held-out ROI monitor smoke: when
  `train.roi_bank_gamma.heldout_roi_library_path` points to an HDF5 ROI bank,
  ROI projection gamma updates sample fixed held-out posterior values and write
  held-out initial/final/loss-delta monitor fields. This is an explicit
  configured held-out path; automatic held-out split construction remains
  future work.
- Slice 5.6 automatic held-out split from configured ROI bank: when
  `auto_heldout_min_rois`/`auto_heldout_max_rois` are set and no explicit
  held-out HDF5 path is provided, the high-fidelity hook keeps a deterministic
  held-out tail split from the configured ROI bank and uses the remaining ROI
  records for the selected gamma objective.
- Slice 5.7 report/summary diagnostics: ROI projection gamma updates write
  `gamma_alternation_summary.json` and `gamma_update_monitor.md` under the run
  artifact tree and link those paths from `gamma_update_metrics.jsonl`.
- Slice 5.8 raw-vs-reconstruction PNG smoke diagnostics: ROI projection gamma
  updates write a compact `raw_vs_recon.png` artifact from the selected ROI
  raw frame and the current gamma projection. This is a smoke diagnostic, not
  the full historical figure pack.
- Slice 5.9 auto ROI-bank construction entrypoint wiring: when
  `train.roi_bank_gamma.auto_build_roi_bank=true` and
  `auto_build_source_path` is provided, the high-fidelity entrypoint builds a
  native ROI bank through the existing raw-frame ROI builder, then runs the
  same selected/held-out gamma monitor path. This is a smoke wiring route;
  the legacy path remains supported.
- Slice 5.10 dataset-agnostic ROI-bank source contract: the high-fidelity
  entrypoint accepts `train.roi_bank_gamma.roi_bank_source` as a generic
  auto-build mapping with `raw_path`, `frame_range`, `roi_size_px`,
  `candidate_mode`, and `domains`. The materialized microtube config is covered
  as the first compatibility fixture via the legacy
  `roi_bank_source: loc_infer_raw_tiff` alias. Scientific ROI candidate
  selection validation remains future work.
- Slice 5.11 domain-aware ROI-bank artifact grouping: existing ROI projection
  gamma artifacts now live under
  `artifacts/roi_bank_gamma/epoch_XXXX/source_<source>/domain_<domain>/`.
  Metrics and summaries record the selected artifact source/domain groups.
  This is path parity for the current summary/report/PNG artifacts, not the
  full historical figure pack.
- Slice 5.12 historical diagnostics parity spec: the old gamma/ZMap
  before-after, raw TIFF patch reconstruction, fixed ROI reconstruction,
  PSF shape-grid, and monitor-summary diagnostics are now mapped in
  `docs/roi_gamma_diagnostics_parity_spec.md`. This is a spec-only slice;
  implementation begins with the Slice 5.13 diagnostics manifest.
- Slice 5.13 diagnostics manifest contract: each grouped ROI-bank gamma
  artifact directory writes `diagnostics/diagnostics_manifest.json`, listing
  current compact artifacts and diagnostics smoke checks as `available`.
- Slices 5.14-5.17 diagnostics smoke checks: ZMap before-after, fixed ROI
  reconstruction, raw TIFF patch reconstruction, and PSF shape-grid diagnostics
  now each write a summary JSON and PNG under the grouped diagnostics
  directory. These are CPU smoke checks, not full historical figure-pack or
  vector-PSF parity claims.
- Runtime run-layout, manifest, stage-status, artifact registry, and launch
  script contracts.
- Peak bootstrap result contract for the active SLURM stage outputs:
  harvest, peak harvest summary, NAT diagnostics summary, export summary,
  coeff-map, zmap, and `peak_nat_zmap_summary.json`.
- Peak bootstrap pipeline orchestration that connects harvest, NAT-LM fit,
  NCC summary, and coeff-map/zmap export backend calls to that contract.
- Default peak backend raw-TIFF harvest with NumPy local-maximum peak finding,
  distance filtering, torch harvest payload export, and peak harvest summary
  writing.
- Default peak backend real raw-patch NAT diagnostics output:
  `real_nat_diagnostics_summary.json` and `real_nat_diagnostics_payload.pt`
  from a harvest payload, with patch extraction, lightweight LM-style Gaussian
  local fitting, alternating gamma-field regression, raw/reconstruction
  MSE/NCC metrics, NCC filter masks, and payload tensors for raw and
  reconstructed patches.
- Default peak backend NCC value summarization from diagnostics summaries,
  including selected values after NCC filtering.
- Default peak backend coeff-map/zmap export that writes
  `alternating_full_roi_zernike_maps_nm.npz`,
  `preferred_full_roi_zernike_maps_nm.npz`,
  `provisional_non_astig_rms_nm.npz`, and `export_nat_zmap_summary.json`.
- Shared optics boundary with order1 NAT field configs, gamma-to-Zernike
  coefficient evaluation, full-ROI coefficient stack generation, and a
  differentiable Gaussian PSF renderer adapter for lightweight smoke paths.
- Tests for import and active config materialization.

## Not Migrated Yet

- Peak bootstrap high-fidelity vector-PSF NAT/LM fitting and the full figure
  report pack used by the historical export scripts.
- Eval/held-out raw TIFF schedule wiring and held-out library construction.
- Real-data scientific validation of the production localizer and richer
  posterior/GMM semantics.
- Full microtube raw-TIFF ROI-bank config-field mapping and scientific ROI
  candidate selection validation inside the high-fidelity training entrypoint.
- Full historical reconstruction/report figure pack parity for gamma updates.
- SLURM launchers.
- Reconstruction and overlay diagnostics.
