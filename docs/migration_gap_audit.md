# Neptune IWAE Migration Gap Audit

## Status

Slice 4.5 checkpoint. This document records what is still not equivalent
between the reference `neptune_iwae` training route and the clean
`neptune_v0.3` implementation.

This audit was first created at the Slice 2.4 checkpoint and is updated as
later fidelity slices close.

`neptune_iwae` remains reference-only. v0.3 must not import old runtime code.

## Current v0.3 Baseline

The current v0.3 active SMLM route is:

```text
materialized microtube_base.yaml
  -> online_train_batch
  -> active_smlm_soft_moe_double_unet
  -> active_smlm_loss
  -> high-fidelity entrypoint smoke training
```

The target contract is now explicit:

```text
v0.3 pxyz_tar: x_px, y_px, z, photons
legacy iwae pxyz_tar: photons, x_px, y_px, z
```

The v0.3 model output contract is the 10-channel SMLM tensor:

```text
p, x_mu, y_mu, z_mu, photons_mu, x_sigma, y_sigma, z_sigma, photons_sigma, bg
```

## Completed Fidelity Checkpoints

### Model And Conditioning

Implemented in v0.3:

- DoubleUNet registry entries:
  - `active_smlm_double_unet`
  - `active_smlm_soft_moe_double_unet`
- Depthwise conv path.
- FiLM conditioning.
- SoftMoE/domain-aware conditioning.
- Explicit condition vector order:
  - `x_norm`
  - `y_norm`
  - `zernike_nm_mean:n2_m0`
  - `zernike_nm_mean:n3_m1`
  - `zernike_nm_mean:n3_m-1`
  - `zernike_nm_mean:n4_m0`
  - `zernike_nm_mean:n3_m-3`
  - `zernike_nm_mean:n3_m3`
  - optional `domain_onehot:*`

Remaining gap:

- SoftMoE routing is currently a clean v0.3 implementation. It is not yet a
  measured numerical reproduction of any old active route.

### Target Contract

Implemented in v0.3:

- `V03_PXYZ_TARGET_ORDER = ("x_px", "y_px", "z", "photons")`.
- `LEGACY_IWAE_PXYZ_TARGET_ORDER = ("photons", "x_px", "y_px", "z")`.
- `legacy_iwae_pxyz_to_v03()` handles reorder only.
- Active loss target conversion uses centered local xy offsets and explicit
  photon/z scaling.
- Simulator metadata records target order and units.

Remaining gap:

- v0.3 has not reproduced the old `TargetProcess` as a runtime component.
  The old `TargetProcess` mutates legacy order targets by scaling
  `photons` at index 0 with `phot_max` and `z` at index 3 with `z_max`.
  v0.3 instead keeps raw v0.3 order targets and applies scaling inside the
  active loss target convention.

## Major Non-Equivalent Areas

### 1. Loss Fidelity

Old reference:

- `neptune_iwae/smlm_v2a/training/losses.py::GMMLoss`
- `neptune_iwae/neptune_core/localization_runtime/data_factory.py::TargetProcess`

Current v0.3:

- `neptune_v03.localization.losses.ActiveSMLMLoss`
- `neptune_v03.localization.losses.ActiveSMLMGMMLoss`
- `neptune_v03.localization.losses.ActiveSMLMGMMTargetAdapter`
- `neptune_v03.localization.smlm_targets.SMLMTargetConvention`
- `docs/loss_fidelity_spec.md`

Key differences:

- Old `GMMLoss` computes a count likelihood from summed detection
  probabilities and target emitter count.
- Old `GMMLoss` computes a Gaussian mixture likelihood over all image pixels.
- Old `GMMLoss` treats predicted xy means as local offsets, then adds pixel
  centers plus runtime xy offsets to compare against absolute targets.
- Old `GMMLoss` supports `gmm_target_chunk`, `gmm_component_chunk`, and
  `gmm_backend`.
- Current `active_smlm_loss` is a minimal smoke training loss:
  detection BCE, sparse pxyz Gaussian NLL at target pixels, dense background
  MSE, and sigma regularization.
- Current `active_smlm_loss` intentionally filters old `gmm_*` parameters
  because they do not apply to the minimal loss.
- Slice 4.1 specifies the old `GMMLoss` count, mixture, background,
  chunking, target preprocessing, and parity fixture contracts in
  `docs/loss_fidelity_spec.md`.
- Slice 4.2 implements `active_smlm_gmm_loss` with explicit v0.3 target
  adaptation and small CPU parity fixtures for chunked and mixture backends.
- GMMLoss hardening now covers non-zero `xyoffset`, `ch_weight` channel
  disabling, old-route `TargetProcess` phot/z scaling plus `disable_attr`,
  direct GMM adapter `disable_attr`, a two-emitter old-route backend parity
  fixture, mask/target shape safety, and an old-route batch loss flow fixture
  from legacy processed targets through `LocalizationTrainBatch`,
  `active_smlm_gmm_loss`, `train_epochs`, and `training_metrics.jsonl`.
- The old-route batch loss flow fixture verifies the JSONL metrics contain
  only the old external GMMLoss component names: `loss_gmm`, `loss_bkg`, and
  `loss_total`. The count and mixture-localization pieces remain internal to
  `loss_gmm`, matching the old 3052 route. It also guards against
  accidentally routing formal old-GMMLoss parity through smoke-only
  `active_smlm_loss` components such as `loss_detect`, `loss_pxyz`, and
  `loss_sigma`.
- Slice 4.3 adds the broader
  `old_route_batched_masked_background_2x3x4` fixture. It covers a batched
  old-route target process case with mixed target masks, non-zero dense
  background, non-uniform output maps, explicit chunking, and backend parity
  across chunked manual, unchunked manual, and `mixture_same_family`.
- Slice 4.4 adds `extreme_probability_sigma_photon_z`, a test-only numerical
  guard for old `GMMLoss` finite/backend behavior under zero or near-one
  detection probabilities, sigmas below `eps`, scaled photon/z targets, and
  non-zero background. It does not add new loss terms or clipping policies.
- Slice 4.5 adds `old_route_real_shaped_batch`, a test-only fixture that
  mirrors the old simulator/training tuple shape
  `frames, detect_frames, bkg_frames, pxyz_tar, mask_tar` and proves it can
  flow through native `LocalizationTrainBatch`, `active_smlm_gmm_loss`,
  `train_epochs`, and persisted GMMLoss metrics without adding a new provider.

Required future slice:

```text
Loss Fidelity
  -> real captured old-route batch fixture beyond hand-authored CPU cases
```

Preferred direction:

- active_smlm_gmm_loss is the formal fidelity route for old GMMLoss.
- active_smlm_loss remains smoke-only.
- active_smlm_composite_loss is not part of the current migration route.

### 2. Simulator Fidelity

Old reference:

- `neptune_iwae/smlm_v2a/simulation/simulator.py`
- old `Simulation.loc2targets()` emits legacy order
  `photons, x_px, y_px, z`.
- old simulator connects camera noise, background generator, and PSF
  convolution.

Current v0.3:

- `neptune_v03.localization.simulator`
- CPU Gaussian renderer for deterministic smoke tests.
- v0.3 target order `x_px, y_px, z, photons`.

Key differences:

- v0.3 simulator is a contract boundary, not the final vector-PSF simulator.
- v0.3 simulator currently uses fixed z=0 for native online smoke batches.
- v0.3 simulator does not yet model camera noise, channel photon distribution,
  or the old PSF calibration stack.

Required future slice:

```text
Simulator Fidelity
  -> vector-PSF/LUT adapter boundary
  -> z sampling range from formal config
  -> photon/background sampling from formal config
  -> camera noise and train normalization parity
```

### 3. Training Runtime Fidelity

Old reference:

- `neptune_iwae/neptune_core/online_train.py`
- runtime construction wires frame processors, background processors,
  `TargetProcess`, eval dataloaders, NAT wake runtime, scheduler behavior,
  and optional checkpoint resume.

Current v0.3:

- `neptune_v03.training.run_high_fidelity`
- `neptune_v03.localization.runtime_config`
- generic `train_epochs`

Key differences:

- v0.3 high-fidelity entrypoint currently trains the localizer path and writes
  manifest, metrics, checkpoints, and stage status.
- Slice 6.1 records the current training runtime contract in
  `resolved_contract.training_runtime`: optimizer name/params, scheduler
  status, configured grad-clip norm, and configured AMP dtype/status. The
  high-fidelity entrypoint writes the same contract into manifest and stage
  status.
- Slice 6.2 wires configured grad clipping through the generic trainer. When
  `resolved_contract.training_runtime.grad_clip.active=true`, `train_epochs`
  passes the configured norm to `train_one_epoch`, which clips gradients after
  `loss.backward()` and before `optimizer.step()`.
- Slice 6.3 hardened the AMP contract. `resolved_contract.training_runtime.amp`
  records `configured`, `dtype`, `active`, and `inactive_reason`.
- Slice 6.4 records scheduler parity in
  `docs/training_scheduler_parity_spec.md`. v0.3 now maps legacy
  `smlm_overrides.lr_scheduler=StepLR`, `lr_step_size=1000`, `lr_gamma=0.9`,
  and `lr_step_unit=optimizer_step` into
  `resolved_contract.training_runtime.scheduler`.
- Slice 6.5 records optimizer parity in
  `docs/training_optimizer_parity_spec.md`. That slice introduced separate
  recording for runtime optimizer state and legacy AdamW target optimizer
  intent in `resolved_contract.training_runtime.legacy_optimizer`.
- Slice 6.6 adds explicit AdamW runtime support to the generic trainer. v0.3
  can instantiate `torch.optim.AdamW` from runtime config, and the existing
  checkpoint helper persists its optimizer state.
- Slice 6.7 switches the materialized microtube active high-fidelity route to
  AdamW when `smlm_overrides.optimizer=AdamW` is present and no explicit
  runtime optimizer overrides it. The manifest/status training runtime contract
  now records `optimizer.name=adamw`, AdamW `lr`/`weight_decay`, and
  `legacy_optimizer.active=true` for that route.
- Slice 6.8 wires the active StepLR scheduler runtime for
  `step_unit=optimizer_step`. The generic trainer now builds StepLR from the
  resolved contract, steps it after each optimizer update, persists
  `scheduler_state_dict` in checkpoints, and restores scheduler state when
  loading checkpoints with a scheduler.
- Slice 6.9 hardens scheduler checkpoint/resume behavior. Best checkpoints
  are covered by scheduler-state tests, inactive scheduler contracts are
  ignored by the runtime factory, and loading with a scheduler now fails fast
  if the checkpoint lacks `scheduler_state_dict`.
- Slice 6.10 records AMP policy in `docs/training_amp_parity_spec.md`.
- Slice 6.11a and Slice 6.11b covered the temporary inactive AMP guard before
  runtime wiring.
- The current trainer wires configured AMP through CUDA `autocast` and
  `torch.amp.GradScaler`, writes `scaler_state_dict` for active CUDA AMP
  checkpoints, and keeps CPU execution plain FP32 without scaler state.
- Slice 6.12 records localizer eval / held-out provider policy in
  `docs/training_eval_heldout_parity_spec.md`. It separates localizer
  `EvalProvider`/`eval_metrics.jsonl`/`checkpoint_best.pt` behavior from
  ROI-bank gamma held-out monitoring and defers provider implementation to
  Slice 6.13.
- Slice 6.13 wires the first localizer eval provider for
  `train.eval.enabled=true, source=online_generation`. Both high-fidelity and
  lightweight localization entrypoints now build fixed native online eval
  batches, write `eval_metrics.jsonl`, save `checkpoint_best.pt`, and record
  the localizer eval contract plus best-checkpoint path in manifest/status.
- Slice 6.14 hardens the online localizer eval contract. Eval batch count and
  batch size must be positive, SoftMoE/domain-onehot eval batches run through
  active model/loss smoke coverage, and eval provider construction now reuses
  the runtime config mapping for FiLM/SoftMoE/domain-aware fields including
  relative `dual_domain_coeff_maps` path resolution.
- Slice 6.15 records a dataset-agnostic materialized localizer eval contract in
  `docs/materialized_localizer_eval_contract.md`. The future preferred source
  is `materialized_dataset`; microtube is only the first compatibility fixture
  or legacy alias. The spec records `dataset_id`, `sample_id`, `source_path`,
  `frame_range`, `crop`, `heldout_split`, no train/eval overlap by default,
  and separation from ROI-bank gamma held-out.
- Slice 6.16 implements the first dataset-agnostic materialized localizer eval
  provider for supervised `.npz` fixtures. `source=materialized_dataset`
  produces fixed eval batches with `model_input`, `detect_tar`, `bkg_tar`,
  `pxyz_tar`, and `mask_tar`; relative `source_path` resolves against the
  config directory; manifest/status record dataset/sample/source/frame/crop
  fields; `materialized_microtube` maps to the same generic contract as a
  compatibility alias.
- Slice 6.17 hardens checkpoint/resume compatibility for localizer eval best
  checkpoints. When training resumes in a run directory that already contains
  `checkpoint_best.pt` with `eval_loss`, `train_epochs` initializes the
  in-memory best eval loss from that checkpoint and does not overwrite the
  historical best checkpoint with a worse resumed eval result.
- v0.3 does not yet implement direct TIFF/HDF5/manifest materialized eval
  ingestion or automatic train/eval overlap comparison against materialized
  train providers.
- v0.3 does not yet wire NAT wake updates into the localizer training loop.
- v0.3 does not yet reproduce old resume/checkpoint initialization semantics
  beyond the generic training checkpoint helpers.

Required future slice:

```text
Training Runtime Fidelity
  -> formal config validation
  -> actual optimizer/scheduler/AMP runtime parity beyond the Slice 6.1
     contract, Slice 6.2 grad-clip wiring, Slice 6.3 AMP inactive contract,
     Slice 6.4 scheduler contract, Slice 6.5 optimizer contract,
     Slice 6.6 explicit AdamW runtime support, and Slice 6.7 active microtube
     AdamW switch, Slice 6.8 active StepLR runtime wiring, and Slice 6.9
     scheduler resume hardening, Slice 6.10 AMP policy/spec, Slice 6.11 AMP
     inactive guards, Slice 6.13 online localizer eval provider wiring,
     Slice 6.14 localizer eval contract hardening, and Slice 6.15
     dataset-agnostic materialized eval spec, plus Slice 6.16 supervised
     `.npz` materialized eval provider and Slice 6.17 best-checkpoint resume
     hardening
  -> direct materialized TIFF/HDF5/manifest eval ingestion
  -> broader checkpoint init/resume compatibility plan
  -> loss component metrics
```

### 4. ROI-bank gamma loop

Old reference:

- active route described in `docs/active_pipeline.md`
- fixed ROI library
- current localizer posterior sampling per scheduled gamma update
- ROI wake objective and gamma-only optimization

Current v0.3:

- ROI bank data model and gamma update boundaries exist.
- Slice 5.1 wires the high-fidelity entrypoint to the scheduled gamma update
  hook when `train.roi_bank_gamma.enabled=true`. The hook writes
  `gamma_update_metrics.jsonl`, records route metadata in manifest/status, and
  verifies gamma updates do not modify localizer parameters.
- Slice 5.2 wires a native ROI projection objective smoke route when
  `train.roi_bank_gamma.smoke_roi_library=true`. The hook builds a tiny ROI
  bank, samples the current localizer posterior, calls
  `GammaProjectionObjective`, and records compact objective metadata in
  `gamma_update_metrics.jsonl`.
- Slice 5.3 wires configured HDF5 ROI libraries through
  `train.roi_bank_gamma.roi_library_path`. The hook loads the fixed ROI bank,
  samples posterior values from the current localizer on ROI batches, calls
  `GammaProjectionObjective`, and records `objective_source=roi_projection_hdf5`
  plus the ROI library path.
- Slice 5.4 adds compact historical monitor field names to ROI projection
  `gamma_update_metrics.jsonl` rows: `best_step`, selected ROI NLL, projected
  photons, sampled emitter count, background mean, held-out fields, and
  checkpoint/report link fields. Held-out values are present as
  `not_configured` placeholders until the held-out ROI split exists.
- Slice 5.5 wires a configured held-out HDF5 ROI bank through
  `train.roi_bank_gamma.heldout_roi_library_path`. The hook samples fixed
  held-out posterior values and records held-out initial loss, final loss,
  delta, percentage delta, and held-out NLL fields in `gamma_update_metrics`.
- Slice 5.6 wires automatic held-out splitting from the configured ROI bank
  when `auto_heldout_min_rois`/`auto_heldout_max_rois` are set and no explicit
  held-out HDF5 path is provided. The selected objective uses the remaining
  ROI records and held-out metrics report the deterministic held-out ROI ids.
- Slice 5.7 writes compact `gamma_alternation_summary.json` and
  `gamma_update_monitor.md` artifacts for ROI projection gamma updates and
  links them from `gamma_update_metrics.jsonl`.
- Slice 5.8 writes a smoke `raw_vs_recon.png` diagnostic from the selected ROI
  raw frame and current gamma projection. This is not the full historical
  report pack.
- Slice 5.9 wires auto ROI-bank construction into the high-fidelity entrypoint
  for smoke runs. When `auto_build_roi_bank=true` and `auto_build_source_path`
  is provided, v0.3 builds a native ROI bank through the raw-frame ROI builder
  and runs the same selected/held-out gamma monitor route.
- Slice 5.10 adds a dataset-agnostic ROI-bank source contract. The
  high-fidelity entrypoint accepts `train.roi_bank_gamma.roi_bank_source` as a
  mapping with `mode`, `raw_path`, `frame_range`, `roi_size_px`,
  `candidate_mode`, and `domains`, resolves relative raw paths against the
  config file, and records the resolved source in manifest and gamma metrics.
  The materialized microtube config is covered as a compatibility fixture via
  the legacy `roi_bank_source: loc_infer_raw_tiff` alias, using
  `train.real_tiff_wake.tiff_path` and domains.
- Slice 5.11 groups the existing ROI-bank gamma artifacts by resolved source
  and selected ROI domain. `gamma_alternation_summary.json`,
  `gamma_update_monitor.md`, and `raw_vs_recon.png` are now written under
  `source_<source>/domain_<domain>/`, with compact source/domain fields also
  recorded in gamma metrics. This is report path parity only, not the full
  historical diagnostics pack.
- Slice 5.12 records the historical diagnostics parity contract in
  `docs/roi_gamma_diagnostics_parity_spec.md`. The spec maps old
  gamma/ZMap before-after, raw TIFF patch reconstruction, fixed ROI
  reconstruction, PSF shape-grid, and monitor-summary diagnostics to explicit
  v0.3 outputs, metrics, non-goals, and future implementation slices.
- Slice 5.13 writes `diagnostics/diagnostics_manifest.json` under each grouped
  ROI-bank gamma artifact directory. The manifest records available compact
  artifacts and explicit `not_run` reasons for ZMap before-after, fixed ROI
  reconstruction, raw TIFF patch reconstruction, and PSF shape-grid diagnostics.
- Slices 5.14-5.17 add CPU smoke diagnostics for ZMap before-after, fixed ROI
  reconstruction, raw TIFF patch reconstruction, and PSF shape-grid summaries.
  Each diagnostic writes a summary JSON and PNG under the grouped diagnostics
  directory and is linked from `diagnostics_manifest.json`.

Key differences:

- Projection objective is wired for tiny smoke ROI banks and configured HDF5
  ROI banks. Compact monitor payload fields, explicit configured held-out
  metrics, automatic held-out split, compact report artifacts, and a smoke
  raw-vs-reconstruction PNG are wired. Auto-built ROI banks now use a generic
  source contract, with microtube covered as the first compatibility fixture.
  Existing report artifacts are grouped by source and domain, and smoke
  diagnostics are available through the diagnostics manifest.
- Scientific ROI candidate selection validation and implementation of the full
  historical diagnostics pack remain future work. The diagnostics parity spec
  and CPU smoke checks are now written.

Required future slice:

```text
ROI-bank Gamma Loop Fidelity
  -> scientific ROI candidate selection validation
  -> high-fidelity diagnostics parity beyond CPU smoke checks
```

## Recommended Next Slices

### Slice 3.1: Runtime Config Validation

Status: completed.

Acceptance criteria:

- Unknown non-legacy loss params fail fast.
- Legacy `gmm_*` params are accepted only for future GMM/composite losses.
- Active model/loss/provider config emits a compact resolved-contract summary.

### Slice 3.2: Training Runtime Fidelity Smoke

Status: completed.

Acceptance criteria:

- Materialized microtube config runs through high-fidelity entrypoint with
  active SoftMoE and records model, batch provider, loss name, loss params,
  and condition contract in manifest or status payload.
- Loss component metrics are persisted.

### Slice 4.1: Loss Fidelity Spec

Status: completed.

Acceptance criteria:

- Old `GMMLoss` count term, mixture term, background term, chunking, and
  target preprocessing are specified with v0.3 inputs.
- Fixtures define small deterministic cases for numerical parity.

### Slice 4.2: GMM Loss Core

Status: completed.

Acceptance criteria:

- v0.3 implements a clean GMM emitter likelihood module.
- Chunked and unchunked paths match on small CPU fixtures.
- The module consumes v0.3 `x_px, y_px, z, photons` targets through explicit
  adapters.

### Slice 4.3: Broader Old-Route GMMLoss Fixtures

Status: completed.

Acceptance criteria:

- Old-route `TargetProcess` scaling is covered in a batched fixture.
- Mixed per-batch masks and different active emitter counts are covered.
- Non-zero dense background uses old summed MSE semantics.
- Non-uniform output maps match across chunked manual, unchunked manual, and
  `mixture_same_family` backends.

### Slice 4.4: Old GMMLoss Numerical Guard Fixture

Status: completed.

Acceptance criteria:

- Extreme but legal old `GMMLoss` inputs stay finite.
- Chunked manual, unchunked manual, and `mixture_same_family` backends match.
- The fixture is test-only and does not introduce new loss semantics.

### Slice 4.5: Old-Route Real-Shaped Batch Fixture

Status: completed.

Acceptance criteria:

- Old simulator/training tuple shape is represented as a native
  `LocalizationTrainBatch`.
- Legacy-order targets are processed through
  `legacy_iwae_target_process_to_v03`.
- The batch flows through `active_smlm_gmm_loss`, `train_epochs`, and
  `training_metrics.jsonl` with GMMLoss components only.
- No production provider or old runtime bridge is introduced.

### Slice 5: ROI-bank Gamma Loop Integration

### Slice 5.1: High-Fidelity Gamma Hook Wiring

Status: completed.

Acceptance criteria:

- `run_high_fidelity` attaches a scheduled gamma update hook when
  `train.roi_bank_gamma.enabled=true`.
- `gamma_update_metrics.jsonl` is written at scheduled epochs.
- manifest and stage status record the gamma route and update count.
- gamma updates do not modify localizer parameters.

### Slice 5.2: ROI Projection Gamma Objective Smoke Wiring

Status: completed.

Acceptance criteria:

- `run_high_fidelity` can call `GammaProjectionObjective` from the scheduled
  gamma hook.
- The hook path samples detached posterior values from the current localizer.
- `gamma_update_metrics.jsonl` records objective source and compact ROI
  metadata.
- Gamma updates do not modify localizer parameters.

### Slice 5.3: Captured Fixed ROI-Library Gamma Objective Wiring

Status: completed.

Acceptance criteria:

- `run_high_fidelity` can optionally run the captured fixed ROI-bank gamma
  schedule.
- Gamma updates use the configured ROI library rather than the smoke ROI bank.
- Metrics and manifest identify the configured ROI-library objective source.

### Slice 5.4: Gamma Monitor Payload Parity

Status: completed for compact ROI projection payload fields.

Acceptance criteria:

- Monitor payload covers the fields documented in `docs/active_pipeline.md`.
- Held-out fields are present and explicitly marked `not_configured` until the
  held-out ROI split is wired.

### Slice 5.5: Configured Held-Out ROI Monitor Smoke

Status: completed.

Acceptance criteria:

- `train.roi_bank_gamma.heldout_roi_library_path` loads an HDF5 ROI bank for
  held-out monitoring.
- Held-out posterior samples are fixed for the gamma update and do not train
  the localizer.
- `gamma_update_metrics.jsonl` records held-out initial loss, final loss,
  delta, percentage delta, and held-out NLL fields.
- No automatic held-out split, PNG diagnostics, or report pack is introduced.

### Slice 5.6: Auto Held-Out Split From Configured ROI Bank

Status: completed.

Acceptance criteria:

- `auto_heldout_min_rois`/`auto_heldout_max_rois` deterministically split a
  held-out tail from the configured ROI bank when no explicit held-out path is
  provided.
- The selected gamma objective uses the remaining ROI records.
- Metrics and manifest identify the auto-heldout route and held-out ROI ids.

### Slice 5.7: Compact Gamma Report Artifacts

Status: completed.

Acceptance criteria:

- ROI projection gamma updates write `gamma_alternation_summary.json`.
- ROI projection gamma updates write `gamma_update_monitor.md`.
- `gamma_update_metrics.jsonl` links the summary, report, and checkpoint.

### Slice 5.8: Raw-Vs-Reconstruction PNG Smoke Diagnostic

Status: completed.

Acceptance criteria:

- ROI projection gamma updates write a PNG diagnostic artifact.
- The diagnostic is generated from selected ROI raw data and the current gamma
  projection path.
- The implementation remains a smoke diagnostic and does not claim full old
  figure-pack parity.

### Slice 5.9: Auto ROI-Bank Construction Entrypoint Wiring

Status: completed for smoke wiring.

Acceptance criteria:

- `train.roi_bank_gamma.auto_build_roi_bank=true` can build a native ROI bank
  when `auto_build_source_path` is provided.
- The auto-built ROI bank flows through selected ROI projection, automatic
  held-out split, compact report artifacts, and PNG smoke diagnostics.
- Manifest and metrics identify `roi_library_source=auto_built`.
- Legacy `auto_build_source_path` remains supported.

### Slice 5.10: Dataset-Agnostic ROI-Bank Source Contract

Status: completed for smoke wiring and microtube compatibility.

Acceptance criteria:

- `train.roi_bank_gamma.roi_bank_source` can be a mapping with generic
  `mode=auto_build`, `raw_path`, `frame_range`, `roi_size_px`,
  `candidate_mode`, and `domains` fields.
- Relative `raw_path` values resolve against the loaded config file location.
- Manifest, status payload, gamma metrics, and gamma summary artifacts record
  `roi_bank_source_mode`, `roi_bank_raw_path`, `roi_bank_candidate_mode`, and
  `roi_bank_frame_range`.
- The materialized microtube config path with legacy
  `roi_bank_source: loc_infer_raw_tiff` maps to the same generic auto-build
  route through `train.real_tiff_wake.tiff_path` and domains.
- Scientific ROI candidate selection validation remains future work.

### Slice 5.11: Domain-Aware ROI-Bank Artifact Grouping

Status: completed for existing summary/report/PNG artifacts.

Acceptance criteria:

- ROI projection gamma artifacts are written under
  `artifacts/roi_bank_gamma/epoch_XXXX/source_<source>/domain_<domain>/`.
- Gamma metrics and summary artifacts include `artifact_source_group`,
  `artifact_domain_group`, and `selected_domain_names`.
- Dataset-agnostic auto-built ROI banks and materialized microtube compatibility
  fixtures both use the grouped artifact paths.
- No new diagnostics or figure-pack content is added in this slice.

### Slice 5.12: Historical Diagnostics Parity Spec

Status: completed as a specification slice.

Acceptance criteria:

- `docs/roi_gamma_diagnostics_parity_spec.md` maps old diagnostics from
  `gamma_zmap_before_after.py`, `raw_tiff_patch_recon_gpu.py`,
  `roi_recon_visual_diagnostic.py`, `psf_shape_grid_compare.py`, and the gamma
  monitor route into v0.3 output and metric contracts.
- The spec records required summary JSON files, PNG families, and per-domain
  metrics including Poisson NLL, RMS, and NCC where applicable.
- The spec explicitly says Slice 5.12 does not implement the full historical
  figure pack.
- Follow-on slices start with Slice 5.13 diagnostics manifest wiring.

### Slice 5.13: Diagnostics Manifest Contract

Status: completed.

Acceptance criteria:

- `diagnostics/diagnostics_manifest.json` is written under the grouped
  `source_<source>/domain_<domain>/` artifact directory.
- The manifest lists compact monitor and `raw_vs_recon_smoke` artifacts as
  `available`.
- Future diagnostics are listed as `not_run` with explicit reasons.
- Existing compact summary/report/PNG paths remain unchanged.

### Slices 5.14-5.17: Diagnostics Smoke Checks

Status: completed for CPU smoke diagnostics.

Acceptance criteria:

- `zmap_before_after` writes
  `delta_gamma_physical_zmap_before_after_summary.json` and a PNG with delta
  mean/max and dominant-mode metrics.
- `fixed_roi_recon` writes `fixed_roi_recon_summary.json` and a PNG with
  selected ROI count, rendered count, Poisson NLL, and RMS.
- `raw_tiff_patch_recon` writes `raw_tiff_patch_recon_summary.json` and a PNG
  with Poisson NLL, MSE, and NCC.
- `psf_shape_grid` writes `psf_shape_grid_summary.json` and a PNG with PSF sum
  and second-moment metrics.
- The diagnostics manifest marks all four smoke checks as `available`.
- These smoke checks do not claim full old figure-pack or vector-PSF parity.

## Priority Decision

Proceed next with Training Runtime Fidelity before Loss Fidelity if the goal is
to run longer formal experiments soon.

Proceed next with Loss Fidelity before ROI-bank gamma loop if the goal is to
match old localization likelihood behavior before full pipeline integration.

Do not start ROI-bank gamma loop fidelity before the active localizer loss and
runtime metrics are stable enough to diagnose training failures.
