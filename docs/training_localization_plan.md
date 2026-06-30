# Training and Localization Plan

## Purpose

This document is the working checkpoint for the `neptune_v0.3` training and
localization rebuild. Keep it updated whenever a training/localization slice is
completed or the next task changes.

The goal is a clean, testable Neptune-IWAE implementation in `neptune_v0.3`.
`neptune_iwae` is reference-only. Do not bridge, inject, import, or wrap old
runtime functions as part of the v0.3 training path.

## Current Principle

- Build native v0.3 interfaces first, then implement behavior behind them.
- Use TDD for each slice: failing test, minimal implementation, full
  verification.
- Keep `training` generic and small. Put localization-specific batch shape,
  loss adapters, and data generation in `localization`.
- Do not copy old mixed-purpose training entrypoints.
- Runtime artifacts, checkpoints, logs, and caches stay out of source trees.

## Completed

### Generic Training Spine

Implemented in `src/neptune_v03/training/`.

- `TrainingBatch`
- `TrainingConfig`
- `EpochTrainingConfig`
- `TrainingEpochResult`
- `TrainingRunEpochResult`
- `TrainingResumeState`
- `train_one_epoch`
- `train_epochs`
- `evaluate`
- `load_training_checkpoint`
- `TrainerRuntime`
- `build_trainer_runtime`

Supported behavior:

- One epoch and multi-epoch training.
- Custom `loss_fn(model, batch)`.
- Metrics JSONL writing.
- Per-epoch checkpoints.
- Fixed eval batches, eval loss, and best checkpoint.
- Resume from checkpoint for model, optimizer, epoch, step count, and
  `global_step`.
- Runtime factory from explicit registries plus built-in SGD and AdamW
  optimizers.

Tests:

- `tests/test_training_loop.py`
- `tests/test_training_runtime.py`

### Localization Batch Adapter

Implemented in `src/neptune_v03/localization/training_adapter.py`.

- `LocalizationTrainBatch`
- `to_training_batch`
- `make_localization_loss`

The localization loss adapter calls criterion objects with the clean
localization signature:

```text
criterion.forward(y_out, detect_tar, pxyz_tar, mask_tar, bkg_tar)
```

Test:

- `tests/test_localization_training_adapter.py`

### Synthetic and Native Online Batch Providers

Implemented in:

- `src/neptune_v03/localization/synthetic.py`
- `src/neptune_v03/localization/online.py`
- `src/neptune_v03/localization/runtime_config.py`

Current state:

- `deterministic_synthetic_online` emits deterministic localization-shaped
  batches for smoke training.
- `online_train_batch` is a native v0.3 provider with a lightweight Gaussian
  frame simulator for CPU smoke training. It supports `triplet` and
  `sequence_window` metadata, deterministic seeds by epoch/step, and
  PSF-like smooth frames.
- `build_localization_runtime_config` maps
  `train.online_generation` fields into a `build_trainer_runtime` config.
- No old `neptune_iwae` builder or old runtime function is injected or called.

Tests:

- `tests/test_synthetic_online_provider.py`
- `tests/test_online_batch_provider.py`
- `tests/test_localization_runtime_config.py`
- `tests/test_training_runtime.py`

### Clean Localization Model Runtime

Implemented in:

- `src/neptune_v03/localization/model.py`

Current state:

- `SimpleLocalizationModel` is a small native convolutional localizer for CPU
  smoke training.
- `ProductionLocalizationModel` is a native residual encoder-decoder localizer
  with multi-scale encoder stages, decoder skip fusion, refinement residual
  blocks, and separate detection, xy-offset, z, and photon heads.
- `build_localization_model_registry` exposes `simple_localizer` for
  lightweight tests and `production_localizer` for high-fidelity smoke runs.
- The model trains through the existing localization loss adapter and native
  online provider.
- Localization-specific eval/resume smoke coverage uses fixed eval batches,
  best checkpoint writing, checkpoint loading, and resumed `global_step`
  continuation.

Test:

- `tests/test_localization_model_runtime.py`
- `tests/test_production_localizer.py`

### ROI Batches and Posterior Sampling

Implemented in:

- `src/neptune_v03/localization/roi_batches.py`
- `src/neptune_v03/localization/posterior.py`

Current state:

- `build_roi_batch_provider` converts `ROIBank` records into
  `LocalizationTrainBatch` objects.
- `sample_detection_posterior` runs a localizer on a localization batch and
  returns detached masked `xyzph` samples plus logits.

Test:

- `tests/test_localization_roi_posterior.py`

### Gamma-Only Update Hook

Implemented in:

- `src/neptune_v03/gamma_update/hook.py`

Current state:

- `build_gamma_update_hook` creates an epoch-end hook with schedule gating.
- The hook optimizes a gamma parameter with an injected objective.
- The hook verifies localizer parameters are unchanged.
- It writes compact gamma update metrics.

Test:

- `tests/test_gamma_update_hook.py`

### Localization Training Entrypoint

Implemented in:

- `src/neptune_v03/training/run_localization.py`

Current state:

- `neptune-v03-train-localization` runs the lightweight localization smoke
  path from a resolved YAML config.
- The entrypoint creates a run layout, writes a manifest, writes stage status,
  and produces training metrics/checkpoints under the run directory.

Test:

- `tests/test_training_entrypoint.py`

### High-Fidelity Production Boundaries

Implemented in:

- `src/neptune_v03/data/normalization.py`
- `src/neptune_v03/data/tiff.py`
- `src/neptune_v03/localization/microtube_tiff.py`
- `src/neptune_v03/localization/simulator.py`
- `src/neptune_v03/localization/model.py`
- `src/neptune_v03/localization/posterior.py`
- `src/neptune_v03/gamma_update/objective.py`
- `src/neptune_v03/gamma_update/feedback.py`
- `src/neptune_v03/training/run_high_fidelity.py`

Current state:

- Microtube camera normalization has native ADU-to-photon and train-input
  scaling helpers.
- Microtube TIFF training ingestion can read bounded TIFF frame windows,
  normalize ADU frames into train input tensors, and feed the high-fidelity
  entrypoint through `microtube_tiff_train_batch`.
- Native simulator boundary produces deterministic localization batches via a
  CPU Gaussian renderer. This is a replaceable boundary, not the final
  vector-PSF renderer.
- `ProductionLocalizationModel` defines and implements the production localizer
  architecture boundary: residual encoder-decoder features plus detection
  logits/probability, xy offsets, z map, and photon map.
- Posterior decoding converts production output maps into detached masked
  `xyzph` samples.
- Projection gamma objective renders posterior samples into ROI frames and
  differentiates gamma-only parameters.
- Feedback maps can be saved/loaded and exposed through localization runtime
  config.
- `neptune-v03-train-high-fidelity` runs the production localizer with the
  native online provider, writes manifest/status, and produces metrics and
  checkpoints under the run directory.

Tests:

- `tests/test_data_normalization.py`
- `tests/test_microtube_tiff_provider.py`
- `tests/test_localization_simulator.py`
- `tests/test_production_localizer.py`
- `tests/test_localization_roi_posterior.py`
- `tests/test_gamma_projection_objective.py`
- `tests/test_feedback_handoff.py`
- `tests/test_high_fidelity_entrypoint.py`

### Baseline Reference

Frozen baseline 3052 is documented in:

- `docs/baselines/3052_microtube_roi_gamma.md`

Use it as behavioral reference only. It is not a code dependency.

### Loss Fidelity Spec

Implemented in:

- `docs/loss_fidelity_spec.md`
- `src/neptune_v03/localization/losses.py`
- `src/neptune_v03/training/runtime.py`

Current state:

- Slice 4.1 specifies old `GMMLoss` count likelihood, all-pixel Gaussian
  mixture localization likelihood, dense background MSE, backend/chunking
  behavior, and `TargetProcess` preprocessing against v0.3 target order.
- Slice 4.2 implements native `active_smlm_gmm_loss` with explicit
  `ActiveSMLMGMMTargetAdapter` conversion from v0.3 target order into the GMM
  internal order.
- The tests define deterministic CPU fixtures:
  `single_emitter_centered_2x2`, `two_emitters_chunked_3x3`, and
  `empty_frame_count_only`.
- GMMLoss hardening covers non-zero `xyoffset`, `ch_weight` channel disabling,
  old-route `TargetProcess` phot/z scaling plus `disable_attr`, direct GMM
  adapter `disable_attr`, a two-emitter old-route backend parity fixture,
  mask/target shape safety, and an old-route batch loss flow fixture through
  `LocalizationTrainBatch`, `active_smlm_gmm_loss`, `train_epochs`, and
  `training_metrics.jsonl`.
- The old-route batch loss flow fixture verifies persisted training metrics
  expose only the old external GMMLoss components: `loss_gmm`, `loss_bkg`,
  and `loss_total`. The count and mixture-localization pieces remain internal
  to `loss_gmm`, matching the old 3052 route. The fixture guards against
  regressions where the formal old-GMMLoss route accidentally emits smoke-only
  `active_smlm_loss` components such as `loss_detect`, `loss_pxyz`, or
  `loss_sigma`.
- Slice 4.3 adds `old_route_batched_masked_background_2x3x4`, a broader
  old-route CPU fixture with `N=2`, mixed target masks, non-zero dense
  background, non-uniform output maps, `TargetProcess` phot/z scaling,
  explicit chunking, and backend parity across chunked manual, unchunked
  manual, and `mixture_same_family`.
- Slice 4.4 adds `extreme_probability_sigma_photon_z`, a test-only numerical
  guard for old `GMMLoss` behavior with zero or near-one detection
  probabilities, sigmas below `eps`, scaled photon/z targets, and non-zero
  background. It verifies finite outputs and backend parity without changing
  production loss semantics.
- Slice 4.5 adds `old_route_real_shaped_batch`, a test-only fixture matching
  the old simulator/training tuple shape
  `frames, detect_frames, bkg_frames, pxyz_tar, mask_tar`. It verifies the
  native `LocalizationTrainBatch` and `active_smlm_gmm_loss` training metrics
  path without adding a provider or importing old runtime code.
- `active_smlm_gmm_loss` is available through the generic training runtime
  registry and explicit localization runtime config.
- active_smlm_gmm_loss is the formal fidelity route for old GMMLoss.
- active_smlm_loss remains smoke-only.
- active_smlm_composite_loss is not part of the current migration route.
- No v0.3 runtime code imports old `neptune_iwae` loss code.

Test:

- `tests/test_active_smlm_gmm_loss.py`
- `tests/test_training_runtime.py`
- `tests/test_localization_runtime_config.py`
- `tests/test_migration_gap_docs.py`

## Current Verification

Latest full-suite verification:

```bash
PYTEST_ADDOPTS='-o cache_dir=.local/cache/pytest' pytest -q
```

Result:

```text
157 passed
```

Latest targeted high-fidelity production-boundary verification:

```bash
PYTEST_ADDOPTS='-o cache_dir=.local/cache/pytest' pytest -q tests/test_data_normalization.py tests/test_localization_simulator.py tests/test_production_localizer.py tests/test_localization_roi_posterior.py tests/test_gamma_projection_objective.py tests/test_feedback_handoff.py tests/test_high_fidelity_entrypoint.py tests/test_online_batch_provider.py
```

Result:

```text
18 passed, 1 warning
```

Latest targeted microtube TIFF training-provider verification:

```bash
PYTEST_ADDOPTS='-o cache_dir=.local/cache/pytest' pytest -q tests/test_microtube_tiff_provider.py tests/test_localization_runtime_config.py tests/test_high_fidelity_entrypoint.py tests/test_training_runtime.py tests/test_online_batch_provider.py tests/test_data_normalization.py
```

Result:

```text
16 passed, 1 warning
```

Latest targeted production localizer verification:

```bash
PYTEST_ADDOPTS='-o cache_dir=.local/cache/pytest' pytest -q tests/test_production_localizer.py tests/test_localization_roi_posterior.py tests/test_high_fidelity_entrypoint.py tests/test_microtube_tiff_provider.py tests/test_localization_runtime_config.py tests/test_training_runtime.py
```

Result:

```text
17 passed, 1 warning
```

## Not Done Yet

These are not complete and should not be described as complete:

- Real microtube raw TIFF ingestion for eval batches and ROI/gamma schedules.
  Training ingestion now has a native TIFF provider for smoke runs.
- Vector-PSF/GPU simulator renderer behind the native simulator boundary.
- Scientific validation of the residual encoder-decoder production localizer
  against real/vector-PSF data and richer posterior/GMM semantics.
- Auto-built real microtube fixed ROI-library selection inside the training
  epoch schedule. HDF5 ROI libraries can already drive the scheduled projection
  objective, but automatic active-route ROI bank construction is still future
  work.
- Held-out ROI split, captured held-out posterior samples, and held-out
  reconstruction monitor values for gamma updates.
- Feedback maps actively consumed by localization data/model paths; current
  support writes/loads maps and exposes map paths through config.
- SLURM script/config packaging validated on the actual cluster. The launch
  spec writer and high-fidelity Python entrypoint exist, but no cluster job has
  been submitted as part of this slice.
- High-fidelity vector-PSF NAT fitting and richer report pack.

## High-Fidelity Production Stage

The next stage is to replace each lightweight smoke component with a native
v0.3 production boundary. Each boundary must be implemented and tested without
bridging old `neptune_iwae` runtime functions.

Important distinction:

- "Production boundary complete" means v0.3 has a clean API, config mapping,
  tests, and a deterministic CPU-smoke implementation.
- "Scientific high-fidelity complete" additionally requires validating the
  implementation against the full vector-PSF/GPU physical model and real
  microtube runs. That validation is not implied by smoke tests.

Production-stage slices:

1. Microtube normalization: completed for native ADU-to-photon and train-input
   scaling helpers.
2. High-fidelity simulator boundary: completed for a deterministic CPU Gaussian
   renderer behind a native simulator API.
3. Production localizer boundary: completed for model registry, residual
   encoder-decoder architecture, output contract, and train-step coverage.
4. Posterior decoding: completed for production output maps to detached masked
   `xyzph` posterior samples.
5. Projection gamma objective: completed for differentiable gamma-only CPU
   projection objective.
6. Feedback handoff: completed for feedback map persistence and runtime config
   exposure.
7. Formal high-fidelity entrypoint: completed for local Python/script entrypoint
   with run layout, manifest, stage status, metrics, and checkpoint writing.
8. Microtube TIFF training ingestion: completed for native TIFF reading,
   camera normalization, train input scaling, and high-fidelity entrypoint
   routing through `microtube_tiff_train_batch`.
9. ROI-bank gamma hook wiring: completed for high-fidelity entrypoint schedule
   wiring, `gamma_update_metrics.jsonl`, manifest/status route payloads, and
   gamma-only parameter updates that leave the localizer unchanged. The fixed
   ROI-library wake objective remains future work.
10. ROI projection gamma objective smoke wiring: completed for a tiny native
   ROI-bank path that runs current localizer posterior sampling, calls
   `GammaProjectionObjective`, and writes compact objective metadata to
   `gamma_update_metrics.jsonl`. Captured fixed ROI libraries and full monitor
   payload parity remain future work.
11. Configured ROI-library gamma objective wiring: completed for HDF5 ROI
   banks referenced by `train.roi_bank_gamma.roi_library_path`. The
   high-fidelity hook loads the fixed ROI bank, samples posterior values from
   the current localizer, calls `GammaProjectionObjective`, and records
   `objective_source=roi_projection_hdf5` plus the ROI library path. Full
   held-out monitor payload parity remains future work.
12. Gamma monitor payload parity: completed for compact old monitor field names
   in ROI projection `gamma_update_metrics.jsonl` rows, including selected ROI
   NLL, sampled emitter count, projected photons, background mean, held-out
   placeholders, and checkpoint/report link fields. Held-out reconstruction
   values remain future work.
13. Configured held-out ROI monitor smoke: completed for explicit HDF5 held-out
   ROI libraries via `train.roi_bank_gamma.heldout_roi_library_path`. Gamma
   updates sample fixed held-out posterior values and write held-out
   initial/final/delta/percentage/NLL monitor fields. Automatic held-out split
   construction and visual report diagnostics remain future work.
14. Automatic held-out split from configured ROI bank: completed for
   `auto_heldout_min_rois`/`auto_heldout_max_rois` when no explicit held-out
   HDF5 path is configured. The hook keeps a deterministic held-out tail split,
   trains gamma on the remaining selected ROI records, and records held-out ROI
   ids in metrics.
15. Compact gamma report artifacts: completed for ROI projection updates. Each
   update writes `gamma_alternation_summary.json` and
   `gamma_update_monitor.md` under run artifacts and links summary/report paths
   from `gamma_update_metrics.jsonl`.
16. Raw-vs-reconstruction PNG smoke diagnostic: completed for ROI projection
   updates with a compact `raw_vs_recon.png` generated from selected ROI raw
   data and the current gamma projection. Full historical figure-pack parity
   remains future work.
17. Auto ROI-bank construction entrypoint wiring: completed for smoke runs.
   When `train.roi_bank_gamma.auto_build_roi_bank=true` and
   `auto_build_source_path` is provided, the high-fidelity entrypoint builds a
   native ROI bank through the raw-frame ROI builder and runs the same
   selected/held-out gamma monitor path. The legacy path remains supported.
18. Dataset-agnostic ROI-bank source contract: completed for smoke wiring and
   microtube compatibility. `train.roi_bank_gamma.roi_bank_source` can be a
   generic auto-build mapping with `raw_path`, `frame_range`, `roi_size_px`,
   `candidate_mode`, and `domains`; the materialized microtube config maps the
   legacy `roi_bank_source: loc_infer_raw_tiff` alias onto the same route.
   Scientific ROI candidate selection validation remains future work.
19. Domain-aware ROI-bank artifact grouping: completed for existing
   summary/report/PNG artifacts. ROI projection gamma artifacts are written
   under `source_<source>/domain_<domain>/`, and metrics record the selected
   artifact source/domain groups. Full historical diagnostics parity remains a
   separate future slice.
20. Historical diagnostics parity spec: completed as a spec-only slice in
   `docs/roi_gamma_diagnostics_parity_spec.md`. The old gamma/ZMap
   before-after, raw TIFF patch reconstruction, fixed ROI reconstruction,
   PSF shape-grid, and monitor-summary diagnostics are mapped to explicit
   v0.3 outputs, metrics, non-goals, and Slice 5.13+ implementation order.
21. Diagnostics manifest contract: completed. Each grouped ROI-bank gamma
   artifact directory writes `diagnostics/diagnostics_manifest.json`, marking
   compact monitor and diagnostics smoke outputs as available.
22. Diagnostics smoke checks: completed for CPU smoke. ZMap before-after, fixed
   ROI reconstruction, raw TIFF patch reconstruction, and PSF shape-grid
   diagnostics each write a summary JSON and PNG under the grouped diagnostics
   directory. These remain smoke checks, not full historical figure-pack or
   vector-PSF parity.
23. Training runtime contract audit: completed for explicit resolved-contract
   recording. `resolved_contract.training_runtime` now records optimizer
   name/params, scheduler status, configured grad-clip norm, and configured AMP
   dtype/status. The high-fidelity entrypoint writes the same contract into the
   run manifest and stage status. This slice was audit/contract only; actual
   grad-clip wiring is tracked in Slice 6.2.
24. Training runtime grad-clip wiring: completed. Configured
   `train.online_generation.grad_clip_norm` now sets
   `resolved_contract.training_runtime.grad_clip.active=true`, flows through
   `build_trainer_runtime` into `EpochTrainingConfig`, and clips gradients in
   `train_one_epoch` after backward and before optimizer step. AMP remains a
   future runtime slice.
25. AMP contract hardening: completed without enabling AMP runtime behavior.
   `resolved_contract.training_runtime.amp` now records `configured`, `dtype`,
   `active=false`, and an explicit `inactive_reason` so CPU smoke runs with
   `amp_enabled=true` cannot be mistaken for autocast/scaler parity.
26. Scheduler parity audit/spec: completed before runtime wiring.
   `docs/training_scheduler_parity_spec.md` records the old active StepLR
   route and v0.3 maps `smlm_overrides.lr_scheduler`, `lr_step_size`,
   `lr_gamma`, and `lr_step_unit` into
   `resolved_contract.training_runtime.scheduler`.
27. Optimizer parity audit/spec: completed without switching the trainer from
   SGD. `docs/training_optimizer_parity_spec.md` records the old active AdamW
   route and v0.3 now records current runtime optimizer separately from
   `resolved_contract.training_runtime.legacy_optimizer`.
28. AdamW runtime wiring: completed for explicit runtime configs. The generic
   trainer now accepts `adamw`/`AdamW` optimizer specs and persists AdamW state
   through the existing checkpoint path.
29. Microtube active optimizer switch: completed. The materialized microtube
   active high-fidelity route now resolves legacy `smlm_overrides.optimizer`
   to runtime `adamw` when no explicit optimizer override is provided. Its
   manifest/status contract records AdamW `lr`/`weight_decay` and
   `legacy_optimizer.active=true`; AMP remains inactive.
30. StepLR runtime wiring: completed for the active optimizer-step route. The
   generic trainer now builds StepLR from the resolved scheduler contract,
   calls `scheduler.step()` after each optimizer update, persists
   `scheduler_state_dict` in checkpoints, restores scheduler state on resume
   when supplied, and marks the materialized microtube scheduler contract
   active.
31. Scheduler resume hardening: completed. Best checkpoints are covered for
   scheduler state persistence, inactive scheduler contracts remain ignored by
   the runtime factory, and `load_training_checkpoint` now fails fast when a
   scheduler is supplied for a checkpoint without `scheduler_state_dict`.
32. AMP policy/spec: completed. `docs/training_amp_parity_spec.md` defines the
   CUDA `autocast`/`GradScaler` route, CPU plain-FP32 behavior when CUDA is not
   available, and `scaler_state_dict` checkpoint requirements for active CUDA
   AMP.
33. AMP runtime wiring: completed for CUDA. The generic trainer now accepts
   `amp.active=true` contracts, propagates AMP settings into
   `EpochTrainingConfig`, wraps forward/loss computation in CUDA autocast, and
   uses `torch.amp.GradScaler`.
34. Generic inactive AMP coverage: completed. Generic trainer tests cover
   `configured=true, active=false` AMP contracts and verify inactive AMP
   checkpoints do not contain `scaler_state_dict`.
35. Localizer eval / held-out provider spec: completed without implementing
   the provider. `docs/training_eval_heldout_parity_spec.md` separates
   localizer `EvalProvider`/`eval_metrics.jsonl`/`checkpoint_best.pt` behavior
   from ROI-bank gamma held-out monitoring and defines Slice 6.13 acceptance
   criteria.
36. Online localizer eval provider wiring: completed for
   `train.eval.enabled=true, source=online_generation`. The shared
   `training.localizer_eval` helper builds fixed native online eval batches,
   and both high-fidelity and lightweight localization entrypoints pass them
   to `train_epochs`. The entries write `eval_metrics.jsonl`, produce
   `checkpoint_best.pt`, and record localizer eval route metadata plus the
   best-checkpoint path in manifest/status. Materialized microtube eval remains
   future work.
37. Online localizer eval contract hardening: completed without adding
   materialized eval. `train.eval.batch_count` and `train.eval.batch_size`
   must be positive, the eval helper reuses the online runtime config mapping
   for FiLM/SoftMoE/domain-aware fields, relative `dual_domain_coeff_maps`
   resolve against the config directory, and active SoftMoE eval is covered by
   helper and high-fidelity entrypoint smoke tests.
38. Dataset-agnostic materialized localizer eval contract: completed as a
   spec/test slice without implementing a provider.
   `docs/materialized_localizer_eval_contract.md` defines
   `source=materialized_dataset` with `dataset_id`, `sample_id`,
   `source_path`, `frame_range`, `crop`, and `heldout_split`. Microtube is
   explicitly only the first compatibility fixture or legacy alias, not the
   whole design. Runtime guard tests verify materialized sources still fail
   fast until Slice 6.16 wires a provider.
39. Dataset-agnostic materialized localizer eval provider: completed for the
   first supervised `.npz` route. `source=materialized_dataset` now builds
   fixed eval batches from `model_input`, `detect_tar`, `bkg_tar`,
   `pxyz_tar`, and `mask_tar`, resolves relative `source_path` against the
   config directory, records dataset/sample/frame/crop/heldout metadata in
   manifest/status, and writes `eval_metrics.jsonl` plus `checkpoint_best.pt`
   through the generic trainer. `materialized_microtube` maps to the same
   generic contract as a compatibility alias. Direct TIFF/HDF5/manifest
   ingestion remains future work.
40. Best-checkpoint resume hardening: completed for localizer eval. When
   resuming in a run directory with an existing `checkpoint_best.pt` carrying
   `eval_loss`, `train_epochs` now initializes the best eval threshold from
   that checkpoint and does not overwrite the historical best checkpoint with
   a worse resumed eval result. This is targeted runtime hardening, not the
   full old checkpoint initialization compatibility plan.

## Next Work Plan

### Phase 1: Native Online Batch Provider

Status: completed for the lightweight CPU smoke path. High-fidelity simulator
integration remains future work.

Goal: replace the placeholder deterministic online provider with a clean v0.3
native provider.

Acceptance criteria:

- The provider is implemented entirely under `neptune_v03`.
- No imports from `neptune_iwae`.
- Config fields come from `train.online_generation` and simulator config.
- It returns `TrainingBatch` objects with `LocalizationTrainBatch.inputs`.
- It supports deterministic seeds by epoch and step.
- It has small CPU tests using lightweight v0.3 simulator pieces.

Suggested tests first:

- Provider emits correct tensor shapes for sequence-window and triplet modes.
- Provider is deterministic for same seed/epoch/step.
- Provider changes generated samples across steps.
- Invalid dimensions and `steps_per_epoch <= 0` fail clearly.

### Phase 2: Clean Localization Model Runtime

Status: completed for the lightweight CPU smoke model. Production model
architecture remains future work.

Goal: add a native localizer model factory and output/loss contract that can
train through `TrainerRuntime`.

Acceptance criteria:

- Model factory is registered explicitly.
- The model accepts the native online batch input shape.
- Loss works through `make_localization_loss`.
- A toy or small real localizer completes one train step and updates
  parameters.
- Runtime config builds model, optimizer, batch provider, loss, and epochs.

Suggested tests first:

- Runtime factory builds a localization training runtime from config.
- One train epoch writes metrics and checkpoint.
- Loss decreases or parameters update on a deterministic toy localization
  target.

### Phase 3: Evaluation and Resume Hardening for Localization

Status: completed for the lightweight CPU smoke localization runtime. This is
generic trainer eval/resume coverage, not the Slice 6.13 high-fidelity
localizer eval provider.

Goal: prove the real localization runtime uses the generic evaluation and resume
features correctly.

Acceptance criteria:

- Fixed eval batches are generated once and reused.
- Eval loss is written to `eval_metrics.jsonl`.
- Best checkpoint updates only on improvement.
- Resume restores model, optimizer, epoch, and `global_step`.
- Resumed training continues without overwriting previous metrics incorrectly.

Suggested tests first:

- Train for one epoch, resume, train next epoch, assert `global_step`.
- Eval provider remains fixed across epochs.
- Best checkpoint corresponds to lowest eval loss.

### Phase 4: Fixed ROI Library and Posterior Sampling

Status: completed for lightweight ROI batch conversion and detached detection
posterior smoke sampling.

Goal: connect localization training to fixed ROI records without gamma updates
yet.

Acceptance criteria:

- Fixed ROI library records can be loaded into localization/eval batches.
- Current localizer can run on selected ROI records.
- Posterior parameter extraction has a native v0.3 contract.
- Detached posterior samples are reproducible for a fixed seed.

Suggested tests first:

- Small in-memory ROI library produces valid localization input batches.
- Posterior sampler returns masked `xyzph` tensors with expected shapes.
- Samples are detached from localizer gradients.

### Phase 5: Gamma-Only Update Hook

Status: completed for the scheduled hook, injected objective smoke path, and
native projection objective boundary.

Goal: add gamma update as an epoch-end hook without coupling it into the
generic trainer.

Acceptance criteria:

- Hook runs only on configured epochs.
- Hook uses fixed ROI records and detached posterior samples.
- Gamma optimizer updates gamma parameters only.
- Metrics and monitor payload are written.
- Localizer weights do not change during gamma update.

Suggested tests first:

- Hook schedule fires at expected epochs.
- Gamma parameter changes; localizer parameter remains unchanged.
- Over-cut crop changes objective region as configured.

### Phase 6: Training Entrypoint

Status: completed for the lightweight local smoke entrypoint and the formal
high-fidelity local Python/script entrypoint. SLURM script generation exists in
`neptune_v03.launch.spec`; cluster execution remains future validation work.

Goal: create a minimal local/SLURM entrypoint for the clean training path.

Acceptance criteria:

- Entrypoint materializes config or accepts resolved config.
- Creates run layout, manifest, metrics, checkpoints, and status files.
- Supports smoke config locally.
- Does not write runtime artifacts into source directories.

Suggested tests first:

- CLI parses a smoke config and writes run artifacts under a temp output dir.
- Failure records stage status.
- Smoke training produces checkpoint and metrics.

## Working Rules for Future Sessions

- Start by reading this document, `docs/architecture.md`, and
  `docs/active_pipeline.md`.
- For neptune_iwae parity work, also read
  `docs/migration_gap_audit.md` before choosing the next slice.
- If touching training/localization behavior, run the targeted tests first.
- Add or update tests before implementation.
- After each completed slice, update this document's `Completed`, `Not Done`,
  and `Next Work Plan` sections.
- Before claiming completion, run the full pytest command above.
- Move generated `__pycache__` and test cache residue under `.local/cache/`.

## Do Not Do

- Do not bridge old `neptune_iwae` runtime functions into v0.3.
- Do not import old `build_online_train_batch` or old runtime setup.
- Do not copy `cached_window_train.py` wholesale.
- Do not make `training` depend on localization internals beyond registered
  provider/loss factories.
- Do not mark training complete until real localization model, native online
  batches, eval, resume, and checkpoint paths all pass tests.
