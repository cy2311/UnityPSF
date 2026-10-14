# UnityPSF Codebase Simplification Review

> Status: Superseded on 2026-08-21 by
> [unity-v0.2-development-overdesign-review-20260821.md](unity-v0.2-development-overdesign-review-20260821.md).
> This file preserves the original baseline and implementation history; do not
> use its phase numbering or line counts as the current roadmap.

Date: 2026-08-20

Scope: `/home/guest/Others/main/race/unity`

Goal: identify how to reduce code size and maintenance cost without changing training, inference, checkpoint, physical-update, or artifact behavior.

## Executive Summary

UnityPSF has a sound top-level ownership direction: new implementation lives under `unity_psf`, while `double_helix` and `neptune_v04` are compatibility surfaces. The architecture test preventing imports from the new package back into legacy package roots is an important constraint and should remain.

The primary maintainability risk is not the number of packages. It is responsibility concentration inside a few modules:

- `src/unity_psf/training/high_fidelity/engine.py`: 4,297 lines.
- `src/unity_psf/localization/data/online.py`: 2,878 lines.
- `src/unity_psf/localization/runtime/config.py`: 1,229 lines.
- `src/unity_psf/peak/nat_optimizer.py`: 976 lines.
- `src/unity_psf/peak/vector_nat_fit.py`: 891 lines.
- `src/unity_psf/training/loop.py`: 825 lines.

These modules combine orchestration, policy, compatibility, numerical operations, diagnostics, persistence, and configuration parsing. This makes otherwise local changes expensive to review and increases regression risk.

The safest optimization is incremental extraction with characterization tests. Do not rewrite the training pipeline, merge compatibility formats, or replace the current configuration contract in one change. The recommended sequence is:

1. Lock current behavior with contract and golden-output tests.
2. Extract pure diagnostics and artifact-writing code.
3. Split runtime configuration by responsibility while retaining one public facade.
4. Separate online batch strategies and rendering backends behind the existing provider API.
5. Consolidate common modality runtime construction and training reporting.
6. Inventory legacy surfaces using measured usage before considering removal.

## Review Method

This review used static repository inspection rather than behavioral modification. Evidence included:

- line counts across `src`, `scripts`, and `tests`;
- public console-script definitions in `pyproject.toml`;
- class and function inventories in the largest modules;
- searches for legacy and compatibility references;
- package-boundary tests;
- configuration layout and repeated training keys;
- existing test organization.

Project snapshot:

- 238 Python files across source, scripts, and tests;
- 38 `test_*.py` test modules;
- approximately 64,523 lines across Python, CUDA, shell, and Slurm files;
- 60 files under `scripts/archive` and `src/double_helix/legacy`;
- 11 published console entry points;
- three installed package roots: `unity_psf`, `double_helix`, and `neptune_v04`.

Static inspection cannot prove that a compatibility wrapper or archive script is unused. Removal recommendations below therefore require runtime or user confirmation.

## Cleanup Completed On 2026-08-20

The first cleanup pass removed only reproducible runtime material. No production source, configuration, test, training launcher, checkpoint, or active output was deleted.

Moved to the recoverable quarantine directory:

```text
.local/trash/redundant-runtime-20260820/
```

Contents isolated:

- 3,179 Python bytecode files (`.pyc`/`.pyo`);
- source and test `__pycache__` directories;
- `.pytest_cache`;
- two old wheel-install temporary directories;
- migration bytecode under `.local/cache/unity-migration`;
- seven zero-byte historical Slurm log files.

Measured quarantine size: 3,373 files and approximately 78.6 MB.

The original source-tree bytecode was additionally preserved in `source-pycache.tar.gz` inside the quarantine directory. This is recoverable local material, not project source and not a Git submission candidate.

## Files Deliberately Retained

The following candidates were checked and were not classified as completely redundant:

| Candidate | Why it remains |
|---|---|
| `src/double_helix/run_calibration.py` and `run_evaluation.py` | Referenced by archived Slurm workflows and provide compatibility entry points. |
| `src/double_helix/legacy/` | Historical DH calibration, simulation, and evaluation scripts are referenced by archive jobs. |
| `scripts/archive/` | README and reproducibility documentation present these as runnable historical workflows. |
| `src/neptune_v04/` | Retained import compatibility surface and documented in ADRs and package configuration. |
| `src/unity_psf/contracts/*legacy*` | Legacy checkpoint payloads and schema versions have active compatibility tests. |
| `src/unity_psf/localization/legacy_decode.py` | Legacy target order and evaluation paths are still part of training/evaluation compatibility behavior. |
| `scripts/globloc/mex.h` | Included by the upstream CUDA integration even though the local shim is intentionally minimal. |
| `tests/cli/__init__.py` | Empty package marker; removing it would change test-package behavior on supported Python environments. |

No formal source file was deleted in this pass because each apparent candidate had at least one import, script, documentation, test, checkpoint, or compatibility reference. Archive and legacy code require a measured usage inventory before removal.

## Post-Cleanup Verification

Focused regression tests remain green after the cleanup:

```text
16 passed, 1 warning
```

Command:

```bash
/home/guest/anaconda3/bin/python -m pytest -q \
  unity/tests/configs/test_default_three_expert_scheme.py \
  unity/tests/models/test_astigmatism_expert.py \
  unity/tests/contracts/test_modality.py
```

The warning was a CUDA initialization warning from the host environment; it did not fail a test. The cleanup did not alter the current training output tree.

## Findings

### P1: High-fidelity training is a monolithic change surface

Evidence: `src/unity_psf/training/high_fidelity/engine.py` is 4,297 lines and contains all of the following:

- CLI parsing and the top-level `main` orchestration around line 250;
- resume and runtime construction;
- peak z-map bootstrap;
- gamma-update routing and feedback construction;
- physical-state persistence and checkpoint metadata around line 1,026;
- Zernike map merging and artifact export around line 1,169;
- posterior sampling and ROI projection objectives;
- gamma monitor Markdown and JSON generation around line 2,515;
- raw-TIFF and reconstruction diagnostics;
- hand-written PNG serialization around line 3,273;
- ROI-bank construction and inference adapters.

Risk:

- Any edit reloads a very large context and makes ownership difficult to identify.
- Pure formatting or diagnostic changes can accidentally affect the training orchestration module.
- Private helpers are effectively a large implicit API because tests and future changes must reason about the whole file.
- Numerical logic, filesystem side effects, and presentation logic cannot be tested independently as cleanly as they should be.

Recommended boundary, retaining existing behavior and call order:

```text
training/high_fidelity/
  engine.py                 thin orchestration and existing public entry point
  bootstrap.py              peak z-map bootstrap and config resolution
  physical_state.py         physical-state read/write/hash/checkpoint extras
  gamma_runtime.py          gamma route, hooks, update scheduling
  posterior_update.py       posterior sampling and ROI update contexts
  projection.py             projection objective construction and tensors
  diagnostics.py            metric assembly and diagnostic decisions
  diagnostic_rendering.py   PNG arrays, tiling, scaling, serialization
  roi_sources.py            ROI-bank source resolution and inference adapters
```

Safety rule: move functions first without changing signatures or branching. Keep re-exports or forwarding imports in `engine.py` until all tests and scripts use stable module ownership.

Expected benefit: reduce `engine.py` to approximately 400-700 lines of orchestration and make numerical, persistence, and diagnostics behavior independently testable.

### P1: Online data generation mixes strategy, backend, cache, and conditioning

Evidence: `src/unity_psf/localization/data/online.py` is 2,878 lines and contains:

- vector renderer and LUT caches starting around line 114;
- LUT interpolation and optional Triton backends;
- `OnlineBatchProviderConfig` around line 645;
- environment-variable feature switches;
- provider construction around line 813;
- physical-versioned LUT lifecycle around line 899;
- native and LUT simulation paths;
- cached-window sequence classes and batching around line 1,234;
- pxyz target conversion around line 2,183;
- camera readout;
- domain and conditioning feature construction;
- condition-provider loading and coordinate calculations.

Risk:

- `cached_window`, native simulation, and LUT simulation are behaviorally distinct strategies but share one implementation module.
- Performance-sensitive code is coupled to configuration normalization and target formatting.
- Backend-specific optimizations are hard to benchmark or disable independently.
- A change intended for one modality can affect all online providers.

Recommended boundary:

```text
localization/data/online/
  __init__.py               preserve build_online_batch_provider API
  config.py                 OnlineBatchProviderConfig and validation
  provider.py               strategy selection only
  cached_window.py          cached-window lifecycle and slicing
  native.py                 native vector simulation sequence path
  lut.py                    LUT bank, interpolation, and LUT sequence path
  projection.py             patch shifting and frame projection backends
  camera.py                 Poisson/readout conversion
  conditioning.py           condition stores, origins, feature vectors
  targets.py                pxyz ordering and padding/finalization
```

Safety rule: retain `build_online_batch_provider`, `OnlineBatchProviderConfig`, deterministic seed derivation, sequence ordering, tensor dtype/device, and metric keys exactly. Extracting cached-window code is especially valuable because it is now a validated Double Helix performance-critical path.

### P1: Runtime configuration is both a facade and a migration engine

Evidence: `src/unity_psf/localization/runtime/config.py` is 1,229 lines and performs:

- top-level runtime materialization;
- modality/expert detection;
- loss selection and legacy loss translation around line 314;
- input and conditioning contract construction;
- optimizer and scheduler resolution around line 607;
- compatibility with legacy optimizer fields;
- online provider materialization around line 775;
- microtube and raw data provider configuration;
- density and physical range conversion;
- path resolution and coefficient-map loading.

Risk:

- New configuration fields can interact with legacy translation far from the field definition.
- Mapping-based data is repeatedly normalized, making it difficult to distinguish accepted input schema from resolved runtime schema.
- Error-message behavior and default behavior are easy to change during cleanup.

Recommended boundary:

```text
localization/runtime/
  config.py                 public build_localization_runtime_config facade
  schema.py                 typed resolved contracts, no YAML parsing
  model_config.py           expert/model selection
  loss_config.py            current and legacy loss translation
  optimizer_config.py       optimizer/scheduler translation
  provider_config.py        online/microtube/raw-TIFF provider resolution
  conditioning_config.py    condition dimension and field contracts
  paths.py                  base-dir-aware path resolution
```

Safety rule: first add table-driven tests that snapshot the fully resolved runtime dictionaries for every committed modality configuration. Preserve dictionary keys, default values, exception types, and important error strings during extraction.

### P1: Current tests are not proportionate to refactor blast radius

There are 38 test modules, which is meaningful coverage, but the largest and most coupled modules require stronger characterization before structural refactoring.

Gaps to close before extraction:

- Golden resolved-runtime snapshots for every formal configuration.
- Seed determinism tests for native, LUT, cached-window, and physical-version changes.
- Tensor contract tests covering shape, dtype, device, target order, and metric keys.
- Artifact tree snapshots for high-fidelity diagnostics and physical-state files.
- Resume equivalence tests comparing uninterrupted and resumed training state.
- CLI smoke tests for every published entry point in `pyproject.toml`.
- Import tests for both modern and compatibility package roots.

The existing package-boundary test in `tests/architecture/test_package_boundaries.py` should be expanded, not removed. It correctly enforces that `unity_psf` does not import `double_helix` or `neptune_v04`.

### P2: Sequential and expert-parallel entry points duplicate lifecycle concepts

Evidence:

- `training/entrypoints/train_modality_joint.py` builds modality runtimes, trains epochs, accumulates metrics, writes checkpoints, validates loadability, writes summaries, and updates stage status.
- `training/entrypoints/train_modality_expert_parallel.py` calls the same runtime builder and epoch trainer but separately defines metric initialization, accumulation, shard status, release publication, and failure status.
- `_build_modality_runtime` is private in one entry-point module but imported by the other, meaning it is already shared implementation with misleading ownership.

Recommended change:

- Move `_build_modality_runtime` and related runtime-audit helpers to `training/modality_runtime.py`.
- Introduce a small `ModalityProgress` value object for metric initialization and accumulation.
- Keep sequential orchestration and distributed coordination separate; do not force them through one generalized runner.
- Centralize summary schema construction and checkpoint validation, while retaining mode-specific fields.

This reduces duplication without hiding the important difference between local sequential execution and rank-coordinated expert-parallel execution.

### P2: Compatibility packages are correctly isolated but lack an explicit retirement policy

Evidence:

- `pyproject.toml` installs `unity_psf`, `double_helix`, and `neptune_v04`.
- `src/double_helix/run_calibration.py` and `run_evaluation.py` are explicitly deprecated wrappers.
- `src/neptune_v04/__init__.py` preserves legacy imports.
- `src/double_helix/legacy` and `scripts/archive` remain referenced by archived Slurm scripts.
- Checkpoint, target-order, and model compatibility paths have active tests.

Conclusion: these files are not automatically dead code. Some are compatibility commitments and some are historical reproducibility assets.

Recommended classification:

```text
Compatibility API
  Import wrappers and checkpoint readers required by old users/checkpoints.
  Keep installed and tested until a documented deprecation deadline.

Reproducibility archive
  Old training and evaluation scripts needed to reproduce named experiments.
  Keep in an archive package or tagged release; exclude from the primary API narrative.

Obsolete implementation
  Code with no import, script, documented experiment, or checkpoint dependency.
  Remove only after an inventory and reference test prove it is unused.
```

Add `docs/compatibility.md` containing supported checkpoint versions, import aliases, deprecated commands, and the removal criteria. This turns implicit compatibility into an explicit contract.

### P2: Configuration composition is only partially normalized

Training values such as epochs, batch size, loss, optimizer, scheduler, simulation backend, modality contract, and provider details appear across base, experiment, modality, and override YAML files. This is expected in experiment code, but ownership of each value is not obvious from the file hierarchy alone.

Recommended rules:

- Base configs own stable physical and runtime defaults.
- Modality configs own expert, PSF, target, loss, and provider behavior.
- Experiment configs own composition, seeds, epochs, steps, rank assignment, and run metadata.
- Overrides own only intentional deltas and should remain short.
- Materialized configs are runtime artifacts and must never be hand-edited or committed as source.

Add tests that reject duplicated ownership for a small set of critical keys. Do not aggressively introduce YAML anchors: cross-file anchors are poorly supported and can make provenance harder to understand. Prefer the existing structured materialization path.

### P2: Scripts contain both supported product entry points and one-off research programs

The `scripts` tree includes formal training launchers, GUI/web submission utilities, inference pipelines, GlobLoc integration, diagnostics, and archived sweeps. These have different maintenance expectations but are presented as peers.

Recommended organization:

```text
scripts/
  train/          supported launchers
  infer/          supported inference/reconstruction programs
  diagnostics/    supported inspection tools
  integrations/   GlobLoc and other external integrations
  tools/          local utilities and smoke tools
  archive/        frozen reproducibility scripts
```

Each supported script should either call a package entry point or contain only orchestration that is inherently environment-specific. Reusable parsing, numerical, and IO logic should live under `src/unity_psf`. Archive scripts should be considered frozen except for security or reproducibility fixes.

### P2: Public API breadth is wider than documented ownership

`pyproject.toml` publishes 11 console commands spanning config materialization, diagnostics, localization training, high-fidelity training, multichannel and joint training, checkpoints, and Double Helix calibration/evaluation.

Recommended improvement:

- Document every command with status: primary, advanced, compatibility, or internal.
- Ensure each command accepts `--help` without optional scientific data.
- Add a console-entry-point smoke test that resolves every target callable.
- Prefer one canonical command per workflow; retain aliases only as documented compatibility wrappers.

Do not merge all commands into one large CLI until workflows and option contracts are stable. A large subcommand framework would change more code than it removes.

### P3: Diagnostics and PNG serialization are reusable infrastructure hidden in training code

High-fidelity training includes display scaling, tiling, Poisson NLL, NCC, raw-versus-reconstruction rendering, PNG chunk serialization, JSON summaries, and Markdown rendering. These functions are valuable but are not training orchestration.

Move them into diagnostics modules with direct unit tests. Preserve byte output only where consumers require byte identity; otherwise preserve image dimensions, dtype, content metrics, paths, and manifest schema.

Before changing PNG implementation, benchmark the existing standard-library serializer against the already-declared Matplotlib dependency. Do not add Pillow solely for convenience unless it produces a measured simplification and dependency policy accepts it.

### P3: Mapping-heavy internal APIs increase repetitive validation

Many internal functions accept `Mapping[str, Any]` and repeat `_mapping`, string conversion, optional field checks, and dictionary assembly. This is appropriate at YAML and checkpoint boundaries but expensive inside resolved runtime logic.

Recommended approach:

- Keep mappings at external boundaries and serialized artifacts.
- Introduce frozen dataclasses for stable resolved internal contracts only after current dictionaries are characterized.
- Convert back to the existing dictionary schema at public/runtime boundaries.
- Avoid a repository-wide typing rewrite.

Best initial candidates are physical-state identity, resolved optimizer/scheduler config, provider strategy config, and modality progress metrics.

### P3: Test modules mirror accumulated feature history more than current ownership

Some test modules are large, particularly `test_modality_joint_training.py` at 906 lines and `test_dual_modality_dual_channel_300epoch.py` at 490 lines. Large tests make it difficult to see which contract protects which subsystem.

Split tests by behavior after production boundaries are extracted:

- runtime construction;
- epoch scheduling;
- checkpoint assembly and resume;
- held-out evaluation;
- physical-state isolation;
- formal config contract;
- expert-parallel coordination.

Do not split tests merely by line count. Each new module should correspond to a stable behavior boundary.

## Proposed Target Architecture

The target should retain the existing domain packages while making execution dependencies one-directional:

```text
configs / CLI
      |
      v
runtime config facade
      |
      +--> typed resolved contracts
      |
      v
training orchestration
      |
      +--> modality runtime and training loops
      +--> data provider strategies
      +--> physical/gamma services
      +--> diagnostics/artifact services
      |
      v
models / optics / localization primitives
```

Compatibility packages may depend on `unity_psf`. `unity_psf` must continue to avoid importing compatibility roots.

## Phased Refactoring Plan

### Phase 0: Establish behavior baselines

No production movement yet.

1. Add resolved-config golden tests for all committed formal configs.
2. Add seed and tensor-contract tests for all online batch strategies.
3. Add representative artifact manifest snapshots.
4. Record checkpoint round-trip and resume-equivalence baselines.
5. Record import and `--help` behavior for all console scripts.

Exit gate: full test suite passes and baseline artifacts are stored as small fixtures or normalized JSON, not large binary outputs.

### Phase 1: Extract pure diagnostics

Move rendering, display conversion, metric calculation, Markdown rendering, and PNG serialization out of `high_fidelity/engine.py`.

Why first: these functions have limited control-flow coupling and can be tested with small tensors. This provides meaningful line reduction with low training risk.

Exit gate:

- diagnostic relative paths unchanged;
- manifest keys unchanged;
- image dimensions and pixel checks unchanged;
- high-fidelity smoke tests unchanged.

### Phase 2: Extract physical-state persistence

Move physical-state read/write/hash/checkpoint-extra logic into a dedicated service module.

Exit gate:

- schema versions unchanged;
- hashes unchanged for identical inputs;
- artifact paths unchanged;
- legacy and current checkpoint tests pass;
- resume from an existing checkpoint remains supported.

### Phase 3: Split runtime configuration internals

Retain `build_localization_runtime_config` as the public facade. Extract loss, optimizer, provider, conditioning, and path logic behind it.

Exit gate: deep equality of resolved runtime dictionaries for every golden config, including error cases.

### Phase 4: Split online provider strategies

Extract config, cached-window, LUT, native, conditioning, target, and camera modules. Preserve the current factory API.

Exit gate:

- deterministic outputs for fixed seeds;
- identical target order and masks;
- identical tensor shapes, dtypes, and devices;
- cached-window throughput does not regress beyond an agreed tolerance;
- DH default remains `cached_window`, vector batch size 1024, and FP16 AMP where configured.

### Phase 5: Consolidate modality runtime ownership

Move shared runtime construction out of the sequential entry-point module. Extract progress aggregation and release summary construction.

Exit gate:

- sequential and expert-parallel summaries keep their schemas;
- rank status and timeout behavior remain unchanged;
- shard resume behavior remains unchanged;
- release checkpoint activation audit remains unchanged.

### Phase 6: Classify and reduce legacy surface

Create an explicit manifest of legacy modules and archive scripts. For each item record:

- importing code or command;
- experiment/release that requires it;
- checkpoint or data format dependency;
- replacement path;
- removal eligibility.

Only delete entries proven obsolete. Move reproducibility-only implementations out of the installed package only after verifying archived scripts can run from a documented environment or tag.

## Behavior-Consistency Gates

Every refactoring commit should satisfy all applicable gates below.

### API and import behavior

- Existing `unity_psf` imports continue to work.
- Documented `double_helix` and `neptune_v04` compatibility imports continue to work.
- Console entry-point names and argument behavior remain stable.
- `unity_psf` continues not to import legacy package roots.

### Numerical behavior

- Fixed-seed model initialization matches.
- Fixed-seed batch sampling matches.
- Tensor shape, dtype, device, and target order match.
- Loss and representative gradient values match within declared tolerances.
- AMP behavior and router/expert dtype behavior remain tested.

### Training behavior

- Optimizer and scheduler state match after representative steps.
- Epoch and global-step accounting match.
- Skipped-step and non-finite handling match.
- Sequential and expert-parallel outputs retain schema and semantics.

### Persistence behavior

- Checkpoint schemas and compatibility loaders remain stable.
- Resume produces the same next epoch/global step and equivalent optimizer state.
- Physical-state hashes, coefficient-map identity, and artifact references remain valid.
- Atomic-write behavior is preserved.

### Data and performance behavior

- `cached_window` ordering and reuse semantics remain stable.
- Native and LUT provider density and photon contracts remain stable.
- Raw-TIFF physical-update path continues to use the configured source and camera conversion.
- Benchmark DH samples/second, GPU utilization, memory, and time per epoch before and after relevant changes.

### Artifact behavior

- Directory layout and relative paths remain stable.
- JSON and Markdown schema keys remain stable.
- Diagnostic images remain nonblank and correctly shaped.
- Runtime outputs remain ignored by Git.

## Suggested Commit Sequence

Keep each change independently reviewable and reversible:

1. `test: characterize resolved UnityPSF runtime configs`
2. `test: lock online provider seed and tensor contracts`
3. `refactor: extract high-fidelity diagnostic rendering`
4. `refactor: isolate physical-state persistence`
5. `refactor: split runtime config resolvers`
6. `refactor: isolate cached-window provider strategy`
7. `refactor: isolate LUT and native provider backends`
8. `refactor: centralize modality runtime construction`
9. `refactor: centralize modality progress reporting`
10. `docs: define UnityPSF compatibility lifecycle`

Avoid commits combining refactoring with new training behavior, parameter changes, performance tuning, or checkpoint schema changes.

## What Not To Do

- Do not rewrite the 4,297-line engine in one change.
- Do not replace mapping configs with dataclasses across the repository at once.
- Do not delete legacy checkpoint handling because modern checkpoints pass tests.
- Do not remove `double_helix` wrappers solely because `unity_psf` no longer imports them.
- Do not deduplicate sequential and distributed orchestration into a highly generic framework.
- Do not introduce a new dependency unless it removes measured complexity or improves correctness.
- Do not change config defaults during a structural refactor.
- Do not move archived scripts into active package APIs.
- Do not optimize GPU code without benchmark receipts from equivalent Slurm hardware.

## Priority Matrix

| Priority | Work item | Expected benefit | Main risk |
|---|---|---|---|
| P0 | Characterization tests | Enables all later simplification | Fixture brittleness if outputs are not normalized |
| P1 | Extract high-fidelity diagnostics | Large low-coupling line reduction | Artifact path/schema drift |
| P1 | Split runtime config internals | Clearer contracts and safer config changes | Default/error behavior drift |
| P1 | Isolate cached-window/LUT/native providers | Safer performance work and modality isolation | Seed/order/performance drift |
| P2 | Centralize modality runtime construction | Removes misleading private cross-module API | Sequential/distributed behavior conflation |
| P2 | Add compatibility lifecycle manifest | Makes deletion decisions evidence-based | Premature deprecation commitments |
| P2 | Reclassify supported and archive scripts | Clearer supported surface | Reproducibility path breakage |
| P3 | Introduce typed internal contracts | Reduces repeated mapping validation | Excessive migration scope |
| P3 | Split large test modules by behavior | Easier ownership and review | Cosmetic splitting without better contracts |

## Recommended First Increment

The runtime-artifact cleanup described above is complete. The next implementation increment should now be deliberately narrow:

1. Add characterization tests for high-fidelity diagnostic helpers.
2. Move only image scaling, tiling, metric helpers, PNG serialization, and Markdown rendering to `training/high_fidelity/diagnostic_rendering.py`.
3. Keep forwarding imports in `engine.py` for one release or until all internal imports are updated.
4. Verify the full focused suite and one small GPU/Slurm smoke run.

This should remove several hundred lines from the largest module without touching optimizer steps, physical updates, data generation, or checkpoint behavior.

## Expected Code Reduction

The cleanup already removed approximately 78.6 MB of redundant local runtime material, but it did not reduce tracked source lines. Formal code reduction should be measured separately from module extraction:

| Stage | Expected tracked-line reduction | Notes |
|---|---:|---|
| Extract diagnostics from `high_fidelity/engine.py` | 100-250 net lines | Most of the 500-900 moved lines remain as clearer modules; deletion comes from shared rendering and artifact helpers. |
| Isolate runtime config resolvers | 150-350 net lines | Removes repeated mapping normalization and modality-specific branching after golden tests exist. |
| Split online provider strategies | 200-500 net lines | Shared range, target, conditioning, and projection helpers can replace repeated branches. |
| Centralize modality runtime/progress code | 200-400 net lines | Consolidates sequential and expert-parallel setup, metrics, and summary construction. |
| Test fixtures and assertion helpers | 200-500 net lines | Reduces repeated runtime, checkpoint, and metric setup without weakening assertions. |
| Proven obsolete compatibility/archive code | 0-4,000+ lines | Optional and only after external usage and reproducibility requirements are confirmed. |

Conservative target without removing supported compatibility or reproducibility behavior: 850-2,000 net tracked lines. A broader 4,000-8,000 line reduction is possible only if a later usage audit proves that substantial legacy/archive behavior is no longer required.

Structural simplification will be larger than the net line reduction. The immediate goal is to reduce the three dominant files from a combined 8,404 lines into independently owned modules, while keeping their public facades stable.

## Implementation Progress

### Phase 1 completed: diagnostic rendering and orchestration extraction

Completed on 2026-08-20:

- added `src/unity_psf/training/high_fidelity/diagnostic_rendering.py`;
- added `src/unity_psf/training/high_fidelity/diagnostics.py`;
- moved gamma-monitor Markdown rendering;
- moved raw-versus-reconstruction PNG composition;
- moved linear and background-anchored uint8 scaling;
- moved frame tiling;
- moved Poisson NLL and NCC diagnostic metrics;
- moved dependency-free grayscale PNG serialization;
- kept the original private helper names inside `engine.py` through explicit import aliases, so existing call sites and call order did not change;
- added `tests/training/test_high_fidelity_diagnostic_rendering.py` to characterize numerical values, PNG dimensions/signature, and Markdown field order.
- moved diagnostics manifest construction, Gamma monitor report orchestration, raw-TIFF diagnostic crop selection, reconstruction montage writers, vector PSF/z-map summaries, artifact grouping, path normalization, and metadata normalization out of `engine.py`;
- retained only `_artifact_group_metrics`, `_path_token`, and `_write_gamma_monitor_report` imports in `engine.py`, which are the diagnostics entry points used by training orchestration.

Measured result:

- `engine.py` reduced from 4,297 to 3,422 lines;
- diagnostics implementation moved out of the training orchestration module as one mechanically preserved block;
- net reduction inside the orchestration module: 875 lines;
- extracted diagnostics module: 174 lines;
- extracted diagnostics orchestration module: approximately 762 lines;
- seven focused behavior tests covering numerical rendering, PNG structure, Markdown fields, artifact grouping, path tokens, raw-TIFF domain cropping, and metadata normalization.

This increment intentionally improves ownership more than total repository line count. The extracted module and characterization tests create the safety boundary required for later deduplication.

Phase 1 is complete. ROI posterior sampling, gamma objectives, physical-state updates, data-provider behavior, optimizer behavior, checkpoint behavior, and training control flow remain in `engine.py` and were not changed.

The initial full-suite verification after Phase 1 produced 186 passes and three failures in pre-existing runtime-contract tests. Those failures were subsequently resolved in the dedicated baseline-contract pass documented below; none required changing diagnostics, physical-state persistence, or training numerical behavior.

### Phase 2 completed: physical-state persistence extraction

Completed on 2026-08-20:

- added `src/unity_psf/training/high_fidelity/physical_state.py`;
- moved initial physical-state publication;
- moved runtime coefficient-map entry normalization;
- moved legacy/no-context current physical-state JSON publication;
- moved run-manifest initial/latest physical-state hash updates;
- moved checkpoint-extra physical artifact and hash validation;
- moved current physical-state loading;
- retained the original private helper names through explicit imports in `engine.py`, preserving internal call sites and existing external tests that import those helpers.

Measured result:

- `engine.py` reduced from the Phase 1 result of 3,422 lines to 3,283 lines;
- net reduction inside the orchestration module during Phase 2: 139 lines;
- extracted physical-state module: 155 lines;
- cumulative reduction of `engine.py` from the original 4,297 lines: 1,014 lines;
- added focused tests for legacy physical-state payloads, exact manifest hashing, initial-hash preservation, coefficient-map normalization, missing-state behavior, and empty checkpoint extras.

Behavior retained:

- `current_physical_state.json` path and JSON fields;
- deterministic SHA-256 input serialization;
- `initial_physical_state_hash` set-once semantics;
- `latest_physical_state_hash` update semantics;
- atomic JSON publication;
- channel identity, coefficient-map hash, and peak-zmap hash validation;
- checkpoint extra keys and omission behavior;
- `ChannelTrainingContext` ownership and schema behavior;
- training, gamma feedback, resume, optimizer, data-provider, and checkpoint control flow.

Phase 2 is complete. Zernike delta calculation and gamma coefficient-map export remain in `engine.py` because they are numerical physical-update operations, not physical-state persistence.

### Phase 3 completed: runtime configuration resolver extraction

Completed on 2026-08-20:

- added `src/unity_psf/localization/runtime/optimizer_config.py` for optimizer, legacy optimizer, scheduler, AMP/gradient runtime contract, localization overrides, and z-activation normalization;
- added `src/unity_psf/localization/runtime/conditioning_config.py` for expert aliases, single-channel normalization, condition dimensions, domain one-hot policy, and soft-MoE dimension validation;
- added `src/unity_psf/localization/runtime/paths.py` for mapping, optional mapping, range/pair/grid normalization, and base-directory-aware path resolution;
- retained `runtime/config.py` as the public facade and preserved its private helper aliases so existing internal and test imports remain valid;
- kept provider materialization, modality contract construction, loss translation, and numerical physical configuration in the facade for the next independently gated phases.

Measured result:

- `runtime/config.py` reduced from 1,249 to 963 lines;
- extracted resolver helpers are independently importable and side-effect free;
- the public `build_localization_runtime_config` and `resolve_localization_model_config` signatures and resolved dictionary schema are unchanged;
- focused runtime/baseline verification: `39 passed, 1 warning`;
- full-suite verification after Phase 3: `191 passed, 4 warnings`.

Behavior retained:

- optimizer and scheduler defaults, legacy optimizer activation matching, AMP and gradient-clip contract fields;
- condition dimensions, domain counts, channel aliases, and single-channel coefficient-map binding;
- path resolution relative to `config_base_dir`, range validation, and original error messages;
- DH, astigmatism, emitter-2D, microtube, raw-TIFF, and legacy soft-MoE runtime routes;
- serialized runtime keys, checkpoint-facing contracts, target order, seed handling, and training provider behavior.

Phase 3 deliberately stops before splitting `_online_provider_config`, loss translation, and modality contract construction. Those areas have stronger cross-responsibility coupling and are scheduled for separately characterized increments.

### Phase 4 completed: online conditioning strategy extraction

Completed on 2026-08-20:

- added `src/unity_psf/localization/data/online_conditioning.py` for condition-field aliases, condition feature ordering, condition vector projection, domain selection, feature-dimension normalization, and vector padding/truncation;
- added `src/unity_psf/localization/data/online_targets.py` for pxyz target-order normalization, legacy/v03 conversion, target padding, and detection target construction;
- retained `build_online_batch_provider`, `OnlineBatchProviderConfig`, `_TrainingBatchSequence`, `_CachedWindowEpochBatches`, LUT lifecycle, native simulation, and camera/readout paths in `online.py`;
- kept private compatibility aliases in `online.py`, so all existing provider call sites and metadata keys remain unchanged;
- did not alter cached-window sequence ordering, deterministic seed derivation, LUT/native renderer selection, target order, tensor dtype/device, or metric visibility.

Measured result:

- `online.py` reduced from 2,878 to 2,767 lines;
- condition and target helper implementations are isolated in 70-line and 58-line side-effect-free modules;
- focused localization/provider verification: `32 passed, 4 warnings`;
- full-suite verification after Phase 4: `191 passed, 4 warnings`.

This is intentionally a conservative Phase 4 implementation. The performance-sensitive cached-window lifecycle, LUT bank, native renderer, projection kernels, and camera conversion remain in their original execution module until each receives a dedicated characterization/benchmark slice. The next Phase 4 increment should extract projection and camera helpers only after recording tensor shape, dtype, device, and frame-level equivalence on representative DH and astigmatism configurations.

### Phase 5 completed: modality runtime and progress ownership

Completed on 2026-08-21:

- added `src/unity_psf/training/modality_progress.py` for empty metric state, resume progress reconstruction, channel progress serialization, epoch accumulation, held-out accumulation, and all-channel held-out enablement checks;
- added `src/unity_psf/training/modality_runtime.py` for the shared runtime contract, provider-to-modality batch adapter, channel condition-store provider override, and physical-state snapshot helper;
- changed `train_modality_expert_parallel.py` to consume shared progress helpers while retaining rank identity, shard persistence, coordination timeout, and release publication locally;
- changed `train_modality_joint.py` to consume shared runtime helpers while retaining formal runtime audit, `_build_modality_runtime`, sequential scheduling, and joint summary/release behavior locally;
- preserved the existing CLI compatibility imports from `unity_psf.cli.train_modality_joint` and `unity_psf.cli.train_modality_expert_parallel`.

Measured result:

- `train_modality_joint.py` reduced from 744 to 673 lines;
- `train_modality_expert_parallel.py` reduced from 759 to 615 lines;
- extracted shared modules total 148 lines;
- focused modality/CLI verification: `29 passed, 1 warning`;
- full-suite verification after Phase 5: `191 passed, 4 warnings`.

Behavior retained:

- sequential and expert-parallel scheduling remain separate;
- rank assignment, status files, coordination timeout, shard checkpoint schema, resume progress, and release publication remain unchanged;
- metric keys, per-channel counters, held-out history copies, and summary JSON schemas remain unchanged;
- formal modality channel contracts, optimizer/loss/AMP audits, checkpoint loadability, and stage-status behavior remain unchanged.

The next Phase 5 increment can move the large formal audit and `_build_modality_runtime` body into a dedicated `modality_runtime.py` facade only after adding a resolved-runtime snapshot test for every formal modality. This keeps the current extraction small and avoids coupling the shared progress object to modality-specific physical validation.

## Baseline Failures Resolved

The post-Phase-2 full-suite run initially reported three failures. They were behavior-contract defects exposed by the refactor verification, not diagnostic or physical-state regressions.

### Emitter-2D formal channel contract

`train_modality_joint.py` treated every non-DH modality as requiring both `left` and `right` channels. The formal emitter-2D single-channel contract only requires `left`; astigmatism remains the modality that requires both channels. The audit now uses an explicit modality-to-channel mapping:

```text
double_helix -> {main}
emitter_2d   -> {left}
astigmatism  -> {left, right}
```

This preserves strict completeness checks while accepting the committed emitter-2D formal configuration.

### Double Helix missing `train.model`

The DH resolver previously called `_mapping(train.model, ...)` even though `feature_channels=32` was already the intended default. It now treats a missing or non-mapping model block as an empty override and retains the default. Explicit `in_channels` and `feature_channels` overrides continue to be honored.

### Double Helix single-channel runtime contract

For single-channel online runtimes, the resolver now derives DH `condition_dim` and `domain_count` from the same normalized online contract used by the provider. A soft-MoE single-channel runtime therefore exposes `condition_dim=5` and `domain_count=1` consistently in both model and provider params; the formal channel-conditioned DH configuration remains `condition_dim=0` and `domain_count=1`.

`DoubleHelixRuntimeModel` accepts these contract values as metadata only. They do not change its forward path, parameters, state dict, checkpoint metadata, or direct-XYZ loss behavior. The DH modality contract also now binds `expert_instance` to the configured channel instead of always writing `main`, preventing channel identity drift in isolated runtimes.

### Verification receipt

The corrected baseline and full-suite checks are:

```text
34 passed, 1 warning
191 passed, 4 warnings
```

Commands:

```bash
/home/guest/anaconda3/bin/python -m pytest -q \
  unity/tests/baseline \
  unity/tests/training/test_modality_joint_training.py \
  unity/tests/training/test_astigmatism_runtime.py \
  unity/tests/training/test_emitter_2d_runtime.py \
  unity/tests/configs/test_default_three_expert_scheme.py

/home/guest/anaconda3/bin/python -m pytest -q unity/tests
```

`git diff --check` also passes. Remaining warnings are environmental CUDA initialization and the existing tifffile RGB-shape deprecation warning; no test is skipped or weakened.

## Follow-up Simplification Plan

The next work should continue in small, behavior-gated phases. No phase should change serialized runtime keys, checkpoint schemas, target ordering, random-seed derivation, or training defaults.

### Phase 3: split runtime configuration resolvers

Create internal modules behind the existing `build_localization_runtime_config` and `resolve_localization_model_config` facade:

- `runtime/model_config.py`: modality/expert model selection and model parameter defaults;
- `runtime/loss_config.py`: current and legacy loss translation;
- `runtime/optimizer_config.py`: optimizer and scheduler compatibility resolution;
- `runtime/conditioning_config.py`: condition fields, feature dimensions, domain contracts, and single-channel normalization;
- `runtime/provider_config.py`: online, microtube, and raw-TIFF provider materialization;
- `runtime/paths.py`: base-directory-aware artifact path resolution.

Execution order: first add normalized runtime snapshots for every committed modality configuration; then move one resolver family at a time with forwarding imports. Preserve exception types, error strings, default values, and dictionary key order where downstream manifests depend on them.

### Phase 4: isolate online provider strategies

Keep `build_online_batch_provider` and `OnlineBatchProviderConfig` stable while extracting the strategy implementations from `localization/data/online.py`:

- cached-window lifecycle and sequence slicing;
- LUT bank/interpolation and vector renderer backends;
- native simulation;
- camera/readout conversion;
- conditioning/domain feature construction;
- pxyz target conversion and finalization.

Start with characterization tests for seed determinism, cached-window ordering/reuse, tensor shape/dtype/device, target order, and physical-version changes. Benchmark the DH `cached_window` path before and after extraction on equivalent Slurm hardware; do not optimize based on CPU timings.

### Phase 5: centralize modality runtime and progress ownership

Move shared `_build_modality_runtime`, runtime audit, metric initialization/accumulation, checkpoint validation, and summary-schema helpers into a small `training/modality_runtime.py` plus a `ModalityProgress` value object. Keep sequential orchestration and expert-parallel coordination separate so rank barriers, shard status, and release publication remain explicit.

The acceptance gate is equivalent sequential and expert-parallel output schemas, checkpoint loadability, stage-status transitions, and unchanged formal channel contracts.

### Phase 6: measured compatibility and archive audit

Build a usage inventory for `src/double_helix`, `src/neptune_v04`, `src/double_helix/legacy`, and `scripts/archive` using imports, launcher references, documentation, checkpoint fixtures, and explicit user workflows. Classify each item as supported compatibility API, reproducibility archive, or obsolete implementation. Remove only the last class after references and reproducibility requirements are proven absent; otherwise document a deprecation deadline in `docs/compatibility.md`.

### Planned reduction

The realistic near-term target remains approximately 850-2,000 net tracked lines without removing supported compatibility or reproducibility behavior. The larger benefit is ownership clarity: runtime config, online strategies, modality lifecycle, and archive policy become independently testable while public entry points and functional behavior remain unchanged.

## Review Verdict

UnityPSF does not need a wholesale redesign. Its domain separation is already visible, and the modern package has a useful dependency-direction test. The codebase needs responsibility extraction, stronger characterization around performance-critical behavior, and explicit compatibility ownership.

The proposed refactoring is feasible without changing functionality if it is executed as small mechanical moves with strict behavioral gates. The highest-value work is to reduce the three dominant modules while preserving their public facades and current serialized contracts.
