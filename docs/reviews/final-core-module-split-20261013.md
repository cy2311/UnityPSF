# Final core module split

## Scope

The approved five core files now delegate to responsibility-specific owners.
Scientific numerical kernels, loss equations, optimizer policy, target units,
checkpoint schemas and CUDA launch policy are not rewritten.

| Previous owner | Before | After | Implementation owners |
| --- | ---: | ---: | --- |
| training/loop.py | 826 | 35 | loop_types, epoch_loop, training_run, evaluation, resume, checkpoint_io |
| training/modality_joint.py | 739 | 56 | modality_types, modality_epoch, modality_evaluation, modality_checkpoint, modality_physical_state |
| training/entrypoints/train_modality_expert_parallel.py | 660 | 76 | expert_parallel_cli, coordination, worker, release, artifacts |
| localization/runtime/config.py | 979 | 173 | model_config, loss_config, modality_config, conditioning_config, provider_config, contract_config, simulator_range_config |
| localization/data/online.py | 1825 | 121 | online_types, provider, native, cached, cached_sequences, cached_payload, cached_window, camera, physical, geometry, conditioning |

## Boundaries

- Generic and modality facades contain exports, not training implementations.
- Internal consumers import canonical owners rather than generic/modality/online facades.
- New owner interfaces are public; historical private names remain explicit compatibility exports.
- Runtime configuration resolves scales and targets once per build and passes the resolved values to owners.
- Cached scheduling, sequence materialization, payload precomputation and target slicing are separately owned.
- Expert Parallel CLI owns dispatch, workers own modality execution, coordination owns rank completion, release owns publication.
- Atomic Torch checkpoint writing is shared; the Expert Parallel adapter retains its existing Path return value.
- Architecture tests enforce facade sizes, owner sizes, import direction and canonical-object identity.

## Verification

Final integrated regression on October 13, 2026: **423 passed, 1 skipped**
(103.19 seconds). Scoped Ruff passes; the configured six-module mypy gate,
the eight split runtime resolver modules, compileall and project-scoped diff
whitespace check pass. The selection CLI plan and internal entrypoint help
also run successfully. This is CPU
engineering verification, not a CUDA acceptance receipt.

Tests cover AMP skips, scheduler advancement, checkpoint inventories and RNG order,
resume equivalence, held-out aggregation, canonical launch locking, rank completion,
cached/native target metadata and nm/um scale equivalence. The online split was
also checked by its worker against pre-split bytecode and CPU native differential
outputs; this does not establish CUDA correctness.

Run the full suite with `PYTHONPATH=src`; use a temporary directory outside the
checkout for pytest fixtures. Putting `--basetemp` under this checkout changes
the ancestor-project detection exercised by test_config_paths.

## Acceptance limits

The new split owners pass scoped Ruff, and compileall succeeds. Full-tree Ruff
still reports existing high-fidelity alias/name errors and unused imports in
gamma_workflow, heldout_monitor, posterior_workflow and roi_projection, plus two
unused locals in localizer_eval. These were identified independently as outside
this structural split and are not silently fixed or counted as passing.

No CUDA job or formal GPU receipt is produced by this refactor. Existing root
caches and the parent repository's untracked project state are retained; no
commit, reset or unrelated cleanup is performed.
