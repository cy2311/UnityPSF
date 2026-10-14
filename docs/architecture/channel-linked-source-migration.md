# UnityPSF v0.3 source boundary

- Status: Phase 0 completed
- Date: 2026-09-24
- Canonical root: `/home/guest/Others/main/race/unity_v0.3`

## Canonical surfaces

```text
unity_v0.3/
  src/       installable Python packages and all new production code
  tests/     package and contract tests
  scripts/   reusable entrypoints and orchestration
  configs/   active experiment and modality configuration
  docs/      durable architecture, plans, ADRs, and reviews
  output/    runtime results and checkpoints, local-only
  logs/      runtime logs, local-only
  .local/    caches, generated local configs, vendor trees, scratch, local-only
```

The current `unity/src`, `unity/tests`, and `unity/scripts` working trees were
copied into the corresponding canonical surfaces on 2026-09-24. The copy
preserves the current source content, including user modifications and new
files, while excluding Python bytecode and cache directories. The exact source
boundary is recorded in `docs/architecture/unity-v0.3-source-migration-manifest.md`.

## Mainline rules

1. New production code, tests, and reusable scripts are edited only under
   `unity_v0.3`.
2. `unity/src` is a read-only migration reference. It is not an import source,
   test path, or feature branch for this mainline.
3. `output`, `logs`, `.local`, caches, checkpoints, and generated reports are
   not source modules and must not be imported by package code.
4. Active configs must resolve paths relative to `unity_v0.3` or use explicit
   externally versioned artifacts with provenance. Do not introduce new
   `/home/guest/Others/main/race/unity/...` paths.
5. Historical/archive scripts remain runnable only when their purpose is
   reproduction or compatibility; they are not evidence of the active mainline.
6. The channel-linked refinement work begins in `src/unity_psf/channel_linking`
   and must preserve the existing per-channel output contract.

## Migration limitations

This is a source-boundary migration, not a claim that every historical script
or absolute artifact path is already clean. The first verification gate checks
package import and architecture boundaries. Path cleanup and removal of legacy
surfaces happen in later, separately tested slices. No source file in `unity`
was deleted or reverted by this migration.
