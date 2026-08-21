# Unity v0.2 Compatibility and Archive Lifecycle

Status: Phase 5 complete on 2026-08-21

This inventory defines what remains runnable for compatibility, historical
replay, and physical reproduction. It is deliberately a manifest, not a new
runtime registry. The formal implementation remains under `unity_psf`.

## Lifecycle rules

- `unity_psf` is the only namespace for new formal training, calibration, and
  evaluation code.
- `double_helix` is a deprecated import and CLI compatibility namespace. It
  forwards to the canonical UnityPSF implementation and receives no new
  features.
- `double_helix.legacy` is retained only for historical Slurm, checkpoint
  replay, physical calibration, and paper/experiment reproduction workflows.
- `scripts/archive` is not a supported default training route. Its scripts are
  kept runnable while their experiment or artifact provenance is still needed.
- Removal requires all of: no active consumer, no replay or reproduction
  requirement, a verified canonical replacement, and at least one release
  cycle with migration evidence. A text search alone is not removal evidence.

## Inventory

| Surface | Formal owner | Known consumers / purpose | Data / weight dependency | Canonical replacement | Status | Last verified | Removal gate | Removal date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `src/double_helix/*.py` numerical/data wrappers | `src/unity_psf/optics/psf/double_helix/*` | Historical imports and archive jobs; wrapper identity is tested | DH calibration stacks, LUTs, and old import paths | Matching `unity_psf.optics.psf.double_helix` module | Deprecated compatibility | 2026-08-21, static identity test | Zero imports and archive references; one release cycle after migration | TBD - owner confirmation required |
| `src/double_helix/run_calibration.py` | `src/unity_psf/cli/double_helix_calibration.py` | Archive calibration Slurm jobs and existing `python -m` callers | Microscope 1 calibration TIFF and optional warm-start gamma | `unity_psf.cli.double_helix_calibration:main` | Retained CLI compatibility | 2026-08-21, static entrypoint test | Migrate every archive caller and verify calibration artifact parity | TBD - migration evidence required |
| `src/double_helix/run_evaluation.py` | `src/unity_psf/cli/double_helix_evaluation.py` | Archive evaluation and density workflows | Calibration LUT, gamma map, and historical checkpoints | `unity_psf.cli.double_helix_evaluation:main` | Retained CLI compatibility | 2026-08-21, static entrypoint test | Migrate every archive caller and verify evaluation artifact parity | TBD - migration evidence required |
| `src/double_helix/legacy/run_fd_zmap.py` | UnityPSF DH calibration/data contracts | Historical Microscope 1 FD z-map generation and replay | Simulated DH stacks and exported z-map NPZ | Canonical DH calibration modules plus a future explicit CLI | Retained historical reproduction | 2026-08-21, archive consumer audit | Replay parity on required datasets and no published-result dependency | TBD - reproduction owner required |
| `src/double_helix/legacy/run_field_gamma.py` | `unity_psf.optics.psf.double_helix.field_gamma` | Field-gamma archive sweeps and coefficient-map reproduction | Field observations and gamma coefficient NPZ | Canonical field-gamma API | Retained physics reproduction | 2026-08-21, archive consumer audit | Archive scripts migrated; coefficient-map parity verified | TBD - physics owner required |
| `src/double_helix/legacy/run_physical_update.py` | `unity_psf.optics.psf.double_helix.physical_update` | Full-FOV physical update and checkpoint artifact replay | Raw TIFF, proposal pairs, checkpoints, coefficient maps | Canonical physical-update API | Retained physics reproduction | 2026-08-21, archive consumer audit | Physical artifact parity and no replay consumer | TBD - physics owner required |
| `src/double_helix/legacy/run_lg_calibration.py` | `unity_psf.optics.psf.double_helix.lg_calibration` | LG residual calibration reproduction | Real T-cell raw TIFF and residual calibration artifacts | Canonical LG calibration API | Retained physics reproduction | 2026-08-21, archive consumer audit | Paper/reproduction sign-off and parity evidence | TBD - reproduction owner required |
| `src/double_helix/legacy/run_pixel_pupil_calibration.py` | `unity_psf.optics.psf.double_helix.pixel_pupil_calibration` | Independent pixel-pupil calibration archive jobs | Raw TIFF, pupil initialization, and fitted pupil artifacts | Canonical pixel-pupil API | Retained physics reproduction | 2026-08-21, archive consumer audit | Calibration artifact parity and no active dataset consumer | TBD - physics owner required |
| Remaining `double_helix/legacy/*.py` | UnityPSF optics/training modules as imported by each script | Wavefront diagnostics, z-bin evaluation, shared-carrier studies, training simulation, summary reports | Historical checkpoints, raw TIFF, calibration maps, diagnostic outputs | No single replacement; use the named canonical module for new work | Retained archive-only | 2026-08-21, archive consumer audit | Per-module consumer audit, then one release cycle | TBD - per-module audit required |
| `scripts/archive/double_helix_sweeps/*` | Archive/reproduction ownership | Microscope 1, real T-cell, density, z-bin, calibration, physical-update, and diagnostic runs | Historical raw data, checkpoints, NPZ maps, and output figures | Current `scripts/train` and UnityPSF CLI only where behavior is formally equivalent | Archived, runnable | 2026-08-21, shell/import audit | No result/checkpoint reproduction need and explicit retirement decision | TBD - owner decision required |
| `scripts/archive/neptune_standard/*` | Archive/reproduction ownership | Historical Neptune standard/3371/3367 submission and inference routes | Raw TIFF, z-maps, checkpoints, SLURM environment | Current formal UnityPSF training and inference scripts | Archived, runnable | 2026-08-21, shell audit | Migrate or freeze provenance; verify output parity before removal | TBD - owner decision required |

## Boundary verification

`tests/compatibility/test_double_helix_lifecycle.py` checks the boundaries
without running Slurm or requiring external datasets:

- each top-level wrapper exports objects from its canonical UnityPSF module;
- both compatibility CLI modules preserve only the `main` entry contract;
- every `python -m double_helix...` reference in archive scripts resolves to a
  source module;
- formal `unity_psf` source does not import `double_helix.legacy`.

The test intentionally does not assert that archived workflows are the default
route, nor does it execute historical jobs. Those are separate scientific
reproduction checks and require their original data, checkpoints, and Slurm
environment.

## Follow-up order

1. For each retained legacy module, record a named reproduction owner and the
   last verified artifact/checkpoint in the table above.
2. Migrate archive scripts only when a canonical command and artifact-parity
   test exist; do not rewrite scripts for style alone.
3. After a release cycle with no consumers, delete an individual wrapper or
   legacy entrypoint and remove its lifecycle row and boundary test.
4. Keep Phase 6 fixture work separate from this lifecycle boundary.
