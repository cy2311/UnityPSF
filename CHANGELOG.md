# Changelog

All notable changes to UnityPSF are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- Removed the unused `neptune_v04` namespace alias from the v0.3 distribution.
- Migrated internal localization imports to `localization.runtime.config` and
  `localization.data.online`; the old module paths remain only as thin import
  shims for external callers.

## [0.3.0] - 2026-09-24

### Added

- Established `unity_v0.3` as the active UnityPSF source boundary.
- Migrated the PPG/PCA physical backend, tests, calibration artifacts, and
  SLURM helpers under the v0.3 checkout.
- Added `scripts/submit_sbatch.sh`, which submits jobs from the project root
  so SLURM logs consistently land under `logs/`.

### Changed

- Unified the active package metadata, checkpoint defaults, and module
  descriptions on UnityPSF `0.3.0`.
- Kept historical Double Helix adapters and legacy checkpoint readers as
  compatibility surfaces only.
- Recorded the active DH decision: PPG/PCA is the physical backend; legacy DH
  localization compatibility code is not part of the active route.

## [0.4.0] - 2026-08-06

### Added

- Introduced the UnityPSF multimodal SMLM localization package.
- Added modality- and channel-routed experts for 2D emitter and astigmatism
  PSFs, with Double Helix optics and calibration infrastructure.
- Added single-process and Expert Parallel joint training entry points.
- Added integrity-checked joint checkpoint v2 release and resume contracts.
- Added visible validation reports, structured experiment configurations, and
  a 170-test contract suite.

### Changed

- Replaced the Neptune public package and command names with `unity_psf` and
  `unity-psf-*`.
- Retained thin `double_helix` compatibility adapters for existing research
  workflows.

[Unreleased]: https://github.com/cy2311/UnityPSF/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/cy2311/UnityPSF/releases/tag/v0.3.0
[0.4.0]: https://github.com/cy2311/UnityPSF/releases/tag/v0.4.0
