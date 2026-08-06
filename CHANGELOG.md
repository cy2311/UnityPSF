# Changelog

All notable changes to UnityPSF are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Retained thin `neptune_v04` and `double_helix` compatibility adapters for
  existing research workflows.

[Unreleased]: https://github.com/cy2311/UnityPSF/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/cy2311/UnityPSF/releases/tag/v0.4.0
