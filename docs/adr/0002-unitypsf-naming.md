# ADR 0002: UnityPSF Naming and Namespace

- Status: Accepted
- Date: 2026-08-02
- Scope: Active v0.4 names and Python package namespace

## Decision

The active model family is named **UnityPSF**. It denotes one localization
model and one checkpoint contract spanning multiple PSF modalities. The active
Python namespace is `unity_psf`, and the distribution name is `unity-psf` with
version `0.4.0`.

The version is kept in package metadata and release tags, not in the import
namespace. The old `neptune_v04` namespace and command names remain as
temporary compatibility aliases while modules and entry points are migrated.

Historical identifiers such as `neptune_v03_*`, old schema names, and existing
result metadata are not renamed. They are compatibility data rather than
active project branding.

## Migration rule

Each subsystem must be migrated in a small verified slice. A slice is complete
only when both `unity_psf.<module>` and the legacy import path behave as
expected, editable and wheel installs pass, and the corresponding entry points
still load. The legacy namespace is removed only after all consumers are
migrated.
