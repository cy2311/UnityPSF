# ADR 0002: UnityPSF Naming and Namespace

> Historical naming decision. The active package version is now `0.3.0`; the
> The `neptune_v04` namespace was removed from the v0.3 distribution after
> repository consumers migrated to `unity_psf`.

- Status: Accepted
- Date: 2026-08-02
- Scope: Active v0.4 names and Python package namespace

## Decision

The active model family is named **UnityPSF**. It denotes one localization
model and one checkpoint contract spanning multiple PSF modalities. The active
Python namespace is `unity_psf`, and the distribution name is `unity-psf`.

The version is kept in package metadata and release tags, not in the import
namespace. The old `neptune_v04` namespace and command names are not part of
the v0.3 distribution.

Historical identifiers such as `neptune_v03_*`, old schema names, and existing
result metadata are not renamed. They are compatibility data rather than
active project branding.

## Migration rule

Each subsystem was migrated in a small verified slice. The legacy namespace was
removed after repository consumers and checkpoint references were audited.
