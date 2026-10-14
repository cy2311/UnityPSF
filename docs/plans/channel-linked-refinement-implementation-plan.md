# Channel-linked refinement implementation plan

- Governing ADR: [ADR 0004](../adr/0004-channel-linked-refinement-without-network-fusion.md)
- Technical design: [channel-linked self-calibrating localization](../architecture/channel-linked-self-calibrating-localization.md)
- Status: Ready for Phase 0 implementation
- Scope: `unity_v0.3` only

## Non-negotiable design rules

1. Keep the existing UnityPSF neural architecture as the starting point.
2. Keep the per-channel 10-channel output order and the existing `localize(...)`
   behavior.
3. Add registration and linking after per-channel neural inference.
4. Use left as the coordinate reference and learn only right-to-left mapping.
5. Start with affine/first-order Legendre; permit second order only after a
   held-out residual gate.
6. Keep channel-local physical state, photons, and background separate.
7. Do not introduce DECODE-Plex-style feature concatenation, a 12/13-channel
   output head, or a cross-channel consistency loss in the first route.
8. A failed or unidentifiable registration must leave independent localization
   available and must never silently force links.

## Phase 0: establish the v0.3 implementation boundary

The installable `unity_v0.3/src/unity_psf` package and focused test layout have
now been established by copying the current `unity/src` and `unity/tests`
working trees with runtime caches excluded. The migration manifest records the
boundary. Do not point the new package at `unity/src` through an implicit
import path.

Acceptance:

- A clean environment imports `unity_psf` from `unity_v0.3`.
- A baseline single-channel forward/localize smoke passes.
- Existing checkpoint/output contracts are tested before adding linked logic.
- The migration manifest names every copied module and its reason.

Recorded files:

- `pyproject.toml`
- `src/unity_psf/` and `tests/`
- `docs/architecture/channel-linked-source-migration.md`

## Phase 1: freeze paired-data and coordinate contracts

Define typed records for paired frame identity, full-sensor/crop coordinates,
channel metadata, candidate provenance, and transform versions. Add a loader
that rejects missing frame pairing, inconsistent crop/orientation metadata, and
non-overlapping fields. Establish the independent-inference baseline report.

Acceptance:

- Pixel-center coordinate round trips are exact within declared tolerance.
- Crop offsets and flips are applied once and are visible in provenance.
- Disjoint-channel examples are marked unidentifiable rather than registered.
- The baseline report preserves left/right metrics separately.

Likely files:

- `src/unity_psf/channel_linking/contracts.py`
- `src/unity_psf/channel_linking/coordinates.py`
- `tests/channel_linking/test_coordinates.py`
- `tests/channel_linking/test_paired_frames.py`

## Phase 2: affine and Legendre transform fitting

Implement a small, serializable transform state. Fit affine parameters from
high-confidence matched candidates using robust residual weighting. Add the
Legendre basis and second-order terms behind an explicit degree setting, with
regularization and local-invertibility checks.

Acceptance:

- A synthetic known affine transform is recovered within a declared pixel
  tolerance from held-out points.
- Degenerate, too-sparse, and one-region matches fail with actionable status.
- State round trips with bounds, basis degree, coefficients, support, hashes,
  covariance, and metrics intact.
- Per-update displacement and Jacobian limits are enforced.

Likely files:

- `src/unity_psf/channel_linking/legendre.py`
- `src/unity_psf/channel_linking/fit.py`
- `src/unity_psf/channel_linking/state.py`
- `tests/channel_linking/test_legendre_transform.py`
- `tests/channel_linking/test_transform_fit.py`

## Phase 3: linking and coordinate refinement

Convert existing 10-channel outputs into candidates, propagate covariance, and
solve one-to-one links with unmatched costs. Refine shared coordinates by
inverse covariance weighting after right-to-left transformation. Preserve
unmatched and ambiguous candidates as explicit result states.

Acceptance:

- Known synthetic pairs link correctly under position uncertainty and transform
  uncertainty.
- Crossing candidates, duplicate detections, and unmatched detections do not
  create many-to-one links.
- Refined coordinates improve lateral error against the independent baseline
  on held-out paired data.
- Per-channel photometry and provenance remain recoverable.

Likely files:

- `src/unity_psf/channel_linking/link.py`
- `src/unity_psf/channel_linking/refine.py`
- `src/unity_psf/channel_linking/result.py`
- `tests/channel_linking/test_linking.py`
- `tests/channel_linking/test_refinement.py`

## Phase 4: physical-update integration and rollback

At an existing physical-update trigger, freeze network/gamma state, fit a
registration proposal from the designated training subset, and evaluate it on
held-out frame blocks/field regions. Persist accepted state atomically and
restore the previous state on rejection. Keep registration metrics separate
from channel-local gamma metrics.

Acceptance:

- A rejected transform proposal leaves the prior state byte-equivalent or
  semantically equivalent under its state schema.
- A restart reloads the accepted transform and its version/metrics.
- Gamma, network weights, and channel-local condition stores are unchanged
  during a transform-only proposal.
- The update log records support, residuals, acceptance decision, and reason.

Likely files:

- `src/unity_psf/channel_linking/update.py`
- existing v0.3 physical-update integration point
- `tests/channel_linking/test_update_gate.py`

## Phase 5: optional raw-pixel joint refinement

Only after Phase 3 is scientifically positive, fit the shared emitter position
to both raw channel images through the existing PSF renderers. Keep photons and
background channel-specific and use the transform only to project the shared
position into each image. Compare runtime and held-out likelihood against the
coordinate-only method before enabling it by default.

Acceptance:

- The joint fit improves a predeclared held-out localization or reconstruction
  endpoint without degrading either channel materially.
- The camera/Poisson likelihood is in the correct physical units.
- Runtime and memory stay within the declared budget.
- Alternating transform/gamma updates are separately gated and rollback-safe.

Likely files:

- `src/unity_psf/channel_linking/joint_physical.py`
- `tests/channel_linking/test_joint_physical.py`
- a versioned experiment config under `configs/experiments/`

## Required ablation report

Every scientific run must compare the same paired held-out split across:

1. independent UnityPSF localization;
2. independent localization plus a fixed transform;
3. transform plus linking and coordinate refinement;
4. optional raw-pixel joint refinement.

Report registration residual by field region, linking precision/recall,
emitter-count error, lateral/axial error, uncertainty coverage, per-channel
metrics, raw-image likelihood/reconstruction error, unmatched counts, and
runtime. A lower registration residual or training loss alone is not evidence
of improved localization precision.

## Stop conditions

Pause the route and review the design if any of these occur:

- paired emitters do not identify a stable transform;
- the transform improves one field region while degrading another;
- linking requires forcing ambiguous candidates;
- coordinate refinement improves synthetic data but not held-out real data;
- registration and gamma updates compensate for each other;
- the new result cannot preserve independent/unmatched channel observations.
