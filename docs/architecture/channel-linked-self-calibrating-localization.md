# UnityPSF channel-linked self-calibrating localization

- Status: Accepted target design, not yet implemented
- Date: 2026-09-23
- Governing decision: [ADR 0004](../adr/0004-channel-linked-refinement-without-network-fusion.md)
- Scope: future implementation and experiments under `unity_v0.3` only

## 1. Objective and boundary

Improve the precision of emitters observed in two synchronized measurement
channels by combining their evidence after the existing UnityPSF network has
localized each channel. Preserve the current network architecture, modality
routing, FiLM conditions, 10-channel per-channel output, and single-channel API.
Do not adopt DECODE-Plex's cross-channel feature concatenation or replace the
output with a 12/13-channel head as part of this route.

The target workflow is:

```text
paired raw frames + crop/orientation metadata
  -> existing UnityPSF per-channel localizations and uncertainties
  -> sample-derived Legendre channel transform
  -> one-to-one cross-channel emitter linking
  -> shared-coordinate refinement, channel-specific photometry
  -> optional two-image physical PSF likelihood refinement
```

The canonical implementation now lives at `unity_v0.3/src/unity_psf`, with
tests under `unity_v0.3/tests`. It was migrated from the current `unity/src`
working tree with runtime caches excluded. Do not silently edit `unity/src/`
while claiming the new route belongs to `unity_v0.3`.

## 2. Data and identifiability contract

A linked observation consists of left/right images from the same exposure,
source frame ID, crop offsets, flips/rotations, pixel size, PSF modality, and
per-channel physical state. Use full-sensor physical coordinates for the
transform and convert to local crop coordinates only at image boundaries.
Positions must consistently use pixel-center conventions; record whether
transforms operate in pixels or nanometers.

Fix left as the reference coordinate system: `T_left = identity`; learn only
`T_right_to_left`. This removes the arbitrary common translation/rotation
gauge. A known crop/orientation and coarse overlap prior are necessary to
initialize matching. Pairing requires a population of the same physical
emitters visible in both channels. If sample chemistry gives unrelated
left/right emitters, a sample-only transform cannot be identified: obtain a
shared fiducial/overlap subset or leave channels independent. A frame with
right-channel camera background alone is not a valid dual-channel example.

The training/validation split must be by independent frame blocks or fields,
not by randomly separating candidate pairs from the same frame. Require
spatial support across the intended field. Never extrapolate a fitted
second-order field beyond its observed support without marking it invalid.

## 3. Transform model

Normalize the right-channel full-field coordinates to `(u,v)` in `[-1,1]^2`.
For Legendre basis functions `P_i`, define

```text
T_x(x,y) = x + sum_{i+j<=d} a_ij P_i(u) P_j(v)
T_y(x,y) = y + sum_{i+j<=d} b_ij P_i(u) P_j(v)
```

`d=1` is an affine transform including translation, rotation, scale, and
shear. Initialize its translation from coarse registration and fit it robustly
to high-confidence pairs. `d=2` adds smooth field curvature; enable it only
when affine held-out residuals have a reproducible spatial pattern and the
paired emitters cover the field. No dense x-y map or higher order by default.

Store the field bounds, coordinate convention, basis order, coefficients,
reference channel, training-data hash, fit support, parameter covariance,
version, and acceptance metrics. Apply regularization to nonlinear terms and
bound per-update displacement/Jacobian change. A transform must remain locally
invertible on the validated field. Its uncertainty is part of downstream
matching/refinement, not an invisible systematic error.

## 4. Candidate linking and coordinate refinement

Run the existing channel-specific localization path first. Convert every
candidate to the common coordinate frame, retaining frame ID, modality,
channel ID, position, z-valid flag, photons, background, detection score,
position covariance, and provenance. Calibrate reported uncertainty before
using it as a quantitative weight.

For each simultaneous frame, form candidate edges only within a registration
search radius. Use the residual and combined covariance
`S = Sigma_left + J_T Sigma_right J_T^T + Sigma_registration` to score each
edge. Apply one-to-one assignment with an explicit unmatched cost. Use z
agreement only when both channel models have comparable, valid z estimates;
do not require equal photon counts across differently weighted channels.
Reject ambiguous links rather than forcing every detection into a pair.

For a reliable linked pair, estimate one common lateral location using the
two covariance matrices, including propagated transform uncertainty. In the
scalar diagonal case this reduces to inverse-variance weighting; detection
probability is a pairing prior, not a substitute for localization variance.
Keep each channel's photons/background separate. If z is valid and shares a
physical coordinate convention, refine it with the same principle. Preserve
single-channel/unmatched localizations unchanged and mark their provenance.

The result contract is a list/set of emitters with shared xyz, covariance,
link state (`paired`, `left_only`, `right_only`, `ambiguous`), per-channel
photometry, source candidates, and transform version. It is not a 12-channel
image tensor. A future `localize_linked(...)` may accept paired images and
conditions while `localize(...)` remains stable. Final API signatures should
be frozen after an offline prototype proves the semantics.

## 5. Physical-update lifecycle

At a scheduled physical-update trigger:

1. Freeze the localizer and gamma/PSF state. Collect high-confidence paired
   candidates from a fixed training ROI/frame subset.
2. Estimate a small update to the current affine transform; use robust
   outlier handling and regularization. Propose second-order terms only after
   the affine baseline passes the spatial-residual gate.
3. Re-link with the candidate transform, then evaluate on untouched held-out
   frames/field regions. Check registration residual, link precision or a
   defensible proxy, emitter count, uncertainty coverage, and physical
   reconstruction quality. Accept only a predeclared improvement with no
   material per-channel regression; otherwise restore the previous transform.
4. Record accepted/rejected state and reasons atomically. A rejected update
   must not change the transform seen by training or inference.

Do not estimate an unrestricted transform and gamma in the same first-stage
optimization: each can compensate for errors in the other. Once the
coordinate-only loop is validated, alternate transform and gamma proposals,
each with its own held-out gate and rollback. The transform update cadence
may share the physical-update trigger, but state, metrics, and acceptance
remain distinct from channel-local gamma updates.

An optional later stage fits the linked emitter's shared xyz to raw pixels in
both channels using the existing channel-specific PSF renderer and appropriate
camera likelihood. Photons and background remain channel-specific; the
transform maps the shared point into each image. Compare its held-out benefit
and runtime against coordinate-only refinement before enabling it by default.

## 6. Staged work and acceptance checks

| Phase | Deliverable under `unity_v0.3` | Exit check |
| --- | --- | --- |
| 0 | Establish package source, tests, and baseline command in this directory; inventory what was copied/migrated from `unity/src` | Existing single-channel inference and checkpoint tests pass without paths into `unity/src` |
| 1 | Freeze paired-data coordinate/crop contract and independent-inference baseline | Paired frame IDs and full-field coordinates round-trip; invalid/non-overlapping channels are identified |
| 2 | Robust affine Legendre registration with persistent versioned state | Synthetic known transform recovered; held-out residual improves over coarse alignment; sparse/degenerate matches fail clearly |
| 3 | One-to-one linking and uncertainty-aware coordinate refinement | Synthetic paired/unpaired/overlap cases pass; no forced links; held-out precision and uncertainty coverage reported |
| 4 | Gated registration updates at physical-update triggers | Accepted state survives restart; rejected state rolls back; gamma and network states stay fixed during transform proposal |
| 5 | Optional joint raw-pixel physical fit, then alternating transform/gamma study | Held-out localization or reconstruction improves over phase 3 within predeclared runtime budget |

Use a small paired synthetic benchmark first, then a real paired acquisition
with an independent validation source. Report at least four ablations:
independent UnityPSF; plus transform; plus linking/coordinate refinement; plus
raw-pixel fit when available. Primary endpoints are lateral/axial localization
error where reference locations exist, linking precision/recall, and emitter
count error. Registration residual, uncertainty coverage, Poisson/camera
likelihood, per-channel metrics, and runtime diagnose failure. Do not claim a
precision improvement from registration residual alone.

Before each implementation phase, write its exact tests, dataset split, metric
tolerances, and rollback criteria in the phase task. Do not treat a synthetic
smoke or a lower training loss as scientific validation.

## 7. Prior code and literature boundary

The existing UnityPSF `DoubleUNet` already concatenates first-stage features
across time-window inputs, but current `UnityPSF.localize` is one measurement
channel per call. The new module links post-network emitter evidence across
physical channels. This is the intended experiment, independent of a neural
fusion architecture.

ClimeLoc motivates channel mapping and joint emitter reasoning, but its
published workflow uses prior channel/PSF calibration. This design replaces
the separate channel-transform calibration acquisition only when sample data
provide identifiable shared emitters. DECODE-Plex remains a comparison for
high-density multichannel localization, not a template for UnityPSF's network
or output tensor. Its published dual-channel example has 13, not 12, output
channels: `[1, 5, 5, 2]`.
