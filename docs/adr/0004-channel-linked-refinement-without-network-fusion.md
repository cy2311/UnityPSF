# ADR 0004: Channel-linked localization without DECODE-Plex network fusion

- Status: Accepted as the target design; implementation pending
- Date: 2026-09-23
- Scope: UnityPSF multichannel localization in `unity_v0.3`

## Context

UnityPSF currently produces a 10-channel localization output for one measurement
channel at a time. Its DoubleUNet already uses a shared first U-Net, concatenates
features, and runs a union U-Net over the input time window. That concatenation
does not jointly process left and right measurement channels when inference calls
are channel-specific.

DECODE-Plex uses multichannel feature fusion and a different output contract.
Its two-channel example has heads of widths `[1, 5, 5, 2]` (13 outputs: one
detection probability, two photon means, shared xyz, corresponding uncertainties,
and two backgrounds). Copying that network would change the UnityPSF checkpoint,
training target, and decoder at the same time as registration and linking.

The research question here is narrower: can observations of the same emitter in
two channels improve localization precision using UnityPSF's existing networks
and physical models, without a separate pre-calibration acquisition?

## Decision

Keep the existing per-channel neural inference and 10-channel output contract.
Add a channel-linked localization layer with four explicit steps:

1. Learn a right-to-left 2D coordinate transform from matched sample emitters.
   Start with affine (first-order Legendre); use second-order Legendre only when
   held-out spatial residuals justify it.
2. Link candidate emitters across simultaneous channels with one-to-one matching,
   uncertainty-aware distances, and an unmatched-emitter path.
3. Refine a linked emitter's shared position from both channel estimates and
   propagated uncertainty. Channel-specific photons and backgrounds stay separate.
4. After the coordinate-only method passes its gate, optionally refine linked
   emitters using both raw images and the existing physical PSF renderers.

The reference channel fixes the coordinate gauge. Transform parameters are
global to a paired acquisition or specified field domain, not free per-ROI maps.
Registration is updated at physical-update triggers by small, separately gated
steps. Gamma and network parameters remain fixed during the first registration
experiments. Registration and gamma may alternate only after each update can be
evaluated independently.

This is sample-driven self-calibration, not a claim of zero calibration
information. It requires paired frames with enough emitters actually visible in
both channels, known crop/orientation/pixel scale, and a fixed reference frame.
Two colors labeling disjoint structures do not satisfy this requirement.

All new code, tests, configs, and design records for this route belong under
`unity_v0.3`. Its canonical package source is now `unity_v0.3/src`; the former
`unity/src` tree is a migration reference and must not receive new feature work.

## Consequences

- Existing single-channel checkpoints and `localize(...)` retain their meaning.
- A linked result is a new API/result type; it must not be represented as a
  misleading 12- or 13-channel tensor.
- An acquisition without reliable shared emitters falls back to independently
  reported localizations and marks the transform unidentifiable.
- Any future cross-channel neural fusion needs a separate ablation showing a
  gain beyond this channel-linked baseline before it becomes part of the route.
- This decision extends the earlier modality-routed blueprint's first-version
  exclusion of cross-channel consistency loss; it does not alter its modality
  router or per-channel physical-state ownership.

## Verification

Compare on held-out paired data: independent localization; independent
localization plus registration; registration plus linking and coordinate
refinement; and, if implemented, raw-pixel joint refinement. Report registration
residual by field region, linking precision/recall, emitter count error,
lateral/axial localization error where ground truth exists, uncertainty
coverage, raw-image reconstruction likelihood, unmatched-emitter behavior, and
runtime. A new stage is accepted only if its predeclared quality gate passes
without hiding degradation in either channel.

The implementation details and phase gates are in
[`../architecture/channel-linked-self-calibrating-localization.md`](../architecture/channel-linked-self-calibrating-localization.md).
