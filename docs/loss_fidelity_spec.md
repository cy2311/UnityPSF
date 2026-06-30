# Loss Fidelity Spec

## Status

Slice 4.5 checkpoint. This document specifies how the old
`neptune_iwae` localization loss maps into native `neptune_v0.3` inputs.

This spec was introduced at Slice 4.1 and updated when the Slice 4.2 GMM core
landed. Slice 4.3 broadened the deterministic old-route parity fixtures.
Slice 4.4 adds a numerical guard for old `GMMLoss` finite/backend behavior; it
does not introduce a new loss design.
Slice 4.5 adds an old-route real-shaped batch fixture; it does not add a new
batch provider.

Slice 4.2 implements `active_smlm_gmm_loss`. It does not implement
`active_smlm_composite_loss`.

`neptune_iwae` remains reference-only. v0.3 runtime code must not import old
loss or data-processing modules.

## Reference Surfaces

Old reference files:

- `neptune_iwae/smlm_v2a/training/losses.py::GMMLoss`
- `neptune_iwae/smlm_v2a/data/data_factory.py::TargetProcess`

Current v0.3 files:

- `neptune_v03.localization.losses.ActiveSMLMLoss`
- `neptune_v03.localization.losses.ActiveSMLMGMMLoss`
- `neptune_v03.localization.losses.ActiveSMLMGMMTargetAdapter`
- `neptune_v03.localization.smlm_targets`
- `neptune_v03.localization.smlm_output`

## Target Order Contract

Old `GMMLoss` consumes processed legacy-order targets:

```text
photons, x_px, y_px, z
```

v0.3 providers emit native-order targets:

```text
x_px, y_px, z, photons
```

The v0.3 GMM-compatible path uses explicit adapters:

- `legacy_iwae_pxyz_to_v03(...)` for reference fixtures that start in old
  order.
- `legacy_iwae_target_process_to_v03(...)` reproduces old `TargetProcess`
  order explicitly: legacy-order `disable_attr`, then `phot_max`/`z_max`
  scaling, then conversion to v0.3 target order.
- `ActiveSMLMGMMTargetAdapter.v03_to_gmm_order(...)` converts native order into the
  internal GMM channel order used by the numerical likelihood.

No loss implementation may infer order from tensor position without naming the
contract at the call boundary.

## TargetProcess Parity

Old `TargetProcess.forward(...)` mutates targets in legacy order:

- optional `disable_attr` zeros selected legacy target channels.
- `phot_max` divides channel `0`, the photon channel.
- `z_max` divides channel `3`, the z channel.

The equivalent v0.3 preprocessing must be explicit and non-mutating by
default:

- photon scaling applies to native channel `3`.
- z scaling applies to native channel `2`.
- old-route `disable_attr` is handled by `legacy_iwae_target_process_to_v03`
  before v0.3 conversion.
- direct GMM target adapter `disable_attr` applies after conversion into
  legacy GMM order and is intended only for already-native v0.3 targets.
- xy values remain absolute pixel coordinates until the GMM mean construction
  adds local offsets to pixel centers.

The old active route's formal microtube configuration uses `phot_max` through
training normalization and `z_max` through target scaling. Slice 4.2 must not
silently reuse the sparse `ActiveSMLMLoss` target convention because that loss
compares only target pixels, while GMMLoss compares each target against all
image pixels as mixture components.

## GMMLoss Terms

Old `GMMLoss.forward(...)` receives:

```text
output:     (N, 10, H, W)
detect_tar: (N, H, W)
pxyz_tar:   (N, M, 4)
mask:       (N, M)
bkg_tar:    (N, H, W)
```

It splits output as:

```text
p        = output[:, 0]
pxyz_mu  = output[:, 1:5]
pxyz_sig = output[:, 5:9]
bkg      = output[:, 9]
```

The returned tensor has two channels per batch:

```text
2 * [loss_gmm, loss_bkg] * ch_weight
```

For v0.3 formal training metrics, the GMM route exposes only the old external
loss contract:

- `loss_gmm`: old GMM likelihood term after sign convention.
- `loss_bkg`: old dense background MSE contribution.
- `loss_total`: weighted total used by the trainer.

The old `loss_gmm` computation internally contains a count likelihood
contribution and an all-pixel mixture localization likelihood contribution.
Those may appear in tests or one-off diagnostics as internal decomposition
terms, but they are not separate formal training loss components in the old
3052 route.

### internal count contribution

Old count likelihood:

```text
nlocs_mean = p.sum over H,W
nlocs_var  = clamp((p - p**2).sum over H,W, min=eps)
nlocs_tar  = mask.sum over M
log_prob  += Normal(nlocs_mean, sqrt(nlocs_var)).log_prob(nlocs_tar) * nlocs_tar
```

The multiplier by `nlocs_tar` is part of the old behavior. Empty frames
therefore do not get a count penalty from this term.

### internal mixture localization contribution

For every active target, old `GMMLoss` evaluates a Gaussian mixture over all
pixels. Mixture weights are derived from `p`, clamped non-negative, normalized
through logits/logsumexp in the manual backend, and protected by `eps`.

Mean construction is channel-specific:

```text
mu_photons = pxyz_mu[:, 0]
mu_x       = pxyz_mu[:, 1] + pixel_x_center + xyoffset_x
mu_y       = pxyz_mu[:, 2] + pixel_y_center + xyoffset_y
mu_z       = pxyz_mu[:, 3]
```

Pixel centers are `col + 0.5` and `row + 0.5`. This is different from the
current sparse `ActiveSMLMLoss`, which samples only the target pixel and uses
integer-centered local offsets.

The returned localization contribution is:

```text
internal_mixture_localization_contribution = -sum_active_targets(log mixture probability)
```

### loss_bkg

Old background term is dense summed MSE:

```text
loss_bkg = ((bkg - bkg_tar) ** 2).sum over H,W
```

This differs from current `ActiveSMLMLoss`, which uses a per-batch mean over
pixels before applying `background_weight`.

## Backend And Chunking Contract

Old `GMMLoss` has two backends:

- `mixture_same_family`
- `manual_chunked`

Both must match on small CPU fixtures within tolerance. `manual_chunked` is the
memory-safe formal route and must support:

- `gmm_target_chunk`: chunk size along target dimension `M`.
- `gmm_component_chunk`: chunk size along mixture component dimension `H * W`.

Chunking must not change numerics except for normal floating-point tolerance.
Invalid mask/target shapes must fail fast.

## Proposed v0.3 Loss Names

### active_smlm_gmm_loss

Status: implemented.

Purpose:

- Reproduce old `GMMLoss` behavior with v0.3 target contracts.
- Return per-batch loss suitable for `make_localization_loss`.
- Record component metrics in `last_components`.

Required params:

- `xyoffset`
- `ch_weight`
- `photon_scale`
- `z_scale`
- `gmm_target_chunk`
- `gmm_component_chunk`
- `gmm_backend`
- `eps`

### active_smlm_composite_loss

Status: not part of the current migration route.

active_smlm_gmm_loss is the formal fidelity route for old GMMLoss.
active_smlm_loss remains smoke-only. active_smlm_composite_loss is not part of
the current migration route and must not be used to replace old `GMMLoss`
parity unless a separate experiment branch is explicitly requested.

## Deterministic Fixtures For Slice 4.2

Fixtures should live in tests, not in runtime modules. Each fixture should be
small enough to run on CPU and should compare both backends when applicable.

### single_emitter_centered_2x2

Purpose:

- Verify pixel-center offset behavior and one-target mixture likelihood.

Shape:

```text
N=1, H=2, W=2, M=1
```

Use one target at a known absolute coordinate and one high-probability pixel.
Set `xyoffset=(0, 0)`, finite sigma, and simple background. The expected test
must fail if x/y means are compared as sparse local targets instead of
pixel-center-adjusted mixture means.

### two_emitters_chunked_3x3

Purpose:

- Verify `manual_chunked` equals `mixture_same_family` with multiple targets
  and multiple mixture components.

Shape:

```text
N=1, H=3, W=3, M=2
```

Run at least two chunk settings:

```text
gmm_target_chunk=1, gmm_component_chunk=2
gmm_target_chunk=0, gmm_component_chunk=0
```

Both settings must match the unchunked backend within tolerance.

### empty_frame_count_only

Purpose:

- Lock old empty-frame behavior.

Shape:

```text
N=1, H=2, W=2, M=1
mask all false
```

Expected behavior:

- no internal mixture localization contribution.
- the internal count contribution follows old multiplication by `nlocs_tar`;
  with zero targets it contributes zero.
- only `loss_bkg` can be non-zero.

This fixture prevents accidentally adding a modern empty-frame count penalty
inside the parity loss.

## Broader Fixtures For Slice 4.3

### old_route_batched_masked_background_2x3x4

Purpose:

- Verify old-route `TargetProcess` scaling on a batched fixture, not only a
  single-frame fixture.
- Verify per-batch target masks with different active emitter counts.
- Verify non-zero dense background loss uses old summed MSE semantics.
- Verify non-uniform detection probabilities, means, z/photon means, and
  sigmas still agree across `manual_chunked`, unchunked manual, and
  `mixture_same_family` backends.

Shape:

```text
N=2, H=3, W=4, M=3
```

The fixture starts from legacy-order targets, applies
`legacy_iwae_target_process_to_v03(phot_max=100.0, z_max=0.6)`, uses
`xyoffset=(20.0, -7.0)`, and compares:

```text
gmm_target_chunk=2, gmm_component_chunk=5
gmm_target_chunk=0, gmm_component_chunk=0
mixture_same_family
```

The expected values independently check:

- processed v0.3 target order and scaling.
- `2 * loss_gmm` in output channel 0.
- `2 * loss_bkg` in output channel 1.
- `last_components["loss_bkg"]` and `last_components["loss_total"]`.

## Numerical Guard Fixtures For Slice 4.4

### extreme_probability_sigma_photon_z

Purpose:

- Guard old `GMMLoss` numerical behavior for legal but extreme inputs.
- Keep the route aligned with the old implementation's existing
  `clamp(..., min=eps)`, `torch.log(prob + eps)`, manual chunking, and
  `MixtureSameFamily` semantics.
- Verify finite outputs and backend parity without adding new loss terms,
  clipping policies, or fallback logic.

Shape:

```text
N=1, H=2, W=3, M=2
```

The fixture includes:

- detection probabilities containing zero, near-one, and tiny positive values.
- per-channel sigmas below `eps` to exercise old-style sigma clamping.
- large photon and z targets using explicit `photon_scale=100.0` and
  `z_scale=0.6`.
- non-zero background differences using old summed MSE semantics.

It compares chunked manual, unchunked manual, and `mixture_same_family`
backends and requires all returned losses and recorded components to be finite.

## Acceptance For Slice 4.2

Slice 4.2 is complete when:

- `ActiveSMLMGMMLoss` implements the count, all-pixel GMM localization, and
  dense background terms.
- `manual_chunked` and `mixture_same_family` match on deterministic CPU
  fixtures.
- `active_smlm_gmm_loss` is available through the training runtime registry.
- explicit runtime configs can derive `photon_scale` and `z_scale` for the GMM
  target adapter.

Hardening after Slice 4.2 additionally locks:

- non-zero `xyoffset` pixel-center shifts.
- `ch_weight` channel disabling.
- old-route `TargetProcess` phot/z scaling and `disable_attr` before v0.3
  conversion.
- direct GMM adapter `disable_attr` for already-native v0.3 targets.
- two-emitter old-route fixture with TargetProcess scaling, non-zero
  `xyoffset`, and `manual_chunked` versus `mixture_same_family` parity.
- mask/target shape safety failures.
- old-route batch loss flow from legacy `TargetProcess` order through
  `legacy_iwae_target_process_to_v03`, `LocalizationTrainBatch`,
  `active_smlm_gmm_loss`, `train_epochs`, and `training_metrics.jsonl`.
- training metrics expose only the old external GMMLoss component names for
  this route: `loss_gmm`, `loss_bkg`, and `loss_total`.
  Smoke-only `active_smlm_loss` component names such as `loss_detect`,
  `loss_pxyz`, and `loss_sigma` are not emitted by the formal GMM route.

Acceptance for Slice 4.3:

- broader old-route fixture
  `old_route_batched_masked_background_2x3x4` covers multi-batch,
  multi-target, mixed mask, non-zero background, non-uniform output maps, and
  explicit chunking behavior.
- the broader fixture passes without importing or executing `neptune_iwae`
  runtime code.
- `active_smlm_gmm_loss` remains the only formal old-GMMLoss fidelity route.

Acceptance for Slice 4.4:

- `extreme_probability_sigma_photon_z` locks finite/backend parity for old
  `GMMLoss` edge values.
- the fixture remains a test-only numerical guard and does not change
  production loss semantics.

## Old-Route Batch Flow Fixture For Slice 4.5

### old_route_real_shaped_batch

Purpose:

- Verify the old simulator/training batch shape can be represented by a native
  v0.3 `LocalizationTrainBatch` without importing old runtime code.
- Lock the tuple-style old batch contract:

```text
frames, detect_frames, bkg_frames, pxyz_tar, mask_tar
```

- Verify old legacy-order `pxyz_tar` is processed with
  `legacy_iwae_target_process_to_v03(phot_max=100.0, z_max=0.6)` before the
  formal GMM loss route consumes it.
- Verify `train_epochs` writes GMMLoss component metrics from this batch shape.

Shape:

```text
N=3, C=3, H=4, W=5, M=2
```

The fixture intentionally stays in tests. It is not a production batch
provider and does not bridge to `neptune_iwae`.

Acceptance for Slice 4.5:

- old-shaped `frames/detect/bkg/pxyz/mask` fixture flows through
  `LocalizationTrainBatch`, `active_smlm_gmm_loss`, `train_epochs`, and
  `training_metrics.jsonl`.
- persisted metrics contain GMMLoss components only.
- no new runtime dependency on `neptune_iwae` is introduced.
