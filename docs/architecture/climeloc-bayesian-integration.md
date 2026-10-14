# climeLoc-aligned Bayesian multi-emitter fitting

## What is being adopted

The climeLoc paper (Han et al., *Optics Express*, 2025,
doi:10.1364/OE.568128) combines pre-calibrated channel transformations with
Bayesian multi-emitter fitting. The UnityPSF integration keeps the existing
per-channel neural localizer and adds a physical linked-fitting layer after
candidate detection.

The first implementation lives in:

- `src/unity_psf/localization/channel_linking.py`
- `src/unity_psf/localization/bayesian_multifit.py`

## Model

For emitter `k`, the latent shared position is `q_k = (x_k, y_k, z_k)`. Each
channel has its own photon and background parameters. A channel renderer maps
the shared position into that channel's coordinate system, evaluates its zmap
PSF, applies camera units, and renders every emitter in the ROI:

```text
mu_c(p) = background_c + sum_k photon[k,c] * PSF_c(p | transform_c(q_k))
```

The posterior objective is:

```text
sum_c Poisson(image_c | mu_c)
  + Gaussian position prior
  + log-photon prior
  + log-background prior
```

The implementation requires a differentiable renderer callback per channel.
This keeps the optimizer independent of whether the renderer uses a zmap LUT,
an empirical spline, or a vector PSF model.

## How it differs from the current fusion

Current fusion averages two already-localized coordinates. The new fitter uses
the raw pixels and jointly explains all emitters in an ROI. It therefore can
use a weak channel when it helps identify emitter count or resolve overlap,
while retaining separate photon/background values for each channel.

The neural detections are initialization and a weak prior, not the final
answer. The right-to-left transformation is fixed during the first fitting
stage. This prevents the transform from absorbing localization errors.

## Integration order

1. Fit and version a held-out Legendre transform.
2. Link candidates with propagated covariance and explicit unmatched states.
3. Build channel renderers from the left/right zmap physical states.
4. Run `fit_bayesian_multi_emitter` on linked emitters in a local ROI.
5. Compare against independent localization and coordinate-only refinement.
6. Only after a held-out gain is demonstrated, expand to overlapping emitter
   groups and full-ROI fitting.
7. At physical-update triggers, propose a small transform update using a fixed
   high-confidence subset; validate on held-out frames and roll back on failure.

## Required controls

Every experiment must report the same held-out split for:

1. independent left/right localization;
2. fixed transform plus coordinate refinement;
3. Bayesian joint fitting with raw pixels;
4. overlap-aware multi-emitter fitting.

Report per-channel XYZ RMSE, paired-link precision/recall, emitter-count error,
posterior coverage, raw-image likelihood, unmatched counts, and runtime. A
lower registration residual alone is not evidence that localization improved.

## Scope boundary

This is climeLoc-aligned, not a claim that climeLoc performs online
self-calibration. The published abstract explicitly describes
**pre-calibrated channel transformations**. Online map updates are a separate
UnityPSF extension and must be independently gated.
