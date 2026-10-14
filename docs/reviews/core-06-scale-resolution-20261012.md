# CORE-06: Unified Photon/Z Scale Resolution

## Decision

All training and inference paths resolve physical scales through
`unity_psf.localization.scale_resolver.resolve_physical_scales`.

The priority is:

1. `train.loss.params.photon_scale` / `z_scale`;
2. `train.scaling.photon_max` / `z_max` and legacy aliases;
3. legacy `train.normalization.photon_scale` or `train.train_params.z_max`;
4. `train.online_generation.photon_scale` / `z_scale`.

The selected source, normalized values, units, and conflicting lower-priority
values are included in the resolved contract. Conflicts are not silently
discarded, which makes existing mixed configurations observable while keeping
legacy runs compatible.

## Z Units

`z_scale_unit` accepts `um` or `nm` and may be placed beside the explicit loss
scale, scaling value, legacy train parameter, or online-generation value. When
omitted, the legacy compatibility rule is centralized in the resolver: values
up to `10` are interpreted as micrometers; larger values as nanometers. All
physical posterior and ROI-inference paths consume the resolver's explicit
`z_scale_nm` result.

## Integrated Paths

- localization runtime loss and online-provider configuration;
- localizer evaluation and target decoding;
- high-fidelity posterior sampling, held-out monitoring, ROI projection, and
  ROI-bank inference;
- legacy posterior/decoder z conversion helpers;
- online simulation range defaults.

## Verification

- `pytest -q`: `359 passed, 1 skipped`;
- `python -m compileall -q src tests`: passed;
- no CUDA or multi-GPU validation was run in this environment.
