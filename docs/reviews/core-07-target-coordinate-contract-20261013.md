# CORE-07: Unified Target, Coordinate, and Unit Contract

## Decision

`SMLMTargetContract` is the single semantic contract for localization targets.
Every target tensor has an explicit source order and is converted at the
boundary to the canonical physical representation:

```text
[x_px, y_px, z_nm, photons]
```

The model's fixed output order remains explicit in the contract:

```text
[photons, x_px, y_px, z_normalized]
```

Legacy `legacy_iwae` and current `v03` orders are compatibility/input formats,
not independently interpreted formats inside evaluation or posterior code.

## Integrated Paths

- `SMLMTargetContract` provides validated order, activation, scale, and unit
  metadata plus bidirectional conversion to canonical physical coordinates;
- runtime resolved contracts now include `target_contract`;
- online provider batches carry target contract metadata;
- localizer evaluation decodes predictions and targets through the same
  contract;
- posterior metadata records the resolved contract;
- GMM target adaptation uses the contract for its single legacy conversion
  boundary;
- ROI and high-fidelity posterior paths consume the resolved z scale in nm.

## Compatibility

Existing configuration aliases remain readable at the boundary. The
`legacy_iwae` representation is still emitted when explicitly requested, but
its conversion is centralized and round-trips through the canonical physical
representation without changing z units or photon values.

## Verification

- target contract round-trip tests cover `legacy_iwae` and `v03`;
- evaluation tests verify equal physical coordinates for both target orders;
- runtime contract tests verify order, units, scales, and activation metadata;
- full validation must be run with `PYTHONPATH=src` so the test process cannot
  accidentally import the adjacent `unity` checkout.
