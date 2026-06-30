# ROI Gamma Diagnostics Parity Spec

## Status

Slice 5.12 completed as a specification slice. This document defines the
historical diagnostics contract that v0.3 should preserve before implementing
the richer figure pack. It does not implement the full historical figure pack.

## Scope

The active route is the ROI-bank gamma loop documented in
`docs/baselines/3052_microtube_roi_gamma.md`. v0.3 currently writes compact
gamma monitor artifacts:

- `gamma_alternation_summary.json`
- `gamma_update_monitor.md`
- `raw_vs_recon.png`

Since Slice 5.11 these live under:

```text
artifacts/roi_bank_gamma/epoch_XXXX/source_<source>/domain_<domain>/
```

Slice 5.12 records what the old diagnostics measured, which outputs should
exist in the clean route, and which acceptance tests should guard future
implementation slices. The implementation starts in Slice 5.13 or later.

## Historical Inputs

The old diagnostics were not a single clean API. The useful contract is the
data they consumed and the metrics they produced:

- Fixed ROI-bank records with raw photon ROIs, smoothed background, frame
  windows, domain names, ROI origins, and persisted emitter/posterior metadata.
- Initial peak-bootstrap coefficient or ZMap files per domain.
- Final ROI-bank gamma feedback coefficient or ZMap files per domain.
- Current-localizer posterior samples from selected or held-out ROI records.
- Raw TIFF domain crops for per-domain reconstruction checks.
- Renderer configuration sufficient to project sampled emitters back into raw
  photon ROIs.

The active microtube baseline uses two domains, `left` and `right`, but the
v0.3 contract must remain per-domain and not hard-code those names.

## Historical Diagnostic Families

### 1. Gamma Monitor Summary

Old reference:

- `neptune_iwae/scripts/diagnostics/gamma_update_monitor.py`
- report writer inside `neptune_iwae/neptune_core/roi_bank_gamma_alternation.py`

Required data:

- epoch
- ROI bank path
- domain name or domain group
- selected step and steps completed
- gamma before/after norm and gamma delta norm
- selected ROI count
- selected sampled emitter count
- selected projected photons
- selected background mean
- selected Poisson NLL
- held-out availability, source, sample count, initial NLL, final NLL, and
  delta
- per-domain ROI record, emitter, grid-cell, and coverage summaries when
  available

v0.3 status:

- Compact JSONL/JSON/MD monitor artifacts are implemented.
- Selected and held-out NLL fields are implemented for the current smoke
  objective.
- Domain/source grouped artifact paths are implemented.
- Full per-domain grid coverage and persisted-emitter summaries are not yet
  implemented.

### 2. Initial vs Final Gamma/ZMap

Old reference:

- `neptune_iwae/scripts/diagnostics/gamma_zmap_before_after.py`
- baseline directory:
  `neptune_iwae/output/3052_initial_vs_epoch300_zmap`
- summary file:
  `delta_gamma_physical_zmap_before_after_summary.json`

Required outputs:

- per-domain delta-gamma before/after plot
- per-domain physical ZMap before/after/delta summary PNG
- per-domain top-delta mode PNG
- per-domain delta-by-Zernike-mode bar PNG
- summary JSON with paths and scalar metrics

Required metrics:

- delta gamma before norm
- delta gamma after norm
- delta gamma step norm
- top delta gamma indices
- physical coefficient-map delta abs mean
- physical coefficient-map delta abs max
- dominant delta Zernike mode
- mode-order consistency check

v0.3 implementation boundary:

- Must compare peak-derived base maps to base plus ROI-bank gamma feedback.
- Must report values per-domain.
- Must treat missing base/final maps as a clear unavailable diagnostic, not as
  a successful zero delta.

### 3. Raw TIFF Patch Reconstruction

Old reference:

- `neptune_iwae/scripts/diagnostics/raw_tiff_patch_recon_gpu.py`
- baseline directory:
  `neptune_iwae/output/3052_epoch300_raw_tiff_patch_recon_gpu`
- summary file:
  `raw_tiff_patch_recon_summary.json`

Required outputs:

- per-domain raw/reconstruction patch figure
- per-domain loss curve figure
- per-domain coefficient map figure or coeff-map path
- summary JSON containing per-domain paths, metrics, selection metrics, and
  fit steps

Required metrics:

- harvest count
- selected patch count
- initial loss
- final loss
- Poisson NLL mean
- patch MSE mean/median when available
- patch NCC mean/median when available
- selection metrics used to choose stable patches

v0.3 implementation boundary:

- This diagnostic is sanity evidence only. It must not replace fixed held-out
  ROI-bank monitoring.
- Raw patch reconstruction may use GPU/vector PSF later, but the contract must
  also support a CPU smoke backend for tests.

### 4. Fixed ROI Reconstruction

Old reference:

- historical `roi_recon_visual_diagnostic.py` diagnostic named in
  `neptune_iwae/docs/ROI_BANK_GAMMA_OPERATIONAL_DEFAULTS.md`
- baseline directories:
  `neptune_iwae/output/3052_epoch300_raw_tiff_roi128_recon_gpu_first5`
  and
  `neptune_iwae/output/3052_epoch300_raw_tiff_roi128_recon_gpu_frames100_110_roi5`

Required outputs:

- per-domain fixed-ROI raw image panels
- per-domain projected reconstruction panels
- per-domain residual or difference panels
- compact summary JSON linking all PNGs

Required metrics:

- selected ROI count
- selected emitter count
- candidate count
- target emitters reached
- per-domain ROI count
- rendered count
- initial NLL
- final NLL
- initial RMS
- final RMS

v0.3 implementation boundary:

- The diagnostic must operate on fixed ROI-bank records, not ad hoc newly
  sampled training batches.
- It must preserve ROI geometry, frame windows, background handling, and
  over-cut settings from the ROI bank.
- It must report before/after values in the same terms as the baseline even if
  renderer internals differ.

### 5. Representative PSF Shape Grid

Old reference:

- `psf_shape_grid_compare.py` named in
  `neptune_iwae/docs/ROI_BANK_GAMMA_OPERATIONAL_DEFAULTS.md`
- related single-emitter diagnostic:
  `neptune_iwae/scripts/diagnostics/roi_single_emitter_psf_diagnostic.py`

Required outputs:

- PSF shape grids at representative field positions
- multiple z planes around the active axial range
- before/after or baseline/current comparisons where gamma feedback is
  available
- summary JSON with figure paths and renderer metadata

Required metrics:

- rendered PSF sum
- centroid or moment summaries
- shape-width or second-moment summaries
- optional NCC or difference metrics for before/after comparison

v0.3 implementation boundary:

- This is lower priority than monitor, ZMap before/after, and fixed ROI
  reconstruction.
- It should be implemented only after renderer parity is stable enough to make
  the plots meaningful.

## Output Layout Contract

Future diagnostics should extend the Slice 5.11 grouped layout without moving
the existing compact artifacts:

```text
artifacts/roi_bank_gamma/epoch_XXXX/source_<source>/domain_<domain>/
  gamma_alternation_summary.json
  gamma_update_monitor.md
  raw_vs_recon.png
  diagnostics/
    zmap_before_after/
    raw_tiff_patch_recon/
    fixed_roi_recon/
    psf_shape_grid/
```

Every diagnostic subdirectory should contain a machine-readable summary JSON
and one or more PNG files. Summary JSON paths should be linked from
`gamma_update_metrics.jsonl` only when the diagnostic was actually executed.

## Acceptance Plan

### Slice 5.13: Diagnostics Manifest Contract

Status: completed.

- A small diagnostics manifest writer lives under the grouped artifact
  directory.
- The manifest lists available diagnostics and explicit `not_run` reasons when
  a future diagnostic is not yet implemented.
- Existing compact summary/report/PNG paths remain unchanged.
- Tests use tiny smoke data and do not require GPU rendering.

### Slice 5.14: ZMap Before/After Smoke

Status: completed for CPU smoke diagnostics.

- Adds a CPU smoke implementation for per-domain before/after coefficient maps.
- Emits `delta_gamma_physical_zmap_before_after_summary.json`.
- Includes delta mean/max and dominant mode metrics.
- Do not claim vector-PSF scientific parity.

### Slice 5.15: Fixed ROI Reconstruction Smoke

Status: completed for CPU smoke diagnostics.

- Extends the current `raw_vs_recon.png` path into a structured fixed-ROI
  reconstruction diagnostic.
- Emits per-domain selected ROI count, rendered count, Poisson NLL, and RMS.
- Keep held-out monitor semantics unchanged.

### Slice 5.16: Raw TIFF Patch Reconstruction Spec/Smoke

Status: completed for CPU smoke diagnostics.

- Adds the raw patch reconstruction summary contract using tiny CPU fixtures.
- Includes Poisson NLL, MSE, and NCC fields.
- Keep it explicitly separate from the fixed ROI-bank held-out monitor.

### Slice 5.17: PSF Shape Grid Spec/Smoke

Status: completed for CPU smoke diagnostics.

- Adds representative PSF grid summary and PNG smoke fixture.
- Remains a lightweight Gaussian smoke check, not a vector-PSF parity claim.

## Non-Goals

- Do not import old `neptune_iwae` runtime code.
- Do not write generated PNGs or summaries into source-controlled docs.
- Do not promote smoke diagnostics to scientific validation.
- Do not change the gamma objective, held-out monitor semantics, or ROI-bank
  construction route in a diagnostics slice.
- Do not implement the full historical figure pack before the manifest and
  before/after contracts are test-backed.
