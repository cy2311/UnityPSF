# Neptune v0.4 Baseline Freeze

Baseline ID: `neptune_v04_baseline_20260802`
Package version: `0.4.0`
Freeze scope: runtime behavior and public workflow contracts

## Verified baseline

- `neptune_v04` is installed from the `src/` layout.
- All 81 Python modules under `src/neptune_v04` import successfully.
- `scripts/archive/neptune_standard/standard.py --check` passes for the microtube standard route.
- All 47 shell and SLURM entry files pass `bash -n`.
- The CPU high-fidelity dry run completes one epoch and writes a checkpoint.
- The latest checkpoint restores model, optimizer, and scheduler state.
- The CPU inference contract returns the expected `(N, 10, H, W)` output shape.
- Formal 3371 CUDA inference is not part of this local receipt because this host
  has no usable CUDA device.

## Frozen contracts

- Distribution name: `neptune-v04`.
- Import package: `neptune_v04`.
- Localization output: the existing 10-channel SMLM output contract.
- Standard configuration: `configs/base/microtube.yaml` plus the standard ROI
  gamma and batch-budget overrides.
- Runtime artifacts remain local-only under `.local/`, `logs/`, and `output/`.
- `neptune_v0.3` is outside this baseline and must remain untouched.

## Known boundary debt at freeze time

- The reusable `double_helix` package is still stored at the project root and
  was previously importable primarily from the repository working directory.
- This freeze intentionally records that debt before package discovery is
  widened to install the package independently.
- Historical `neptune_v03` strings in schema, training-mode, and result
  metadata are compatibility identifiers, not active Python imports.

## Reproduction commands

```bash
PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH \
  python scripts/archive/neptune_standard/standard.py --check

PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH \
  python -c 'import neptune_v04; print(neptune_v04.__version__)'

PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH \
  python scripts/run_high_fidelity_dry_run.py \
    --run-root .local/tmp/acceptance/high_fidelity_v4 \
    --run-name cpu_smoke --epochs 1 --device cpu
```

Any structural migration after this point must preserve these contracts or
add an explicit compatibility adapter and a new baseline receipt.
