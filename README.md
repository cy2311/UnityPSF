# Neptune v0.3

Neptune v0.3 is a research package for SMLM localization training with online
simulation, ROI-bank posterior sampling, and physical-model gamma updates from
real raw TIFF data.

This directory is a cleaned reconstruction of the active `neptune_iwae`
workflow. The supported high-fidelity workflow is the microtube ROI-bank gamma
loop:

```text
raw TIFF -> peak bootstrap -> loc training -> fixed ROI library -> posterior sampling -> gamma update -> feedback maps
```

## Repository Layout

```text
configs/        Base and override YAML configs.
docs/           Architecture notes, parity specs, and baseline references.
scripts/        Reusable diagnostics and SLURM submission scripts.
src/            Installable neptune_v03 Python package.
tests/          Unit and smoke tests.
standard.py     Default batch-budget route validator/submission helper.
```

Runtime artifacts are intentionally local-only:

```text
.local/         Local caches, temporary resolved configs, one-off helpers.
logs/           SLURM stdout/stderr.
output/         Training runs, checkpoints, metrics, figures, ROI banks.
```

Those runtime directories are ignored by Git and should not be committed.

## Quick Start

```bash
cd neptune_v0.3
python -m pip install -e ".[dev]"
python -m pytest

export NEPTUNE_V03_RAW_TIFF_PATH=/path/to/raw_tiff_or_stack_dir

python -m neptune_v03.config.materialize \
  --base configs/microtube_base.yaml \
  --override configs/overrides/standard_roi_gamma.yaml \
  --override configs/overrides/standard_roi_gamma_batch_budget.yaml \
  --output .local/tmp/standard/resolved_standard_roi_gamma_batch_budget.yaml

python standard.py --check
python scripts/run_high_fidelity_dry_run.py
python scripts/run_high_fidelity_dry_run.py \
  --raw-tiff /path/to/raw_stack.tif \
  --run-name real_tiff_smoke
python -m neptune_v03.diagnostics.gamma_update_monitor --run-dir output/some_run
```

The current default route is batch-budget based, not epoch-budget based:

- 10,000 training batches by default.
- ROI-bank gamma starts at batch 2,000.
- Gamma updates run every 500 batches.
- ROI size is 128 px in the default route.
- Posterior sampling uses 25 samples per posterior group.
- Gamma optimization uses 100 Adam steps at learning rate 0.025.
- Held-out ROI loss is monitored, not used as a hard rejection gate.

The SLURM helper is:

```bash
python standard.py --submit
```

The helper validates the resolved config before calling `sbatch`.

## Current Scope

Implemented:

- Installable `src/neptune_v03` package.
- YAML base/override config materialization.
- High-fidelity training entrypoint and batch-budget default route.
- Online cached-window localization batch provider.
- SMLM U-Net runtime and active GMM-style localization loss.
- Legacy-style localization decoding and 3D greedy eval metrics.
- Peak bootstrap and field-dependent conditioning handoff.
- ROI-bank HDF5 model, raw-TIFF harvesting, and posterior sampling.
- Per-domain ROI-bank gamma updates with feedback coeff-map export.
- Gamma monitor metrics, held-out loss monitoring, and diagnostic artifacts.
- Tests for config, runtime, localization, ROI bank, gamma update, peak, and training loop behavior.

Out of scope for the public repository:

- Raw TIFF datasets.
- SLURM logs.
- Training outputs, checkpoints, ROI banks, and generated figures.
- Local one-off diagnostic scripts under `.local/`.
