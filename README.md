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

## Standard End-to-End Entry

```bash
cd neptune_v0.3
python -m pip install -e ".[dev]"
python -m pytest
```

The current standard route is the 3371-style fast route:

```text
raw TIFF
  -> high-quality left/right initial zmap from raw TIFF
  -> fast-route training
  -> left channel infer/filter/recon
  -> right channel infer/filter/recon
  -> union raw-ratio dual-channel reconstruction
```

Submit the full standard pipeline:

```bash
export NEPTUNE_V03_RAW_TIFF_PATH=/path/to/raw_stack.ome.tif
bash run_standard_pipeline.sh
```

The training and pipeline entry points infer the high-quality initial-zmap
sample preset from the raw TIFF path before submitting work to SLURM. The
currently supported high-quality zmap presets are `microtube`, `paint`, and
`ncp`. If a path looks like `paint` but `ZMAP_SAMPLE_KIND=microtube` is passed,
submission is rejected before any bootstrap output is produced. Paths that look
like `dynamin` or `membrane` are also rejected until dedicated zmap presets are
added, rather than silently falling back to the microtube preset.

For manual submission, override only after validating the preset:

```bash
ZMAP_SAMPLE_KIND=paint NEPTUNE_V03_RAW_TIFF_PATH=/path/to/paint_stack.ome.tif sbatch train_standard_3367_hqzmap.sh
```

For the local GUI submitter:

```bash
./run_gui_submit.sh
```

The GUI previews the first TIFF frame, draws the left/right crop boxes, infers
the sample kind from the selected TIFF path, and blocks mismatched submissions.

Default training settings:

- ROI/input size: 96 x 96.
- PSF patch: 25 x 25.
- Epoch route: 300 epochs, batch size 24, 417 steps per epoch.
- LR schedule: 3052 parity, StepLR with step size 10 epochs and gamma 0.9.
- Physical update: start epoch 30, every 5 epochs.
- Physical update target: 5000 projected emitters.
- Initial physical model: freshly recomputed high-quality left/right zmap from raw TIFF.
- Fast route: global-field LUT, LUT epoch prewarm, cached window, fused projection.

Default infer/recon settings:

- Full raw TIFF inference up to 8000 frames.
- ROI/input size: 96 x 96.
- Valid core: 80 x 80.
- Cut edge: 8 px.
- Input preprocessing: FD-DeepLoc-style recenter.
- Final filter/recon: probability >= 0.9, no localization-precision gate.

The pipeline submits dependent SLURM jobs and prints the training run directory
and final left, right, and dual-channel output directories. To run infer/recon
from an existing completed training run:

```bash
PIPELINE_MODE=infer_only \
RUN_DIR=/path/to/output/<run_tag>/<run_name_jobid> \
bash run_standard_pipeline.sh
```

Low-level training scripts remain available for controlled experiments:

```bash
sbatch train_standard_3367_hqzmap.sh
sbatch train_standard_3367.sh
sbatch train_standard_3367_peakbootstrap.sh
python standard.py --check
```

## Multicolor Reconstruction

The default v0.3 ratiometric multicolor reconstruction is a union-based raw
ratio route, matching the validated Neptune-IWAE multicolor workflow:

```text
left infer set + right infer set -> union duplicate suppression -> raw TIFF left/right intensity -> ratio_right threshold -> two-color render
```

Key defaults:

- Use the unfiltered `infer/predictions_merged.h5` outputs from both channels.
- Union left/right emitter sets with `union_dist_px=2.0`.
- For duplicate detections, keep the right-channel position, z, probability,
  and localization precision.
- Re-measure left/right intensity directly from the raw TIFF over the local
  emitter coordinate, instead of using localization photon estimates.
- Classify color with `ratio_right = I_right / (I_left + I_right)` and
  `ratio_threshold=0.4`.
- Render with right-priority localization precision and no locprec gate.

Standard submission:

```bash
sbatch scripts/infer/run_3371_union_raw_ratio_bicolor.sbatch
```

Override the input sets when needed:

```bash
LEFT_PREDICTIONS=/path/to/left/infer/predictions_merged.h5 \
RIGHT_PREDICTIONS=/path/to/right/infer/predictions_merged.h5 \
SAMPLE_TIFF=/path/to/raw.ome.tif \
RUN_NAME=my_union_raw_ratio_bicolor \
sbatch scripts/infer/run_3371_union_raw_ratio_bicolor.sbatch
```

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
- Union-based raw-TIFF ratiometric multicolor reconstruction.
- Tests for config, runtime, localization, ROI bank, gamma update, peak, and training loop behavior.

Out of scope for the public repository:

- Raw TIFF datasets.
- SLURM logs.
- Training outputs, checkpoints, ROI banks, and generated figures.
- Local one-off diagnostic scripts under `.local/`.
