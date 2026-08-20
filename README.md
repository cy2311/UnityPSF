# UnityPSF v0.4

UnityPSF v0.4 is a research package for SMLM localization training with online
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
configs/        Base, modality, experiment, calibration, and override configs.
scripts/        Reusable diagnostics, SLURM jobs, and archived legacy entry points.
src/            Installable UnityPSF Python packages.
src/double_helix/  Deprecated DH compatibility wrappers.
scripts/archive/double_helix_sweeps/  Historical DH calibration and sweep jobs.
scripts/archive/neptune_standard/  Historical Neptune standard workflow.
```

The active public namespace is `unity_psf`. The legacy `neptune_v04` namespace
remains as an advisory compatibility alias while consumers are migrated. New
installations publish only `unity-psf-*` command names. The model family name is **UnityPSF**: one localization model for
multiple PSF modalities.

The package build discovers only packages under `src`: the active
`src/unity_psf` package, the thin `src/neptune_v04` compatibility shim, and
the installable `src/double_helix` entry points. Historical SLURM commands
remain runnable from `double_helix.legacy` modules in
`scripts/archive/double_helix_sweeps/`, while reusable physics modules are owned
by the UnityPSF optics namespace. The top-level `double_helix` package contains
only compatibility adapters.

## Dual-Modality PSF MoE

The current UnityPSF milestone is one modality- and channel-routed hard MoE
model with one release checkpoint:

```text
unitypsf_joint.ckpt
    +-- Emitter2DExpert(main)
    +-- AstigmatismExpert(left)
    +-- AstigmatismExpert(right)
```

Each entry is a complete expert instance with its own localization backbone,
FiLM parameters, calibration, and physical state. During training, every
instance also owns its optimizer and scheduler. The Astigmatism left and right
instances may start from the same prototype, but they do not share parameters
or training state. There is no shared image stem in this formal path.

The public model is `UnityPSF`. Its router selects exactly one instance by
`(modality, channel_id)`, and every expert returns the common 10-channel SMLM
output contract. The release format is `unity_psf.joint_checkpoint.v2`: one
network per PSF modality with independently stored measurement-channel state.
The v1 per-instance format remains readable for compatibility and assembly.

```python
import torch
from unity_psf.models import UnityPSF

model = UnityPSF.from_checkpoint("unitypsf_joint.ckpt", device="cuda:0")
images = torch.randn(1, 3, 96, 96)
conditions = torch.zeros(1, 4)  # zernike_0, zernike_1, field_x, field_y
result = model.localize(
    images,
    modality="astigmatism",
    channel_id="left",
    conditions=conditions,
)
```

The single-process reference trainer and three-rank Expert Parallel trainer
use the same model and checkpoint contract. The default formal scheme is the
three-expert route: emitter 2D, astigmatism, and double helix. Each modality
owns one GPU; DH first performs a raw-TIFF physical update, then trains with
the direct-XYZ LUT contract, cached-window generation, vector batch size 1024,
and FP16 AMP.

```bash
unity-psf-train-joint \
  --config configs/experiments/unitypsf_dual_modality_multichannel_smoke.yaml

sbatch scripts/train/unitypsf_default_3expert.sbatch

unity-psf-checkpoint verify /path/to/unitypsf_joint.ckpt
unity-psf-checkpoint inspect /path/to/unitypsf_joint.ckpt
```

The default formal training modalities are `emitter_2d`, `astigmatism`, and
`double_helix`. Routing remains explicit and deterministic. The DH route uses
the raw-TIFF physical-update coefficient map generated at job start; it does
not infer a modality from image content.

For a smoke test or a legacy two-modality experiment, use the explicitly named
configs and scripts under `configs/experiments/` and `scripts/train/` rather
than the default entrypoint.

Runtime artifacts are intentionally local-only:

```text
.local/         Local caches, temporary resolved configs, one-off helpers.
logs/           SLURM stdout/stderr.
output/         Training runs, checkpoints, metrics, figures, ROI banks.
```

Those runtime directories are ignored by Git and should not be committed.

## Standard End-to-End Entry

```bash
git clone <UnityPSF repository URL>
cd UnityPSF
python -m pip install -e .
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
export UNITY_V04_RAW_TIFF_PATH=/path/to/raw_stack.ome.tif
bash scripts/archive/neptune_standard/run_standard_pipeline.sh
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
ZMAP_SAMPLE_KIND=paint UNITY_V04_RAW_TIFF_PATH=/path/to/paint_stack.ome.tif \
  sbatch scripts/archive/neptune_standard/train_standard_3367_hqzmap.sh
```

For the local GUI submitter:

```bash
./scripts/archive/neptune_standard/run_gui_submit.sh
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
bash scripts/archive/neptune_standard/run_standard_pipeline.sh
```

Low-level training scripts remain available for controlled experiments:

```bash
sbatch scripts/archive/neptune_standard/train_standard_3367_hqzmap.sh
sbatch scripts/archive/neptune_standard/train_standard_3367.sh
sbatch scripts/archive/neptune_standard/train_standard_3367_peakbootstrap.sh
python scripts/archive/neptune_standard/standard.py --check
```

## Multicolor Reconstruction

The default v0.4 ratiometric multicolor reconstruction is a union-based raw
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

- Installable `src/unity_psf` package with a temporary `neptune_v04` import shim.
- One top-level `UnityPSF` model with exact `(modality, channel_id)` hard routing.
- Complete and independent `Emitter2DExpert(main)`,
  `AstigmatismExpert(left)`, and `AstigmatismExpert(right)` instances.
- Atomic, integrity-checked `unity_psf.joint_checkpoint.v2` release/resume files,
  with compatibility loading for v1 checkpoints.
- Single-process round-robin and three-rank Expert Parallel joint training.
- Modality/channel-separated training reports, overlays, reconstructions, and
  physical-state availability panels.
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

Engineering status:

- The complete synthetic contract suite passes, and the 3-GPU SLURM smoke run
  completed with one joint checkpoint and successful route reloads.
- This is an engineering milestone, not a scientific performance claim. The
  first scientific baseline still requires real Origami and Astigmatism
  left/right data, real peak-zmap/gamma state, and human review of the report.

Out of scope for the public repository:

- Raw TIFF datasets.
- SLURM logs.
- Training outputs, checkpoints, ROI banks, and generated figures.
- Local one-off diagnostic scripts under `.local/`.
