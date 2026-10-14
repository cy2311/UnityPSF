# UnityPSF v0.3

## Select modalities from the CLI

`unity-psf-train` selects the modalities and measurement channels for one
Expert Parallel run. Each selected modality uses one GPU and produces a single
v2 UnityPSF checkpoint. 2D and Astigmatism accept `left`, `right`, or both;
Double Helix accepts only `main`.

```bash
unity-psf-train plan --route emitter_2d:left,right --route astigmatism:left,right --run-id my-run
unity-psf-train smoke --route emitter_2d:left --run-id my-smoke
unity-psf-train smoke --route emitter_2d:left,right --route astigmatism:left,right --route double_helix:main --dh-ppg-map /absolute/path/to/ppg_map.npz --require-cuda --run-id gpu-smoke
unity-psf-train run --route emitter_2d:left,right --route astigmatism:left,right --run-id my-run
unity-psf-train resume --route emitter_2d:left,right --route astigmatism:left,right --run-id my-run
```

The default 2D initialization is the explicit circular `circular_2d_50nm`
anchor, with a 50 nm Gaussian-sigma-equivalent spot radius. The default
Astigmatism initialization is its 660 nm, 100 nm anchor and does not require a
coefficient map. To use a calibrated sample field for a selected 2D or
Astigmatism channel, add
`--map emitter_2d:left=/absolute/path/to/map.npz`. Double Helix uses the
beads calibrated PPG/PCA map and its adjacent approved JSON manifest:

```bash
unity-psf-train plan --route double_helix:main --run-id dh-run
unity-psf-train run --route double_helix:main --dh-ppg-map /absolute/path/to/ppg_map.npz --run-id dh-run
```

`plan` checks inputs and prints the route and GPU plan without creating a run.
`smoke` runs one epoch of up to eight steps per channel; `run` defaults to 300
epochs and 417 steps per channel. Override with `--epochs`,
`--steps-per-epoch`, `--batch-size`, and `--run-root`. `resume` must use the
same arguments and external inputs as the original `run`; the resolved config
and input hashes are checked before launching. Outputs are under
`output/unitypsf/<run-id>/`, including `resolved/` and
`checkpoints/unitypsf_joint.ckpt`. Smoke metrics establish interface health,
not localization quality. A successful smoke also writes
`metadata/short_run_gate.json` after checking finite losses, an optimizer
update and held-out loss for every channel, matching checkpoint routes, and
resume shard progress. Use `--require-cuda` for the release smoke gate; it
requires one visible CUDA device per selected modality and never enables the
CPU fallback. Every released checkpoint records a
`unitypsf.physical_provenance.v1` entry per modality/channel, including anchor
profile and value, coefficient-map or DH PCA/beads input hashes, and whether a
fixed anchor was enabled. Coefficient maps contain absolute Zernike values;
combining one with an additional fixed Astigmatism anchor is rejected. The DH PPG map is fixed during this CLI route; the
legacy `ppg_physical_update` template setting is not run by this trainer.

## Channel-Linked Development Direction

For future multichannel localization work in this directory, follow
[ADR 0004](docs/adr/0004-channel-linked-refinement-without-network-fusion.md)
and the [channel-linked design](docs/architecture/channel-linked-self-calibrating-localization.md):
keep the existing UnityPSF network and 10-channel per-channel output, then add
a sample-derived Legendre transform, cross-channel emitter linking, and joint
coordinate refinement. This is a target design, not a shipped capability.
The canonical package source and tests now live under `src/` and `tests/` in
this directory. The source was migrated from the current `unity/src` working
tree with caches excluded; future feature work must target this checkout.

## Local Quality Gate

Run the CPU quality checks from this checkout using one Python environment:

```bash
python -m pip install -e ".[dev]"
bash scripts/quality_check.sh
```

The gate checks Ruff, the six typed boundary modules, and the full pytest
suite. It stops at the first failure and reports missing tools before running
checks. Set `PYTHON=/path/to/python` to select an interpreter. For tools
installed with `pip install --target`, set `QUALITY_TOOLS_DIR`; the default is
`.local/quality-tools`. Caches are kept under `.local/cache/`, and the gate
does not write Python bytecode into the source tree.

Tests must not import modules from sibling research checkouts. The CUDA
renderer sources are included as wheel package data, but that does not verify
CUDA compilation or GPU performance. Default training templates still live in
the checkout's `configs/` directory, so standalone wheel installation is not
yet a supported replacement for the editable checkout. The nested GitHub
workflow is not active in the parent repository; the local gate is not a
remote CI enforcement mechanism. See [the quality baseline](docs/reviews/engineering-quality-baseline-20261012.md)
for verification evidence and remaining priorities.

UnityPSF v0.3 is a research package for SMLM localization training with online
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
scripts/        Reusable diagnostics and current SLURM jobs.
src/            Installable UnityPSF Python packages.
src/double_helix/  DH calibration and physics compatibility package.
```

The active public namespace is `unity_psf`; `neptune_v04` was removed from the
v0.3 distribution because the repository has no active consumers. New
installations publish only `unity-psf-*` command names. The model family name is **UnityPSF**: one localization model for
multiple PSF modalities.

The remaining legacy import and checkpoint surfaces are listed in
[`docs/compatibility.md`](docs/compatibility.md).

The package build discovers only packages under `src`: the active
`src/unity_psf` package, the canonical `src/ppg` PPG/PCA physical backend, and the current
`src/double_helix` calibration helpers. Historical archive and legacy replay
entry points have been removed from the final v0.3 tree.

## Dual-Modality PSF MoE

The current microtube training route uses modality- and channel-conditioned
Soft-MoE FiLM experts with one release checkpoint:

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

The public model is `UnityPSF`. Its routing is explicit by `(modality,
channel_id)`, and every expert returns the common 10-channel SMLM output
contract. The release format is `unity_psf.joint_checkpoint.v2`: one network
per PSF modality with independently stored measurement-channel state.
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
owns one GPU. The Double Helix route consumes a staged, sample-derived physical
map and calibrated pupil, then trains with the direct CUDA-GEMM simulation backend, cached-window generation, vector batch
size 1024, and FP16 AMP.

Stage the six external calibration inputs and pass the three GPU smoke gate as
described in [the formal GPU gate](docs/formal-gpu-gate.md) before formal training.

```bash
unity-psf-train run \
  --route emitter_2d:left,right \
  --route astigmatism:left,right \
  --route double_helix:main \
  --dh-ppg-map configs/calibration/ppg/ppg_map_initial.npz \
  --require-cuda \
  --run-id formal-3expert

bash scripts/submit_sbatch.sh scripts/train/unitypsf_default_3expert.sbatch

unity-psf-checkpoint verify /path/to/unitypsf_joint.ckpt
unity-psf-checkpoint inspect /path/to/unitypsf_joint.ckpt
```

The default formal training modalities are `emitter_2d`, `astigmatism`, and
`double_helix`. Routing remains explicit and deterministic. The DH route uses
the raw-TIFF physical-update coefficient map generated at job start; it does
not infer a modality from image content.

For a smoke test or a formal experiment, use `unity-psf-train` or the scripts
under `scripts/train/`. The former joint trainer modules were removed; there is
one training implementation and one release path.

Runtime artifacts are intentionally local-only:

```text
.local/         Local caches, temporary resolved configs, one-off helpers.
logs/           SLURM stdout/stderr.
output/         Training runs, checkpoints, metrics, figures, ROI banks.
```

Those runtime directories are ignored by Git and should not be committed.

## v0.3 Microtube Training Default

The default `configs/base/microtube.yaml` route uses the measured LiteLoc-style
fixed-cache schedule. CUDA GEMM renders one 10,000-center cache for the current
physical-model version, materializes it in pinned CPU memory, and asynchronously
prefetches one batch to the GPU. Training consumes the cache through a temporal
cursor; it does not reshuffle frames or break the three-frame local context.
When a physical-model update publishes a new condition version, the old cache is
invalidated and regenerated before the next batch is requested.

The cache removes repeated simulation from the steady-state loop, but it does not
make UnityPSF a LiteLoc-sized network. UnityPSF's default soft-MoE path still
runs both experts for each batch, so forward and backward remain the dominant
cost. The measured comparison and optimization priorities are recorded in
[`docs/performance/unitypsf-v03-liteloc-training.md`](docs/performance/unitypsf-v03-liteloc-training.md).

## Standard End-to-End Entry

```bash
git clone <UnityPSF repository URL>
cd UnityPSF
python -m pip install -e .
```

The current standard route is the CUDA-GEMM three-expert training job:

```bash
bash scripts/submit_sbatch.sh scripts/train/unitypsf_default_3expert.sbatch
```

The launcher delegates to the current three-modality raw-TIFF configuration
and writes checkpoints, metrics, and logs under the v0.3 runtime directories.
For a direct local or single-process run, use the corresponding `unity-psf-*`
entry point with a materialized experiment config.

Default training settings:

- ROI/input size: 64 x 64 for the microtube cached training route.
- PSF patch: 25 x 25.
- Epoch route: 300 epochs, batch size 10, 500 steps per epoch.
- LR schedule: AdamW with StepLR at optimizer step 1000 and gamma 0.9.
- Physical update: start epoch 30, every 5 epochs.
- Physical update target: 5000 projected emitters.
- Initial physical model: freshly recomputed high-quality left/right zmap from raw TIFF.
- Fast route: CUDA GEMM simulation, fixed cached window, one-batch GPU prefetch.

Default infer/recon settings:

- Full raw TIFF inference up to 8000 frames.
- ROI/input size: 96 x 96.
- Valid core: 80 x 80.
- Cut edge: 8 px.
- Input preprocessing: FD-DeepLoc-style recenter.
- Final filter/recon: probability >= 0.9, no localization-precision gate.

To run single-channel infer/recon from an existing completed training run:

```bash
RUN_DIR=/path/to/output/<run_tag>/<run_name_jobid> SIDE=left \
  bash scripts/submit_sbatch.sh scripts/infer/standard_channel_infer_recon.sbatch
```

Current training scripts remain available for controlled experiments:

```bash
bash scripts/submit_sbatch.sh scripts/train/unitypsf_default_3expert.sbatch
bash scripts/submit_sbatch.sh scripts/train/unitypsf_cuda_gemm_roi_gamma.sbatch
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
bash scripts/submit_sbatch.sh scripts/infer/run_3371_union_raw_ratio_bicolor.sbatch
```

Override the input sets when needed:

```bash
LEFT_PREDICTIONS=/path/to/left/infer/predictions_merged.h5 \
RIGHT_PREDICTIONS=/path/to/right/infer/predictions_merged.h5 \
SAMPLE_TIFF=/path/to/raw.ome.tif \
RUN_NAME=my_union_raw_ratio_bicolor \
bash scripts/submit_sbatch.sh scripts/infer/run_3371_union_raw_ratio_bicolor.sbatch
```

## Current Scope

Implemented:

- Installable `src/unity_psf` package with canonical localization runtime and data imports.
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
