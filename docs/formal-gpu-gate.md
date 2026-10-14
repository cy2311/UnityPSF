# Formal three GPU training gate

The five coefficient maps and the Double Helix pupil are external calibration inputs. Stage them into `.local/formal-inputs/` with the Python environment used for training:

```bash
PYTHONPATH=src python -m unity_psf.training.formal_gate stage \
  --destination .local/formal-inputs \
  --emitter-2d-left /path/to/emitter_left.npz \
  --emitter-2d-right /path/to/emitter_right.npz \
  --astigmatism-left /path/to/astig_left.npz \
  --astigmatism-right /path/to/astig_right.npz \
  --double-helix-map /path/to/dh_coeff_maps.npz \
  --double-helix-pupil /path/to/dh_complex_pupil_128.npz
```

The command checks NPZ structure, copies each file, and writes `manifest.json` with source paths and SHA256 values. The pupil must be 128 × 128. The checked in `tcell_dh_bead_01_complex_pupil_npupil64.npz` is 64 × 64 and cannot satisfy this contract. Review the source and SHA256 values before treating a run as a reproduction of a prior experiment.

Run the short three GPU gate first:

```bash
bash scripts/submit_sbatch.sh scripts/train/unitypsf_three_modality_gpu_smoke.sbatch
```

The smoke uses the five formal instances with eight steps per instance, one epoch, and three CUDA ranks. Success requires `output/unitypsf/gpu-smoke-<job-id>/gpu-gate-receipt.json`, confirmed CUDA on all ranks, optimizer updates for each modality, a joint checkpoint, and a report. Inspect the SLURM log before submitting the 300 epoch job:

```bash
export UNITYPSF_GPU_GATE_RECEIPT="output/unitypsf/gpu-smoke-<job-id>/gpu-gate-receipt.json"
bash scripts/submit_sbatch.sh scripts/train/unitypsf_default_3expert.sbatch
```

Use an absolute path for `UNITYPSF_GPU_GATE_RECEIPT` when submitting from outside this project directory. The formal launcher verifies that the receipt still matches the staged inputs and the smoke checkpoint.

The smoke verifies execution, not localization accuracy or the scientific identity of supplied calibration files. This project currently sits inside a larger Git repository, so its nested `.github/workflows/quality.yml` is not active at the repository root. The SLURM receipt is the GPU release gate until a GPU runner is connected there.

For route-selectable validation, use `scripts/train/unitypsf_selection_cuda_smoke.sbatch`. It runs the CLI selector over 2D, Astigmatism, and DH with `--require-cuda`; the selected modalities determine the required rank count and CPU fallback is disabled.
