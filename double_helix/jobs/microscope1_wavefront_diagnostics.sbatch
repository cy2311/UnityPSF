#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_wavefront
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/dh_m1_wavefront-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V03_ROOT=${RACE_ROOT}/neptune_v0.3
DATASET_ROOT=${RACE_ROOT}/datasets/training_sets/double_helix/Simulated_datasets_Microscope1
EVALUATION_ROOT=${V03_ROOT}/output/double_helix/microscope1/evaluation

mkdir -p "${V03_ROOT}/logs/slurm" "${EVALUATION_ROOT}"
nvidia-smi
export PYTHONPATH="${V03_ROOT}:${V03_ROOT}/src"
export MPLCONFIGDIR="${V03_ROOT}/.local/cache/matplotlib"
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.render_wavefront_diagnostics \
  --dataset-root "${DATASET_ROOT}" \
  --gamma-path "${V03_ROOT}/output/double_helix/microscope1/field_gamma/arrays/gamma_coefficients.npz" \
  --localizations-path "${EVALUATION_ROOT}/arrays/independent_localizations.npz" \
  --evaluation-manifest-path "${EVALUATION_ROOT}/metadata/manifest.json" \
  --output-dir "${EVALUATION_ROOT}" \
  --frames 1,1667,3333,5000 \
  --device cuda
