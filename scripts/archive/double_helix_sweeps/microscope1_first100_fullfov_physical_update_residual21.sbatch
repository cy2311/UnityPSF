#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_first100_r21
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_first100_r21-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
DATASET_ROOT=${RACE_ROOT}/datasets/training_sets/double_helix/Simulated_datasets_Microscope1
GLOBAL_ROOT=${V04_ROOT}/output/double_helix/microscope1_shared_carrier_residual21_20260725
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/microscope1_shared_carrier_residual21_physical_update_first100_20260725
PHYSICAL_ROOT=${OUTPUT_ROOT}/physical_update
EVALUATION_ROOT=${OUTPUT_ROOT}/evaluation
CARRIER_PATH=${GLOBAL_ROOT}/calibration/arrays/shared_double_helix_carrier.npz
INITIAL_GAMMA_PATH=${GLOBAL_ROOT}/calibration/arrays/residual_gamma_initialization.npz
INITIAL_LOCALIZATIONS_PATH=${GLOBAL_ROOT}/evaluation/arrays/independent_localizations.npz
FINAL_GAMMA_PATH=${PHYSICAL_ROOT}/arrays/final_gamma_coefficients.npz

mkdir -p \
  "${V04_ROOT}/logs/slurm" \
  "${RACE_ROOT}/.local/cache/matplotlib" \
  "${PHYSICAL_ROOT}" \
  "${EVALUATION_ROOT}"
export MPLCONFIGDIR=${RACE_ROOT}/.local/cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${V04_ROOT}/src

nvidia-smi
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); assert torch.cuda.is_available()'

/home/guest/anaconda3/bin/python -m double_helix.legacy.run_physical_update \
  --dataset-root "${DATASET_ROOT}" \
  --carrier-complex "${CARRIER_PATH}" \
  --initial-gamma "${INITIAL_GAMMA_PATH}" \
  --initial-localizations "${INITIAL_LOCALIZATIONS_PATH}" \
  --output-dir "${PHYSICAL_ROOT}" \
  --fixed-first-frame-count 100 \
  --update-count 5 \
  --heldout-fraction 0.1 \
  --gamma-steps 100 \
  --gamma-lr 0.025 \
  --frame-batch-size 8 \
  --localization-refinement-steps 30 \
  --seed 20260725 \
  --device cuda

/home/guest/anaconda3/bin/python -m double_helix.run_evaluation \
  --dataset-root "${DATASET_ROOT}" \
  --gamma-path "${FINAL_GAMMA_PATH}" \
  --carrier-complex "${CARRIER_PATH}" \
  --output-dir "${EVALUATION_ROOT}" \
  --device cuda \
  --npupil 128 \
  --max-frames 5000 \
  --refinement-steps 30 \
  --batch-size 64 \
  --threshold-sigma 15 \
  --minimum-ncc 0.35 \
  --lateral-tolerance-nm 500 \
  --axial-tolerance-nm 500

/home/guest/anaconda3/bin/python -m double_helix.legacy.render_wavefront_diagnostics \
  --dataset-root "${DATASET_ROOT}" \
  --gamma-path "${FINAL_GAMMA_PATH}" \
  --carrier-complex "${CARRIER_PATH}" \
  --localizations-path "${EVALUATION_ROOT}/arrays/independent_localizations.npz" \
  --evaluation-manifest-path "${EVALUATION_ROOT}/metadata/manifest.json" \
  --output-dir "${EVALUATION_ROOT}" \
  --frames 1,1667,3333,5000 \
  --device cuda
