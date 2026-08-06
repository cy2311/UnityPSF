#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_carrier_r21
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_carrier_r21-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
DATASET_ROOT=${RACE_ROOT}/datasets/training_sets/double_helix/Simulated_datasets_Microscope1
FAILED_DIRECT21_ROOT=${V04_ROOT}/output/double_helix/microscope1_sample_mode21_steps3000_20260725
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/microscope1_shared_carrier_residual21_20260725
CALIBRATION_ROOT=${OUTPUT_ROOT}/calibration
FIELD_ROOT=${OUTPUT_ROOT}/field_gamma
EVALUATION_ROOT=${OUTPUT_ROOT}/evaluation
CARRIER_PATH=${CALIBRATION_ROOT}/arrays/shared_double_helix_carrier.npz
RESIDUAL_INITIALIZATION=${CALIBRATION_ROOT}/arrays/residual_gamma_initialization.npz

mkdir -p \
  "${V04_ROOT}/logs/slurm" \
  "${RACE_ROOT}/.local/cache/matplotlib" \
  "${CALIBRATION_ROOT}" \
  "${FIELD_ROOT}" \
  "${EVALUATION_ROOT}"
export MPLCONFIGDIR=${RACE_ROOT}/.local/cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${V04_ROOT}/src

nvidia-smi
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); assert torch.cuda.is_available()'

/home/guest/anaconda3/bin/python -m double_helix.legacy.run_microscope1_shared_carrier \
  --dataset-root "${DATASET_ROOT}" \
  --initialization "${FAILED_DIRECT21_ROOT}/calibration/arrays/gamma_coefficients.npz" \
  --output-dir "${CALIBRATION_ROOT}" \
  --device cuda \
  --alternating-rounds 2 \
  --local-steps 100 \
  --phase-adam-steps 500 \
  --lbfgs-steps 80 \
  --phase-learning-rate 0.03 \
  --phase-anchor-weight 0.0001 \
  --seed 20260725

/home/guest/anaconda3/bin/python -m double_helix.legacy.run_field_gamma \
  --dataset-root "${DATASET_ROOT}" \
  --calibration-gamma "${RESIDUAL_INITIALIZATION}" \
  --carrier-complex "${CARRIER_PATH}" \
  --output-dir "${FIELD_ROOT}" \
  --device cuda \
  --npupil 128 \
  --adam-steps 600 \
  --batch-size 64 \
  --max-training-emitters 3000 \
  --max-validation-emitters 1200 \
  --seed 20260725

/home/guest/anaconda3/bin/python -m double_helix.run_evaluation \
  --dataset-root "${DATASET_ROOT}" \
  --gamma-path "${FIELD_ROOT}/arrays/gamma_coefficients.npz" \
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
  --gamma-path "${FIELD_ROOT}/arrays/gamma_coefficients.npz" \
  --carrier-complex "${CARRIER_PATH}" \
  --localizations-path "${EVALUATION_ROOT}/arrays/independent_localizations.npz" \
  --evaluation-manifest-path "${EVALUATION_ROOT}/metadata/manifest.json" \
  --output-dir "${EVALUATION_ROOT}" \
  --frames 1,1667,3333,5000 \
  --device cuda
