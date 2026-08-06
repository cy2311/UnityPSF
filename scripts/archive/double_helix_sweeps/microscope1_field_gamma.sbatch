#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_field
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_field-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
DATASET_ROOT=${RACE_ROOT}/datasets/training_sets/double_helix/Simulated_datasets_Microscope1
CALIBRATION_GAMMA=${V04_ROOT}/output/double_helix/microscope1/calibration/arrays/gamma_coefficients.npz
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/microscope1/field_gamma

mkdir -p "${V04_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
nvidia-smi
export PYTHONPATH="${V04_ROOT}/src"
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.legacy.run_field_gamma \
  --dataset-root "${DATASET_ROOT}" \
  --calibration-gamma "${CALIBRATION_GAMMA}" \
  --output-dir "${OUTPUT_ROOT}" \
  --device cuda \
  --npupil 128 \
  --adam-steps 600 \
  --batch-size 64 \
  --max-training-emitters 3000 \
  --max-validation-emitters 1200 \
  --seed 20260723
