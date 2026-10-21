#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_field
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/dh_m1_field-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V03_ROOT=${RACE_ROOT}/neptune_v0.3
DATASET_ROOT=${RACE_ROOT}/datasets/training_sets/double_helix/Simulated_datasets_Microscope1
CALIBRATION_GAMMA=${V03_ROOT}/output/double_helix/microscope1/calibration/arrays/gamma_coefficients.npz
OUTPUT_ROOT=${V03_ROOT}/output/double_helix/microscope1/field_gamma

mkdir -p "${V03_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
nvidia-smi
export PYTHONPATH="${V03_ROOT}:${V03_ROOT}/src"
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.run_field_gamma \
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
