#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_high
#SBATCH --array=64,72,84,92
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_high%A_%a.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
DATASET_ROOT=${RACE_ROOT}/datasets/training_sets/double_helix/Simulated_datasets_Microscope1
MODE_COUNT=${SLURM_ARRAY_TASK_ID}
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/microscope1_mode${MODE_COUNT}/calibration

mkdir -p "${V04_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
nvidia-smi
export PYTHONPATH="${V04_ROOT}/src"
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.run_calibration \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --mode-count "${MODE_COUNT}" \
  --device cuda \
  --wavelength-nm 660 \
  --na 1.27 \
  --refractive-index 1.33 \
  --npupil 128 \
  --adam-steps 300 \
  --restart-count 4 \
  --seed 20260723
