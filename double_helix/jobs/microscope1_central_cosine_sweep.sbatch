#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_central
#SBATCH --array=21,30,42
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/dh_m1_central%A_%a.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V03_ROOT=${RACE_ROOT}/neptune_v0.3
DATASET_ROOT=${RACE_ROOT}/datasets/training_sets/double_helix/Simulated_datasets_Microscope1
MODE_COUNT=${SLURM_ARRAY_TASK_ID}
OUTPUT_ROOT=${V03_ROOT}/output/double_helix/microscope1_mode${MODE_COUNT}_steps2000_cosine_z1000/calibration

mkdir -p "${V03_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
nvidia-smi
export PYTHONPATH="${V03_ROOT}:${V03_ROOT}/src"
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
  --adam-steps 2000 \
  --fit-z-range-nm 1000 \
  --learning-rate-schedule cosine \
  --minimum-learning-rate-ratio 0.1 \
  --restart-count 4 \
  --seed 20260723
