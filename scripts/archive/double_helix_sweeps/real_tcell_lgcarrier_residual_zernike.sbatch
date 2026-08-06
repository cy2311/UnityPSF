#!/usr/bin/env bash
#SBATCH --job-name=dh_real_lg_res
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_real_lg_res-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
SOURCE_TIFF=${RACE_ROOT}/datasets/training_sets/double_helix/Real_dataset_Tcell/Calib.tif
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/real_tcell_lgcarrier_residual_zernike/calibration

mkdir -p "${V04_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
nvidia-smi
export MPLCONFIGDIR="${RACE_ROOT}/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${V04_ROOT}/src"
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.legacy.run_lg_calibration \
  --source-tiff "${SOURCE_TIFF}" \
  --output-dir "${OUTPUT_ROOT}" \
  --device cuda \
  --mode-count 13 \
  --maximum-beads 5 \
  --roi-size 17 \
  --npupil 128 \
  --fit-z-range-nm 2000 \
  --alternating-rounds 4 \
  --local-steps 100 \
  --global-steps 500
