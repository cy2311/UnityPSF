#!/usr/bin/env bash
#SBATCH --job-name=dh_pixel_pupil
#SBATCH --array=0-4%4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_pixel_pupil-%A_%a.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
SOURCE_TIFF=${RACE_ROOT}/datasets/training_sets/double_helix/Real_dataset_Tcell/Calib.tif
INITIALIZATION=${V04_ROOT}/output/double_helix/real_tcell_deepz_mode64_weight_sweep_z2000/w5434/calibration/arrays/gamma_coefficients.npz
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/real_tcell_independent_pixel_pupils_w5434_init_z2000/calibration
BEAD_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))

mkdir -p "${V04_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
export MPLCONFIGDIR="${RACE_ROOT}/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${V04_ROOT}/src"

nvidia-smi
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); assert torch.cuda.is_available()'
/home/guest/anaconda3/bin/python -m double_helix.legacy.run_pixel_pupil_calibration \
  --source-tiff "${SOURCE_TIFF}" \
  --initialization "${INITIALIZATION}" \
  --output-dir "${OUTPUT_ROOT}" \
  --bead-number "${BEAD_NUMBER}" \
  --device cuda \
  --fit-z-range-nm 2000 \
  --z-step-nm 40 \
  --roi-size 19 \
  --npupil 128 \
  --alternating-rounds 2 \
  --local-steps 100 \
  --phase-adam-steps 500 \
  --lbfgs-steps 80 \
  --phase-learning-rate 0.03 \
  --phase-anchor-weight 0.0001
