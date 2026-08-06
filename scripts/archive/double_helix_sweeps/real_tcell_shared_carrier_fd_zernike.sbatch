#!/usr/bin/env bash
#SBATCH --job-name=dh_shared_fd
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_shared_fd-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
PIXEL_PUPIL_ROOT=${V04_ROOT}/output/double_helix/real_tcell_independent_pixel_pupils_w5434_init_z2000/calibration
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/real_tcell_shared_carrier_fd_zernike_mode21_complexcircle_z2000/calibration

mkdir -p "${V04_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
export MPLCONFIGDIR="${RACE_ROOT}/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${V04_ROOT}/src"

nvidia-smi
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); assert torch.cuda.is_available()'
/home/guest/anaconda3/bin/python -m double_helix.legacy.run_shared_carrier_field \
  --pixel-pupil-dir "${PIXEL_PUPIL_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --device cuda \
  --mode-count 21 \
  --alternating-rounds 6 \
  --gamma-steps 300 \
  --gamma-learning-rate-nm 1.0 \
  --field-ridge 0.001
