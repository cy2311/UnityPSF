#!/usr/bin/env bash
#SBATCH --job-name=dh_real_tcell
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_real_tcell_%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
CALIBRATION_TIFF=${V04_ROOT}/output/double_helix/real_tcell/raw_calibration/real_tcell_calib_cut1_19px.tif
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/real_tcell_mode42_steps2000_cosine_z1000/calibration

mkdir -p "${V04_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
nvidia-smi
export PYTHONPATH="${V04_ROOT}/src"
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.run_calibration \
  --calibration-tiff "${CALIBRATION_TIFF}" \
  --output-dir "${OUTPUT_ROOT}" \
  --mode-count 42 \
  --device cuda \
  --wavelength-nm 660 \
  --na 1.27 \
  --refractive-index 1.33 \
  --pixel-size-nm 207 \
  --z-step-nm 40 \
  --z-index-origin 1 \
  --z-sign 1 \
  --npupil 128 \
  --psf-size 19 \
  --adam-steps 2000 \
  --fit-z-range-nm 1000 \
  --learning-rate-schedule cosine \
  --minimum-learning-rate-ratio 0.1 \
  --restart-count 4 \
  --seed 20260724
