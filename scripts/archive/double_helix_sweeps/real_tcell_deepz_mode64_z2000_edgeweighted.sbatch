#!/usr/bin/env bash
#SBATCH --job-name=dh_real_z2k_edgew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_real_z2k_edgew_%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
CALIBRATION_TIFF=${V04_ROOT}/output/double_helix/real_tcell/raw_calibration_17px/real_tcell_calib_cut1_17px.tif
WARM_START_GAMMA=${V04_ROOT}/output/double_helix/real_tcell_deepz_mode64_warm42_roi17_zopt/calibration/arrays/gamma_coefficients.npz
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/real_tcell_deepz_mode64_warm64_roi17_zopt_z2000_edgew/calibration

mkdir -p "${V04_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
nvidia-smi
export PYTHONPATH="${V04_ROOT}/src"
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.run_calibration \
  --calibration-tiff "${CALIBRATION_TIFF}" \
  --warm-start-gamma "${WARM_START_GAMMA}" \
  --output-dir "${OUTPUT_ROOT}" \
  --mode-count 64 \
  --device cuda \
  --wavelength-nm 660 \
  --na 1.27 \
  --refractive-index 1.33 \
  --pixel-size-nm 207 \
  --z-step-nm 40 \
  --z-index-origin 1 \
  --z-sign 1 \
  --npupil 128 \
  --psf-size 17 \
  --adam-steps 2000 \
  --fit-z-range-nm 2000 \
  --learning-rate-schedule cosine \
  --minimum-learning-rate-ratio 0.1 \
  --restart-count 4 \
  --warm-start-noise-nm 5 \
  --optimize-z-calibration \
  --z-calibration-learning-rate 0.01 \
  --deep-z-loss \
  --poisson-loss-weight 1 \
  --ncc-loss-weight 2 \
  --lobe-geometry-loss-weight 3 \
  --z-bin-edges-nm 500 1000 1500 \
  --z-bin-weights 1 1.5 3 5 \
  --seed 20260725
