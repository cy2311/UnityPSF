#!/usr/bin/env bash
#SBATCH --job-name=dh_pixel_aggregate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_pixel_aggregate-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/real_tcell_independent_pixel_pupils_w5434_init_z2000/calibration

export MPLCONFIGDIR="${RACE_ROOT}/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${V04_ROOT}/src"

/home/guest/anaconda3/bin/python -m double_helix.legacy.run_pixel_pupil_calibration \
  --output-dir "${OUTPUT_ROOT}" \
  --aggregate-only
