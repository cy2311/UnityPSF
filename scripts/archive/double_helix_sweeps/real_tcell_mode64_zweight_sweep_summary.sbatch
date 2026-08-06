#!/usr/bin/env bash
#SBATCH --job-name=dh_m64_zwsum
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_m64_zwsum_%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
OUTPUT_BASE=${V04_ROOT}/output/double_helix
SWEEP_ROOT=${OUTPUT_BASE}/real_tcell_deepz_mode64_weight_sweep_z2000
SUMMARY_ROOT=${SWEEP_ROOT}/summary

mkdir -p "${V04_ROOT}/logs/slurm" "${SUMMARY_ROOT}"
export MPLCONFIGDIR="${RACE_ROOT}/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${V04_ROOT}/src"
/home/guest/anaconda3/bin/python -m double_helix.legacy.summarize_weight_sweep \
  --candidate edge_11535 1 1.5 3 5 "${OUTPUT_BASE}/real_tcell_deepz_mode64_warm64_roi17_zopt_z2000_edgew/calibration" \
  --candidate balanced_4334 4 3 3 4 "${OUTPUT_BASE}/real_tcell_deepz_mode64_warm64_roi17_zopt_z2000_balancedw/calibration" \
  --candidate sweep_3334 3 3 3 4 "${SWEEP_ROOT}/w3334/calibration" \
  --candidate sweep_4434 4 4 3 4 "${SWEEP_ROOT}/w4434/calibration" \
  --candidate sweep_5434 5 4 3 4 "${SWEEP_ROOT}/w5434/calibration" \
  --candidate sweep_6534 6 5 3 4 "${SWEEP_ROOT}/w6534/calibration" \
  --output-dir "${SUMMARY_ROOT}" \
  --central-z-limit-nm 600 \
  --minimum-heldout-ncc 0.90 \
  --maximum-edge-flux 0.12
