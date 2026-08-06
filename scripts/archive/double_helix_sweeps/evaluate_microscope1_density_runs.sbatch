#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_eval-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_eval-%j.err

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
OLD_ROOT=${V04_ROOT}/output/double_helix/microscope1_neptune_single_channel_train300_20260725
RESUME_ROOT=${V04_ROOT}/output/double_helix/microscope1_neptune_single_channel_train300_first100tyx_resume_20260725
FRESH_ROOT=${V04_ROOT}/output/double_helix/microscope1_neptune_single_channel_train300_first100tyx_fresh_20260725
OUTPUT_DIR=${V04_ROOT}/output/double_helix/microscope1_neptune_single_channel_train300_final_evaluation_20260725

mkdir -p "${V04_ROOT}/logs/slurm" "${RACE_ROOT}/.local/cache/matplotlib" "${RACE_ROOT}/.local/cache/pycache"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite evaluation output: ${OUTPUT_DIR}" >&2
  exit 2
fi

export MPLCONFIGDIR=${RACE_ROOT}/.local/cache/matplotlib
export PYTHONPYCACHEPREFIX=${RACE_ROOT}/.local/cache/pycache
export PYTHONPATH=${V04_ROOT}/src
export PYTHONUNBUFFERED=1

/home/guest/anaconda3/bin/python -u -m double_helix.legacy.evaluate_training_runs \
  --run density0p3 0.3 "${FRESH_ROOT}/density0p3/run" \
  --run density0p5 0.5 "${RESUME_ROOT}/density0p5/run" \
  --run density1p0 1.0 "${RESUME_ROOT}/density1p0/run" \
  --prefix-run density0p5 "${OLD_ROOT}/density0p5/run" \
  --prefix-run density1p0 "${OLD_ROOT}/density1p0/run" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-epochs 300 \
  --steps-per-epoch 417 \
  --expected-gamma-updates 28
