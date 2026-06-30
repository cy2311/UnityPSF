#!/usr/bin/env bash
#SBATCH --job-name=nv03_std_rgb
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=neptune_v0.3/logs/standard_roi_gamma_batch_budget_%j.out
#SBATCH --error=neptune_v0.3/logs/standard_roi_gamma_batch_budget_%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "${PROJECT_ROOT}"
export PYTHONPATH=src
export MPLCONFIGDIR=.local/cache/matplotlib
export APPEND_DOMAIN_ONEHOT=1
export DOMAIN_BALANCE_MODE=alternate_step

RUN_NAME="standard_roi_gamma_batch_budget_start2000_lr025_steps100_batches10000_${SLURM_JOB_ID:-local}"
RUN_ROOT="output"
RESOLVED_CONFIG=".local/tmp/standard/resolved_standard_roi_gamma_batch_budget_${SLURM_JOB_ID:-local}.yaml"

mkdir -p .local/cache/matplotlib .local/tmp/standard logs "${RUN_ROOT}/${RUN_NAME}"

"${PYTHON_BIN}" -m neptune_v03.config.materialize \
  --base configs/microtube_base.yaml \
  --override configs/overrides/standard_roi_gamma.yaml \
  --override configs/overrides/standard_roi_gamma_batch_budget.yaml \
  --output "${RESOLVED_CONFIG}"

"${PYTHON_BIN}" -u -m neptune_v03.training.run_high_fidelity \
  --config "${RESOLVED_CONFIG}" \
  --run-root "${RUN_ROOT}" \
  --run-name "${RUN_NAME}" \
  --seed 0
