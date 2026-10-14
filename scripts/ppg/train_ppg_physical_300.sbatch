#!/usr/bin/env bash
#SBATCH --job-name=ppg_phys300
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/ppg_phys300-%j.out
#SBATCH --error=logs/ppg_phys300-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-ppg-dh-gmm-physical-300-lunar-bg10-${SLURM_JOB_ID}}"
RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
DENSITY_ARGS=()
if [[ -n "${DENSITY:-}" ]]; then
  DENSITY_ARGS+=(--density "$DENSITY")
fi
EPOCH_ARGS=()
if [[ -n "${EPOCHS:-}" ]]; then
  EPOCH_ARGS+=(--epochs "$EPOCHS")
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
/home/guest/anaconda3/bin/python -m ppg.train_dh_physical \
  --config "$ROOT/configs/modalities/double_helix/ppg_physical.yaml" \
  --run-root "$ROOT/output/ppg_unitypsf" \
  --run-id "$RUN_ID" \
  --device cuda:0 \
  "${DENSITY_ARGS[@]}" \
  "${EPOCH_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
