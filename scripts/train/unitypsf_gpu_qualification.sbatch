#!/usr/bin/env bash
#SBATCH --job-name=unitypsf_gpu_qualification
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/unitypsf_gpu_qualification-%j.out
#SBATCH --error=logs/slurm/unitypsf_gpu_qualification-%j.err
set -euo pipefail
ROOT="${UNITYPSF_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda-11.7
export PATH="$CUDA_HOME/bin:$PATH"
export MPLCONFIGDIR="$ROOT/.local/cache/matplotlib"
export NEPTUNE_TORCH_EXTENSIONS_DIR="$ROOT/.local/cache/torch_extensions"
exec /home/guest/anaconda3/bin/python scripts/train/qualify_gpu_baseline.py \
  --run-root "$ROOT/output/unitypsf" \
  --run-id "${UNITYPSF_QUALIFICATION_RUN_ID:-gpu-qualification-${SLURM_JOB_ID}}" \
  --epochs "${UNITYPSF_QUALIFICATION_EPOCHS:-3}" \
  --steps-per-epoch "${UNITYPSF_QUALIFICATION_STEPS_PER_EPOCH:-157}" \
  --batch-size "${UNITYPSF_QUALIFICATION_BATCH_SIZE:-64}" \
  --input-size "${UNITYPSF_QUALIFICATION_INPUT_SIZE:-64}" \
  --astigmatism-map "astigmatism:left=${UNITYPSF_ASTIGMATISM_MAP_LEFT:?}" \
  --astigmatism-map "astigmatism:right=${UNITYPSF_ASTIGMATISM_MAP_RIGHT:?}"
