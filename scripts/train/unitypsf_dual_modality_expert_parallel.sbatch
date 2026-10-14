#!/usr/bin/env bash
#SBATCH --job-name=unitypsf_ep
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/unitypsf_ep-%j.out
#SBATCH --error=logs/slurm/unitypsf_ep-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON:-/home/guest/anaconda3/bin/python}"
RUN_ID="${UNITYPSF_RUN_ID:-dual-modality-ep-${SLURM_JOB_ID:-local}}"
RUN_ROOT="${UNITYPSF_RUN_ROOT:-$PROJECT_ROOT/output/unitypsf}"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/.local/cache/matplotlib" "$PROJECT_ROOT/logs/slurm" "$RUN_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export MPLCONFIGDIR="$PROJECT_ROOT/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print({"cuda_available": True, "device_count": torch.cuda.device_count()})'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec "$PYTHON_BIN" -m unity_psf.cli.train_selection \
  run \
  --route emitter_2d:left,right \
  --route astigmatism:left,right \
  --epochs "${UNITYPSF_EPOCHS:-300}" \
  --steps-per-epoch "${UNITYPSF_STEPS_PER_EPOCH:-417}" \
  --require-cuda \
  --run-root "$RUN_ROOT" \
  --run-id "$RUN_ID"
