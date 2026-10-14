#!/usr/bin/env bash
# Formal UnityPSF dual-modality, mixed-channel Expert Parallel training.

#SBATCH --job-name=unitypsf_2gpu_300
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm/unitypsf_2gpu_300-%j.out
#SBATCH --error=logs/slurm/unitypsf_2gpu_300-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON:-/home/guest/anaconda3/bin/python}"
RUN_ROOT="${UNITYPSF_RUN_ROOT:-$PROJECT_ROOT/output/unitypsf}"
RUN_ID="${UNITYPSF_RUN_ID:-dual-modality-mixed-channel-300epoch-${SLURM_JOB_ID:-local}}"

cd "$PROJECT_ROOT"
mkdir -p \
  "$PROJECT_ROOT/.local/cache/matplotlib" \
  "$PROJECT_ROOT/.local/tmp" \
  "$PROJECT_ROOT/logs/slurm" \
  "$RUN_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src"
export MPLCONFIGDIR="$PROJECT_ROOT/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1

echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "run_root=${RUN_ROOT}"
echo "run_id=${RUN_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: formal training refuses CPU fallback"; count = torch.cuda.device_count(); assert count >= 2, f"expected two visible GPUs, got {count}"; print({"cuda_available": True, "device_count": count, "devices": [torch.cuda.get_device_name(i) for i in range(count)]})'
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
