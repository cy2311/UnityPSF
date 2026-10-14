#!/usr/bin/env bash
# UnityPSF mixed-channel, two-modality CUDA gate before formal training.

#SBATCH --job-name=unitypsf_2gpu_smoke
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/unitypsf_2gpu_smoke-%j.out
#SBATCH --error=logs/slurm/unitypsf_2gpu_smoke-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ROOT="${UNITYPSF_RUN_ROOT:-$PROJECT_ROOT/output/unitypsf}"
RUN_ID="${UNITYPSF_RUN_ID:-dual-modality-mixed-channel-gpu-smoke-${SLURM_JOB_ID:-local}}"
PYTHON_BIN="${PYTHON:-/home/guest/anaconda3/bin/python}"

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
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-scheduler-managed}"

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: smoke refuses CPU fallback"; count = torch.cuda.device_count(); assert count >= 2, f"expected two visible GPUs, got {count}"; print({"cuda_available": True, "device_count": count, "devices": [torch.cuda.get_device_name(i) for i in range(count)]})'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec "$PYTHON_BIN" -m unity_psf.cli.train_selection \
  smoke \
  --route emitter_2d:left,right \
  --route astigmatism:left,right \
  --epochs 1 \
  --steps-per-epoch 8 \
  --batch-size 1 \
  --require-cuda \
  --run-root "$RUN_ROOT" \
  --run-id "$RUN_ID"
