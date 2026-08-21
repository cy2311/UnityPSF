#!/usr/bin/env bash
# Formal UnityPSF dual-modality, dual-channel Expert Parallel training.

#SBATCH --job-name=unitypsf_2gpu_300
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_2gpu_300-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_2gpu_300-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/guest/Others/main/race/unity}"
PYTHON_BIN="${PYTHON:-/home/guest/anaconda3/bin/python}"
TORCHRUN_BIN="${TORCHRUN:-/home/guest/anaconda3/bin/torchrun}"
CONFIG="${UNITYPSF_JOINT_CONFIG:-$PROJECT_ROOT/configs/experiments/unitypsf_dual_modality_dual_channel_300epoch.yaml}"
RUN_ROOT="${UNITYPSF_RUN_ROOT:-$PROJECT_ROOT/output/unitypsf}"
RUN_ID="${UNITYPSF_RUN_ID:-dual-modality-dual-channel-300epoch-${SLURM_JOB_ID:-local}}"

cd "$PROJECT_ROOT"
mkdir -p \
  "$PROJECT_ROOT/.local/cache/matplotlib" \
  "$PROJECT_ROOT/.local/tmp" \
  "$PROJECT_ROOT/logs/slurm" \
  "$RUN_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src"
export MPLCONFIGDIR="$PROJECT_ROOT/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "config=${CONFIG}"
echo "run_root=${RUN_ROOT}"
echo "run_id=${RUN_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: formal training refuses CPU fallback"; count = torch.cuda.device_count(); assert count >= 2, f"expected two visible GPUs, got {count}"; print({"cuda_available": True, "device_count": count, "devices": [torch.cuda.get_device_name(i) for i in range(count)]})'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN_BIN" \
  --standalone \
  --nproc-per-node=2 \
  -m unity_psf.cli.train_joint_expert_parallel \
  --config "$CONFIG" \
  --run-root "$RUN_ROOT" \
  --run-id "$RUN_ID" \
  --coordination-timeout-seconds "${UNITYPSF_COORDINATION_TIMEOUT_SECONDS:-21600}"
