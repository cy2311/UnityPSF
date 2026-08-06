#!/usr/bin/env bash
#SBATCH --job-name=unitypsf_ep
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_ep-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_ep-%j.err

set -euo pipefail

PROJECT_ROOT="/home/guest/Others/main/race/unity"
PYTHON_BIN="${PYTHON:-/home/guest/anaconda3/bin/python}"
TORCHRUN_BIN="${TORCHRUN:-/home/guest/anaconda3/bin/torchrun}"
CONFIG="${UNITYPSF_JOINT_CONFIG:-$PROJECT_ROOT/configs/experiments/unitypsf_dual_modality_multichannel_smoke.yaml}"
RUN_ID="${UNITYPSF_RUN_ID:-dual-modality-ep-${SLURM_JOB_ID}}"
RUN_ROOT="${UNITYPSF_RUN_ROOT:-$PROJECT_ROOT/output/unitypsf}"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/.local/cache/matplotlib" "$PROJECT_ROOT/logs/slurm" "$RUN_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export MPLCONFIGDIR="$PROJECT_ROOT/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print({"cuda_available": True, "device_count": torch.cuda.device_count()})'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN_BIN" --standalone --nproc-per-node=3 \
  -m unity_psf.cli.train_joint_expert_parallel \
  --config "$CONFIG" \
  --run-root "$RUN_ROOT" \
  --run-id "$RUN_ID"
