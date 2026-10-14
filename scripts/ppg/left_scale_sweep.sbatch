#!/usr/bin/env bash
#SBATCH --job-name=ppg_leftscale
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ppg_leftscale-%j.out
#SBATCH --error=logs/ppg_leftscale-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
TRAIN_JOB_ID="${TRAIN_JOB_ID:-5617}"
RUN_DIR="$ROOT/output/ppg_unitypsf/ppg-dh-gmm-physical-300-lunar-bg10-${TRAIN_JOB_ID}"
OUT="$ROOT/outputs/ppg_left_scale_sweep_20260922"

test -f "$RUN_DIR/checkpoints/double_helix.resume.ckpt"
test -f "$RUN_DIR/artifacts/ppg_physical_updates/epoch_300/sample_roi_bank.h5"
test -f "$ROOT/outputs/ppg_left_half_random_rois_20260922/left_half_random_rois_summary.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

/home/guest/anaconda3/bin/python -m ppg.diagnose_left_scale_sweep \
  --checkpoint "$RUN_DIR/checkpoints/double_helix.resume.ckpt" \
  --summary "$ROOT/outputs/ppg_left_half_random_rois_20260922/left_half_random_rois_summary.json" \
  --roi-bank "$RUN_DIR/artifacts/ppg_physical_updates/epoch_300/sample_roi_bank.h5" \
  --output-dir "$OUT" \
  --device cuda:0 \
  --prob-threshold 0.3 \
  --display-threshold 0.7 \
  --scales 1.0 1.25 1.5 2.0 3.0 4.0
