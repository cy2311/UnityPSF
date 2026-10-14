#!/usr/bin/env bash
#SBATCH --job-name=ppg_leftroi
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ppg_leftroi-%j.out
#SBATCH --error=logs/ppg_leftroi-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
TRAIN_JOB_ID="${TRAIN_JOB_ID:-5617}"
RUN_DIR="$ROOT/output/ppg_unitypsf/ppg-dh-gmm-physical-300-lunar-bg10-${TRAIN_JOB_ID}"

test -f "$RUN_DIR/checkpoints/double_helix.resume.ckpt"
test -f "$ROOT/datasets/training_sets/Zeiss_2/microtubule_2026_08_20__17_02_49_2x_binmean.tif"
test -f "$RUN_DIR/artifacts/ppg_physical_updates/epoch_300/sample_roi_bank.h5"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

/home/guest/anaconda3/bin/python -m ppg.diagnose_left_half_rois \
  --checkpoint "$RUN_DIR/checkpoints/double_helix.resume.ckpt" \
  --sample-tiff "$ROOT/datasets/training_sets/Zeiss_2/microtubule_2026_08_20__17_02_49_2x_binmean.tif" \
  --roi-bank "$RUN_DIR/artifacts/ppg_physical_updates/epoch_300/sample_roi_bank.h5" \
  --output-dir "$ROOT/outputs/ppg_left_half_random_rois_20260922" \
  --device cuda:0 \
  --seed 20260922 \
  --sample-count 8 \
  --prob-threshold 0.3 \
  --display-threshold 0.7 \
  --left-width-px 256 \
  --boundary-margin-px 24
