#!/usr/bin/env bash
#SBATCH --job-name=ppg_sigscale
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ppg_sigscale-%j.out
#SBATCH --error=logs/ppg_sigscale-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
TRAIN_JOB_ID="${TRAIN_JOB_ID:-5617}"
RUN_DIR="$ROOT/output/ppg_unitypsf/ppg-dh-gmm-physical-300-lunar-bg10-${TRAIN_JOB_ID}"
CHECKPOINT="$RUN_DIR/checkpoints/double_helix.resume.ckpt"
SAMPLE_TIFF="$ROOT/datasets/training_sets/Zeiss_2/microtubule_2026_08_20__17_02_49_2x_binmean.tif"
ROI_BANK="$RUN_DIR/artifacts/ppg_physical_updates/epoch_300/sample_roi_bank.h5"
OUTPUT_DIR="$ROOT/outputs/ppg_signal_scale_diagnostic_20260922"

test -f "$CHECKPOINT"
test -f "$SAMPLE_TIFF"
test -f "$ROI_BANK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

/home/guest/anaconda3/bin/python -m ppg.diagnose_signal_scale \
  --checkpoint "$CHECKPOINT" \
  --sample-tiff "$SAMPLE_TIFF" \
  --roi-bank "$ROI_BANK" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --prob-threshold 0.3 \
  --display-threshold 0.7 \
  --scales 0.05 0.10 0.15 0.25 0.50 1.00
