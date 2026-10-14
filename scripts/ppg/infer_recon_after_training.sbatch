#!/usr/bin/env bash
#SBATCH --job-name=ppg_inf300
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ppg_infer300-%j.out
#SBATCH --error=logs/ppg_infer300-%j.err
set -euo pipefail

: "${TRAIN_JOB_ID:?set TRAIN_JOB_ID to the completed training job id}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1

RUN_DIR="$ROOT/output/ppg_unitypsf/ppg-dh-gmm-physical-300-lunar-bg10-${TRAIN_JOB_ID}"
CHECKPOINT="$RUN_DIR/checkpoints/double_helix.resume.ckpt"
SAMPLE_TIFF="$ROOT/datasets/training_sets/Zeiss_2/microtubule_2026_08_20__17_02_49_2x_binmean.tif"
INFER_DIR="$RUN_DIR/infer_recon"

test -f "$CHECKPOINT"
test -f "$SAMPLE_TIFF"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for BACKGROUND_PERCENTILE in 3 5 7; do
  /home/guest/anaconda3/bin/python -m ppg.infer_recon_sample \
    --checkpoint "$CHECKPOINT" \
    --sample-tiff "$SAMPLE_TIFF" \
    --output-dir "$INFER_DIR/darkest_${BACKGROUND_PERCENTILE}" \
    --device cuda:0 \
    --batch-size 1 \
    --chunk-frames 64 \
    --prob-threshold 0.3 \
    --render-prob-threshold 0.7 \
    --background-percentile "$BACKGROUND_PERCENTILE"
done
