#!/usr/bin/env bash
#SBATCH --job-name=ppg_eval_png
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/ppg_eval_png-%j.out
#SBATCH --error=logs/ppg_eval_png-%j.err
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
TRAIN_JOB_ID="${TRAIN_JOB_ID:-5617}"
RUN_DIR="$ROOT/output/ppg_unitypsf/ppg-dh-gmm-physical-300-lunar-bg10-${TRAIN_JOB_ID}"
/home/guest/anaconda3/bin/python -m ppg.render_training_detection \
  --config "$ROOT/configs/modalities/double_helix/ppg_physical.yaml" \
  --checkpoint "$RUN_DIR/checkpoints/double_helix.resume.ckpt" \
  --output "$RUN_DIR/training_eval_raw_plus_prediction_points.png" \
  --device cuda:0
