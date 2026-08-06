#!/usr/bin/env bash
# Full 8000-frame 3371 infer -> filter/recon.
# Geometry: 96x96 input ROI, keep center 80x80 valid region, cut_edge=8, edgecover stride 80.

#SBATCH --job-name=nv04_3371_ifr
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/nv04_3371_ifr-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/nv04_3371_ifr-%j.err

set -euo pipefail

ROOT="${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}}}"
UNITY_DIR="$ROOT/unity"
PYTHON="${UNITY_V04_PYTHON:-${NEPTUNE_V04_PYTHON:-${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}}}"

mkdir -p "$UNITY_DIR/logs/slurm" "$UNITY_DIR/output"
cd "$ROOT"

export PYTHONPATH="$UNITY_DIR/src:$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="$UNITY_DIR/.local/cache/matplotlib"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-3371_full8000_infer_filter_recon_roi96_valid80_cut8}"
OUTPUT_DIR="${OUTPUT_DIR:-$UNITY_DIR/output/$RUN_NAME}"
mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint "$CHECKPOINT")
fi

"$PYTHON" "$UNITY_DIR/scripts/infer/run_3371_full8000_infer_filter_recon.py" \
  "${CHECKPOINT_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --device "${DEVICE:-cuda:0}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --frame-block "${FRAME_BLOCK:-128}" \
  --max-frames "${MAX_FRAMES:-8000}" \
  --roi-size 96 \
  --valid-roi-size 80 \
  --input-preprocess "${INPUT_PREPROCESS:-fd_deeploc_recenter}" \
  --filter-prob-min "${FILTER_PROB_MIN:-0.70}" \
  --render-pixel-nm "${RENDER_PIXEL_NM:-20.0}" \
  --spot-radius-nm "${SPOT_RADIUS_NM:-28.0}" \
  --radius-mode "${RADIUS_MODE:-fixed}" \
  --side "${SIDE:-both}" \
  --infer-amp
