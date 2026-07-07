#!/usr/bin/env bash
# Standard Neptune v0.3 single-channel infer -> filter/recon.
#
# Required:
#   RUN_DIR=/path/to/training/run
#   SIDE=left|right
#
# Standard geometry:
#   ROI input: 96 x 96
#   valid core: 80 x 80
#   cut edge: 8 px
#   input preprocess: fd_deeploc_recenter
#   final filter/recon: prob >= 0.9, no locprec gate

#SBATCH --job-name=nv03_ch_infer
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/nv03_ch_infer-%j.out
#SBATCH --error=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/nv03_ch_infer-%j.err

set -euo pipefail

ROOT="${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}"
NEPTUNE_DIR="$ROOT/neptune_v0.3"
PYTHON="${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}"
INFER_RECON_ROOT="${INFER_RECON_ROOT:-$NEPTUNE_DIR/src/neptune_v03/infer_recon}"
SIDE="${SIDE:-left}"

if [[ "$SIDE" != "left" && "$SIDE" != "right" ]]; then
  echo "SIDE must be left or right, got: $SIDE" >&2
  exit 2
fi
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR is required and must point to a completed Neptune v0.3 training run." >&2
  exit 2
fi
if [[ ! -s "$RUN_DIR/checkpoints/checkpoint_latest.pt" ]]; then
  echo "Missing checkpoint_latest.pt under RUN_DIR: $RUN_DIR" >&2
  exit 3
fi
if [[ ! -s "$RUN_DIR/metadata/current_physical_state.json" ]]; then
  echo "Missing current physical state under RUN_DIR: $RUN_DIR" >&2
  exit 3
fi

mkdir -p "$NEPTUNE_DIR/logs/slurm" "$NEPTUNE_DIR/output"
cd "$ROOT"

export PYTHONPATH="$ROOT:$NEPTUNE_DIR/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$NEPTUNE_DIR/.local/cache/matplotlib"
export PYTHONUNBUFFERED=1

SAMPLE_TIFF="${SAMPLE_TIFF:-${NEPTUNE_V03_RAW_TIFF_PATH:-$ROOT/neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}}"
CONFIG="${CONFIG:-$NEPTUNE_DIR/configs/microtube_base.yaml}"
RUN_BASENAME="$(basename "$RUN_DIR")"
RUN_NAME="${RUN_NAME:-${RUN_BASENAME}_${SIDE}_infer_recon_full8000_roi96_valid80_cut8_prob090_no_locprec}"
OUTPUT_DIR="${OUTPUT_DIR:-$NEPTUNE_DIR/output/$RUN_NAME}"
mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

"$PYTHON" "$NEPTUNE_DIR/scripts/infer/run_3371_full8000_infer_filter_recon.py" \
  --run-dir "$RUN_DIR" \
  --config "$CONFIG" \
  --checkpoint "${CHECKPOINT:-$RUN_DIR/checkpoints/checkpoint_latest.pt}" \
  --sample-tiff "$SAMPLE_TIFF" \
  --output-dir "$OUTPUT_DIR" \
  --infer-recon-root "$INFER_RECON_ROOT" \
  --device "${DEVICE:-cuda:0}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --frame-block "${FRAME_BLOCK:-128}" \
  --max-frames "${MAX_FRAMES:-8000}" \
  --roi-size "${ROI_SIZE:-96}" \
  --valid-roi-size "${VALID_ROI_SIZE:-80}" \
  --input-preprocess "${INPUT_PREPROCESS:-fd_deeploc_recenter}" \
  --prob-threshold "${PROB_THRESHOLD:-0.70}" \
  --raw-th "${RAW_TH:-0.5}" \
  --split-th "${SPLIT_TH:-0.6}" \
  --filter-prob-min "${FILTER_PROB_MIN:-0.90}" \
  --side "$SIDE" \
  --infer-amp
