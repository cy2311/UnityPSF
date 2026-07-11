#!/usr/bin/env bash
# Reprocess an existing localization H5 with the current Neptune v0.3
# degrid -> RCC -> filter -> reconstruction contract.

#SBATCH --job-name=nv03_reprocess
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/nv03_reprocess-%j.out
#SBATCH --error=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/nv03_reprocess-%j.err

set -euo pipefail

ROOT="${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}"
NEPTUNE_DIR="$ROOT/neptune_v0.3"
PYTHON="${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}"

: "${SOURCE_PREDICTIONS:?SOURCE_PREDICTIONS is required}"
: "${RUNTIME_STATE:?RUNTIME_STATE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${SIDE:?SIDE is required}"

WIDTH_PX="${WIDTH_PX:-600}"
HEIGHT_PX="${HEIGHT_PX:-1200}"
PIXEL_NM_X="${PIXEL_NM_X:-101.11}"
PIXEL_NM_Y="${PIXEL_NM_Y:-98.83}"
INFER_DIR="$OUTPUT_DIR/infer"
FILTER_DIR="$OUTPUT_DIR/$SIDE/filter_recon_prob070_current_standard"

mkdir -p "$INFER_DIR" "$FILTER_DIR" "$NEPTUNE_DIR/logs/slurm" "$NEPTUNE_DIR/.local/cache/matplotlib"
export PYTHONPATH="$NEPTUNE_DIR:$NEPTUNE_DIR/src:$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="$NEPTUNE_DIR/.local/cache/matplotlib"
export PYTHONUNBUFFERED=1

"$PYTHON" "$NEPTUNE_DIR/src/neptune_v03/infer_recon/degrid.py" \
  --predictions "$SOURCE_PREDICTIONS" \
  --output "$INFER_DIR/predictions_degrid.h5" \
  --summary-json "$INFER_DIR/degrid_summary.json" \
  --histogram-png "$INFER_DIR/degrid_offset_histograms.png" \
  --pixel-size-nm-x "$PIXEL_NM_X" \
  --pixel-size-nm-y "$PIXEL_NM_Y" \
  --rescale-bins "${DEGRID_RESCALE_BINS:-20}" \
  --threshold "${DEGRID_THRESHOLD:-0.01}" \
  --min-bin-count "${DEGRID_MIN_BIN_COUNT:-32}" \
  --spatial-bins-x "${DEGRID_SPATIAL_BINS_X:-6}" \
  --spatial-bins-y "${DEGRID_SPATIAL_BINS_Y:-12}" \
  --field-width-px "$WIDTH_PX" \
  --field-height-px "$HEIGHT_PX"

"$PYTHON" "$NEPTUNE_DIR/scripts/analysis/run_rcc_drift_diagnostic.py" \
  --predictions "$INFER_DIR/predictions_degrid.h5" \
  --output-dir "$INFER_DIR" \
  --frame-block-size "${RCC_FRAME_BLOCK_SIZE:-500}" \
  --camera-pixel-nm-x "$PIXEL_NM_X" \
  --camera-pixel-nm-y "$PIXEL_NM_Y" \
  --width-px "$WIDTH_PX" \
  --height-px "$HEIGHT_PX" \
  --rcc-pixel-nm "${RCC_PIXEL_NM:-50.0}" \
  --rcc-sigma-px "${RCC_SIGMA_PX:-1.0}" \
  --upsample-factor "${RCC_UPSAMPLE_FACTOR:-10}" \
  --max-pair-gap "${RCC_MAX_PAIR_GAP:-8}" \
  --max-shift-nm "${RCC_MAX_SHIFT_NM:-1000.0}" \
  --min-pair-correlation "${RCC_MIN_PAIR_CORRELATION:-0.02}"

"$PYTHON" "$NEPTUNE_DIR/src/neptune_v03/infer_recon/filter/apply_filter_recon.py" \
  --infer-dir "$INFER_DIR" \
  --runtime-state "$RUNTIME_STATE" \
  --predictions "$INFER_DIR/predictions_degrid_rcc_corrected.h5" \
  --output-dir "$FILTER_DIR" \
  --width-px "$WIDTH_PX" \
  --height-px "$HEIGHT_PX" \
  --filter-profile basic \
  --prob-min "${FILTER_PROB_MIN:-0.70}" \
  --render-pixel-nm "${RENDER_PIXEL_NM:-20.0}" \
  --spot-radius-nm "${SPOT_RADIUS_NM:-28.0}" \
  --renderer integrated_gaussian \
  --render-weight count \
  --gamma 1.0 \
  --display-mode quantile \
  --display-imax-min "${DISPLAY_IMAX_MIN:--2.5228787452803374}" \
  --radius-mode fixed \
  --filtered-format h5 \
  --keep-filtered-predictions \
  --z-min -600 \
  --z-max 600 \
  --suffix current_standard_prob070_degrid_rcc

printf 'current_standard_reprocess_complete\noutput_dir=%s\nside=%s\n' "$OUTPUT_DIR" "$SIDE"
