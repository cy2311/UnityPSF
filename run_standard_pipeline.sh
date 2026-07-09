#!/usr/bin/env bash
# Neptune v0.3 standard end-to-end pipeline submitter.
#
# Default route:
#   train with the 3371-style fresh high-quality zmap route
#     -> left channel infer/filter/recon
#     -> right channel infer/filter/recon
#     -> dual-channel union raw-ratio bicolor reconstruction
#
# Usage:
#   bash run_standard_pipeline.sh
#
# Infer from an existing training run:
#   PIPELINE_MODE=infer_only RUN_DIR=/path/to/run bash run_standard_pipeline.sh
#
# Main outputs:
#   output/<RUN_TAG>/<RUN_TAG>_<train_jobid>/
#   output/<RUN_TAG>_<train_jobid>_left_infer_recon_full8000_roi96_valid80_cut8_prob090_no_locprec/
#   output/<RUN_TAG>_<train_jobid>_right_infer_recon_full8000_roi96_valid80_cut8_prob090_no_locprec/
#   output/<RUN_TAG>_<train_jobid>_union_raw_ratio_bicolor_unfiltered_thr040_right_priority/

set -euo pipefail

ROOT="${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}"
NEPTUNE_DIR="$ROOT/neptune_v0.3"
PIPELINE_MODE="${PIPELINE_MODE:-train_infer}"
CHANNEL_MODE="${CHANNEL_MODE:-dual}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$NEPTUNE_DIR/train_standard_3367_hqzmap.sh}"
CHANNEL_INFER_SCRIPT="${CHANNEL_INFER_SCRIPT:-$NEPTUNE_DIR/scripts/infer/standard_channel_infer_recon.sbatch}"
DUAL_SCRIPT="${DUAL_SCRIPT:-$NEPTUNE_DIR/scripts/infer/run_3371_union_raw_ratio_bicolor.sbatch}"
SAMPLE_TIFF="${SAMPLE_TIFF:-${NEPTUNE_V03_RAW_TIFF_PATH:-$ROOT/neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}}"
PIPELINE_TAG="${PIPELINE_TAG:-standard_v03_$(date +%Y%m%d_%H%M%S)}"

infer_zmap_sample_kind_from_path() {
  local path_lc
  path_lc="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "$path_lc" == *"dynamin"* ]]; then
    printf 'dynamin'
  elif [[ "$path_lc" == *"membrane"* ]]; then
    printf 'membrane'
  elif [[ "$path_lc" == *"ncp"* ]]; then
    printf 'ncp'
  elif [[ "$path_lc" == *"paint"* || "$path_lc" == *"lh1"* ]]; then
    printf 'paint'
  elif [[ "$path_lc" == *"microtube"* || "$path_lc" == *"spool_800mw"* || "$path_lc" == *"3d_7_1"* ]]; then
    printf 'microtube'
  else
    printf ''
  fi
}

INFERRED_ZMAP_SAMPLE_KIND="$(infer_zmap_sample_kind_from_path "$SAMPLE_TIFF")"
if [[ -z "${ZMAP_SAMPLE_KIND+x}" || -z "${ZMAP_SAMPLE_KIND}" ]]; then
  ZMAP_SAMPLE_KIND="${INFERRED_ZMAP_SAMPLE_KIND:-microtube}"
fi
if [[ -n "$INFERRED_ZMAP_SAMPLE_KIND" && "$INFERRED_ZMAP_SAMPLE_KIND" != "$ZMAP_SAMPLE_KIND" ]]; then
  echo "SAMPLE_TIFF appears to be sample kind '$INFERRED_ZMAP_SAMPLE_KIND', but ZMAP_SAMPLE_KIND='$ZMAP_SAMPLE_KIND'." >&2
  echo "Refusing to submit because this would build an initial zmap with the wrong preset." >&2
  echo "sample_tiff=${SAMPLE_TIFF}" >&2
  exit 2
fi
case "$ZMAP_SAMPLE_KIND" in
  microtube|microtubule|paint|ncp|dynamin|membrane) ;;
  *)
    echo "Unsupported ZMAP_SAMPLE_KIND=$ZMAP_SAMPLE_KIND. Supported high-quality zmap presets are: microtube, paint, ncp, dynamin, membrane." >&2
    exit 2
    ;;
esac
if [[ "$ZMAP_SAMPLE_KIND" == "ncp" ]]; then
  LEFT_DOMAIN_CROP_LEFT="${LEFT_DOMAIN_CROP_LEFT:-100}"
  LEFT_DOMAIN_CROP_TOP="${LEFT_DOMAIN_CROP_TOP:-400}"
  LEFT_DOMAIN_CROP_WIDTH="${LEFT_DOMAIN_CROP_WIDTH:-400}"
  LEFT_DOMAIN_CROP_HEIGHT="${LEFT_DOMAIN_CROP_HEIGHT:-400}"
  RIGHT_DOMAIN_CROP_LEFT="${RIGHT_DOMAIN_CROP_LEFT:-700}"
  RIGHT_DOMAIN_CROP_TOP="${RIGHT_DOMAIN_CROP_TOP:-400}"
  RIGHT_DOMAIN_CROP_WIDTH="${RIGHT_DOMAIN_CROP_WIDTH:-400}"
  RIGHT_DOMAIN_CROP_HEIGHT="${RIGHT_DOMAIN_CROP_HEIGHT:-400}"
else
  LEFT_DOMAIN_CROP_LEFT="${LEFT_DOMAIN_CROP_LEFT:-0}"
  LEFT_DOMAIN_CROP_TOP="${LEFT_DOMAIN_CROP_TOP:-0}"
  LEFT_DOMAIN_CROP_WIDTH="${LEFT_DOMAIN_CROP_WIDTH:-600}"
  LEFT_DOMAIN_CROP_HEIGHT="${LEFT_DOMAIN_CROP_HEIGHT:-1200}"
  RIGHT_DOMAIN_CROP_LEFT="${RIGHT_DOMAIN_CROP_LEFT:-600}"
  RIGHT_DOMAIN_CROP_TOP="${RIGHT_DOMAIN_CROP_TOP:-0}"
  RIGHT_DOMAIN_CROP_WIDTH="${RIGHT_DOMAIN_CROP_WIDTH:-600}"
  RIGHT_DOMAIN_CROP_HEIGHT="${RIGHT_DOMAIN_CROP_HEIGHT:-1200}"
fi
RUN_TAG="${RUN_TAG:-standard_3371_${ZMAP_SAMPLE_KIND}_fast_route_roi96_psf25_fresh_hqzmap_emit500_round20_p1baseline_3052lr_start30_interval5_emit5000_epoch300_bs24_steps417_${PIPELINE_TAG}}"

mkdir -p "$NEPTUNE_DIR/logs/slurm" "$NEPTUNE_DIR/output"

if [[ "$PIPELINE_MODE" != "train_infer" && "$PIPELINE_MODE" != "infer_only" ]]; then
  echo "PIPELINE_MODE must be train_infer or infer_only, got: $PIPELINE_MODE" >&2
  exit 2
fi
if [[ "$CHANNEL_MODE" != "dual" && "$CHANNEL_MODE" != "left" && "$CHANNEL_MODE" != "right" ]]; then
  echo "CHANNEL_MODE must be dual, left, or right, got: $CHANNEL_MODE" >&2
  exit 2
fi
if [[ ! -s "$CHANNEL_INFER_SCRIPT" ]]; then
  echo "Missing channel infer script: $CHANNEL_INFER_SCRIPT" >&2
  exit 3
fi
if [[ ! -s "$DUAL_SCRIPT" ]]; then
  echo "Missing dual-channel script: $DUAL_SCRIPT" >&2
  exit 3
fi

submit_export() {
  local dependency="$1"
  local script="$2"
  shift 2
  local exports="ALL"
  local item
  for item in "$@"; do
    exports="${exports},${item}"
  done
  if [[ -n "$dependency" ]]; then
    sbatch --parsable --dependency="$dependency" --export="$exports" "$script"
  else
    sbatch --parsable --export="$exports" "$script"
  fi
}

train_job=""
if [[ "$PIPELINE_MODE" == "train_infer" ]]; then
  if [[ ! -s "$TRAIN_SCRIPT" ]]; then
    echo "Missing training script: $TRAIN_SCRIPT" >&2
    exit 3
  fi
  train_job="$(submit_export "" "$TRAIN_SCRIPT" \
    NEPTUNE_V03_ROOT="$ROOT" \
    NEPTUNE_V03_RAW_TIFF_PATH="$SAMPLE_TIFF" \
    ZMAP_SAMPLE_KIND="$ZMAP_SAMPLE_KIND" \
    EPOCHS="${EPOCHS:-300}" \
    BATCH_SIZE="${BATCH_SIZE:-24}" \
    STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-417}" \
    ROI_SIZE="${ROI_SIZE:-96}" \
    PSF_SIZE="${PSF_SIZE:-25}" \
    ROI_STRIDE="${ROI_STRIDE:-88}" \
    START_EPOCH="${START_EPOCH:-30}" \
    UPDATE_INTERVAL_EPOCHS="${UPDATE_INTERVAL_EPOCHS:-5}" \
    TARGET_PROJECTED_EMITTERS="${TARGET_PROJECTED_EMITTERS:-5000}" \
    HQ_MAX_EMITTERS="${HQ_MAX_EMITTERS:-500}" \
    HQ_ALTERNATING_ROUNDS="${HQ_ALTERNATING_ROUNDS:-20}" \
    HQ_SPATIAL_BALANCE_GRID_PX="${HQ_SPATIAL_BALANCE_GRID_PX:-}" \
    LEFT_DOMAIN_CROP_LEFT="$LEFT_DOMAIN_CROP_LEFT" \
    LEFT_DOMAIN_CROP_TOP="$LEFT_DOMAIN_CROP_TOP" \
    LEFT_DOMAIN_CROP_WIDTH="$LEFT_DOMAIN_CROP_WIDTH" \
    LEFT_DOMAIN_CROP_HEIGHT="$LEFT_DOMAIN_CROP_HEIGHT" \
    RIGHT_DOMAIN_CROP_LEFT="$RIGHT_DOMAIN_CROP_LEFT" \
    RIGHT_DOMAIN_CROP_TOP="$RIGHT_DOMAIN_CROP_TOP" \
    RIGHT_DOMAIN_CROP_WIDTH="$RIGHT_DOMAIN_CROP_WIDTH" \
    RIGHT_DOMAIN_CROP_HEIGHT="$RIGHT_DOMAIN_CROP_HEIGHT" \
    RUN_TAG="$RUN_TAG")"
  RUN_DIR="$NEPTUNE_DIR/output/$RUN_TAG/${RUN_TAG}_${train_job}"
  infer_dependency="afterok:${train_job}"
else
  if [[ -z "${RUN_DIR:-}" ]]; then
    echo "PIPELINE_MODE=infer_only requires RUN_DIR=/path/to/completed/training/run" >&2
    exit 2
  fi
  if [[ ! -s "$RUN_DIR/checkpoints/checkpoint_latest.pt" ]]; then
    echo "Missing checkpoint_latest.pt under RUN_DIR: $RUN_DIR" >&2
    exit 3
  fi
  RUN_BASENAME="$(basename "$RUN_DIR")"
  RUN_TAG="${RUN_TAG:-$RUN_BASENAME}"
  infer_dependency=""
fi

RUN_BASENAME="$(basename "$RUN_DIR")"
LEFT_RUN_NAME="${LEFT_RUN_NAME:-${RUN_BASENAME}_left_infer_recon_full8000_roi96_valid80_cut8_prob090_no_locprec}"
RIGHT_RUN_NAME="${RIGHT_RUN_NAME:-${RUN_BASENAME}_right_infer_recon_full8000_roi96_valid80_cut8_prob090_no_locprec}"
DUAL_RUN_NAME="${DUAL_RUN_NAME:-${RUN_BASENAME}_union_raw_ratio_bicolor_unfiltered_thr040_right_priority}"

LEFT_OUTPUT_DIR="${LEFT_OUTPUT_DIR:-$NEPTUNE_DIR/output/$LEFT_RUN_NAME}"
RIGHT_OUTPUT_DIR="${RIGHT_OUTPUT_DIR:-$NEPTUNE_DIR/output/$RIGHT_RUN_NAME}"
DUAL_OUTPUT_DIR="${DUAL_OUTPUT_DIR:-$NEPTUNE_DIR/output/$DUAL_RUN_NAME}"

left_job="skipped"
right_job="skipped"
dual_job="skipped"

submit_channel_infer() {
  local side="$1"
  local run_name="$2"
  local output_dir="$3"
  submit_export "$infer_dependency" "$CHANNEL_INFER_SCRIPT" \
    NEPTUNE_V03_ROOT="$ROOT" \
    SAMPLE_TIFF="$SAMPLE_TIFF" \
    RUN_DIR="$RUN_DIR" \
    SIDE="$side" \
    RUN_NAME="$run_name" \
    ROI_SIZE="${ROI_SIZE:-96}" \
    VALID_ROI_SIZE="${VALID_ROI_SIZE:-80}" \
    PROB_THRESHOLD="${PROB_THRESHOLD:-0.70}" \
    RAW_TH="${RAW_TH:-0.5}" \
    SPLIT_TH="${SPLIT_TH:-0.6}" \
    FILTER_PROB_MIN="${FILTER_PROB_MIN:-0.90}" \
    LEFT_DOMAIN_CROP_LEFT="$LEFT_DOMAIN_CROP_LEFT" \
    LEFT_DOMAIN_CROP_TOP="$LEFT_DOMAIN_CROP_TOP" \
    LEFT_DOMAIN_CROP_WIDTH="$LEFT_DOMAIN_CROP_WIDTH" \
    LEFT_DOMAIN_CROP_HEIGHT="$LEFT_DOMAIN_CROP_HEIGHT" \
    RIGHT_DOMAIN_CROP_LEFT="$RIGHT_DOMAIN_CROP_LEFT" \
    RIGHT_DOMAIN_CROP_TOP="$RIGHT_DOMAIN_CROP_TOP" \
    RIGHT_DOMAIN_CROP_WIDTH="$RIGHT_DOMAIN_CROP_WIDTH" \
    RIGHT_DOMAIN_CROP_HEIGHT="$RIGHT_DOMAIN_CROP_HEIGHT" \
    OUTPUT_DIR="$output_dir"
}

if [[ "$CHANNEL_MODE" == "dual" || "$CHANNEL_MODE" == "left" ]]; then
  left_job="$(submit_channel_infer left "$LEFT_RUN_NAME" "$LEFT_OUTPUT_DIR")"
fi

if [[ "$CHANNEL_MODE" == "dual" || "$CHANNEL_MODE" == "right" ]]; then
  right_job="$(submit_channel_infer right "$RIGHT_RUN_NAME" "$RIGHT_OUTPUT_DIR")"
fi

if [[ "$CHANNEL_MODE" == "dual" ]]; then
  dual_dependency="afterok:${left_job}:${right_job}"
  dual_job="$(submit_export "$dual_dependency" "$DUAL_SCRIPT" \
    NEPTUNE_V03_ROOT="$ROOT" \
    SAMPLE_TIFF="$SAMPLE_TIFF" \
    LEFT_PREDICTIONS="$LEFT_OUTPUT_DIR/left/infer/predictions_merged.h5" \
    RIGHT_PREDICTIONS="$RIGHT_OUTPUT_DIR/right/infer/predictions_merged.h5" \
    RUN_NAME="$DUAL_RUN_NAME" \
    RIGHT_CROP_LEFT="$RIGHT_DOMAIN_CROP_LEFT" \
    WIDTH_PX="$LEFT_DOMAIN_CROP_WIDTH" \
    HEIGHT_PX="$LEFT_DOMAIN_CROP_HEIGHT" \
    OUTPUT_DIR="$DUAL_OUTPUT_DIR")"
fi

cat <<EOF
submitted_neptune_v03_standard_pipeline
mode=$PIPELINE_MODE
channel_mode=$CHANNEL_MODE
train_job=${train_job:-skipped}
left_infer_job=$left_job
right_infer_job=$right_job
dual_channel_job=$dual_job
run_dir=$RUN_DIR
left_output=$LEFT_OUTPUT_DIR
right_output=$RIGHT_OUTPUT_DIR
dual_output=$DUAL_OUTPUT_DIR
sample_tiff=$SAMPLE_TIFF
EOF
