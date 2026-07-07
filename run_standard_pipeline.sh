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
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$NEPTUNE_DIR/train_standard_3367_hqzmap.sh}"
CHANNEL_INFER_SCRIPT="${CHANNEL_INFER_SCRIPT:-$NEPTUNE_DIR/scripts/infer/standard_channel_infer_recon.sbatch}"
DUAL_SCRIPT="${DUAL_SCRIPT:-$NEPTUNE_DIR/scripts/infer/run_3371_union_raw_ratio_bicolor.sbatch}"
SAMPLE_TIFF="${SAMPLE_TIFF:-${NEPTUNE_V03_RAW_TIFF_PATH:-$ROOT/neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}}"
PIPELINE_TAG="${PIPELINE_TAG:-standard_v03_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-standard_3371_fast_route_roi96_psf25_fresh_hqzmap_emit500_round20_p1baseline_3052lr_start30_interval5_emit5000_epoch300_bs24_steps417_${PIPELINE_TAG}}"

mkdir -p "$NEPTUNE_DIR/logs/slurm" "$NEPTUNE_DIR/output"

if [[ "$PIPELINE_MODE" != "train_infer" && "$PIPELINE_MODE" != "infer_only" ]]; then
  echo "PIPELINE_MODE must be train_infer or infer_only, got: $PIPELINE_MODE" >&2
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

left_job="$(submit_export "$infer_dependency" "$CHANNEL_INFER_SCRIPT" \
  NEPTUNE_V03_ROOT="$ROOT" \
  SAMPLE_TIFF="$SAMPLE_TIFF" \
  RUN_DIR="$RUN_DIR" \
  SIDE=left \
  RUN_NAME="$LEFT_RUN_NAME" \
  OUTPUT_DIR="$LEFT_OUTPUT_DIR")"

right_job="$(submit_export "$infer_dependency" "$CHANNEL_INFER_SCRIPT" \
  NEPTUNE_V03_ROOT="$ROOT" \
  SAMPLE_TIFF="$SAMPLE_TIFF" \
  RUN_DIR="$RUN_DIR" \
  SIDE=right \
  RUN_NAME="$RIGHT_RUN_NAME" \
  OUTPUT_DIR="$RIGHT_OUTPUT_DIR")"

dual_dependency="afterok:${left_job}:${right_job}"
dual_job="$(submit_export "$dual_dependency" "$DUAL_SCRIPT" \
  NEPTUNE_V03_ROOT="$ROOT" \
  SAMPLE_TIFF="$SAMPLE_TIFF" \
  LEFT_PREDICTIONS="$LEFT_OUTPUT_DIR/left/infer/predictions_merged.h5" \
  RIGHT_PREDICTIONS="$RIGHT_OUTPUT_DIR/right/infer/predictions_merged.h5" \
  RUN_NAME="$DUAL_RUN_NAME" \
  OUTPUT_DIR="$DUAL_OUTPUT_DIR")"

cat <<EOF
submitted_neptune_v03_standard_pipeline
mode=$PIPELINE_MODE
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
