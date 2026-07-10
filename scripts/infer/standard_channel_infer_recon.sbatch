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
#   quality stage: disabled by default; enable QUALITY_METRICS=true only for diagnostics
#   final filter/recon: prob >= 0.7, SMAP-style fixed sigma 14 nm, no locprec/LLrel/PSF-size hard gate
#   formal decode: fixed LiteLoc NMS contract, integrated probability > 0.7

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
RUN_NAME="${RUN_NAME:-${RUN_BASENAME}_${SIDE}_infer_recon_full8000_roi96_valid80_cut8_prob070_smap_fixed14nm}"
OUTPUT_DIR="${OUTPUT_DIR:-$NEPTUNE_DIR/output/$RUN_NAME}"
mkdir -p "$MPLCONFIGDIR"

EXTRA_ARGS=()
if [[ -n "${LOCPREC_XY_NM_MAX:-}" ]]; then
  EXTRA_ARGS+=(--locprec-xy-nm-max "$LOCPREC_XY_NM_MAX")
fi
if [[ -n "${X_SIG_PX_MAX:-}" ]]; then
  EXTRA_ARGS+=(--x-sig-px-max "$X_SIG_PX_MAX")
fi
if [[ -n "${Y_SIG_PX_MAX:-}" ]]; then
  EXTRA_ARGS+=(--y-sig-px-max "$Y_SIG_PX_MAX")
fi
if [[ -n "${LLREL_MIN:-}" ]]; then
  EXTRA_ARGS+=(--llrel-min "$LLREL_MIN")
fi
if [[ -n "${PSF_XY_NM_MAX:-}" ]]; then
  EXTRA_ARGS+=(--psf-xy-nm-max "$PSF_XY_NM_MAX")
fi
if [[ -n "${REQUIRE_FIT_STATUS:-}" && "${REQUIRE_FIT_STATUS:-0}" != "0" && "${REQUIRE_FIT_STATUS,,}" != "false" ]]; then
  EXTRA_ARGS+=(--require-fit-status)
fi
if [[ -n "${ALLOW_OVERWRITE_OUTPUT:-}" && "${ALLOW_OVERWRITE_OUTPUT:-0}" != "0" && "${ALLOW_OVERWRITE_OUTPUT,,}" != "false" ]]; then
  EXTRA_ARGS+=(--overwrite-output)
fi
if [[ -n "${DISPLAY_IMAX:-}" ]]; then
  EXTRA_ARGS+=(--display-imax "$DISPLAY_IMAX")
fi
if [[ -n "${NORMALIZATION_FOV:-}" ]]; then
  EXTRA_ARGS+=(--normalization-fov "$NORMALIZATION_FOV")
fi

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
  --filter-prob-min "${FILTER_PROB_MIN:-0.70}" \
  --filter-prob-sweep "${FILTER_PROB_SWEEP:-}" \
  --render-pixel-nm "${RENDER_PIXEL_NM:-20.0}" \
  --spot-radius-nm "${SPOT_RADIUS_NM:-28.0}" \
  --radius-mode "${RADIUS_MODE:-fixed}" \
  --display-mode "${DISPLAY_MODE:-quantile}" \
  --display-imax-min "${DISPLAY_IMAX_MIN:--2.5228787452803374}" \
  --quality-metrics="${QUALITY_METRICS:-false}" \
  --quality-metric-mode "${QUALITY_METRIC_MODE:-moment}" \
  --quality-roi-radius-px "${QUALITY_ROI_RADIUS_PX:-3}" \
  --degrid="${DEGRID_ENABLED:-true}" \
  --degrid-rescale-bins "${DEGRID_RESCALE_BINS:-20}" \
  --degrid-threshold "${DEGRID_THRESHOLD:-0.01}" \
  --degrid-min-bin-count "${DEGRID_MIN_BIN_COUNT:-32}" \
  --rcc-drift="${RCC_DRIFT_ENABLED:-true}" \
  --rcc-frame-block-size "${RCC_FRAME_BLOCK_SIZE:-500}" \
  --rcc-pixel-nm "${RCC_PIXEL_NM:-50.0}" \
  --rcc-sigma-px "${RCC_SIGMA_PX:-1.0}" \
  --rcc-upsample-factor "${RCC_UPSAMPLE_FACTOR:-10}" \
  --rcc-max-pair-gap "${RCC_MAX_PAIR_GAP:-8}" \
  --rcc-max-shift-nm "${RCC_MAX_SHIFT_NM:-1000.0}" \
  --rcc-min-pair-correlation "${RCC_MIN_PAIR_CORRELATION:-0.02}" \
  --side "$SIDE" \
  --left-crop-left "${LEFT_DOMAIN_CROP_LEFT:-0}" \
  --left-crop-top "${LEFT_DOMAIN_CROP_TOP:-0}" \
  --left-crop-width "${LEFT_DOMAIN_CROP_WIDTH:-600}" \
  --left-crop-height "${LEFT_DOMAIN_CROP_HEIGHT:-1200}" \
  --right-crop-left "${RIGHT_DOMAIN_CROP_LEFT:-600}" \
  --right-crop-top "${RIGHT_DOMAIN_CROP_TOP:-0}" \
  --right-crop-width "${RIGHT_DOMAIN_CROP_WIDTH:-600}" \
  --right-crop-height "${RIGHT_DOMAIN_CROP_HEIGHT:-1200}" \
  "${EXTRA_ARGS[@]}" \
  --infer-amp
