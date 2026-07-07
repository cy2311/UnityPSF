#!/usr/bin/env bash
# Standard v0.3 multicolor reconstruction for 3371-style dual-channel outputs.
# Method: union left/right emitter sets, measure raw TIFF left/right intensity,
# threshold ratio_right at 0.4, and keep right-channel coordinates/precision for duplicates.

#SBATCH --job-name=nv03_union_ratio
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/nv03_union_ratio-%j.out
#SBATCH --error=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/nv03_union_ratio-%j.err

set -euo pipefail

ROOT="${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}"
NEPTUNE_DIR="$ROOT/neptune_v0.3"
PYTHON="${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}"

LEFT_PREDICTIONS="${LEFT_PREDICTIONS:-$NEPTUNE_DIR/output/3371_left_root_infer_recon_latestckpt_recenter_full8000_roi96_valid80_cut8_p070/left/infer/predictions_merged.h5}"
RIGHT_PREDICTIONS="${RIGHT_PREDICTIONS:-$NEPTUNE_DIR/output/3371_right_infer_recon_latestckpt_recenter_full8000_roi96_valid80_cut8_prob090_no_locprec/right/infer/predictions_merged.h5}"
SAMPLE_TIFF="${SAMPLE_TIFF:-$ROOT/neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}"
RUN_NAME="${RUN_NAME:-3371_union_raw_ratio_bicolor_unfiltered_thr040_right_priority}"
OUTPUT_DIR="${OUTPUT_DIR:-$NEPTUNE_DIR/output/$RUN_NAME}"

mkdir -p "$NEPTUNE_DIR/logs/slurm" "$OUTPUT_DIR"
cd "$ROOT"

export PYTHONPATH="$NEPTUNE_DIR/src:$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="$NEPTUNE_DIR/.local/cache/matplotlib"
export PYTHONUNBUFFERED=1

"$PYTHON" "$NEPTUNE_DIR/scripts/infer/render_union_raw_ratio_bicolor.py" \
  --left-predictions "$LEFT_PREDICTIONS" \
  --right-predictions "$RIGHT_PREDICTIONS" \
  --sample-tiff "$SAMPLE_TIFF" \
  --output-dir "$OUTPUT_DIR" \
  --ratio-threshold "${RATIO_THRESHOLD:-0.4}" \
  --union-dist-px "${UNION_DIST_PX:-2.0}" \
  --min-total-intensity "${MIN_TOTAL_INTENSITY:-0.0}" \
  --right-crop-left "${RIGHT_CROP_LEFT:-600}" \
  --device "${DEVICE:-cuda:0}" \
  --suffix "${SUFFIX:-union_raw_ratio_unfiltered_thr040_right_priority}"
