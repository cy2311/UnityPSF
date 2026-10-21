#!/usr/bin/env bash
#SBATCH --job-name=mt_zmap_v04
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/guest/Others/main/race/results/microtube/microtube_dual_2/zmap-rebuild-v04-%j.out
#SBATCH --error=/home/guest/Others/main/race/results/microtube/microtube_dual_2/zmap-rebuild-v04-%j.err

set -euo pipefail

ROOT=/home/guest/Others/main/race
V04="$ROOT/unity_v0.4"
TARGET="$ROOT/results/microtube/microtube_dual_2"
RAW_TIFF="$ROOT/datasets/training_sets/microtubule_3d_800mw_7_1"
ZMAP_ROOT="$TARGET/zmap_microtube_raw_v04"

test -r "$RAW_TIFF"
test ! -e "$ZMAP_ROOT/left"
test ! -e "$ZMAP_ROOT/right"

export PYTHONPATH="$V04/src"
export MPLCONFIGDIR="$V04/.local/cache/matplotlib"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$ZMAP_ROOT"

run_side() {
  local side="$1"
  /home/guest/anaconda3/bin/python -u "$V04/scripts/train/run_high_quality_initial_zmap.py" \
    --side "$side" \
    --run-root "$ZMAP_ROOT/$side" \
    --repo-root "$ROOT" \
    --raw-tiff "$RAW_TIFF" \
    --zmap-sample microtube \
    --device cuda:0 \
    --max-emitters 500 \
    --alternating-rounds 20 \
    --alternating-local-steps 100 \
    --alternating-global-steps 100 \
    --selection-pool-multiplier 50 \
    --baseline-percentile 1.0 \
    --baseline-frame-start 0 \
    --baseline-frame-stop 100
}

run_side left
run_side right

echo "v0.4 microtube dual Z-map rebuild complete"
