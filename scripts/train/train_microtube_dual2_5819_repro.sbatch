#!/usr/bin/env bash
#SBATCH --job-name=mt_train_v04_from0
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=120:00:00
#SBATCH --output=/home/guest/Others/main/race/results/microtube/microtube_dual_2/v04_5819_repro_from0-train-%j.out
#SBATCH --error=/home/guest/Others/main/race/results/microtube/microtube_dual_2/v04_5819_repro_from0-train-%j.err

set -euo pipefail

ROOT=/home/guest/Others/main/race
V04="$ROOT/unity_v0.4"
cd "$V04"
TARGET="$ROOT/results/microtube/microtube_dual_2"
SOURCE_CONFIG="$TARGET/retry_5815_zmap/resolved_config.yaml"
RAW_TIFF="$ROOT/datasets/training_sets/microtubule_3d_800mw_7_1"
ZMAP_ROOT="$TARGET/zmap_microtube_raw_v04"
RUN_ROOT="$TARGET/v04_5819_repro_from0"
CONFIG="$RUN_ROOT/resolved_config.yaml"

test -f "$SOURCE_CONFIG"
test -r "$RAW_TIFF"
test -f "$ZMAP_ROOT/left/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz"
test -f "$ZMAP_ROOT/right/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz"
test ! -e "$RUN_ROOT"

export PYTHONPATH="$V04/src"
export MPLCONFIGDIR="$V04/.local/cache/matplotlib"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MPLCONFIGDIR" "$RUN_ROOT"

/home/guest/anaconda3/bin/python -u "$V04/scripts/train/prepare_microtube_dual2_5819_config.py" \
  --base-config "$SOURCE_CONFIG" \
  --raw-tiff "$RAW_TIFF" \
  --zmap-root "$ZMAP_ROOT" \
  --output "$CONFIG"

/home/guest/anaconda3/bin/python -u -m neptune_v03.training.run_high_fidelity \
  --config "$CONFIG" \
  --run-root "$RUN_ROOT" \
  --run-name run \
  --seed 20260711
