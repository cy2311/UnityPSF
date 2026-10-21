#!/usr/bin/env bash
#SBATCH --job-name=mt_v04_oneclick
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=120:00:00
#SBATCH --output=/home/guest/Others/main/race/unity_v0.4/results/microtube_dual2_v04_from0_cuda_gemm/logs/oneclick-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity_v0.4/results/microtube_dual2_v04_from0_cuda_gemm/logs/oneclick-%j.err

set -euo pipefail

ROOT=/home/guest/Others/main/race
V04="$ROOT/unity_v0.4"
OUTPUT_ROOT="$V04/results/microtube_dual2_v04_from0_cuda_gemm"
RUN_ROOT="$OUTPUT_ROOT/runs/${SLURM_JOB_ID:?Submit this script with sbatch}"
RAW_TIFF="${RAW_TIFF:-/share/home/guest/race/datasets/samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT/results/microtube/microtube_dual_2/retry_5815_zmap/resolved_config.yaml}"
ZMAP_ROOT="$RUN_ROOT/zmap_microtube_raw_v04"
TRAIN_ROOT="$RUN_ROOT/training"
CONFIG="$TRAIN_ROOT/resolved_config.yaml"

cd "$V04"
test -r "$RAW_TIFF"
test -r "$BASE_CONFIG"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT" "$V04/.local/cache/matplotlib"

export PYTHONPATH="$V04/src"
export MPLCONFIGDIR="$V04/.local/cache/matplotlib"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${PYTHON:-/home/guest/anaconda3/bin/python}"

echo "run_root=$RUN_ROOT"
echo "raw_tiff=$RAW_TIFF"
echo "stage=zmap_left"
"$PYTHON" -u "$V04/scripts/train/run_high_quality_initial_zmap.py" \
  --side left \
  --run-root "$ZMAP_ROOT/left" \
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

echo "stage=zmap_right"
"$PYTHON" -u "$V04/scripts/train/run_high_quality_initial_zmap.py" \
  --side right \
  --run-root "$ZMAP_ROOT/right" \
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

LEFT_MAP="$ZMAP_ROOT/left/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz"
RIGHT_MAP="$ZMAP_ROOT/right/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz"
test -s "$LEFT_MAP"
test -s "$RIGHT_MAP"

echo "stage=prepare_training_config"
mkdir -p "$TRAIN_ROOT"
"$PYTHON" -u "$V04/scripts/train/prepare_microtube_dual2_5819_config.py" \
  --base-config "$BASE_CONFIG" \
  --raw-tiff "$RAW_TIFF" \
  --zmap-root "$ZMAP_ROOT" \
  --output "$CONFIG"

echo "stage=train_from_epoch_0"
"$PYTHON" -u -m neptune_v03.training.run_high_fidelity \
  --config "$CONFIG" \
  --run-root "$TRAIN_ROOT" \
  --run-name run \
  --seed 20260711

printf 'completed\n' > "$RUN_ROOT/PIPELINE_STATUS"
echo "pipeline_status=completed"
echo "zmap_root=$ZMAP_ROOT"
echo "training_root=$TRAIN_ROOT"
