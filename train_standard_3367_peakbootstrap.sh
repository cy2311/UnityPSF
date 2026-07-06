#!/usr/bin/env bash
# Neptune v0.3 run-3367 fast route with peak-bootstrap initial zmap.
#
# Submit with:
#   sbatch /home/guest/Others/main/race/neptune_v0.3/train_standard_3367_peakbootstrap.sh

#SBATCH --job-name=nv03_3367pb
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/nv03_3367pb-%j.out
#SBATCH --error=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/nv03_3367pb-%j.err

set -euo pipefail

ROOT="${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}"
NEPTUNE_DIR="$ROOT/neptune_v0.3"

export NEPTUNE_V03_ROOT="$ROOT"
export NEPTUNE_V03_PYTHON="${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}"
export NEPTUNE_V03_RAW_TIFF_PATH="${NEPTUNE_V03_RAW_TIFF_PATH:-$ROOT/neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}"

export NAT_CONFIG_KIND="${NAT_CONFIG_KIND:-order1_13}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPEND_DOMAIN_ONEHOT="${APPEND_DOMAIN_ONEHOT:-1}"
export DOMAIN_BALANCE_MODE="${DOMAIN_BALANCE_MODE:-alternate_step}"
export NEPTUNE_V03_PROJECTION_BACKEND="${NEPTUNE_V03_PROJECTION_BACKEND:-triton_fused}"
export NEPTUNE_V03_CACHED_WINDOW_PRECOMPUTE="${NEPTUNE_V03_CACHED_WINDOW_PRECOMPUTE:-1}"
export NEPTUNE_V03_LUT_EPOCH_PREWARM="${NEPTUNE_V03_LUT_EPOCH_PREWARM:-1}"
export NEPTUNE_V03_LUT_SHIFT_BACKEND="${NEPTUNE_V03_LUT_SHIFT_BACKEND:-fourier}"
export NEPTUNE_V03_PROFILE_TIMING="${NEPTUNE_V03_PROFILE_TIMING:-1}"
export NEPTUNE_V03_PROFILE_SYNC_CUDA="${NEPTUNE_V03_PROFILE_SYNC_CUDA:-1}"

export BATCH_SIZE=24
export STEPS_PER_EPOCH=417
export EPOCHS=300
export ROI_SIZE=96
export PSF_SIZE=25
export ROI_STRIDE=88
export START_EPOCH=30
export UPDATE_INTERVAL_EPOCHS=5
export TARGET_PROJECTED_EMITTERS=5000

export LR_STEP_UNIT=epoch
export LR_STEP_SIZE=10
export LR_GAMMA=0.9

export SEQUENCE_COUNT=417
export CACHED_WINDOW_ORDER=auto
export CACHED_WINDOW_MAX_GPU_SEQUENCES=2
export FIELD_ORIGIN_SAMPLING_MODE=sliding_window
export NAT_GRID_SIZE_X=2
export NAT_GRID_SIZE_Y=2

export BOOTSTRAP_MAX_EMITTERS=500
export BOOTSTRAP_ALTERNATING_ROUNDS=30
export BOOTSTRAP_ALTERNATING_LOCAL_STEPS=100
export BOOTSTRAP_ALTERNATING_GLOBAL_STEPS=100
export BOOTSTRAP_NCC_THRESHOLD=0.8
export BOOTSTRAP_MIN_DISTANCE_PX=15.0
export BOOTSTRAP_MAX_PATCH_PEAK_DISTANCE_PX=1.2
export BOOTSTRAP_MAX_SECONDARY_PEAK_FRACTION=0.25
export BOOTSTRAP_SPATIAL_BALANCE_GRID_PX=100
export BOOTSTRAP_FORMAL_EXPORT_STAGE=alternating

export RUN_TAG="fast_route_roi96_psf25_peakbootstrap_3052depth_qc_emit500_3052lr_start30_interval5_emit5000_epoch300_bs24_steps417"

exec bash "$NEPTUNE_DIR/scripts/train/fast_route_3367_peakbootstrap.sbatch"
