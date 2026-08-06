#!/usr/bin/env bash
# UnityPSF v0.4 default run-3371 route: fast training with freshly recomputed raw-TIFF high-quality initial zmap.
#
# Submit with:
#   sbatch /home/guest/Others/main/race/unity/scripts/archive/neptune_standard/train_standard_3367_hqzmap.sh
#
# Stage 1:
#   recompute left/right initial zmap in parallel on one GPU from raw TIFF with zmap_main high-quality settings
#   emit500, alternating round20/local100/global100, per-channel 1% baseline over frames 0:100
# Stage 2:
#   run the 3367 fast route using the zmap NPZ files produced by Stage 1

#SBATCH --job-name=nv04_3367hq
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/nv04_3367hq-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/nv04_3367hq-%j.err

set -euo pipefail

ROOT="${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}}}"
UNITY_DIR="$ROOT/unity"
PY="${UNITY_V04_PYTHON:-${NEPTUNE_V04_PYTHON:-${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}}}"
JOB_SUFFIX="${SLURM_JOB_ID:-local}"

export NEPTUNE_V04_ROOT="$ROOT"
export NEPTUNE_V04_PYTHON="$PY"
export NEPTUNE_V04_RAW_TIFF_PATH="${NEPTUNE_V04_RAW_TIFF_PATH:-${NEPTUNE_V03_RAW_TIFF_PATH:-$ROOT/neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}}"

HQ_ZMAP_PARALLEL_MODE="${HQ_ZMAP_PARALLEL_MODE:-same_gpu}"
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  if [[ "$HQ_ZMAP_PARALLEL_MODE" == "two_gpu" ]]; then
    export CUDA_VISIBLE_DEVICES="0,1"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
else
  export CUDA_VISIBLE_DEVICES
fi
export APPEND_DOMAIN_ONEHOT="${APPEND_DOMAIN_ONEHOT:-1}"
export DOMAIN_BALANCE_MODE="${DOMAIN_BALANCE_MODE:-alternate_step}"
export NEPTUNE_V04_PROJECTION_BACKEND="${NEPTUNE_V04_PROJECTION_BACKEND:-${NEPTUNE_V03_PROJECTION_BACKEND:-triton_fused}}"
export NEPTUNE_V04_CACHED_WINDOW_PRECOMPUTE="${NEPTUNE_V04_CACHED_WINDOW_PRECOMPUTE:-${NEPTUNE_V03_CACHED_WINDOW_PRECOMPUTE:-1}}"
export NEPTUNE_V04_LUT_EPOCH_PREWARM="${NEPTUNE_V04_LUT_EPOCH_PREWARM:-${NEPTUNE_V03_LUT_EPOCH_PREWARM:-1}}"
export NEPTUNE_V04_LUT_SHIFT_BACKEND="${NEPTUNE_V04_LUT_SHIFT_BACKEND:-${NEPTUNE_V03_LUT_SHIFT_BACKEND:-fourier}}"
export NEPTUNE_V04_PROFILE_TIMING="${NEPTUNE_V04_PROFILE_TIMING:-${NEPTUNE_V03_PROFILE_TIMING:-1}}"
export NEPTUNE_V04_PROFILE_SYNC_CUDA="${NEPTUNE_V04_PROFILE_SYNC_CUDA:-${NEPTUNE_V03_PROFILE_SYNC_CUDA:-1}}"

export NAT_CONFIG_KIND="${NAT_CONFIG_KIND:-order1_13}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-417}"
export EPOCHS="${EPOCHS:-300}"
export ROI_SIZE="${ROI_SIZE:-96}"
export PSF_SIZE="${PSF_SIZE:-25}"
export ROI_STRIDE="${ROI_STRIDE:-88}"
export START_EPOCH="${START_EPOCH:-30}"
export UPDATE_INTERVAL_EPOCHS="${UPDATE_INTERVAL_EPOCHS:-5}"
export TARGET_PROJECTED_EMITTERS="${TARGET_PROJECTED_EMITTERS:-5000}"

export HQ_MAX_EMITTERS="${HQ_MAX_EMITTERS:-500}"
export HQ_ALTERNATING_ROUNDS="${HQ_ALTERNATING_ROUNDS:-20}"
export HQ_ALTERNATING_LOCAL_STEPS="${HQ_ALTERNATING_LOCAL_STEPS:-100}"
export HQ_ALTERNATING_GLOBAL_STEPS="${HQ_ALTERNATING_GLOBAL_STEPS:-100}"
export HQ_SPATIAL_BALANCE_GRID_PX="${HQ_SPATIAL_BALANCE_GRID_PX:-}"
export HQ_SPATIAL_BALANCE_MIN_PER_CELL="${HQ_SPATIAL_BALANCE_MIN_PER_CELL:-}"
export HQ_SPATIAL_BALANCE_MAX_PER_CELL="${HQ_SPATIAL_BALANCE_MAX_PER_CELL:-}"
export HQ_SELECTION_POOL_MULTIPLIER="${HQ_SELECTION_POOL_MULTIPLIER:-50}"
export HQ_LEFT_SELECTION_POOL_MULTIPLIER="${HQ_LEFT_SELECTION_POOL_MULTIPLIER:-$HQ_SELECTION_POOL_MULTIPLIER}"
export HQ_RIGHT_SELECTION_POOL_MULTIPLIER="${HQ_RIGHT_SELECTION_POOL_MULTIPLIER:-$HQ_SELECTION_POOL_MULTIPLIER}"
export HQ_BASELINE_PERCENTILE="${HQ_BASELINE_PERCENTILE:-1.0}"
export HQ_BASELINE_FRAME_START="${HQ_BASELINE_FRAME_START:-0}"
export HQ_BASELINE_FRAME_STOP="${HQ_BASELINE_FRAME_STOP:-100}"

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

validate_supported_zmap_sample_kind() {
  case "$1" in
    microtube|microtubule|paint|ncp|dynamin|membrane) ;;
    *)
      echo "Unsupported ZMAP_SAMPLE_KIND=$1. Supported high-quality zmap presets are: microtube, paint, ncp, dynamin, membrane." >&2
      exit 2
      ;;
  esac
}

INFERRED_ZMAP_SAMPLE_KIND="$(infer_zmap_sample_kind_from_path "$NEPTUNE_V04_RAW_TIFF_PATH")"
if [[ -z "${ZMAP_SAMPLE_KIND+x}" || -z "${ZMAP_SAMPLE_KIND}" ]]; then
  export ZMAP_SAMPLE_KIND="${INFERRED_ZMAP_SAMPLE_KIND:-microtube}"
else
  export ZMAP_SAMPLE_KIND
fi
if [[ -n "$INFERRED_ZMAP_SAMPLE_KIND" && "$INFERRED_ZMAP_SAMPLE_KIND" != "$ZMAP_SAMPLE_KIND" ]]; then
  echo "Raw TIFF path appears to be sample kind '$INFERRED_ZMAP_SAMPLE_KIND', but ZMAP_SAMPLE_KIND='$ZMAP_SAMPLE_KIND'." >&2
  echo "Refusing to submit because this would build an initial zmap with the wrong preset." >&2
  echo "raw_tiff=${NEPTUNE_V04_RAW_TIFF_PATH}" >&2
  exit 2
fi
validate_supported_zmap_sample_kind "$ZMAP_SAMPLE_KIND"

export HQ_LEFT_CROP_X0="${HQ_LEFT_CROP_X0:-}"
export HQ_LEFT_CROP_X1="${HQ_LEFT_CROP_X1:-}"
export HQ_RIGHT_CROP_X0="${HQ_RIGHT_CROP_X0:-}"
export HQ_RIGHT_CROP_X1="${HQ_RIGHT_CROP_X1:-}"
export HQ_CROP_Y0="${HQ_CROP_Y0:-}"
export HQ_CROP_Y1="${HQ_CROP_Y1:-}"
export HQ_LEFT_ROI_X_MIN_PX="${HQ_LEFT_ROI_X_MIN_PX:-}"
export HQ_LEFT_ROI_X_MAX_PX="${HQ_LEFT_ROI_X_MAX_PX:-}"
export HQ_RIGHT_ROI_X_MIN_PX="${HQ_RIGHT_ROI_X_MIN_PX:-}"
export HQ_RIGHT_ROI_X_MAX_PX="${HQ_RIGHT_ROI_X_MAX_PX:-}"
export HQ_ROI_Y_MIN_PX="${HQ_ROI_Y_MIN_PX:-}"
export HQ_ROI_Y_MAX_PX="${HQ_ROI_Y_MAX_PX:-}"

if [[ "${ZMAP_SAMPLE_KIND:-microtube}" == "ncp" ]]; then
  export LEFT_DOMAIN_CROP_LEFT="${LEFT_DOMAIN_CROP_LEFT:-100}"
  export LEFT_DOMAIN_CROP_TOP="${LEFT_DOMAIN_CROP_TOP:-400}"
  export LEFT_DOMAIN_CROP_WIDTH="${LEFT_DOMAIN_CROP_WIDTH:-400}"
  export LEFT_DOMAIN_CROP_HEIGHT="${LEFT_DOMAIN_CROP_HEIGHT:-400}"
  export RIGHT_DOMAIN_CROP_LEFT="${RIGHT_DOMAIN_CROP_LEFT:-700}"
  export RIGHT_DOMAIN_CROP_TOP="${RIGHT_DOMAIN_CROP_TOP:-400}"
  export RIGHT_DOMAIN_CROP_WIDTH="${RIGHT_DOMAIN_CROP_WIDTH:-400}"
  export RIGHT_DOMAIN_CROP_HEIGHT="${RIGHT_DOMAIN_CROP_HEIGHT:-400}"
else
  export LEFT_DOMAIN_CROP_LEFT="${LEFT_DOMAIN_CROP_LEFT:-0}"
  export LEFT_DOMAIN_CROP_TOP="${LEFT_DOMAIN_CROP_TOP:-0}"
  export LEFT_DOMAIN_CROP_WIDTH="${LEFT_DOMAIN_CROP_WIDTH:-600}"
  export LEFT_DOMAIN_CROP_HEIGHT="${LEFT_DOMAIN_CROP_HEIGHT:-1200}"
  export RIGHT_DOMAIN_CROP_LEFT="${RIGHT_DOMAIN_CROP_LEFT:-600}"
  export RIGHT_DOMAIN_CROP_TOP="${RIGHT_DOMAIN_CROP_TOP:-0}"
  export RIGHT_DOMAIN_CROP_WIDTH="${RIGHT_DOMAIN_CROP_WIDTH:-600}"
  export RIGHT_DOMAIN_CROP_HEIGHT="${RIGHT_DOMAIN_CROP_HEIGHT:-1200}"
fi
if [[ "$LEFT_DOMAIN_CROP_TOP" -ne "$RIGHT_DOMAIN_CROP_TOP" || "$LEFT_DOMAIN_CROP_HEIGHT" -ne "$RIGHT_DOMAIN_CROP_HEIGHT" ]]; then
  echo "HQ bootstrap currently requires matching left/right y-domain crops; got left top/height ${LEFT_DOMAIN_CROP_TOP}/${LEFT_DOMAIN_CROP_HEIGHT}, right top/height ${RIGHT_DOMAIN_CROP_TOP}/${RIGHT_DOMAIN_CROP_HEIGHT}" >&2
  exit 2
fi
export HQ_LEFT_CROP_X0="${HQ_LEFT_CROP_X0:-$LEFT_DOMAIN_CROP_LEFT}"
export HQ_LEFT_CROP_X1="${HQ_LEFT_CROP_X1:-$((LEFT_DOMAIN_CROP_LEFT + LEFT_DOMAIN_CROP_WIDTH))}"
export HQ_RIGHT_CROP_X0="${HQ_RIGHT_CROP_X0:-$RIGHT_DOMAIN_CROP_LEFT}"
export HQ_RIGHT_CROP_X1="${HQ_RIGHT_CROP_X1:-$((RIGHT_DOMAIN_CROP_LEFT + RIGHT_DOMAIN_CROP_WIDTH))}"
export HQ_CROP_Y0="${HQ_CROP_Y0:-$LEFT_DOMAIN_CROP_TOP}"
export HQ_CROP_Y1="${HQ_CROP_Y1:-$((LEFT_DOMAIN_CROP_TOP + LEFT_DOMAIN_CROP_HEIGHT))}"
export HQ_LEFT_ROI_X_MIN_PX="${HQ_LEFT_ROI_X_MIN_PX:-$HQ_LEFT_CROP_X0}"
export HQ_LEFT_ROI_X_MAX_PX="${HQ_LEFT_ROI_X_MAX_PX:-$HQ_LEFT_CROP_X1}"
export HQ_RIGHT_ROI_X_MIN_PX="${HQ_RIGHT_ROI_X_MIN_PX:-$HQ_RIGHT_CROP_X0}"
export HQ_RIGHT_ROI_X_MAX_PX="${HQ_RIGHT_ROI_X_MAX_PX:-$HQ_RIGHT_CROP_X1}"
export HQ_ROI_Y_MIN_PX="${HQ_ROI_Y_MIN_PX:-$HQ_CROP_Y0}"
export HQ_ROI_Y_MAX_PX="${HQ_ROI_Y_MAX_PX:-$HQ_CROP_Y1}"

export LR_STEP_UNIT="${LR_STEP_UNIT:-epoch}"
export LR_STEP_SIZE="${LR_STEP_SIZE:-10}"
export LR_GAMMA="${LR_GAMMA:-0.9}"

export SEQUENCE_COUNT="${SEQUENCE_COUNT:-$STEPS_PER_EPOCH}"
export CACHED_WINDOW_ORDER="${CACHED_WINDOW_ORDER:-auto}"
export CACHED_WINDOW_MAX_GPU_SEQUENCES="${CACHED_WINDOW_MAX_GPU_SEQUENCES:-2}"
export FIELD_ORIGIN_SAMPLING_MODE="${FIELD_ORIGIN_SAMPLING_MODE:-sliding_window}"
export NAT_GRID_SIZE_X="${NAT_GRID_SIZE_X:-2}"
export NAT_GRID_SIZE_Y="${NAT_GRID_SIZE_Y:-2}"

HQ_BALANCE_TAG=""
if [[ -n "$HQ_SPATIAL_BALANCE_GRID_PX" ]]; then
  HQ_BALANCE_TAG="_grid${HQ_SPATIAL_BALANCE_GRID_PX}"
  if [[ -n "$HQ_SPATIAL_BALANCE_MIN_PER_CELL" ]]; then
    HQ_BALANCE_TAG="${HQ_BALANCE_TAG}_mincell${HQ_SPATIAL_BALANCE_MIN_PER_CELL}"
  fi
fi
HQ_RUN_TAG="${ZMAP_SAMPLE_KIND}_formal_order2_frame1000_emit${HQ_MAX_EMITTERS}_round${HQ_ALTERNATING_ROUNDS}_p1baseline${HQ_BALANCE_TAG}_recomputed_${JOB_SUFFIX}"
HQ_ZMAP_ROOT="$UNITY_DIR/output/high_quality_initial_zmap/$HQ_RUN_TAG"
export ZMAP_LEFT="$HQ_ZMAP_ROOT/left/export_nat_zmap/alternating_full_roi_zernike_maps_nm.npz"
export ZMAP_RIGHT="$HQ_ZMAP_ROOT/right/export_nat_zmap/alternating_full_roi_zernike_maps_nm.npz"
export INITIAL_ZMAP_TAG="high_quality_initial_zmap_recomputed_emit${HQ_MAX_EMITTERS}_round${HQ_ALTERNATING_ROUNDS}_p1baseline${HQ_BALANCE_TAG}_order2_${JOB_SUFFIX}"
export STANDARD_NOTE="Run 3371 default fast route after recomputing high-quality initial zmap from raw TIFF with zmap_main emit${HQ_MAX_EMITTERS} round${HQ_ALTERNATING_ROUNDS} p1baseline per-channel baseline; no reused zmap artifact and no inline peak bootstrap."
export RUN_TAG="${RUN_TAG:-standard_3367_fast_route_roi${ROI_SIZE}_psf${PSF_SIZE}_fresh_hqzmap_emit${HQ_MAX_EMITTERS}_round${HQ_ALTERNATING_ROUNDS}_p1baseline${HQ_BALANCE_TAG}_3052lr_start${START_EPOCH}_interval${UPDATE_INTERVAL_EPOCHS}_emit${TARGET_PROJECTED_EMITTERS}_epoch${EPOCHS}_bs${BATCH_SIZE}_steps${STEPS_PER_EPOCH}_${JOB_SUFFIX}}"

mkdir -p "$UNITY_DIR/logs/slurm" "$HQ_ZMAP_ROOT"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "stage1=high_quality_initial_zmap_from_scratch"
echo "raw_tiff=${NEPTUNE_V04_RAW_TIFF_PATH}"
echo "inferred_zmap_sample_kind=${INFERRED_ZMAP_SAMPLE_KIND:-unknown}"
echo "zmap_sample_kind=${ZMAP_SAMPLE_KIND}"
echo "hq_zmap_root=${HQ_ZMAP_ROOT}"
echo "hq_settings=sample=${ZMAP_SAMPLE_KIND} max_emitters=${HQ_MAX_EMITTERS} alternating_rounds=${HQ_ALTERNATING_ROUNDS} local_steps=${HQ_ALTERNATING_LOCAL_STEPS} global_steps=${HQ_ALTERNATING_GLOBAL_STEPS} baseline=per_channel_p${HQ_BASELINE_PERCENTILE}_frames_${HQ_BASELINE_FRAME_START}_${HQ_BASELINE_FRAME_STOP} roi_stride_aligned_grid_px=${HQ_SPATIAL_BALANCE_GRID_PX} spatial_balance_min_per_cell=${HQ_SPATIAL_BALANCE_MIN_PER_CELL} spatial_balance_max_per_cell=${HQ_SPATIAL_BALANCE_MAX_PER_CELL} selection_pool_multiplier=${HQ_SELECTION_POOL_MULTIPLIER} left_right_parallel=1 parallel_mode=${HQ_ZMAP_PARALLEL_MODE} left_crop_x=${HQ_LEFT_CROP_X0}:${HQ_LEFT_CROP_X1} right_crop_x=${HQ_RIGHT_CROP_X0}:${HQ_RIGHT_CROP_X1} crop_y=${HQ_CROP_Y0}:${HQ_CROP_Y1}"
echo "training_domain_crops=left:${LEFT_DOMAIN_CROP_LEFT},${LEFT_DOMAIN_CROP_TOP},${LEFT_DOMAIN_CROP_WIDTH},${LEFT_DOMAIN_CROP_HEIGHT} right:${RIGHT_DOMAIN_CROP_LEFT},${RIGHT_DOMAIN_CROP_TOP},${RIGHT_DOMAIN_CROP_WIDTH},${RIGHT_DOMAIN_CROP_HEIGHT}"
echo "stage2_run_tag=${RUN_TAG}"

export PYTHONPATH="$ROOT:$UNITY_DIR/src"

stage1_start="$(date +%s)"
left_device="cuda:0"
right_device="cuda:0"
if [[ "$HQ_ZMAP_PARALLEL_MODE" == "two_gpu" ]]; then
  right_device="cuda:1"
fi

left_hq_args=(
  --side left
  --run-root "$HQ_ZMAP_ROOT/left"
  --repo-root "$ROOT"
  --raw-tiff "$NEPTUNE_V04_RAW_TIFF_PATH"
  --zmap-sample "$ZMAP_SAMPLE_KIND"
  --device "$left_device"
  --max-emitters "$HQ_MAX_EMITTERS"
  --alternating-rounds "$HQ_ALTERNATING_ROUNDS"
  --alternating-local-steps "$HQ_ALTERNATING_LOCAL_STEPS"
  --alternating-global-steps "$HQ_ALTERNATING_GLOBAL_STEPS"
  --selection-pool-multiplier "$HQ_LEFT_SELECTION_POOL_MULTIPLIER"
  --baseline-percentile "$HQ_BASELINE_PERCENTILE"
  --baseline-frame-start "$HQ_BASELINE_FRAME_START"
  --baseline-frame-stop "$HQ_BASELINE_FRAME_STOP"
)
right_hq_args=(
  --side right
  --run-root "$HQ_ZMAP_ROOT/right"
  --repo-root "$ROOT"
  --raw-tiff "$NEPTUNE_V04_RAW_TIFF_PATH"
  --zmap-sample "$ZMAP_SAMPLE_KIND"
  --device "$right_device"
  --max-emitters "$HQ_MAX_EMITTERS"
  --alternating-rounds "$HQ_ALTERNATING_ROUNDS"
  --alternating-local-steps "$HQ_ALTERNATING_LOCAL_STEPS"
  --alternating-global-steps "$HQ_ALTERNATING_GLOBAL_STEPS"
  --selection-pool-multiplier "$HQ_RIGHT_SELECTION_POOL_MULTIPLIER"
  --baseline-percentile "$HQ_BASELINE_PERCENTILE"
  --baseline-frame-start "$HQ_BASELINE_FRAME_START"
  --baseline-frame-stop "$HQ_BASELINE_FRAME_STOP"
)
if [[ -n "$HQ_SPATIAL_BALANCE_GRID_PX" ]]; then
  left_hq_args+=(--spatial-balance-grid-px "$HQ_SPATIAL_BALANCE_GRID_PX")
  right_hq_args+=(--spatial-balance-grid-px "$HQ_SPATIAL_BALANCE_GRID_PX")
fi
if [[ -n "$HQ_SPATIAL_BALANCE_MIN_PER_CELL" ]]; then
  left_hq_args+=(--spatial-balance-min-per-cell "$HQ_SPATIAL_BALANCE_MIN_PER_CELL")
  right_hq_args+=(--spatial-balance-min-per-cell "$HQ_SPATIAL_BALANCE_MIN_PER_CELL")
fi
if [[ -n "$HQ_SPATIAL_BALANCE_MAX_PER_CELL" ]]; then
  left_hq_args+=(--spatial-balance-max-per-cell "$HQ_SPATIAL_BALANCE_MAX_PER_CELL")
  right_hq_args+=(--spatial-balance-max-per-cell "$HQ_SPATIAL_BALANCE_MAX_PER_CELL")
fi
if [[ -n "$HQ_LEFT_CROP_X0" ]]; then
  left_hq_args+=(--crop-x0 "$HQ_LEFT_CROP_X0")
fi
if [[ -n "$HQ_LEFT_CROP_X1" ]]; then
  left_hq_args+=(--crop-x1 "$HQ_LEFT_CROP_X1")
fi
if [[ -n "$HQ_RIGHT_CROP_X0" ]]; then
  right_hq_args+=(--crop-x0 "$HQ_RIGHT_CROP_X0")
fi
if [[ -n "$HQ_RIGHT_CROP_X1" ]]; then
  right_hq_args+=(--crop-x1 "$HQ_RIGHT_CROP_X1")
fi
if [[ -n "$HQ_CROP_Y0" ]]; then
  left_hq_args+=(--crop-y0 "$HQ_CROP_Y0")
  right_hq_args+=(--crop-y0 "$HQ_CROP_Y0")
fi
if [[ -n "$HQ_CROP_Y1" ]]; then
  left_hq_args+=(--crop-y1 "$HQ_CROP_Y1")
  right_hq_args+=(--crop-y1 "$HQ_CROP_Y1")
fi
if [[ -n "$HQ_LEFT_ROI_X_MIN_PX" ]]; then
  left_hq_args+=(--roi-x-min-px "$HQ_LEFT_ROI_X_MIN_PX")
fi
if [[ -n "$HQ_LEFT_ROI_X_MAX_PX" ]]; then
  left_hq_args+=(--roi-x-max-px "$HQ_LEFT_ROI_X_MAX_PX")
fi
if [[ -n "$HQ_RIGHT_ROI_X_MIN_PX" ]]; then
  right_hq_args+=(--roi-x-min-px "$HQ_RIGHT_ROI_X_MIN_PX")
fi
if [[ -n "$HQ_RIGHT_ROI_X_MAX_PX" ]]; then
  right_hq_args+=(--roi-x-max-px "$HQ_RIGHT_ROI_X_MAX_PX")
fi
if [[ -n "$HQ_ROI_Y_MIN_PX" ]]; then
  left_hq_args+=(--roi-y-min-px "$HQ_ROI_Y_MIN_PX")
  right_hq_args+=(--roi-y-min-px "$HQ_ROI_Y_MIN_PX")
fi
if [[ -n "$HQ_ROI_Y_MAX_PX" ]]; then
  left_hq_args+=(--roi-y-max-px "$HQ_ROI_Y_MAX_PX")
  right_hq_args+=(--roi-y-max-px "$HQ_ROI_Y_MAX_PX")
fi

echo "start_zmap_side=left device=${left_device}"
"$PY" -u "$UNITY_DIR/scripts/train/run_high_quality_initial_zmap.py" "${left_hq_args[@]}" &
left_pid=$!

echo "start_zmap_side=right device=${right_device}"
"$PY" -u "$UNITY_DIR/scripts/train/run_high_quality_initial_zmap.py" "${right_hq_args[@]}" &
right_pid=$!

left_rc=0
right_rc=0
wait "$left_pid" || left_rc=$?
echo "finished_zmap_side=left"
wait "$right_pid" || right_rc=$?
echo "finished_zmap_side=right"
if [[ "$left_rc" -ne 0 || "$right_rc" -ne 0 ]]; then
  echo "High-quality zmap stage failed: left_rc=${left_rc}, right_rc=${right_rc}" >&2
  exit 4
fi
stage1_end="$(date +%s)"
echo "stage1_elapsed_sec=$((stage1_end - stage1_start))"

if [[ ! -s "$ZMAP_LEFT" ]]; then
  echo "Missing freshly computed left zmap: $ZMAP_LEFT" >&2
  exit 3
fi
if [[ ! -s "$ZMAP_RIGHT" ]]; then
  echo "Missing freshly computed right zmap: $ZMAP_RIGHT" >&2
  exit 3
fi

echo "stage2=training_fast_route_3367"
echo "initial_zmap_left=${ZMAP_LEFT}"
echo "initial_zmap_right=${ZMAP_RIGHT}"

export PYTHONPATH="$UNITY_DIR/src"
exec bash "$UNITY_DIR/scripts/train/fast_route_3052_strength_oldzmap.sbatch"
