#!/usr/bin/env bash
# Fast-route training with 3052-strength physical update settings:
# - old 3052 zmap as initial physical model
# - ROI128, PSF51, start epoch 30, update every 5 epochs
# - 5000 projected emitters and posterior probability quality gate

#SBATCH --job-name=nv04_f3052
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/nv04_f3052-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/nv04_f3052-%j.err

set -euo pipefail

ROOT="${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}}}"
UNITY_DIR="$ROOT/unity"
PY="${UNITY_V04_PYTHON:-${NEPTUNE_V04_PYTHON:-${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}}}"
NAT_CONFIG_KIND="${NAT_CONFIG_KIND:-order1_13}"

ZMAP_LEFT="${ZMAP_LEFT:-$ROOT/neptune_iwae/test_data/microtube/zmap/left/alternating_full_roi_zernike_maps_nm.npz}"
ZMAP_RIGHT="${ZMAP_RIGHT:-$ROOT/neptune_iwae/test_data/microtube/zmap/right/alternating_full_roi_zernike_maps_nm.npz}"
export ZMAP_LEFT
export ZMAP_RIGHT

export PYTHONPATH="$UNITY_DIR/src"
export MPLCONFIGDIR="$UNITY_DIR/.local/cache/matplotlib"
export NEPTUNE_V04_RAW_TIFF_PATH="${NEPTUNE_V04_RAW_TIFF_PATH:-${NEPTUNE_V03_RAW_TIFF_PATH:-$ROOT/neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPEND_DOMAIN_ONEHOT="${APPEND_DOMAIN_ONEHOT:-1}"
export DOMAIN_BALANCE_MODE="${DOMAIN_BALANCE_MODE:-alternate_step}"
export NEPTUNE_V04_PROJECTION_BACKEND="${NEPTUNE_V04_PROJECTION_BACKEND:-${NEPTUNE_V03_PROJECTION_BACKEND:-triton_fused}}"
export NEPTUNE_V04_CACHED_WINDOW_PRECOMPUTE="${NEPTUNE_V04_CACHED_WINDOW_PRECOMPUTE:-${NEPTUNE_V03_CACHED_WINDOW_PRECOMPUTE:-1}}"
export NEPTUNE_V04_LUT_EPOCH_PREWARM="${NEPTUNE_V04_LUT_EPOCH_PREWARM:-${NEPTUNE_V03_LUT_EPOCH_PREWARM:-1}}"
export NEPTUNE_V04_LUT_SHIFT_BACKEND="${NEPTUNE_V04_LUT_SHIFT_BACKEND:-${NEPTUNE_V03_LUT_SHIFT_BACKEND:-fourier}}"
export NEPTUNE_V04_PROFILE_TIMING="${NEPTUNE_V04_PROFILE_TIMING:-${NEPTUNE_V03_PROFILE_TIMING:-1}}"
export NEPTUNE_V04_PROFILE_SYNC_CUDA="${NEPTUNE_V04_PROFILE_SYNC_CUDA:-${NEPTUNE_V03_PROFILE_SYNC_CUDA:-1}}"

BATCH_SIZE="${BATCH_SIZE:-24}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-417}"
EPOCHS="${EPOCHS:-300}"
ROI_SIZE="${ROI_SIZE:-128}"
PSF_SIZE="${PSF_SIZE:-51}"
ROI_STRIDE="${ROI_STRIDE:-88}"
START_EPOCH="${START_EPOCH:-30}"
UPDATE_INTERVAL_EPOCHS="${UPDATE_INTERVAL_EPOCHS:-5}"
TARGET_PROJECTED_EMITTERS="${TARGET_PROJECTED_EMITTERS:-5000}"
ROI_GAMMA_ARTIFACT_RETENTION_POLICY="${ROI_GAMMA_ARTIFACT_RETENTION_POLICY:-compact_latest}"
ROI_GAMMA_DIAGNOSTIC_INTERVAL_UPDATES="${ROI_GAMMA_DIAGNOSTIC_INTERVAL_UPDATES:-10}"
LR_STEP_UNIT="${LR_STEP_UNIT:-epoch}"
LR_STEP_SIZE="${LR_STEP_SIZE:-10}"
LR_GAMMA="${LR_GAMMA:-0.9}"
SEQUENCE_COUNT="${SEQUENCE_COUNT:-$STEPS_PER_EPOCH}"
CACHED_WINDOW_ORDER="${CACHED_WINDOW_ORDER:-auto}"
CACHED_WINDOW_MAX_GPU_SEQUENCES="${CACHED_WINDOW_MAX_GPU_SEQUENCES:-2}"
FIELD_ORIGIN_SAMPLING_MODE="${FIELD_ORIGIN_SAMPLING_MODE:-sliding_window}"
NAT_GRID_SIZE_X="${NAT_GRID_SIZE_X:-2}"
NAT_GRID_SIZE_Y="${NAT_GRID_SIZE_Y:-2}"
LEFT_DOMAIN_CROP_LEFT="${LEFT_DOMAIN_CROP_LEFT:-0}"
LEFT_DOMAIN_CROP_TOP="${LEFT_DOMAIN_CROP_TOP:-0}"
LEFT_DOMAIN_CROP_WIDTH="${LEFT_DOMAIN_CROP_WIDTH:-600}"
LEFT_DOMAIN_CROP_HEIGHT="${LEFT_DOMAIN_CROP_HEIGHT:-1200}"
RIGHT_DOMAIN_CROP_LEFT="${RIGHT_DOMAIN_CROP_LEFT:-600}"
RIGHT_DOMAIN_CROP_TOP="${RIGHT_DOMAIN_CROP_TOP:-0}"
RIGHT_DOMAIN_CROP_WIDTH="${RIGHT_DOMAIN_CROP_WIDTH:-600}"
RIGHT_DOMAIN_CROP_HEIGHT="${RIGHT_DOMAIN_CROP_HEIGHT:-1200}"
export LEFT_DOMAIN_CROP_LEFT LEFT_DOMAIN_CROP_TOP LEFT_DOMAIN_CROP_WIDTH LEFT_DOMAIN_CROP_HEIGHT
export RIGHT_DOMAIN_CROP_LEFT RIGHT_DOMAIN_CROP_TOP RIGHT_DOMAIN_CROP_WIDTH RIGHT_DOMAIN_CROP_HEIGHT

RUN_TAG="${RUN_TAG:-fast_route_roi128_psf51_oldzmap_3052lr_start30_interval5_emit5000_qualitygate_epoch${EPOCHS}_bs${BATCH_SIZE}_steps${STEPS_PER_EPOCH}}"
INITIAL_ZMAP_TAG="${INITIAL_ZMAP_TAG:-3052_old_zmap}"
STANDARD_NOTE="${STANDARD_NOTE:-Fast LUT/prewarm/projection route with ROI128, PSF51, initial zmap, start30 interval5, target_projected_emitters=5000, explicit posterior quality gates, and 3052 scheduler parity: StepLR(step_size=10, gamma=0.9) stepped once per epoch.}"
RUN_ROOT="$UNITY_DIR/output/$RUN_TAG"
RUN_NAME="${RUN_TAG}_${SLURM_JOB_ID:-local}"
OVERRIDE_CONFIG="$UNITY_DIR/.local/tmp/standard/${RUN_TAG}.yaml"
RESOLVED_CONFIG="$UNITY_DIR/.local/tmp/standard/resolved_${RUN_TAG}_${SLURM_JOB_ID:-local}.yaml"

mkdir -p \
  "$UNITY_DIR/.local/cache/matplotlib" \
  "$UNITY_DIR/.local/tmp/standard" \
  "$UNITY_DIR/logs/slurm" \
  "$RUN_ROOT"

if [[ ! -s "$ZMAP_LEFT" ]]; then
  echo "Missing left initial zmap: $ZMAP_LEFT" >&2
  exit 3
fi
if [[ ! -s "$ZMAP_RIGHT" ]]; then
  echo "Missing right initial zmap: $ZMAP_RIGHT" >&2
  exit 3
fi

cat > "$OVERRIDE_CONFIG" <<YAML
train:
  real_tiff_wake:
    tiff_path: ${NEPTUNE_V04_RAW_TIFF_PATH}
    domains:
      - name: left
        crop_left: ${LEFT_DOMAIN_CROP_LEFT}
        crop_top: ${LEFT_DOMAIN_CROP_TOP}
        crop_width: ${LEFT_DOMAIN_CROP_WIDTH}
        crop_height: ${LEFT_DOMAIN_CROP_HEIGHT}
      - name: right
        crop_left: ${RIGHT_DOMAIN_CROP_LEFT}
        crop_top: ${RIGHT_DOMAIN_CROP_TOP}
        crop_width: ${RIGHT_DOMAIN_CROP_WIDTH}
        crop_height: ${RIGHT_DOMAIN_CROP_HEIGHT}
  epochs: ${EPOCHS}
  batch_size: ${BATCH_SIZE}
  camera:
    baseline_mode: per_domain_percentile
    baseline_frame_range: [0, 100]
    baseline_percentile: 1.0
    e_per_adu: 1.020784562122306
    qe: 0.9
    em_gain: 1.0
    spurious_charge: 0.002
  online_generation:
    enabled: true
    steps_per_epoch: ${STEPS_PER_EPOCH}
    height: ${ROI_SIZE}
    width: ${ROI_SIZE}
    batch_strategy: cached_window
    sequence_count: ${SEQUENCE_COUNT}
    sequence_window_chunks: 2
    simulation_backend: lut
    simulation_output_device: renderer
    cached_window_order: ${CACHED_WINDOW_ORDER}
    cached_window_max_gpu_sequences: ${CACHED_WINDOW_MAX_GPU_SEQUENCES}
    field_origin_sampling_mode: ${FIELD_ORIGIN_SAMPLING_MODE}
    field_origin_stride_px: ${ROI_STRIDE}
    emitter_density_um2: 1.0
    lut_simulation:
      field_mode: global_field
      storage_dtype: fp16
      field_stride: 16
      subpixel_bins: 1
      psf_size: ${PSF_SIZE}
    dual_domain_coeff_maps:
      - name: left
        coeff_maps_npz: ${ZMAP_LEFT}
      - name: right
        coeff_maps_npz: ${ZMAP_RIGHT}
    amp_enabled: true
    amp_dtype: float16
    nat_simulation_mode: grid_map
    nat_grid_size: [${NAT_GRID_SIZE_X}, ${NAT_GRID_SIZE_Y}]
    nat_grid_z_steps: 41
  eval:
    enabled: true
    source: online_generation
    batch_count: 4
    batch_size: 4
    seed: 100000
    raw_th: 0.5
    split_th: 0.6
    probability_threshold: 0.5
  peak_zmap_bootstrap:
    enabled: false
    nat_config_kind: ${NAT_CONFIG_KIND}
  nat_wake:
    patch_size_px: 25
    feedback_interval_steps: ${STEPS_PER_EPOCH}
  roi_bank_gamma:
    enabled: true
    nat_config_kind: ${NAT_CONFIG_KIND}
    start_epoch: ${START_EPOCH}
    stop_epoch: ${EPOCHS}
    update_interval_epochs: ${UPDATE_INTERVAL_EPOCHS}
    output_dir: roi_bank_gamma_3052_strength_oldzmap_fast_route
    roi_bank_frame_range: [0, 100]
    roi_size_px: ${ROI_SIZE}
    roi_bank_valid_core_size_px: ${ROI_SIZE}
    roi_bank_over_cut_px: 8
    roi_bank_origin_mode: sliding_window_guided
    roi_bank_origin_stride_px: ${ROI_STRIDE}
    roi_bank_candidate_probability_threshold: 0.3
    roi_bank_probability_threshold: 0.5
    posterior_probability_threshold: 0.7
    posterior_candidate_probability_threshold: 0.3
    posterior_adjacent_probability_threshold: 0.6
    probability_threshold: 0.5
    gamma_steps: 100
    gamma_lr: 0.025
    artifact_retention_policy: ${ROI_GAMMA_ARTIFACT_RETENTION_POLICY}
    diagnostic_interval_updates: ${ROI_GAMMA_DIAGNOSTIC_INTERVAL_UPDATES}
    target_projected_emitters: ${TARGET_PROJECTED_EMITTERS}
    max_sampling_rounds: 20
    num_posterior_samples: 25
    roi_bank_projection_sample_batch_size: 128
    posterior_batch_size: 4
    roi_bank_projection_emitter_chunk_size: 4096
smlm_overrides:
  lr_scheduler: StepLR
  lr_step_size: ${LR_STEP_SIZE}
  lr_gamma: ${LR_GAMMA}
  lr_step_unit: ${LR_STEP_UNIT}
metadata:
  training_mode: neptune_v03_fast_route_3052_strength_oldzmap
  initial_zmap_tag: ${INITIAL_ZMAP_TAG}
  scheduler_step_unit: ${LR_STEP_UNIT}
  scheduler_step_size: ${LR_STEP_SIZE}
  scheduler_gamma: ${LR_GAMMA}
  standard_note: "${STANDARD_NOTE}"
YAML

echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "nat_config_kind=${NAT_CONFIG_KIND}"
echo "batch_size=${BATCH_SIZE}"
echo "steps_per_epoch=${STEPS_PER_EPOCH}"
echo "sequence_count=${SEQUENCE_COUNT}"
echo "epochs=${EPOCHS}"
echo "roi_size=${ROI_SIZE}"
echo "psf_size=${PSF_SIZE}"
echo "roi_stride=${ROI_STRIDE}"
echo "cached_window_order=${CACHED_WINDOW_ORDER}"
echo "cached_window_max_gpu_sequences=${CACHED_WINDOW_MAX_GPU_SEQUENCES}"
echo "field_origin_sampling_mode=${FIELD_ORIGIN_SAMPLING_MODE}"
echo "nat_grid_size=[${NAT_GRID_SIZE_X}, ${NAT_GRID_SIZE_Y}]"
echo "start_epoch=${START_EPOCH}"
echo "update_interval_epochs=${UPDATE_INTERVAL_EPOCHS}"
echo "target_projected_emitters=${TARGET_PROJECTED_EMITTERS}"
echo "lr_step_unit=${LR_STEP_UNIT}"
echo "lr_step_size=${LR_STEP_SIZE}"
echo "lr_gamma=${LR_GAMMA}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "raw_tiff=${NEPTUNE_V04_RAW_TIFF_PATH}"
echo "left_domain_crop=${LEFT_DOMAIN_CROP_LEFT},${LEFT_DOMAIN_CROP_TOP},${LEFT_DOMAIN_CROP_WIDTH},${LEFT_DOMAIN_CROP_HEIGHT}"
echo "right_domain_crop=${RIGHT_DOMAIN_CROP_LEFT},${RIGHT_DOMAIN_CROP_TOP},${RIGHT_DOMAIN_CROP_WIDTH},${RIGHT_DOMAIN_CROP_HEIGHT}"
echo "initial_zmap_left=${ZMAP_LEFT}"
echo "initial_zmap_right=${ZMAP_RIGHT}"
echo "initial_zmap_tag=${INITIAL_ZMAP_TAG}"
echo "projection_backend=${NEPTUNE_V04_PROJECTION_BACKEND}"
echo "lut_epoch_prewarm=${NEPTUNE_V04_LUT_EPOCH_PREWARM}"
echo "run_root=${RUN_ROOT}"
echo "run_name=${RUN_NAME}"

"$PY" - <<'PY'
import os
from pathlib import Path
import numpy as np
import tifffile
import torch
from unity_psf.optics import build_named_nat_config

raw = os.environ["NEPTUNE_V04_RAW_TIFF_PATH"]
cfg = build_named_nat_config(os.environ.get("NAT_CONFIG_KIND", "order1_13"), img_size_x=600, img_size_y=1200)
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
print("raw_tiff_shape", tifffile.TiffFile(raw).series[0].shape)
print("nat_gamma_size", len(cfg.gammas))
print("nat_aberrations", list(cfg.aberrations))
for env_name in ("ZMAP_LEFT", "ZMAP_RIGHT"):
    path = Path(os.environ[env_name])
    payload = np.load(path, allow_pickle=False)
    modes = [tuple(int(v) for v in row) for row in payload["mode_order"]]
    missing = [mode for mode in cfg.aberrations if mode not in modes]
    print(env_name, path, payload["zernike_maps_nm"].shape, modes)
    print(env_name + "_missing_required_modes", missing)
    if missing:
        raise SystemExit(f"{env_name} missing required modes: {missing}")
left_shape = np.load(os.environ["ZMAP_LEFT"], allow_pickle=False)["zernike_maps_nm"].shape[-2:]
right_shape = np.load(os.environ["ZMAP_RIGHT"], allow_pickle=False)["zernike_maps_nm"].shape[-2:]
expected = {
    "ZMAP_LEFT": (int(os.environ["LEFT_DOMAIN_CROP_HEIGHT"]), int(os.environ["LEFT_DOMAIN_CROP_WIDTH"])),
    "ZMAP_RIGHT": (int(os.environ["RIGHT_DOMAIN_CROP_HEIGHT"]), int(os.environ["RIGHT_DOMAIN_CROP_WIDTH"])),
}
actual = {"ZMAP_LEFT": tuple(int(v) for v in left_shape), "ZMAP_RIGHT": tuple(int(v) for v in right_shape)}
print("domain_zmap_shape_check", {"expected_hw": expected, "actual_hw": actual})
for key in expected:
    if actual[key] != expected[key]:
        raise SystemExit(f"{key} zmap shape {actual[key]} does not match configured domain crop H,W {expected[key]}")
PY

"$PY" -m unity_psf.config.materialize \
  --base "$UNITY_DIR/configs/base/microtube.yaml" \
  --override "$UNITY_DIR/configs/overrides/standard_roi_gamma.yaml" \
  --override "$OVERRIDE_CONFIG" \
  --output "$RESOLVED_CONFIG"

echo "resolved_fast_route_3052_strength_config:"
grep -E "real_tiff_wake:|crop_left:|crop_top:|crop_width:|crop_height:|epochs:|batch_size:|steps_per_epoch:|height:|width:|sequence_count:|field_origin_sampling_mode:|field_origin_stride_px:|psf_size:|patch_size_px:|feedback_interval_steps:|start_epoch:|update_interval_epochs:|stop_epoch:|roi_size_px:|roi_bank_valid_core_size_px:|roi_bank_over_cut_px:|gamma_steps:|target_projected_emitters:|posterior_probability_threshold:|posterior_candidate_probability_threshold:|posterior_adjacent_probability_threshold:|roi_bank_probability_threshold:|roi_bank_projection_sample_batch_size:|posterior_batch_size:|dual_domain_coeff_maps:|lr_scheduler:|lr_step_size:|lr_gamma:|lr_step_unit:|enabled:" "$RESOLVED_CONFIG" || true
if grep -E "max_batches:|start_batch:|stop_batch:|update_interval_batches:" "$RESOLVED_CONFIG"; then
  echo "ERROR: batch-budget fields leaked into epoch route config" >&2
  exit 2
fi

"$PY" -u -m unity_psf.training.run_high_fidelity \
  --config "$RESOLVED_CONFIG" \
  --run-root "$RUN_ROOT" \
  --run-name "$RUN_NAME" \
  --seed 0
