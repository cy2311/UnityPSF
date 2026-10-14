#!/usr/bin/env bash
# Dual-channel Soft-MoE training with ROI-library physical updates.
# The first 30 epochs are warmup; the first ROI library is built at epoch 30.
#SBATCH --job-name=unitypsf_roi_gamma
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm/unitypsf_roi_gamma-%j.out
#SBATCH --error=logs/slurm/unitypsf_roi_gamma-%j.err

set -euo pipefail
ROOT="${UNITYPSF_ROOT:-/home/guest/Others/main/race/unity_v0.3}"
cd "$ROOT"
PY="${UNITY_V04_PYTHON:-/home/guest/anaconda3/bin/python}"
RAW_TIFF="${UNITYPSF_RAW_TIFF:-/share/home/guest/race/datasets/samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}"
RUN_NAME="${UNITYPSF_RUN_NAME:-unitypsf_cuda_gemm_roi_gamma_${SLURM_JOB_ID:-local}}"
RUN_ROOT="${UNITYPSF_RUN_ROOT:-$ROOT/output}"
PROFILE_TIMING="${UNITYPSF_PROFILE_TIMING:-0}"
PROFILE_EPOCHS="${UNITYPSF_PROFILE_EPOCHS:-}"
PROFILE_MAX_BATCHES="${UNITYPSF_PROFILE_MAX_BATCHES:-}"
BATCH_SIZE="${UNITYPSF_BATCH_SIZE:-64}"
STEPS_PER_EPOCH="${UNITYPSF_STEPS_PER_EPOCH:-157}"
RESOLVED="$ROOT/.local/tmp/unitypsf_roi_gamma_resolved_${SLURM_JOB_ID:-local}.yaml"

mkdir -p "$ROOT/.local/tmp" "$ROOT/logs/slurm" "$RUN_ROOT"
test -f "$RAW_TIFF"
export PYTHONPATH="$ROOT/src"
export NEPTUNE_TORCH_EXTENSIONS_DIR="$ROOT/.local/cache/torch_extensions"
export MPLCONFIGDIR="$ROOT/.local/cache/matplotlib"
export UNITYPSF_RAW_TIFF="$RAW_TIFF"
mkdir -p "$MPLCONFIGDIR"
MATERIALIZE_ARGS=(
  --base configs/base/microtube.yaml
  --override configs/overrides/standard_roi_gamma.yaml
  --output "$RESOLVED"
  --set "train.batch_size=$BATCH_SIZE"
  --set "train.online_generation.steps_per_epoch=$STEPS_PER_EPOCH"
)
if [[ -n "$PROFILE_EPOCHS" ]]; then
  MATERIALIZE_ARGS+=(--set "train.epochs=$PROFILE_EPOCHS")
fi
if [[ -n "$PROFILE_MAX_BATCHES" ]]; then
  MATERIALIZE_ARGS+=(--set "train.max_batches=$PROFILE_MAX_BATCHES")
fi
"$PY" -m unity_psf.config.materialize "${MATERIALIZE_ARGS[@]}"
if [[ "$PROFILE_TIMING" == "1" ]]; then
  export UNITY_V04_PROFILE_TIMING=1
  export UNITY_V04_PROFILE_SYNC_CUDA=1
fi
"$PY" -u -m unity_psf.training.run_high_fidelity --config "$RESOLVED" --run-root "$RUN_ROOT" --run-name "$RUN_NAME" --seed 0
