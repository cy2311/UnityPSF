#!/usr/bin/env bash
# Build independent Origami left/right Emitter2D coefficient maps from real samples.

#SBATCH --job-name=unitypsf_2d_peak_nat
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_2d_peak_nat-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_2d_peak_nat-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/guest/Others/main/race/unity}"
PYTHON_BIN="${PYTHON:-/home/guest/anaconda3/bin/python}"
RUN_ROOT="${UNITYPSF_CALIBRATION_ROOT:-$PROJECT_ROOT/output/unitypsf/calibration}"
CONFIG_ROOT="${UNITYPSF_CALIBRATION_CONFIG_ROOT:-$PROJECT_ROOT/configs/calibration}"
JOB_ID="${SLURM_JOB_ID:-local}"
CHANNEL_LOG_ROOT="$PROJECT_ROOT/logs/slurm"

cd "$PROJECT_ROOT"
mkdir -p \
  "$PROJECT_ROOT/.local/cache/matplotlib" \
  "$PROJECT_ROOT/.local/tmp" \
  "$CHANNEL_LOG_ROOT" \
  "$RUN_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PROJECT_ROOT/.local/cache/matplotlib"
export OMP_NUM_THREADS=8

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r LEFT_GPU RIGHT_GPU _ <<< "$VISIBLE_GPUS"
if [[ -z "${LEFT_GPU:-}" || -z "${RIGHT_GPU:-}" ]]; then
  echo "expected two scheduler-assigned GPUs, got CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS" >&2
  exit 2
fi

echo "job_id=$JOB_ID"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "run_root=$RUN_ROOT"
echo "left_gpu=$LEFT_GPU"
echo "right_gpu=$RIGHT_GPU"

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: calibration refuses CPU fallback"; assert torch.cuda.device_count() >= 2, f"expected two visible GPUs, got {torch.cuda.device_count()}"; print({"cuda_available": True, "device_count": torch.cuda.device_count(), "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

LEFT_LOG="$CHANNEL_LOG_ROOT/unitypsf_2d_peak_nat-left-$JOB_ID.log"
RIGHT_LOG="$CHANNEL_LOG_ROOT/unitypsf_2d_peak_nat-right-$JOB_ID.log"

CUDA_VISIBLE_DEVICES="$LEFT_GPU" "$PYTHON_BIN" -m unity_psf.peak.run_peak_nat_zmap \
  --run-root "$RUN_ROOT" \
  --run-name origami-emitter-2d-left \
  --config-json "$CONFIG_ROOT/origami_emitter_2d_left_peak_nat.json" \
  >"$LEFT_LOG" 2>&1 &
LEFT_PID=$!

CUDA_VISIBLE_DEVICES="$RIGHT_GPU" "$PYTHON_BIN" -m unity_psf.peak.run_peak_nat_zmap \
  --run-root "$RUN_ROOT" \
  --run-name origami-emitter-2d-right \
  --config-json "$CONFIG_ROOT/origami_emitter_2d_right_peak_nat.json" \
  >"$RIGHT_LOG" 2>&1 &
RIGHT_PID=$!

set +e
wait "$LEFT_PID"
LEFT_STATUS=$?
wait "$RIGHT_PID"
RIGHT_STATUS=$?
set -e

echo "left_status=$LEFT_STATUS left_log=$LEFT_LOG"
echo "right_status=$RIGHT_STATUS right_log=$RIGHT_LOG"
if [[ "$LEFT_STATUS" -ne 0 || "$RIGHT_STATUS" -ne 0 ]]; then
  exit 1
fi

LEFT_MAP="$RUN_ROOT/origami-emitter-2d-left/stages/peak/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz"
RIGHT_MAP="$RUN_ROOT/origami-emitter-2d-right/stages/peak/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz"
test -s "$LEFT_MAP"
test -s "$RIGHT_MAP"
sha256sum "$LEFT_MAP" "$RIGHT_MAP"
