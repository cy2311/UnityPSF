#!/usr/bin/env bash
#SBATCH --job-name=unitypsf_3mod_gpu_smoke
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/unitypsf_3mod_gpu_smoke-%j.out
#SBATCH --error=logs/slurm/unitypsf_3mod_gpu_smoke-%j.err
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${UNITYPSF_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
PYTHON_BIN="${PYTHON:-/home/guest/anaconda3/bin/python}"
RUN_ROOT="${UNITYPSF_RUN_ROOT:-$PROJECT_ROOT/output/unitypsf}"
RUN_ID="${UNITYPSF_RUN_ID:-three-modality-gpu-smoke-${SLURM_JOB_ID:-local}}"
ACTION="${UNITYPSF_ACTION:-smoke}"
EPOCHS="${UNITYPSF_EPOCHS:-1}"
STEPS_PER_EPOCH="${UNITYPSF_STEPS_PER_EPOCH:-8}"
DH_PPG_MAP="${UNITYPSF_DH_PPG_MAP:-$PROJECT_ROOT/configs/calibration/ppg/ppg_map_initial.npz}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs/slurm" "$RUN_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export RUN_ROOT RUN_ID
mkdir -p "$RUN_ROOT/$RUN_ID/metadata"
"$PYTHON_BIN" - <<'PY'
import json, os, subprocess, torch
payload = {
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "driver_query": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines(),
}
receipt_name = f"gpu_environment-{os.environ.get('SLURM_JOB_ID', 'local')}.json"
with open(os.path.join(os.environ["RUN_ROOT"], os.environ["RUN_ID"], "metadata", receipt_name), "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY

exec "$PYTHON_BIN" -m unity_psf.cli.train_selection \
  "$ACTION" \
  --route emitter_2d:left,right \
  --route astigmatism:left,right \
  --route double_helix:main \
  --dh-ppg-map "$DH_PPG_MAP" \
  --epochs "$EPOCHS" \
  --steps-per-epoch "$STEPS_PER_EPOCH" \
  --batch-size 1 \
  --require-cuda \
  --run-root "$RUN_ROOT" \
  --run-id "$RUN_ID"
