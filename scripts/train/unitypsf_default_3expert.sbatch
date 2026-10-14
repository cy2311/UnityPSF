#!/usr/bin/env bash
# UnityPSF default formal training: emitter 2D + astigmatism + double helix.
#SBATCH --job-name=unitypsf_default_3expert
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/unitypsf_default_3expert-%j.out
#SBATCH --error=logs/slurm/unitypsf_default_3expert-%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON:-/home/guest/anaconda3/bin/python}"
RUN_ROOT="${UNITYPSF_RUN_ROOT:-$ROOT/output/unitypsf}"
RUN_ID="${UNITYPSF_RUN_ID:-three-modality-${SLURM_JOB_ID:-local}}"
cd "$ROOT"
mkdir -p "$ROOT/logs/slurm" "$RUN_ROOT"
export PYTHONPATH="$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

exec "$PYTHON_BIN" -m unity_psf.cli.train_selection \
  run \
  --route emitter_2d:left,right \
  --route astigmatism:left,right \
  --route double_helix:main \
  --dh-ppg-map "${UNITYPSF_DH_PPG_MAP:-$ROOT/configs/calibration/ppg/ppg_map_initial.npz}" \
  --epochs "${UNITYPSF_EPOCHS:-300}" \
  --steps-per-epoch "${UNITYPSF_STEPS_PER_EPOCH:-417}" \
  --require-cuda \
  --run-root "$RUN_ROOT" \
  --run-id "$RUN_ID"
