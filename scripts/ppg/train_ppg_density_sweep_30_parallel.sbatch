#!/usr/bin/env bash
#SBATCH --job-name=ppg_density30
#SBATCH --partition=cpu1
# Slurm on this node exposes the physical GPUs to a gpu:1 allocation;
# the launcher below binds the three child processes explicitly.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=144G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/ppg_density30-%j.out
#SBATCH --error=logs/ppg_density30-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1

run_one() {
  local gpu="$1" density="$2" run_id="$3"
  CUDA_VISIBLE_DEVICES="$gpu" \
    /home/guest/anaconda3/bin/python -m ppg.train_dh_physical \
      --config "$ROOT/configs/modalities/double_helix/ppg_physical.yaml" \
      --run-root "$ROOT/output/ppg_unitypsf" \
      --run-id "$run_id" \
      --device cuda:0 \
      --epochs 30 \
      --density "$density" \
      > "$ROOT/logs/ppg_density30-${density//./p}-%j.out" 2> "$ROOT/logs/ppg_density30-${density//./p}-%j.err" &
  echo $!
}

run_one 0 0.2 ppg-dh-gmm-30-lunar-sparse-photonrange-density02-20260922
run_one 1 0.5 ppg-dh-gmm-30-lunar-sparse-photonrange-density05-20260922
run_one 2 1.0 ppg-dh-gmm-30-lunar-sparse-photonrange-density10-20260922
wait
