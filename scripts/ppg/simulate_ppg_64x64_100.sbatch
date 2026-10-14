#!/usr/bin/env bash
#SBATCH --job-name=ppg_sim64
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/ppg_sim64-%j.out
#SBATCH --error=logs/ppg_sim64-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=$ROOT/.local/cache/matplotlib

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
/home/guest/anaconda3/bin/python -m ppg.simulate_ppg_frames \
  --config "$ROOT/configs/modalities/double_helix/ppg_physical.yaml" \
  --output-dir "$ROOT/output/ppg_simulation_64x64_density0p2_z600" \
  --frame-count 100 \
  --roi-size 64 \
  --density 0.2 \
  --z-range-um -0.6 0.6 \
  --seed 20260920
