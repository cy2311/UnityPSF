#!/usr/bin/env bash
#SBATCH --job-name=ppg_anchor
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/ppg_anchor-%j.out
#SBATCH --error=logs/ppg_anchor-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${UNITYPSF_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=$ROOT/.local/cache/matplotlib

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
/home/guest/anaconda3/bin/python -m ppg.sweep_anchor_strength \
  --output "$ROOT/output/ppg_anchor_strength_sweep_0p2_0p5" \
  --strengths 0.1 0.2 0.3 0.4 0.5
