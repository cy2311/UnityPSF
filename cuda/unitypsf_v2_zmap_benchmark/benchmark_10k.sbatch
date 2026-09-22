#!/usr/bin/env bash
#SBATCH --job-name=unitypsf_v2_zmap10k
#SBATCH --partition=cpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/guest/Others/main/race/results/microtube/logs/unitypsf-v2-zmap10k-%j.out
#SBATCH --error=/home/guest/Others/main/race/results/microtube/logs/unitypsf-v2-zmap10k-%j.err
set -euo pipefail

ROOT=/home/guest/Others/main/race
PYTHON=/home/guest/anaconda3/bin/python
OUT="${OUT_OVERRIDE:-$ROOT/results/microtube/unitypsf_v2_cuda_zmap_density1_roi64_npupil64_10k}"
MAP="$ROOT/results/microtube_right_benchmark_20260711/calibration/right_fd_zmap/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz"

mkdir -p "$OUT" "$ROOT/results/microtube/logs"
export PYTHONPATH="$ROOT/neptune_v0.3:$ROOT/neptune_v0.3/src:$ROOT"
export PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR="$ROOT/.local/cache/torch_extensions"
export NEPTUNE_TORCH_EXTENSIONS_DIR="$ROOT/.local/cache/torch_extensions"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"$PYTHON" "$ROOT/scripts/benchmarks/benchmark_unitypsf_v2_cuda_zmap.py" \
  --map-npz "$MAP" \
  --output "$OUT" \
  --frames 10000 \
  --roi 64 \
  --npupil 64 \
  --psf-size 21 \
  --density 1.0 \
  --batch-size 4096 \
  --frame-block 64 \
  --seed 20260922
