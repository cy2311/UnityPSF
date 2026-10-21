#!/usr/bin/env bash
# Formal UnityPSF training with direct CUDA PSF simulation.
#SBATCH --job-name=unitypsf_cuda_gemm
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm/unitypsf_cuda_gemm-%j.out
#SBATCH --error=logs/slurm/unitypsf_cuda_gemm-%j.err

set -euo pipefail
ROOT="${UNITYPSF_ROOT:-/home/guest/Others/main/race/unity_v0.3}"
cd "$ROOT"
PY="${UNITY_V04_PYTHON:-/home/guest/anaconda3/bin/python}"
RAW_TIFF="${UNITYPSF_RAW_TIFF:-/share/home/guest/race/datasets/samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif}"
LEFT_MAP="${UNITYPSF_LEFT_MAP:-$ROOT/../results/microtube_right_benchmark_20260711/runs/neptune_v03_fd_nat/training_dual_anchor99_gamma10_conditionfix/run/artifacts/roi_bank_gamma/latest/feedback/left/coeff_maps.npz}"
RIGHT_MAP="${UNITYPSF_RIGHT_MAP:-$ROOT/../results/microtube_right_benchmark_20260711/runs/neptune_v03_fd_nat/training_dual_anchor99_gamma10_conditionfix/run/artifacts/roi_bank_gamma/latest/feedback/right/coeff_maps.npz}"
RUN_NAME="${UNITYPSF_RUN_NAME:-unitypsf_cuda_gemm_from_scratch_${SLURM_JOB_ID:-local}}"
RUN_ROOT="${UNITYPSF_RUN_ROOT:-$ROOT/output}"
RESOLVED="$ROOT/.local/tmp/unitypsf_cuda_gemm_resolved_${SLURM_JOB_ID:-local}.yaml"

mkdir -p "$ROOT/.local/tmp" "$ROOT/logs/slurm" "$RUN_ROOT"
test -f "$RAW_TIFF"
test -f "$LEFT_MAP"
test -f "$RIGHT_MAP"
export PYTHONPATH="$ROOT/src"
export NEPTUNE_TORCH_EXTENSIONS_DIR="$ROOT/.local/cache/torch_extensions"
export MPLCONFIGDIR="$ROOT/.local/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export UNITYPSF_RAW_TIFF="$RAW_TIFF"
export UNITYPSF_LEFT_MAP="$LEFT_MAP"
export UNITYPSF_RIGHT_MAP="$RIGHT_MAP"

"$PY" -m unity_psf.config.materialize \
  --base configs/base/microtube.yaml \
  --override configs/overrides/cuda_gemm_from_scratch.yaml \
  --output "$RESOLVED"
"$PY" -u -m unity_psf.training.run_high_fidelity --config "$RESOLVED" --run-root "$RUN_ROOT" --run-name "$RUN_NAME" --seed 0
