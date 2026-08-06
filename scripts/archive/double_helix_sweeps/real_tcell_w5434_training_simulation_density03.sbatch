#!/usr/bin/env bash
#SBATCH --job-name=dh_w5434_d03
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_w5434_d03-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
REFERENCE_CONFIG=${RACE_ROOT}/results/microtube_right_benchmark_20260711/runs/neptune_v03_fd_nat/training_dual_anchor99_gamma10_conditionfix/resolved_config.yaml
W5434_ROOT=${V04_ROOT}/output/double_helix/real_tcell_deepz_mode64_weight_sweep_z2000/w5434/calibration
OUTPUT_ROOT=${V04_ROOT}/output/double_helix/real_tcell_w5434_training_sim_z2000_photon20k_density0p3

mkdir -p "${V04_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
nvidia-smi
export MPLCONFIGDIR="${RACE_ROOT}/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${V04_ROOT}/src"
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.legacy.simulate_training_wavefront \
  --reference-config "${REFERENCE_CONFIG}" \
  --source-zmap "${W5434_ROOT}/arrays/alternating_full_roi_zernike_maps_nm.npz" \
  --source-metrics "${W5434_ROOT}/metadata/metrics.json" \
  --output-dir "${OUTPUT_ROOT}" \
  --emitter-density-um2 0.3 \
  --seed 20260725 \
  --device cuda:0
