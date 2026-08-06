#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_lut_pilot
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_lut_pilot-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
REFERENCE_CONFIG=${RACE_ROOT}/results/microtube_right_benchmark_20260711/runs/neptune_v03_fd_nat/training_dual_anchor99_gamma10_conditionfix/resolved_config.yaml
CALIBRATION_ROOT=${V04_ROOT}/output/double_helix/microscope1_shared_carrier_residual21_physical_update_first100_totalsteps1000_20260725/physical_update
CARRIER_PATH=${V04_ROOT}/output/double_helix/microscope1_shared_carrier_residual21_20260725/calibration/arrays/shared_double_helix_carrier.npz
OUTPUT_DIR=${V04_ROOT}/output/double_helix/microscope1_neptune_lut_physical_contract_v2_20260725/density0p4

mkdir -p "${V04_ROOT}/logs/slurm" "${RACE_ROOT}/.local/cache/matplotlib"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite pilot output: ${OUTPUT_DIR}" >&2
  exit 2
fi

nvidia-smi
export MPLCONFIGDIR="${RACE_ROOT}/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${V04_ROOT}/src"

/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.legacy.simulate_training_wavefront \
  --reference-config "${REFERENCE_CONFIG}" \
  --source-zmap "${CALIBRATION_ROOT}/arrays/alternating_full_roi_zernike_maps_nm.npz" \
  --source-metrics "${CALIBRATION_ROOT}/metadata/manifest.json" \
  --shared-carrier "${CARRIER_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --emitter-density-um2 0.4 \
  --sample-count 96 \
  --seed 20260725 \
  --device cuda:0
