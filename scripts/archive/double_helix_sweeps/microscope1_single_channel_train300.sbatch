#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_train300
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=120:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_train300-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_train300-%j.err

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
REFERENCE_CONFIG=${RACE_ROOT}/results/microtube_right_benchmark_20260711/runs/neptune_v03_fd_nat/training_dual_anchor99_gamma10_conditionfix/resolved_config.yaml
CALIBRATION_ROOT=${V04_ROOT}/output/double_helix/microscope1_shared_carrier_residual21_physical_update_first100_totalsteps1000_20260725/physical_update
CARRIER_PATH=${V04_ROOT}/output/double_helix/microscope1_shared_carrier_residual21_20260725/calibration/arrays/shared_double_helix_carrier.npz
DENSITY_UM2=${DENSITY_UM2:?DENSITY_UM2 is required}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
TRAIN_SEED=${TRAIN_SEED:-20260725}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${V04_ROOT}/output/double_helix/microscope1_neptune_single_channel_train300_first100tyx_resume_20260725}
RUN_ROOT=${EXPERIMENT_ROOT}/${RUN_TAG}
ROI_BANK_ROOT=${RUN_ROOT}/fixed_dh_roi_bank
RESOLVED_CONFIG=${RUN_ROOT}/resolved_config.yaml

mkdir -p "${V04_ROOT}/logs/slurm" "${RACE_ROOT}/.local/cache/matplotlib" "${RACE_ROOT}/.local/cache/pycache"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite training run: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint does not exist: ${RESUME_CHECKPOINT}" >&2
  exit 2
fi

export MPLCONFIGDIR=${RACE_ROOT}/.local/cache/matplotlib
export PYTHONPYCACHEPREFIX=${RACE_ROOT}/.local/cache/pycache
export PYTHONPATH=${V04_ROOT}/src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
/home/guest/anaconda3/bin/python -c 'import torch; assert torch.cuda.is_available(); print("cuda_available", True); print("gpu_name", torch.cuda.get_device_name(0))'

/home/guest/anaconda3/bin/python -m double_helix.legacy.simulate_training_wavefront \
  --reference-config "${REFERENCE_CONFIG}" \
  --source-zmap "${CALIBRATION_ROOT}/arrays/alternating_full_roi_zernike_maps_nm.npz" \
  --source-metrics "${CALIBRATION_ROOT}/metadata/manifest.json" \
  --shared-carrier "${CARRIER_PATH}" \
  --output-dir "${ROI_BANK_ROOT}" \
  --emitter-density-um2 "${DENSITY_UM2}" \
  --sample-count 96 \
  --seed "${TRAIN_SEED}" \
  --device cuda:0

/home/guest/anaconda3/bin/python -m double_helix.legacy.train_single_channel \
  --reference-config "${REFERENCE_CONFIG}" \
  --zmap "${ROI_BANK_ROOT}/arrays/residual21_zero_defocus_zernike_maps_nm.npz" \
  --carrier "${CARRIER_PATH}" \
  --roi-tiff "${ROI_BANK_ROOT}/stacks/simulated_training_first100_full_roi_adu.tif" \
  --density-um2 "${DENSITY_UM2}" \
  --seed "${TRAIN_SEED}" \
  --output-config "${RESOLVED_CONFIG}"

TRAIN_ARGS=(
  --config "${RESOLVED_CONFIG}"
  --run-root "${RUN_ROOT}"
  --run-name run
  --seed "${TRAIN_SEED}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
/home/guest/anaconda3/bin/python -u -m unity_psf.training.run_high_fidelity "${TRAIN_ARGS[@]}"
