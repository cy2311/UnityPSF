#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_native_d03
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_native_d03-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/dh_m1_native_d03-%j.err

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V04_ROOT=${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${RACE_ROOT}/unity}}
OUTPUT_DIR=${V04_ROOT}/output/double_helix/microscope1_model_native_physical_eval_density0p3_20260727

D03_ROOT=${V04_ROOT}/output/double_helix/microscope1_neptune_single_channel_train300_first100tyx_fresh_20260725/density0p3
D05_ROOT=${V04_ROOT}/output/double_helix/microscope1_neptune_single_channel_train300_first100tyx_resume_20260725/density0p5
D10_ROOT=${V04_ROOT}/output/double_helix/microscope1_neptune_single_channel_train300_first100tyx_resume_20260725/density1p0

mkdir -p "${V04_ROOT}/logs/slurm" "${OUTPUT_DIR}" "${RACE_ROOT}/.local/cache/matplotlib" "${RACE_ROOT}/.local/cache/pycache"
if [[ -e "${OUTPUT_DIR}/summary.json" ]]; then
  echo "Refusing to overwrite completed model-native evaluation: ${OUTPUT_DIR}/summary.json" >&2
  exit 2
fi

export MPLCONFIGDIR=${RACE_ROOT}/.local/cache/matplotlib
export PYTHONPYCACHEPREFIX=${RACE_ROOT}/.local/cache/pycache
export PYTHONPATH=${V04_ROOT}/src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
/home/guest/anaconda3/bin/python -c 'import torch; assert torch.cuda.is_available(); print("cuda_available", True); print("gpu_name", torch.cuda.get_device_name(0))'

/home/guest/anaconda3/bin/python -u -m double_helix.legacy.run_z_bin_evaluation \
  --run train_density0p3 0.3 "${D03_ROOT}/resolved_config.yaml" "${D03_ROOT}/run/checkpoints/checkpoint_latest.pt" "${D03_ROOT}/run/metadata/current_physical_state.json" \
  --run train_density0p5 0.5 "${D05_ROOT}/resolved_config.yaml" "${D05_ROOT}/run/checkpoints/checkpoint_latest.pt" "${D05_ROOT}/run/metadata/current_physical_state.json" \
  --run train_density1p0 1.0 "${D10_ROOT}/resolved_config.yaml" "${D10_ROOT}/run/checkpoints/checkpoint_latest.pt" "${D10_ROOT}/run/metadata/current_physical_state.json" \
  --evaluation-density-um2 0.3 \
  --output-dir "${OUTPUT_DIR}"
