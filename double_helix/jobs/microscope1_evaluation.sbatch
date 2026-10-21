#!/usr/bin/env bash
#SBATCH --job-name=dh_m1_eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/dh_m1_eval-%j.out

set -euo pipefail

RACE_ROOT=/home/guest/Others/main/race
V03_ROOT=${RACE_ROOT}/neptune_v0.3
DATASET_ROOT=${RACE_ROOT}/datasets/training_sets/double_helix/Simulated_datasets_Microscope1
GAMMA_PATH=${V03_ROOT}/output/double_helix/microscope1/field_gamma/arrays/gamma_coefficients.npz
OUTPUT_ROOT=${V03_ROOT}/output/double_helix/microscope1/evaluation

mkdir -p "${V03_ROOT}/logs/slurm" "${OUTPUT_ROOT}"
nvidia-smi
export PYTHONPATH="${V03_ROOT}:${V03_ROOT}/src"
/home/guest/anaconda3/bin/python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
/home/guest/anaconda3/bin/python -m double_helix.run_evaluation \
  --dataset-root "${DATASET_ROOT}" \
  --gamma-path "${GAMMA_PATH}" \
  --output-dir "${OUTPUT_ROOT}" \
  --device cuda \
  --npupil 128 \
  --max-frames 5000 \
  --refinement-steps 30 \
  --batch-size 64 \
  --threshold-sigma 15 \
  --minimum-ncc 0.35 \
  --lateral-tolerance-nm 500 \
  --axial-tolerance-nm 500
