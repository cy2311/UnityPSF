#!/usr/bin/env bash
#SBATCH --job-name=bead03_zeiss2_infer
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/bead03_zeiss2_infer-%j.out
#SBATCH --error=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/bead03_zeiss2_infer-%j.err
set -euo pipefail
R=/home/guest/Others/main/race
RUN=$R/results/zeiss_bead03_exactlut_density02_training300_20260901/run
CFG=$R/results/zeiss_bead03_exactlut_density02_training300_20260901/inference_config.yaml
export NEPTUNE_V03_ROOT=$R
export RUN_DIR=$RUN
export CHECKPOINT=$RUN/checkpoints/checkpoint_best.pt
export CONFIG=$CFG
export COEFF_MAP=$R/results/zeiss_bead03_exactlut_infer_zero_zernike_maps_640.npz
export MAX_FRAMES=20000
export SIDE=left
export DOMAIN_COUNT=1
export LEFT_DOMAIN_CROP_WIDTH=512
export LEFT_DOMAIN_CROP_HEIGHT=512
export ROI_SIZE=96
export VALID_ROI_SIZE=80
export RECENTER_SCOPE=tile
export DISPLAY_MODE=fixed_imax
export DISPLAY_IMAX=2.0
export DEGRID_ENABLED=false
export RCC_DRIFT_ENABLED=false
for pair in \
  'microtubule_2026_08_20__17_02_49_2x_binmean.tif sample02' \
  'microtubule_2026_08_20__17_13_43_2x_binmean.tif sample13'; do
  set -- $pair
  export SAMPLE_TIFF=$R/datasets/training_sets/Zeiss_2/$1
  export RUN_NAME=bead03_$2_infer_recon_v7_train96_tilerecenter_nospatialdegrid
  export OUTPUT_DIR=$R/results/$RUN_NAME
  bash $R/neptune_v0.3/scripts/infer/standard_channel_infer_recon.sbatch
done
