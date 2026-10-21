#!/bin/bash
#SBATCH --job-name=dh_z03_zernike800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/dh_z03_zernike800-%j.out

set -euo pipefail
cd /home/guest/Others/main/race
export PYTHONPATH=/home/guest/Others/main/race/neptune_v0.3:/home/guest/Others/main/race/neptune_v0.3/src

MODE_COUNT="${MODE_COUNT:?MODE_COUNT must be set to 64 or 92}"
SOURCE_TIFF=/home/guest/Others/main/race/datasets/training_sets/Zeiss/63x1.4_642.tif
WORK_TIFF=/tmp/zeiss_bead03_roi51.tif
OUTPUT_DIR=/home/guest/Others/main/race/neptune_v0.3/output/double_helix/zeiss_bead03_zernike${MODE_COUNT}_z800_20260831/calibration

/home/guest/anaconda3/bin/python -c '
import tifffile
a=tifffile.imread("'"$SOURCE_TIFF"'")
y,x=365,186
r=25
tifffile.imwrite("'"$WORK_TIFF"'", a[:, y-r:y+r+1, x-r:x+r+1])
print("cropped", a.shape, "->", (a[:, y-r:y+r+1, x-r:x+r+1]).shape)
'

/home/guest/anaconda3/bin/python -m double_helix.run_calibration \
  --calibration-tiff "$WORK_TIFF" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --mode-count "$MODE_COUNT" \
  --wavelength-nm 642 \
  --na 1.46 \
  --refractive-index 1.518 \
  --pixel-size-nm 125.21189705655723 \
  --z-step-nm 10 \
  --z-index-origin 0 \
  --npupil 128 \
  --psf-size 51 \
  --adam-steps 700 \
  --fit-z-range-nm 800 \
  --learning-rate-schedule cosine \
  --restart-count 3 \
  --poisson-loss-weight 1 \
  --ncc-loss-weight 3 \
  --lobe-geometry-loss-weight 4 \
  --z-bin-edges-nm 300 600 \
  --z-bin-weights 2 4 5 \
  --seed 20260831
