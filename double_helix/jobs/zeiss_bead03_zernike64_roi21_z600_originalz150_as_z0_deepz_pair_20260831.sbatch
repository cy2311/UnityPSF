#!/bin/bash
#SBATCH --job-name=dh_z03_z600_z0150
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/guest/Others/main/race/neptune_v0.3/logs/slurm/dh_z03_z600_z0150-%j.out

set -euo pipefail
cd /home/guest/Others/main/race
export PYTHONPATH=/home/guest/Others/main/race/neptune_v0.3:/home/guest/Others/main/race/neptune_v0.3/src

SOURCE_TIFF=/home/guest/Others/main/race/datasets/training_sets/Zeiss/63x1.4_642.tif
WORK_TIFF=/tmp/zeiss_bead03_roi21_z600_z0150_${SLURM_JOB_ID}.tif
WARM_GAMMA=/home/guest/Others/main/race/neptune_v0.3/output/double_helix/zeiss_bead03_zernike64_z600_roi21_frame201z0_deepz_optz_pair_20260831/calibration/arrays/gamma_coefficients.npz
OUTPUT_DIR=/home/guest/Others/main/race/neptune_v0.3/output/double_helix/zeiss_bead03_zernike64_z600_roi21_originalz150_as_z0_deepz_pair_20260831/calibration

/home/guest/anaconda3/bin/python -c '
import tifffile
a=tifffile.imread("'"$SOURCE_TIFF"'")
y,x=365,186
r=10
roi=a[155:276, y-r:y+r+1, x-r:x+r+1]
tifffile.imwrite("'"$WORK_TIFF"'", roi)
print("cropped", a.shape, "->", roi.shape)
print("new z=0: original nominal z=+150 nm, zero-based index 215, human frame 216")
print("new range -600..+600 nm: original nominal z=-450..+750 nm, zero-based indices 155..275")
print("fixed z gauge; paired deep optimization: 2000 Adam steps, 5 restarts")
'

/home/guest/anaconda3/bin/python -m double_helix.run_calibration \
  --calibration-tiff "$WORK_TIFF" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --mode-count 64 \
  --warm-start-gamma "$WARM_GAMMA" \
  --wavelength-nm 642 \
  --na 1.46 \
  --refractive-index 1.518 \
  --pixel-size-nm 125.21189705655723 \
  --z-step-nm 10 \
  --z-index-origin 0 \
  --npupil 128 \
  --psf-size 21 \
  --adam-steps 2000 \
  --fit-z-range-nm 600 \
  --deep-z-loss \
  --paired-helix-loss-weight 3 \
  --paired-angle-weight 1 \
  --paired-separation-weight 1 \
  --paired-center-weight 1 \
  --learning-rate-schedule cosine \
  --restart-count 5 \
  --poisson-loss-weight 1 \
  --ncc-loss-weight 3 \
  --lobe-geometry-loss-weight 4 \
  --z-bin-edges-nm 250 450 \
  --z-bin-weights 2 4 5 \
  --seed 20260831

rm -f "$WORK_TIFF"
