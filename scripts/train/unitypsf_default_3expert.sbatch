#!/usr/bin/env bash
# UnityPSF default formal training: emitter 2D + astigmatism + double helix.
# The delegated script performs the raw-TIFF DH physical update and preflight.
#SBATCH --job-name=unitypsf_default_3expert
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_default_3expert-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_default_3expert-%j.err

set -euo pipefail
ROOT=/home/guest/Others/main/race
exec bash "$ROOT/unity/scripts/train/unitypsf_three_modality_physical_update_300epoch.sbatch"
