#!/usr/bin/env bash
#SBATCH --job-name=unitypsf_phys300
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_phys300-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_phys300-%j.err
set -euo pipefail
ROOT=/home/guest/Others/main/race
PROJECT=$ROOT/unity
PYTHON=/home/guest/anaconda3/bin/python
TORCHRUN=/home/guest/anaconda3/bin/torchrun
UPDATE=$PROJECT/output/unitypsf/dh_physical_update_100_gain0p03_psf21_bead01/physical_update
RAW=/share/home/guest/race/datasets/double_helix/Real_dataset_Tcell/derived/tcell_raw_native207_padded96_20260810_full75000/Stack0-14_cell_native207_padded96_final.tif
PAIRS=$ROOT/output/double_helix/references/tcell_neptune_dh_postprocess_full75000_global/proposal_pairs.npz
BASE=/share/home/guest/race_workspace/artifacts/output/double_helix/training/tcell_dh_direct_xyz_domainmatched300_density0p2_gain0p04_20260817_bead01_psf21/resolved_config.yaml
JOINT=$PROJECT/configs/experiments/unitypsf_three_modality_raw_tiff_300epoch.yaml
RUN_ID=${UNITYPSF_RUN_ID:-three-modality-physical-update-300-${SLURM_JOB_ID}}
mkdir -p "$PROJECT/.local/cache/pycache" "$PROJECT/logs/slurm"
export PYTHONPATH=$PROJECT/src:$ROOT
export PYTHONPYCACHEPREFIX=$PROJECT/.local/cache/pycache
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
$PYTHON $ROOT/scripts/double_helix/run_dh_pair_roi_gamma_update.py --pairs "$PAIRS" --raw "$RAW" --config "$BASE" --output-dir "$UPDATE" --device cuda:0 --frame-start 0 --frame-count 100 --signal-gain 0.03
test -f "$UPDATE/run/metadata/current_physical_state.json"
$PYTHON - "$PROJECT/configs/modalities/double_helix/double_helix_raw_tiff_300epoch.yaml" <<'PY'
import sys
from pathlib import Path
import torch

from unity_psf.config import load_config
from unity_psf.localization.model import build_localization_model_registry
from unity_psf.localization.runtime.config import build_localization_runtime_config
from unity_psf.training.runtime import build_runtime_batch_provider, build_runtime_loss

path = Path(sys.argv[1])
config = load_config(path)
runtime = build_localization_runtime_config(config, config_base_dir=path.parent, seed=4311)
if runtime["batch_provider"]["name"] != "dh_online_direct_xyz_batch":
    raise RuntimeError("DH formal training must use the online direct-XYZ provider")
params = dict(runtime["batch_provider"]["params"])
params["batch_size"] = 1
params["steps_per_epoch"] = 1
runtime["batch_provider"] = {"name": "dh_online_direct_xyz_batch", "params": params}
batch = next(iter(build_runtime_batch_provider(runtime)(1)))
model = build_localization_model_registry()["double_helix_expert"](runtime["model"]["params"]).cuda()
loss = build_runtime_loss(runtime).from_output(model(batch.inputs.cuda()), batch)
if not torch.isfinite(loss):
    raise RuntimeError("DH online vector/LUT preflight loss is non-finite")
print({"preflight": "passed", "provider": "dh_online_direct_xyz_batch", "batch": tuple(batch.inputs.shape), "loss": float(loss)}, flush=True)
PY
$TORCHRUN --standalone --nproc-per-node=3 -m unity_psf.cli.train_joint_expert_parallel --config "$JOINT" --run-root "$PROJECT/output/unitypsf" --run-id "$RUN_ID"
