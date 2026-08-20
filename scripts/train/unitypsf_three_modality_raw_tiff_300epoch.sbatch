#!/usr/bin/env bash
#SBATCH --job-name=unitypsf_3mod_300
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_3mod_300-%j.out
#SBATCH --error=/home/guest/Others/main/race/unity/logs/slurm/unitypsf_3mod_300-%j.err
set -euo pipefail
PROJECT_ROOT=/home/guest/Others/main/race/unity
PYTHON_BIN=${PYTHON:-/home/guest/anaconda3/bin/python}
TORCHRUN_BIN=${TORCHRUN:-/home/guest/anaconda3/bin/torchrun}
CONFIG=${UNITYPSF_JOINT_CONFIG:-$PROJECT_ROOT/configs/experiments/unitypsf_three_modality_raw_tiff_300epoch.yaml}
RUN_ID=${UNITYPSF_RUN_ID:-three-modality-raw-tiff-300-${SLURM_JOB_ID}}
RUN_ROOT=${UNITYPSF_RUN_ROOT:-$PROJECT_ROOT/output/unitypsf}
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/.local/cache/matplotlib" "$PROJECT_ROOT/.local/cache/pycache" "$PROJECT_ROOT/logs/slurm" "$RUN_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/.local/cache/pycache"
export MPLCONFIGDIR="$PROJECT_ROOT/.local/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import torch
from unity_psf.config import load_config
from unity_psf.localization.dh_raw_tiff import DHRawTiffBatchProviderConfig, build_dh_raw_tiff_batch_provider, DHDirectXYZLossAdapter
from unity_psf.localization.model import build_localization_model_registry
from unity_psf.localization.runtime_config import build_localization_runtime_config

if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
    raise RuntimeError(f"formal preflight requires 3 CUDA devices, got {torch.cuda.device_count()}")
path = Path('/home/guest/Others/main/race/unity/configs/modalities/double_helix/double_helix_raw_tiff_300epoch.yaml')
cfg = load_config(path)
runtime_cfg = build_localization_runtime_config(cfg, config_base_dir=path.parent, seed=4311)
params = dict(runtime_cfg['batch_provider']['params'])
params.update(batch_size=1, steps_per_epoch=1)
provider = build_dh_raw_tiff_batch_provider(DHRawTiffBatchProviderConfig(**params))
batch = next(iter(provider(1)))
model = build_localization_model_registry()['double_helix_expert'](runtime_cfg['model']['params']).cuda(0)
out = model(batch.inputs.cuda(0))
loss = DHDirectXYZLossAdapter(runtime_cfg['loss']['params']).from_output(out, batch)
assert torch.isfinite(loss).item()
print({'preflight': 'passed', 'frames': tuple(batch.inputs.shape), 'loss': float(loss), 'gpu_count': torch.cuda.device_count()}, flush=True)
PY
"$TORCHRUN_BIN" --standalone --nproc-per-node=3 -m unity_psf.cli.train_joint_expert_parallel --config "$CONFIG" --run-root "$RUN_ROOT" --run-id "$RUN_ID"
