#!/usr/bin/env bash
# GPU smoke test for the PyTorch C++/CUDA vector PSF extension.
# Outputs are intentionally kept under results/ for reproducibility.
#SBATCH --job-name=dh_cuda_ext_smoke
#SBATCH --partition=cpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/guest/Others/main/race/logs/slurm/dh_cuda_ext_smoke-%j.out
#SBATCH --error=/home/guest/Others/main/race/logs/slurm/dh_cuda_ext_smoke-%j.err

set -euo pipefail
R=/home/guest/Others/main/race
P=/home/guest/anaconda3/bin/python
O="$R/results/cuda_extension_smoke_20260907"
mkdir -p "$O" "$R/.local/cache/torch_extensions" "$R/.local/cache/pycache"
export PYTHONPATH="$R/neptune_v0.3:$R/neptune_v0.3/src"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$R/.local/cache/pycache"
export NEPTUNE_TORCH_EXTENSIONS_DIR="$R/.local/cache/torch_extensions"

"$P" - "$O" <<'PY'
from pathlib import Path
import json
import time
import numpy as np
import torch

from neptune_v03.localization.cuda_gemm_renderer import CudaGemmRenderer
from neptune_v03.localization.simulator import LocalizationSimulatorConfig, simulate_localization_batch
from neptune_v03.localization.online import OnlineBatchProviderConfig, build_online_batch_provider
from neptune_v03.optics.vector_psf import VectorPSFParams, build_vector_psf_context, noll_to_nm, render_vector_psf_bank

out = Path(__import__('sys').argv[1])
assert torch.cuda.is_available(), 'CUDA is required for this smoke test'
device = torch.device('cuda')
noll = [1, 4, 5, 6, 7, 8]
ctx = build_vector_psf_context(
    NA=1.46, wavelength_nm=642.0,
    pixel_size_nm_x=125.21189705655723, pixel_size_nm_y=125.21189705655723,
    noll_indices=noll,
    params=VectorPSFParams(npupil=64, psf_size=21, refmed=1.518, refcov=1.518,
                           refimm=1.518, objstage0=0.0, otf_rescale_xy=(0.0, 0.0),
                           zemit0=0.0, batch_size=128), device=device)
renderer = CudaGemmRenderer(ctx)
g = torch.Generator(device='cpu').manual_seed(20260907)
n = 8
coeffs = (torch.randn((n, len(noll)), generator=g, dtype=torch.float32) * 0.08).to(device)
z_m = ((torch.rand((n,), generator=g, dtype=torch.float32) - 0.5) * 1e-6).to(device)
carrier = torch.exp(1j * (torch.randn((64, 64), generator=g, dtype=torch.float32) * 0.3)).to(device=device, dtype=torch.complex64)

t0 = time.perf_counter()
got = renderer.render(coeffs, z_m, carrier)
torch.cuda.synchronize()
compile_and_first_call_s = time.perf_counter() - t0
t1 = time.perf_counter()
got2 = renderer.render(coeffs, z_m, carrier)
torch.cuda.synchronize()
steady_s = time.perf_counter() - t1
ref = render_vector_psf_bank(ctx, coeffs, z_m, pupil_carrier_complex=carrier, out_size=21, batch_size=n, return_torch=True)
delta = (got - ref).abs()
got_np = got.detach().cpu().numpy()
ref_np = ref.detach().cpu().numpy()
frame_indices = torch.tensor([0, 0, 1, 1, 1, 0, 0, 1], device=device, dtype=torch.long)
center_x = torch.tensor([10, 10, 5, 8, 12, 3, 14, 7], device=device, dtype=torch.long)
center_y = torch.tensor([10, 10, 5, 8, 12, 3, 14, 7], device=device, dtype=torch.long)
reduced = renderer.reduce_frames(got, frame_indices=frame_indices, center_x=center_x, center_y=center_y,
                                 frame_count=2, height=21, width=21, background=110.0)
reduced_ref = torch.full((2, 21, 21), 110.0, device=device)
for i in range(n):
    for py in range(21):
        for px in range(21):
            yy = int(center_y[i]) + py - 10
            xx = int(center_x[i]) + px - 10
            if 0 <= yy < 21 and 0 <= xx < 21:
                reduced_ref[int(frame_indices[i]), yy, xx] += got[i, py, px]
reduction_delta = (reduced - reduced_ref).abs()
flat_g = got.reshape(n, -1)
flat_r = ref.reshape(n, -1)
ncc = []
for i in range(n):
    a, b = flat_g[i] - flat_g[i].mean(), flat_r[i] - flat_r[i].mean()
    ncc.append(float((a*b).sum().item() / ((a.square().sum().sqrt() * b.square().sum().sqrt()).item() + 1e-12)))
metrics = {
    'cuda_available': True,
    'gpu': torch.cuda.get_device_name(0),
    'shape': list(got.shape),
    'max_abs': float(delta.max().item()),
    'mae': float(delta.mean().item()),
    'rmse': float(delta.square().mean().sqrt().item()),
    'ncc_min': min(ncc),
    'ncc_mean': float(np.mean(ncc)),
    'compile_and_first_call_s': compile_and_first_call_s,
    'steady_call_s': steady_s,
    'steady_emitters_per_s': n / steady_s,
    'reduction_max_abs': float(reduction_delta.max().item()),
    'reduction_rmse': float(reduction_delta.square().mean().sqrt().item()),
}
maps = torch.zeros((len(noll), 16, 16), dtype=torch.float32)
sim_cfg = LocalizationSimulatorConfig(
    batch_size=2, frames_per_sample=3, height=16, width=16, emitters_per_sample=3,
    seed=20260907, photons=20000.0, background=110.0, psf_type='vector',
    pixel_size_nm_x=125.21189705655723, pixel_size_nm_y=125.21189705655723,
    wavelength_nm=642.0, na=1.46, npupil=64, vector_psf_size=21, vector_batch_size=32,
    refmed=1.518, refcov=1.518, refimm=1.518, z_range=(-0.5, 0.5),
    coeff_maps_nm=maps, coeff_mode_order=tuple(noll_to_nm(j) for j in noll),
    pupil_carrier_complex=carrier, output_device='renderer', simulation_backend='cuda_gemm',
)
integration = simulate_localization_batch(sim_cfg, epoch=0, step=0, source='cuda_extension_smoke')
assert integration.model_input.is_cuda
metrics['integration_shape'] = list(integration.model_input.shape)
metrics['integration_cuda'] = bool(integration.model_input.is_cuda)
online_cfg = OnlineBatchProviderConfig(
    batch_size=2, channels=3, height=16, width=16, emitters_per_sample=3,
    seed=20260907, steps_per_epoch=1, simulation_backend='cuda_gemm',
    pixel_size_nm_x=125.21189705655723, pixel_size_nm_y=125.21189705655723,
    wavelength_nm=642.0, na=1.46, refmed=1.518, refcov=1.518, refimm=1.518,
    npupil=64, vector_psf_size=21, vector_batch_size=32, z_range=(-0.5, 0.5),
    simulation_output_device='renderer', domain_count=1,
)
online_batch = build_online_batch_provider(online_cfg)(0)[0]
assert online_batch.inputs.model_input.is_cuda
metrics['online_provider_shape'] = list(online_batch.inputs.model_input.shape)
metrics['online_provider_cuda'] = bool(online_batch.inputs.model_input.is_cuda)
(out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
np.save(out / 'extension_psf.npy', got_np)
np.save(out / 'reference_psf.npy', ref_np)
print(json.dumps(metrics, indent=2))
PY
