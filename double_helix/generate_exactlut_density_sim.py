from pathlib import Path
import numpy as np, tifffile, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/guest/Others/main/race')
OUT = ROOT / 'results/zeiss_bead03_exactlut_density02_simulation_20260901'
(OUT / 'stacks').mkdir(parents=True, exist_ok=True)
(OUT / 'figures').mkdir(parents=True, exist_ok=True)
(OUT / 'arrays').mkdir(parents=True, exist_ok=True)
d = np.load(ROOT / 'results/zeiss_bead03_saved_complex_pupil_exact_forward_20260901/exact_psf_stack.npz')
psf, z = d['psf_unit_flux'], d['z_nm']
rng = np.random.default_rng(20260901)
frames = []
for sample in range(96):
    f = np.full((3, 96, 96), 110., np.float32)
    n = max(1, rng.poisson(0.2 * 12.0 * 12.0))
    for _ in range(n):
        x, y = rng.integers(11, 85, size=2)
        zz = float(rng.uniform(-500, 500))
        photons = float(np.clip(rng.normal(20000, 1000), 0, 31000))
        for t, dz in enumerate((-100, 0, 100)):
            k = int(np.argmin(abs(z - (zz + dz))))
            patch = psf[k] / psf[k].sum() * photons
            f[t, y-10:y+11, x-10:x+11] += patch
    frames.append(rng.poisson(np.clip(f, 0, None)).astype(np.float32))
frames = np.asarray(frames)
tifffile.imwrite(OUT / 'stacks/simulated_training_triplets_adu.tif', frames, metadata={'axes': 'TCYX'})
tifffile.imwrite(OUT / 'stacks/simulated_training_first100_full_roi_adu.tif', frames.reshape(-1, 96, 96)[:100], metadata={'axes': 'TYX'})
fig, ax = plt.subplots(4, 3, figsize=(9, 12), constrained_layout=True)
for row, i in enumerate((0, 31, 63, 95)):
    for col in range(3):
        ax[row, col].imshow(frames[i, col], cmap='gray', vmin=100, vmax=800)
        ax[row, col].set_title(f'sample {i}, t{col-1:+d}')
        ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
fig.savefig(OUT / 'figures/simulated_training_frames.png', dpi=220); plt.close(fig)
np.savez_compressed(OUT / 'arrays/exactlut_density02.npz', frames_adu=frames, z_nm=z)
print(OUT)
