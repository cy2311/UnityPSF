from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from double_helix.vector_model import DoubleHelixVectorPSF
root=Path('/home/guest/Others/main/race'); src=root/'results/zeiss_bead03_direct_s3_staged_savedpupil_z500_roi21_gpu_20260901'; out=root/'results/zeiss_bead03_saved_complex_pupil_exact_forward_20260901'; out.mkdir(parents=True,exist_ok=True)
with np.load(src/'arrays/bead_complex_pupil.npz',allow_pickle=False) as d: pupil=torch.from_numpy(np.asarray(d['complex_pupil'],np.complex64))
with np.load(src/'arrays/residual_gamma_observations.npz',allow_pickle=False) as d: gamma=torch.from_numpy(np.asarray(d['gamma_nm'][0],np.float32)); modes=tuple(tuple(int(v) for v in row) for row in np.asarray(d['mode_order']))
model=DoubleHelixVectorPSF(mode_order=modes,na=1.46,wavelength_nm=642.0,pixel_size_nm=125.21189705655723,refractive_index=1.518,npupil=128,psf_size=21,device='cuda')
z=torch.arange(-500,501,10,dtype=torch.float32,device='cuda')
with torch.no_grad(): psf=model.render(coefficients_nm=torch.zeros((z.numel(),len(modes)),device='cuda'),z_nm=z,carrier_complex=pupil.to('cuda'))
arr=psf.cpu().numpy().astype(np.float32); np.savez_compressed(out/'exact_psf_stack.npz',psf_unit_flux=arr,z_nm=z.cpu().numpy(),complex_pupil=pupil.numpy())
sel=np.linspace(0,len(z)-1,9,dtype=int); fig,ax=plt.subplots(1,9,figsize=(15,2),constrained_layout=True); v=float(arr.max())
for j,i in enumerate(sel): ax[j].imshow(arr[i],cmap='magma',vmin=0,vmax=v); ax[j].set_title(f'z={z[i].item():.0f}',fontsize=8); ax[j].set_xticks([]); ax[j].set_yticks([])
fig.savefig(out/'exact_saved_pupil_z_slices.png',dpi=250); plt.close(fig)
xz=arr.max(axis=1); yz=arr.max(axis=2); fig,ax=plt.subplots(1,2,figsize=(9,4),constrained_layout=True); extent=(-21*125.21189705655723/2000,21*125.21189705655723/2000,-500,500)
ax[0].imshow(xz,origin='lower',aspect='auto',extent=extent,cmap='magma'); ax[0].set_title('Exact pupil XZ projection'); ax[0].set_xlabel('x (um)'); ax[0].set_ylabel('z (nm)')
ax[1].imshow(yz,origin='lower',aspect='auto',extent=extent,cmap='magma'); ax[1].set_title('Exact pupil YZ projection'); ax[1].set_xlabel('y (um)'); ax[1].set_ylabel('z (nm)')
fig.savefig(out/'exact_saved_pupil_projection.png',dpi=250); plt.close(fig); print(out)
