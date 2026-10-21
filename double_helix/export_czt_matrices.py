from pathlib import Path
import numpy as np, torch
from neptune_v03.optics.vector_psf import VectorPSFParams, build_vector_psf_context
ctx=build_vector_psf_context(NA=1.46,wavelength_nm=642.0,pixel_size_nm_x=125.21189705655723,pixel_size_nm_y=125.21189705655723,noll_indices=[4],params=VectorPSFParams(npupil=128,psf_size=21,refmed=1.518,refcov=1.518,refimm=1.518,objstage0=0,otf_rescale_xy=(0,0),batch_size=1),device=torch.device('cuda:0'))
def matrix(a,b,d):
    n=128; rows=[]
    for j in range(n):
        x=torch.zeros((2,3,1,1,n),device='cuda',dtype=torch.complex64)
        x[0,0,0,0,j]=1
        y=ctx.psf.czt_parallel(x,a,b,d)[0,0,0,0]
        rows.append(y.detach().cpu().numpy())
    return np.stack(rows,axis=1).astype(np.complex64)
out=Path('results/vector128_cuda_alignment_20260905'); out.mkdir(parents=True,exist_ok=True)
axm=matrix(ctx.psf.ax,ctx.psf.bx,ctx.psf.dx); aym=matrix(ctx.psf.ay,ctx.psf.by,ctx.psf.dy)
np.savez(out/'czt_coefficients.npz',ax=ctx.psf.ax.detach().cpu().numpy(),bx=ctx.psf.bx.detach().cpu().numpy(),dx=ctx.psf.dx.detach().cpu().numpy(),ay=ctx.psf.ay.detach().cpu().numpy(),by=ctx.psf.by.detach().cpu().numpy(),dy=ctx.psf.dy.detach().cpu().numpy(),ax_matrix=axm,ay_matrix=aym)
axm.tofile(out/'czt_ax_matrix.bin'); aym.tofile(out/'czt_ay_matrix.bin')
print('saved czt coefficients',ctx.psf.ax.shape,ctx.psf.bx.shape,ctx.psf.dx.shape)
