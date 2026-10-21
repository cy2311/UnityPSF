from __future__ import annotations

import torch

from neptune_v03.localization.cuda_gemm_renderer import CudaGemmRenderer
from neptune_v03.optics.vector_psf import VectorPSFParams, build_vector_psf_context


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GEMM smoke requires a visible CUDA device")

    device = torch.device("cuda:0")
    context = build_vector_psf_context(
        NA=1.4,
        wavelength_nm=660.0,
        pixel_size_nm_x=101.11,
        pixel_size_nm_y=98.83,
        noll_indices=[4, 5, 6],
        params=VectorPSFParams(
            npupil=16,
            psf_size=7,
            refmed=1.518,
            refcov=1.518,
            refimm=1.518,
            objstage0=0.0,
            otf_rescale_xy=(0.0, 0.0),
            batch_size=1,
        ),
        device=device,
    )
    renderer = CudaGemmRenderer(context)
    patches = renderer.render(
        torch.zeros((1, renderer.phase_basis.shape[0]), device=device),
        torch.zeros((1,), device=device),
        carrier=None,
    )
    frames = renderer.reduce_frames(
        patches,
        frame_indices=torch.zeros((1,), device=device, dtype=torch.long),
        center_x=torch.full((1,), 3, device=device, dtype=torch.long),
        center_y=torch.full((1,), 3, device=device, dtype=torch.long),
        frame_count=1,
        height=7,
        width=7,
    )
    torch.cuda.synchronize()
    if patches.shape != (1, 7, 7) or not torch.isfinite(patches).all() or patches.sum() <= 0:
        raise RuntimeError("CUDA GEMM renderer produced invalid PSF output")
    torch.testing.assert_close(frames.sum(), patches.sum(), rtol=1e-5, atol=1e-5)
    print(f"cuda_device={torch.cuda.get_device_name(device)}")
    print(f"patch_sum={float(patches.sum().item()):.6f}")
    print("cuda_gemm_extension_smoke=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
