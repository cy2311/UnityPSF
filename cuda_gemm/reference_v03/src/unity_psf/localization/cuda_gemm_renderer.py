"""Production CUDA extension wrapper for the vector PSF renderer.

The extension receives the same fixed optical tensors used by ``vector_psf``.
Only per-emitter phase and propagation are performed in CUDA; no emitter is
combined with another before intensity evaluation.
"""

from __future__ import annotations

from pathlib import Path
import os
import importlib.util
import shutil
import subprocess
import sys
import threading

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


_LOAD_LOCK = threading.Lock()
_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_gemm renderer requires a CUDA runtime")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build the cuda_gemm extension")
    with _LOAD_LOCK:
        if _EXTENSION is None:
            root = Path(__file__).with_name("cuda_extension")
            cache_root = Path(os.environ.get("NEPTUNE_TORCH_EXTENSIONS_DIR", ".local/cache/torch_extensions"))
            cache_root.mkdir(parents=True, exist_ok=True)
            build_root = cache_root / "neptune_v03_cuda_gemm_v1"
            build_root.mkdir(parents=True, exist_ok=True)
            verbose = bool(int(os.environ.get("NEPTUNE_CUDA_EXT_VERBOSE", "0")))
            if shutil.which("ninja") is not None:
                _EXTENSION = load(
                    name="neptune_v03_cuda_gemm_v1",
                    sources=[str(root / "bindings.cpp"), str(root / "kernels.cu")],
                    build_directory=str(build_root),
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "--use_fast_math"],
                    with_cuda=True,
                    verbose=verbose,
                )
            else:
                subprocess.check_call(
                    [
                        sys.executable,
                        str(root / "setup.py"),
                        "build_ext",
                        "--build-lib",
                        str(build_root),
                        "--build-temp",
                        str(build_root / "temp"),
                    ],
                    cwd=str(root),
                    stdout=None,
                )
                candidates = sorted(build_root.rglob("neptune_v03_cuda_gemm_v1*.so"))
                if not candidates:
                    raise RuntimeError(f"CUDA extension build produced no shared library in {build_root}")
                spec = importlib.util.spec_from_file_location("neptune_v03_cuda_gemm_v1", candidates[-1])
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"cannot load CUDA extension {candidates[-1]}")
                module = importlib.util.module_from_spec(spec)
                sys.modules["neptune_v03_cuda_gemm_v1"] = module
                spec.loader.exec_module(module)
                _EXTENSION = module
    return _EXTENSION


def dense_czt_matrix(psf, *, axis: str, pupil: int, psf_size: int) -> torch.Tensor:
    """Materialize the exact reference CZT as a dense ``[S, P]`` matrix."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    coeff_a = getattr(psf, f"a{axis}")
    coeff_b = getattr(psf, f"b{axis}")
    coeff_d = getattr(psf, f"d{axis}")
    basis = torch.zeros((2, 3, pupil, 1, pupil), device=coeff_a.device, dtype=psf.complex_type)
    basis[0, 0, :, 0, :] = torch.eye(pupil, device=coeff_a.device, dtype=psf.complex_type)
    transformed = psf.czt_parallel(basis, coeff_a, coeff_b, coeff_d)
    # czt_parallel output is [2, 3, P, 1, S], where row P selects input basis.
    return transformed[0, 0, :, 0, :].transpose(0, 1).contiguous()


class CudaGemmRenderer:
    """Reusable fixed-contract CUDA PSF renderer."""

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        psf = ctx.psf
        self.device = ctx.device
        self.pupil = int(psf.npupil)
        self.psf_size = int(psf.psf_size)
        self.phase_basis = (psf.allzernikes / ctx.normfac[:, None, None]).contiguous()
        self.amplitude = psf.amplitude.real.to(dtype=torch.float32).contiguous()
        self.polarization = psf.polarizationvector.reshape(6, self.pupil, self.pupil).contiguous()
        self.kz_medium = psf.wavevectorzmed.real.contiguous()
        self.kz_immersion = psf.wavevectorzimm.real.contiguous()
        self.transform_y = dense_czt_matrix(psf, axis="y", pupil=self.pupil, psf_size=self.psf_size)
        self.transform_x = dense_czt_matrix(psf, axis="x", pupil=self.pupil, psf_size=self.psf_size)
        self.norm_intensity = psf.norm_intensity.detach().reshape(())
        self.objstage0_nm = float(psf.objstage0.detach().cpu().item())
        self.zemit0_nm = float(psf.zemit0.detach().cpu().item())
        self._extension = None

    def render(self, coeffs_rad: torch.Tensor, z_m: torch.Tensor, carrier: torch.Tensor | None) -> torch.Tensor:
        coeffs = coeffs_rad.to(device=self.device, dtype=torch.float32).reshape(-1, self.phase_basis.shape[0]).contiguous()
        z_nm = z_m.to(device=self.device, dtype=torch.float32).reshape(-1).mul(1.0e9).contiguous()
        if carrier is None:
            carrier_t = torch.ones((self.pupil, self.pupil), device=self.device, dtype=torch.complex64)
        else:
            carrier_t = carrier.to(device=self.device, dtype=torch.complex64).contiguous()
            carrier_t = carrier_t / carrier_t.abs().clamp_min(torch.finfo(torch.float32).eps)
        if self._extension is None:
            self._extension = _load_extension()
        patches = []
        for start in range(0, int(coeffs.shape[0]), 4096):
            stop = start + 4096
            patches.append(self._extension.render_psf_bank(
                coeffs[start:stop].contiguous(),
                z_nm[start:stop].contiguous(),
                self.phase_basis,
                self.amplitude,
                self.polarization,
                carrier_t,
                self.kz_medium,
                self.kz_immersion,
                self.transform_y,
                self.transform_x,
                self.norm_intensity,
                self.objstage0_nm,
                self.zemit0_nm,
            ))
        return torch.cat(patches, dim=0)

    def reduce_frames(
        self,
        patches: torch.Tensor,
        *,
        frame_indices: torch.Tensor,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        frame_count: int,
        height: int,
        width: int,
        background: float = 0.0,
    ) -> torch.Tensor:
        if self._extension is None:
            self._extension = _load_extension()
        return self._extension.reduce_frames(
            patches.to(device=self.device, dtype=torch.float32).contiguous(),
            frame_indices.to(device=self.device, dtype=torch.long).reshape(-1).contiguous(),
            center_x.to(device=self.device, dtype=torch.long).reshape(-1).contiguous(),
            center_y.to(device=self.device, dtype=torch.long).reshape(-1).contiguous(),
            int(frame_count), int(height), int(width), float(background),
        )
