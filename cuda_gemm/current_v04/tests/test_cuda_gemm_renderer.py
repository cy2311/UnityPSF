from pathlib import Path

import pytest
import torch

import neptune_v03.localization.cuda_gemm_renderer as cuda_gemm_renderer


def test_render_splits_large_emitter_banks_into_bounded_cuda_calls(monkeypatch) -> None:
    class FakeExtension:
        def __init__(self) -> None:
            self.batch_sizes = []

        def render_psf_bank(self, coeffs, z_nm, *_args):
            self.batch_sizes.append(int(coeffs.shape[0]))
            return torch.zeros((int(coeffs.shape[0]), 3, 3), dtype=torch.float32)

    extension = FakeExtension()
    monkeypatch.setattr(cuda_gemm_renderer, "_load_extension", lambda: extension)

    renderer = cuda_gemm_renderer.CudaGemmRenderer.__new__(cuda_gemm_renderer.CudaGemmRenderer)
    renderer.device = torch.device("cpu")
    renderer.pupil = 2
    renderer.phase_basis = torch.zeros((4, 2, 2))
    renderer.amplitude = torch.zeros((2, 2))
    renderer.polarization = torch.zeros((6, 2, 2))
    renderer.kz_medium = torch.zeros((2, 2))
    renderer.kz_immersion = torch.zeros((2, 2))
    renderer.transform_y = torch.zeros((3, 2))
    renderer.transform_x = torch.zeros((3, 2))
    renderer.norm_intensity = torch.tensor(1.0)
    renderer.objstage0_nm = 0.0
    renderer.zemit0_nm = 0.0
    renderer._extension = None

    patches = renderer.render(torch.zeros((4100, 4)), torch.zeros(4100), carrier=None)

    assert extension.batch_sizes == [4096, 4]
    assert patches.shape == (4100, 3, 3)


def test_fallback_extension_build_uses_absolute_cache_paths(monkeypatch) -> None:
    class BuildObserved(Exception):
        pass

    observed = {}

    def capture_build(command, *, cwd, stdout):
        observed["command"] = command
        observed["cwd"] = cwd
        raise BuildObserved

    monkeypatch.setattr(cuda_gemm_renderer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda_gemm_renderer, "CUDA_HOME", Path("/usr/local/cuda"))
    monkeypatch.setattr(cuda_gemm_renderer.shutil, "which", lambda _name: None)
    monkeypatch.setattr(cuda_gemm_renderer.subprocess, "check_call", capture_build)
    monkeypatch.setattr(cuda_gemm_renderer, "_EXTENSION", None)

    with pytest.raises(BuildObserved):
        cuda_gemm_renderer._load_extension()

    command = observed["command"]
    build_lib = Path(command[command.index("--build-lib") + 1])
    build_temp = Path(command[command.index("--build-temp") + 1])
    assert build_lib.is_absolute()
    assert build_temp.is_absolute()
