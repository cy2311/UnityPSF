from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
setup(
    name="neptune_v03_cuda_gemm_v1",
    ext_modules=[
        CUDAExtension(
            name="neptune_v03_cuda_gemm_v1",
            sources=[str(ROOT / "bindings.cpp"), str(ROOT / "kernels.cu")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
