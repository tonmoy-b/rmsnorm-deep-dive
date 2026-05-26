from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rmsnorm_ext",
    ext_modules=[
        CUDAExtension(
            name="rmsnorm_ext._C",
            sources=["csrc/rmsnorm.cpp", "csrc/rmsnorm_cuda.cu"],
            extra_compile_args={
                "cxx":  ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math",
                         "-gencode=arch=compute_80,code=sm_80",  # A100
                         "-gencode=arch=compute_89,code=sm_89"], # L4/4090
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["rmsnorm"],
)