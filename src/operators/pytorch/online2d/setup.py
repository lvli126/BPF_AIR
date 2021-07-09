from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="online2d", 
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "online2d",[
                "online2d_cuda_kernel.cu",
                "online2d_cuda.cpp",
            ]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    } 

)