from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='points_query',
    ext_modules=[CUDAExtension('points_query', 
                               ['points_query.cu'])],
    cmdclass={'build_ext': BuildExtension}
)